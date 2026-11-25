"""Competing GPU / memory use on a shared machine (v3). Speed only: none of this changes what is computed.

On the GB10, GPU and system memory are one pool, so other users' jobs show up as lower MemAvailable. Three
mechanisms, all logged to <run>/resources.json and listed in the report:

  wait      before a stage, if MemAvailable < watchdog threshold + safety margin (+ model weights not loaded yet),
            poll every `poll_seconds` for up to `max_wait_minutes`; if memory never frees up, stop with a clear
            message (exit code 4, resumable).
  fit       before a stage, scale every batch budget (forward / backward / loss-check / generation tokens) by a
            power-of-two factor so the batch memory measured by the smoke test fits into what is free now; never
            above the back-off cap.
  back-off  if the stage is stopped by the memory watchdog or a CUDA out-of-memory error, halve the cap, drop caches,
            re-arm the watchdog and rerun the stage (finished ablation chunks are kept), up to `max_backoffs` times.
The factor and cap persist in resources.json, so a resumed run keeps the smaller batches.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

from .memory import mem_available_gb, mem_total_gb
from .utils import LOG, read_json, write_json


class ResourceError(RuntimeError):
    """Not enough free memory, even after waiting and backing off."""


def _is_oom(e: BaseException) -> bool:
    try:
        import torch

        if isinstance(e, torch.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


class Resources:
    def __init__(self, run, mem_fn=mem_available_gb, sleep_fn=time.sleep, total_gb: float | None = None):
        self.run = run
        self.cfg = run.cfg["memory"]
        self.path = Path(run.run_dir) / "resources.json"
        self.state = read_json(self.path) if self.path.exists() else {"factor": 1.0, "cap": 1.0, "events": []}
        self.mem, self.sleep = mem_fn, sleep_fn
        total = total_gb if total_gb is not None else (mem_total_gb() or 128.0)
        self.scale = min(1.0, total / 128.0)                 # the margins assume the 128 GB GB10; scale on smaller machines
        self.min_factor = float(self.cfg.get("min_batch_factor", 1 / 16))

    # ---------------------------------------------------------------- state
    @property
    def factor(self) -> float:
        return float(self.state["factor"])

    def _save(self):
        write_json(self.path, self.state)

    def _event(self, kind: str, stage: str, **kw):
        e = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "stage": stage,
             "mem_available_gb": round(float(self.mem()), 2), "factor": self.factor, "cap": float(self.state["cap"]), **kw}
        self.state["events"].append(e)
        self._save()
        return e

    def _threshold(self) -> float:
        wd = getattr(self.run, "wd", None)
        return float(wd.threshold_gb) if wd is not None else 0.0

    def _safety(self) -> float:
        return float(self.cfg.get("safety_gb", 8)) * self.scale

    def _pending_weights(self) -> float:
        """Weights of models this run has not loaded yet (they will take memory when the stage starts)."""
        out = 0.0
        for mc in getattr(self.run, "_m", {}).values():
            if getattr(mc, "_model", None) is None:
                out += float(mc.tuning.get("weights_gb") or 0.0)
        return out

    def _batch_mem(self) -> float | None:
        vals = [float(mc.tuning["batch_mem_gb"]) for mc in getattr(self.run, "_m", {}).values()
                if mc.tuning.get("batch_mem_gb")]
        return max(vals) if vals else None

    # ---------------------------------------------------------------- before a stage
    def prepare(self, stage: str):
        """Wait for free memory, then fit the batch factor to it; re-arm the watchdog."""
        need = self._threshold() + self._safety() + self._pending_weights()
        avail = float(self.mem())
        if avail < need:
            wait_max = float(self.cfg.get("max_wait_minutes", 30)) * 60
            poll = float(self.cfg.get("poll_seconds", 30))
            self._event("wait", stage, need_gb=round(need, 2), note="memory used by other jobs; waiting")
            LOG.warning("[%s] only %.1f GB free (need %.1f GB): waiting up to %.0f min for other jobs to free memory",
                        stage, avail, need, wait_max / 60)
            t0 = time.time()
            waited = 0.0
            while avail < need and waited < wait_max:
                self.sleep(poll)
                waited = time.time() - t0 if self.sleep is time.sleep else waited + poll
                avail = float(self.mem())
            if avail < need:
                self._event("gave_up", stage, need_gb=round(need, 2))
                raise ResourceError(f"{stage}: only {avail:.1f} GB free after waiting {waited / 60:.0f} min (need "
                                    f"{need:.1f} GB: watchdog floor + safety margin + model weights). Other jobs are "
                                    "using the memory; rerun when it is free (finished work is kept).")
            self._event("resumed", stage, waited_s=round(waited, 1))
        self.fit(stage, avail - need)
        wd = getattr(self.run, "wd", None)
        if wd is not None and getattr(wd, "tripped", False):
            wd.tripped, wd.trip_reason = False, ""
            LOG.info("[%s] memory watchdog re-armed", stage)

    def fit(self, stage: str, headroom_gb: float):
        """Largest power-of-two factor <= cap whose batch memory (smoke measurement x factor) fits the headroom."""
        bm = self._batch_mem()
        cap = float(self.state["cap"])
        f = cap
        if bm:
            while f > self.min_factor and bm * f > max(headroom_gb, 0.0):
                f /= 2
        f = max(f, self.min_factor)
        if abs(f - self.factor) > 1e-12:
            old = self.factor
            self.state["factor"] = f
            self._event("fit", stage, previous_factor=old, headroom_gb=round(headroom_gb, 2),
                        batch_mem_gb_at_factor_1=bm)
            LOG.warning("[%s] batch budgets x%g (was x%g) to fit %.1f GB of free memory (speed only)", stage, f, old,
                        headroom_gb)

    # ---------------------------------------------------------------- after a memory stop
    def backoff(self, stage: str, reason: str) -> bool:
        cap = float(self.state["cap"])
        if cap / 2 < self.min_factor - 1e-12:
            return False
        self.state["cap"] = cap / 2
        self.state["factor"] = min(self.factor, cap / 2)
        self._event("backoff", stage, reason=reason)
        LOG.warning("[%s] %s: halving every batch budget (now x%g) and retrying the stage (speed only)", stage, reason,
                    self.factor)
        return True

    def release(self):
        """Drop per-stage caches (prefix keys/values, SAEs) and free cached GPU memory; models stay loaded."""
        for mc in getattr(self.run, "_m", {}).values():
            getattr(mc, "_prefix", {}).clear()
            getattr(mc, "_sae", {}).clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def run_with_backoff(run, stage: str, fn):
    """Run a stage; on a memory-watchdog stop or an out-of-memory error, back off and retry (up to max_backoffs)."""
    res = run.res
    tries = int(run.cfg["memory"].get("max_backoffs", 3))
    for attempt in range(tries + 1):
        res.prepare(stage)
        try:
            return fn()
        except KeyboardInterrupt:
            wd = getattr(run, "wd", None)
            if not (wd is not None and wd.tripped):
                raise                                    # a real Ctrl-C / --stop: not ours to retry
            reason = wd.trip_reason or "memory watchdog stop"
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                raise
            reason = f"out of memory ({type(e).__name__})"
        res.release()
        if attempt == tries or not res.backoff(stage, reason):
            res._event("gave_up", stage, reason=reason)
            raise ResourceError(f"{stage}: {reason}; still failing after {attempt} back-offs (batches x{res.factor:g}). "
                                "Other jobs may be using the memory; rerun when it is free (finished work is kept).")
    return None
