"""Memory safety for unified-memory machines (GB10): a /proc/meminfo watchdog and telemetry.

On the GB10, GPU allocations come out of the same 128 GB pool as the OS, so a runaway
allocation can freeze the machine instead of raising a CUDA OOM. The watchdog polls
MemAvailable every second and, if it falls below the threshold, interrupts the main
thread; the stage runner turns that into a clean abort (finished stages are kept for
--resume). `nvidia-smi` is never used for memory: it reports "Not Supported" here.
"""

from __future__ import annotations

import _thread
import threading
import time

from .utils import LOG


def meminfo_kb() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                out[k.strip()] = int(v.strip().split()[0])
    except OSError:
        pass
    return out


def mem_available_gb() -> float:
    return meminfo_kb().get("MemAvailable", 0) / 1024**2


def mem_total_gb() -> float:
    return meminfo_kb().get("MemTotal", 0) / 1024**2


def effective_threshold_gb(configured_gb: float, target_total_gb: float = 128.0) -> tuple[float, str]:
    """The configured threshold (16 GB) assumes the 128 GB GB10. On smaller machines (only
    used for the CPU smoke check) scale it to the same fraction of total memory."""
    total = mem_total_gb()
    if total <= 0:
        return configured_gb, "MemTotal unavailable; using configured threshold"
    if total >= 0.9 * target_total_gb:
        return configured_gb, "configured"
    frac = configured_gb / target_total_gb
    thr = round(total * frac, 2)
    return thr, (f"scaled to {frac:.1%} of MemTotal={total:.1f} GB because this machine is "
                 f"smaller than the {target_total_gb:.0f} GB target")


class MemoryWatchdog:
    def __init__(self, threshold_gb: float, interval_s: float = 1.0):
        self.threshold_gb = float(threshold_gb)
        self.interval_s = interval_s
        self.tripped = False
        self.trip_reason = ""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._min_avail = float("inf")
        self._peak_used = 0.0
        self._thread: threading.Thread | None = None
        self.enabled = True

    # ------------------------------------------------------------------ control
    def start(self) -> "MemoryWatchdog":
        self._thread = threading.Thread(target=self._run, name="mem-watchdog", daemon=True)
        self._thread.start()
        LOG.info("memory watchdog running: abort the current stage if MemAvailable < %.1f GB "
                 "(polled every %.0f s)", self.threshold_gb, self.interval_s)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _sample(self) -> float:
        info = meminfo_kb()
        avail = info.get("MemAvailable", 0) / 1024**2
        used = (info.get("MemTotal", 0) - info.get("MemAvailable", 0)) / 1024**2
        with self._lock:
            self._min_avail = min(self._min_avail, avail)
            self._peak_used = max(self._peak_used, used)
        return avail

    def reset_peak(self) -> None:
        with self._lock:
            self._min_avail = float("inf")
            self._peak_used = 0.0
        self._sample()

    def peak(self) -> dict:
        self._sample()
        with self._lock:
            return {"min_mem_available_gb": round(self._min_avail, 2) if self._min_avail < float("inf") else None,
                    "peak_mem_used_gb": round(self._peak_used, 2)}

    # ------------------------------------------------------------------ loop
    def _run(self) -> None:
        while not self._stop.is_set():
            avail = self._sample()
            if self.enabled and avail < self.threshold_gb and not self.tripped:
                self.tripped = True
                self.trip_reason = (f"MemAvailable={avail:.1f} GB fell below the watchdog threshold "
                                    f"{self.threshold_gb:.1f} GB")
                LOG.error("MEMORY WATCHDOG: %s; aborting the current stage", self.trip_reason)
                _thread.interrupt_main()
            self._stop.wait(self.interval_s)


def torch_peak_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        pass
    return None


def torch_reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def stage_memory_record(wd: MemoryWatchdog | None) -> dict:
    rec = {"torch_max_memory_allocated_gb": torch_peak_gb(),
           "meminfo_now_available_gb": round(mem_available_gb(), 2)}
    if wd is not None:
        rec.update(wd.peak())
    return rec


def wait_s(s: float) -> None:  # tiny helper for tests
    time.sleep(s)
