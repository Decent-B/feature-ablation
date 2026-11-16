"""Stage orchestration: run ids, --resume, DONE markers, manifest, watchdog, smoke checks.

  python -m pilot.cli --check [--smoke] [--force]      Step 0 only
  python -m pilot.cli [--smoke] [--out DIR] [--resume]  the pilot (runs the check first if needed)
  python -m pilot.cli --verify-smoke --out DIR          smoke acceptance checks (used by run_pilot.sh)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .utils import (LOG, RUNS_DIR, git_commit, load_pilot_config, model_slug, read_json, set_seeds, setup_logging,
                    versions, write_json)

STAGES = ["env", "data", "gate", "cache", "bases", "select", "ablate", "analyze"]
EXIT_ABORT_MEMORY, EXIT_INVALID_HARNESS = 4, 5


def parse(argv=None):
    p = argparse.ArgumentParser(prog="pilot")
    p.add_argument("--check", action="store_true", help="Step 0 truth check only")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    p.add_argument("--out", type=str, default=None, help="run directory")
    p.add_argument("--resume", action="store_true", help="skip stages with a DONE marker")
    p.add_argument("--force", action="store_true", help="--check: override a decision made under other rules")
    p.add_argument("--verify-smoke", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dev-tiny", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _latest_run(kind: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    cands = []
    for d in RUNS_DIR.iterdir():
        rj = d / "run.json"
        if d.is_dir() and rj.exists():
            try:
                if read_json(rj).get("kind") == kind:
                    cands.append((rj.stat().st_mtime, d))
            except Exception:  # noqa: BLE001
                pass
    return max(cands)[1] if cands else None


def _resolve_run_dir(args, kind: str, model_id: str) -> Path:
    if args.out:
        d = Path(args.out)
        d = d if d.is_absolute() else Path.cwd() / d
    elif args.resume:
        d = _latest_run(kind)
        if d is None:
            raise SystemExit(f"--resume: no previous {kind} run found under {RUNS_DIR}")
    else:
        d = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{model_slug(model_id)}"
    if d.exists() and any(d.glob("*/DONE")) and not args.resume:
        raise SystemExit(f"{d} already has finished stages; pass --resume to continue it or choose a new --out")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_model(args, cfg) -> tuple[str, dict | None, str]:
    from . import check as CK

    if args.dev_tiny:
        from .devtiny import ensure_tiny_model

        return str(ensure_tiny_model()), None, "dev tiny random model (plumbing test only)"
    if args.smoke:
        return cfg["model"], None, "smoke run: Qwen3-1.7B forced; the Step 0 decision is neither read nor written"
    if cfg.get("model", "auto") != "auto":
        raise SystemExit("config/pilot.yaml `model` must be `auto`: the model is fixed by the Step 0 decision. To change "
                         "it, edit config/check.yaml and run `bash run_pilot.sh --check --force`.")
    if not CK.decision_path(False).exists():
        LOG.info("no Step 0 decision found: running the check first")
        rc = CK.run_check(smoke=False, force=False)
        if rc != 0:
            raise SystemExit(rc)
    d = CK.verify_decision_for_pilot()
    reason = (f"Step 0 {d['status']}: {d['check_model']} truth check "
              + ("passed all thresholds" if d["status"] == "PASS" else f"failed ({'; '.join(d['failed_conditions'])})")
              + f" -> {d['pilot_model']}")
    return d["pilot_model"], d, reason


def run_pilot(args) -> int:
    from .memory import MemoryWatchdog, effective_threshold_gb, stage_memory_record, torch_reset_peak

    smoke = args.smoke or args.dev_tiny
    cfg = load_pilot_config(smoke)
    if args.dev_tiny:
        from .devtiny import dev_overrides
        from .utils import deep_merge

        cfg = deep_merge(cfg, dev_overrides())
    extra = os.environ.get("PILOT_EXTRA_CONFIG")
    if extra:                              # optional YAML merged last (e.g. enable the mlp arm for a test)
        from .utils import deep_merge, load_yaml

        cfg = deep_merge(cfg, load_yaml(Path(extra)))
    setup_logging(None)
    model_id, decision, reason = _resolve_model(args, cfg)
    kind = "dev" if args.dev_tiny else ("smoke" if args.smoke else "pilot")
    run_dir = _resolve_run_dir(args, kind, model_id)
    setup_logging(run_dir / "log.txt")
    LOG.info("=" * 70)
    LOG.info("pilot run %s (%s) model=%s resume=%s", run_dir.name, kind, model_id, args.resume)
    if not (run_dir / "run.json").exists():
        write_json(run_dir / "run.json", {"kind": kind, "model": model_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (RUNS_DIR / f".last_{kind}").write_text(str(run_dir) + "\n")
    set_seeds(int(cfg["seed"]))
    thr, why = effective_threshold_gb(float(cfg["memory"]["watchdog_min_available_gb"]))
    if why != "configured":
        LOG.warning("memory watchdog threshold %s", why)
    wd = MemoryWatchdog(thr).start()

    from .pipeline import Ctx, HarnessInvalid
    from .utils import update_json

    ctx = Ctx(cfg, run_dir, model_id, smoke, wd)
    ctx.dev = bool(args.dev_tiny)
    base_manifest = {"run_id": run_dir.name, "kind": kind, "model_id": model_id, "model_reason": reason,
                     "seeds": {"seed": cfg["seed"]}, "git_commit": git_commit(), "versions": versions(),
                     "config": cfg, "watchdog": {"threshold_gb": thr, "note": why, "running": wd.running},
                     "dev_tiny_model": bool(args.dev_tiny)}
    if decision is not None:
        base_manifest["decision"] = {"status": decision["status"], "pilot_model": decision["pilot_model"],
                                     "rule_hash": decision["rule_hash"], "file_sha256": decision["_decision_file_sha256"],
                                     "failed_conditions": decision.get("failed_conditions"),
                                     "override": decision.get("override")}
    update_json(ctx.manifest_path, base_manifest)
    t_all = time.time()
    try:
        for stage in STAGES:
            sdir = run_dir / stage
            if args.resume and (sdir / "DONE").exists():
                LOG.info("[%s] already done, skipping", stage)
                continue
            LOG.info("[%s] start", stage)
            wd.reset_peak()
            torch_reset_peak()
            t0 = time.time()
            info = _run_stage(ctx, stage)
            dt = time.time() - t0
            mem = stage_memory_record(wd)
            man = ctx.manifest()
            man.setdefault("stages", {})[stage] = {"seconds": round(dt, 1), "memory": mem}
            write_json(ctx.manifest_path, man)
            sdir.mkdir(parents=True, exist_ok=True)
            write_json(sdir / "DONE", {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(dt, 1),
                                       "info": info or {}})
            LOG.info("[%s] done in %.1f s; peak memory: torch %s GB, /proc/meminfo used %s GB (min available %s GB)",
                     stage, dt, mem["torch_max_memory_allocated_gb"], mem.get("peak_mem_used_gb"),
                     mem.get("min_mem_available_gb"))
    except KeyboardInterrupt:
        if wd.tripped:
            LOG.error("stage aborted by the memory watchdog (%s). Finished stages are kept: rerun with --resume.",
                      wd.trip_reason)
            return EXIT_ABORT_MEMORY
        LOG.error("interrupted; finished stages are kept: rerun with --resume")
        return 130
    except HarnessInvalid as e:
        LOG.error("INVALID HARNESS: %s", e)
        (run_dir / "report.md").write_text(
            f"# Pilot report\n\n**Model: `{model_id}`**\n\n**DECISION (overall): INVALID HARNESS**\n\n- {e}\n\n"
            "Stop interpreting; fix the harness. See log.txt and gate/gate.json.\n")
        return EXIT_INVALID_HARNESS
    finally:
        wd.stop()
    LOG.info("all stages finished in %.1f s; report: %s", time.time() - t_all, run_dir / "report.md")
    return 0


def _run_stage(ctx, stage: str):
    from . import pipeline as PL

    if stage == "env":
        from .env_check import run_env_check

        rec = run_env_check(ctx.cfg, ctx.model_id if not ctx.dev else "Qwen/Qwen3-1.7B", allow_cpu=ctx.smoke)
        ctx.record(env=rec)
        return {"device": rec["device"], "warnings": len(rec["warnings"])}
    if stage == "data":
        return PL.stage_data(ctx)
    if stage == "gate":
        return PL.stage_gate(ctx)
    if stage == "cache":
        return PL.stage_cache(ctx)
    if stage == "bases":
        return PL.stage_bases(ctx)
    if stage == "select":
        return PL.stage_select(ctx)
    if stage == "ablate":
        from .ablate import stage_ablate

        return stage_ablate(ctx)
    if stage == "analyze":
        from .report import write_outputs

        return write_outputs(ctx)
    raise ValueError(stage)


def main(argv=None) -> int:
    args = parse(argv)
    if args.verify_smoke:
        from .smoke import verify

        setup_logging(Path(args.out) / "log.txt")
        return verify(Path(args.out), dev=args.dev_tiny)
    if args.check:
        from . import check as CK

        CK.CHECK_DIR.mkdir(parents=True, exist_ok=True)
        setup_logging(CK.CHECK_DIR / ("log_dev.txt" if args.dev_tiny else ("log_smoke.txt" if args.smoke else "log.txt")))
        return CK.run_check(smoke=args.smoke, force=args.force, dev=args.dev_tiny)
    return run_pilot(args)


if __name__ == "__main__":
    sys.exit(main())
