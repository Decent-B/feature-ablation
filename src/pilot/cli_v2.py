"""v2 orchestration (invoked by `bash run_pilot.sh --v2 ...`).

  --prefetch                  download every model, SAE file and dataset (--smoke runs it first)
  --smoke                     every stage on tiny data for both models + acceptance checks + projections
  --dev                       development run: Qwen3-1.7B truth + refusal, 50% depth, both sites
  --render-prereg             regenerate prereg/PREREGISTRATION.md from prereg/hypotheses.yaml
  --freeze                    hash + commit + tag the preregistration (refuses on uncommitted changes)
  --confirm [--deviation R]   confirmatory run (refuses without a valid freeze)
  --resume / --out DIR        continue a run; --dev-tiny swaps in tiny random models (plumbing test)
Runs land in runs/v2/<phase>/<run_id>/. Stages write DONE markers; --resume skips finished stages.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from .utils import LOG, git_commit, set_seeds, setup_logging, update_json, versions, write_json
from .v2common import (PREREG_DIR, V2_RUNS, HarnessError, load_v2_config, models_in, pair_role, slug,
                       tasks_in)

EXIT_ABORT_MEMORY, EXIT_HARNESS, EXIT_REFUSED = 4, 5, 2


def parse(argv=None):
    p = argparse.ArgumentParser(prog="pilot.cli_v2")
    g = p.add_mutually_exclusive_group(required=True)
    for m in ("prefetch", "smoke", "dev", "freeze", "confirm", "render-prereg", "verify-smoke"):
        g.add_argument(f"--{m}", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--deviation", type=str, default=None)
    p.add_argument("--dev-tiny", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--enforce-only", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _root() -> Path:
    return Path.cwd()


def _run_dir(args, phase: str) -> Path:
    base = V2_RUNS / phase
    if args.out:
        d = Path(args.out)
        d = d if d.is_absolute() else Path.cwd() / d
    elif args.resume:
        last = base / ".last"
        if not last.exists():
            raise SystemExit(f"--resume: no previous {phase} run under {base}")
        d = Path(last.read_text().strip())
    else:
        d = base / f"{time.strftime('%Y%m%d-%H%M%S')}-{phase}{'-devtiny' if args.dev_tiny else ''}"
    if d.exists() and any(d.rglob("DONE")) and not args.resume:
        raise SystemExit(f"{d} already has finished stages; pass --resume or choose a new --out")
    d.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    (base / ".last").write_text(str(d) + "\n")
    return d


def _stage(run, name: str, marker: Path, fn, resume: bool):
    from .memory import stage_memory_record, torch_reset_peak

    if resume and (marker / "DONE").exists():
        LOG.info("[%s] already done, skipping", name)
        return None
    LOG.info("[%s] start", name)
    if run.wd is not None:
        run.wd.reset_peak()
    torch_reset_peak()
    t0 = time.time()
    info = fn()
    dt = time.time() - t0
    mem = stage_memory_record(run.wd)
    man = run.manifest()
    man.setdefault("stages", {})[name] = {"seconds": round(dt, 1), "memory": mem}
    write_json(run.manifest_path, man)
    marker.mkdir(parents=True, exist_ok=True)
    write_json(marker / "DONE", {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(dt, 1),
                                 "info": info or {}})
    LOG.info("[%s] done in %.1f s; peak torch %s GB, /proc/meminfo used %s GB", name, dt,
             mem["torch_max_memory_allocated_gb"], mem.get("peak_mem_used_gb"))
    return info


def _prep_tasks(run, m: str) -> list[str]:
    from .ablate_v2 import included_pairs

    tasks = {p["task"] for p in included_pairs(run) if p["model"] == m}   # cross-task sources are included pairs
    return [t for t in tasks_in(run.cfg) if t in tasks]


def run_phase(phase: str, args) -> int:
    from .engine_v2 import RunCtx
    from .memory import MemoryWatchdog, effective_threshold_gb

    cfg = load_v2_config(phase, dev_tiny=args.dev_tiny)
    root = _root()
    deviation = None
    if phase == "confirm":
        from . import prereg

        ok, v, deviation = prereg.enforce(root, args.deviation)
        if not ok:
            print(f"[confirm] REFUSED: {v['reason']}" + (f"; changed: {', '.join(v['changed'])}" if v["changed"] else "")
                  + ("\nA change after the freeze needs --deviation \"<reason>\"." if v.get("record") else ""))
            return EXIT_REFUSED
        print(f"[confirm] preregistration check: {v['reason']}" + (f"; DEVIATION recorded: {deviation['reason']}" if deviation else ""))
        if args.enforce_only:
            return 0
    if args.dev_tiny:
        from .devtiny_v2 import ensure_tiny_models

        setup_logging(None)
        ensure_tiny_models()
    if phase == "smoke" and not args.resume:
        from .prefetch_v2 import prefetch

        setup_logging(None)
        LOG.info("[prefetch] downloading everything the full run needs (runs before the smoke stages)")
        t0 = time.time()
        rc = prefetch(load_v2_config("prefetch", dev_tiny=args.dev_tiny), args.dev_tiny)
        if rc != 0:
            LOG.error("prefetch incomplete; see the log above")
            return 1
        prefetch_s = time.time() - t0
    else:
        prefetch_s = None
    run_dir = _run_dir(args, phase)
    setup_logging(run_dir / "log.txt")
    LOG.info("=" * 70)
    LOG.info("v2 %s run %s (dev_tiny=%s, resume=%s)", phase, run_dir.name, args.dev_tiny, args.resume)
    if not (run_dir / "run.json").exists():
        write_json(run_dir / "run.json", {"kind": f"v2-{phase}", "dev_tiny": args.dev_tiny,
                                          "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    snap = run_dir / "prereg_snapshot"
    if not snap.exists():
        snap.mkdir()
        for f in ("hypotheses.yaml", "PREREGISTRATION.md", "FROZEN.json"):
            if (PREREG_DIR / f).exists():
                shutil.copy2(PREREG_DIR / f, snap / f)
    set_seeds(int(cfg["seed"]))
    thr, why = effective_threshold_gb(float(cfg["memory"]["watchdog_min_available_gb"]))
    wd = MemoryWatchdog(thr).start()
    run = RunCtx(cfg, run_dir, wd)
    base = {"run_id": run_dir.name, "phase": phase, "dev_tiny": args.dev_tiny, "git_commit": git_commit(),
            "versions": versions(), "config": cfg, "watchdog": {"threshold_gb": thr, "note": why, "running": wd.running}}
    if prefetch_s is not None:
        base["prefetch_seconds"] = round(prefetch_s, 1)
    if deviation:
        man0 = run.manifest()
        base["deviations"] = (man0.get("deviations") or []) + [deviation]
    update_json(run.manifest_path, base)
    t_all = time.time()
    try:
        _run_stages(run, args)
    except KeyboardInterrupt:
        if wd.tripped:
            LOG.error("stage aborted by the memory watchdog (%s); rerun with --resume", wd.trip_reason)
            return EXIT_ABORT_MEMORY
        LOG.error("interrupted; finished stages are kept: rerun with --resume")
        return 130
    except HarnessError as e:
        LOG.error("HARNESS ERROR: %s", e)
        (run_dir / "HARNESS_ERROR.txt").write_text(str(e) + "\n")
        return EXIT_HARNESS
    finally:
        wd.stop()
    update_json(run.manifest_path, {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    LOG.info("all v2 %s stages finished in %.1f s: %s", phase, time.time() - t_all, run_dir)
    return 0


def _run_stages(run, args):
    from .data_v2 import build_text
    from .engine_v2 import MCtx

    cfg = run.cfg
    rd = run.run_dir
    res = args.resume

    def env():
        from .env_check import run_env_check

        rec = run_env_check(cfg, "Qwen/Qwen3-1.7B", allow_cpu=run.phase == "smoke" or run.dev_tiny)
        run.record(env=rec)
        return {"device": rec["device"]}
    _stage(run, "env", rd / "env", env, res)

    def data():
        toks = [run.mctx(m).tok for m in models_in(cfg)]
        s = build_text(cfg, toks, rd / "data")
        write_json(rd / "data" / "data_summary.json", s)
        return {"n_splits": len(s["splits"])}
    _stage(run, "data", rd / "data", data, res)

    from .gate_v2 import inclusion, stage_gate

    gates = {}
    for m in models_in(cfg):
        mc = run.mctx(m)
        _stage(run, f"gate:{mc.slug}", mc.dir / "gate", lambda: stage_gate(mc), res)
        from .utils import read_json

        gates[m] = read_json(mc.dir / "gate" / "gate.json")
    if not (res and (rd / "inclusion.json").exists()):
        inc = inclusion(cfg, gates)
        write_json(rd / "inclusion.json", inc)
        LOG.info("inclusion: %d pairs %s; excluded %s; substitutions %s", len(inc["pairs"]),
                 [(p["model"].split("/")[-1], p["task"]) for p in inc["pairs"]], inc["excluded"], inc["substitutions"])
    if run.phase == "smoke":
        from .budget_v2 import save_tuning, tune_model

        for m in models_in(cfg):
            mc = run.mctx(m)

            def tune(mc=mc):
                rec = tune_model(mc)
                rec["n_layers"] = mc.n_layers
                mt = {t: float(mc.tokenized(t, "eval").n_tokens.mean()) for t in tasks_in(cfg) if t != "refusal"}
                if "refusal" in tasks_in(cfg):
                    mt["refusal"] = float(mc.tokenized("refusal", "eval_harmless").n_tokens.mean())
                mt["generic"] = float(mc.tokenized("generic", "cost").n_tokens.mean())
                rec["mean_tokens"] = mt
                save_tuning(("dev_tiny:" if run.dev_tiny else "") + m, rec)
                return {k: rec[k] for k in ("max_batch_tokens", "grad_batch_tokens", "gen_batch_size")}
            _stage(run, f"tune:{mc.slug}", mc.dir / "tune", tune, res)
    from .prep_v2 import stage_prep

    for m in models_in(cfg):
        mc = run.mctx(m)
        tasks = _prep_tasks(run, m)
        _stage(run, f"prep:{mc.slug}", mc.dir / "prep", lambda: stage_prep(mc, tasks), res)
    if run.phase == "smoke":
        from .smoke_v2 import model_checks

        def checks():
            out = {m: model_checks(run, run.mctx(m)) for m in models_in(cfg)}
            write_json(rd / "checks" / "checks.json", out)
            return {m: all(v.get("pass", True) for v in c.values() if isinstance(v, dict)) for m, c in out.items()}
        _stage(run, "checks", rd / "checks", checks, res)
    from .ablate_v2 import stage_ablate

    _stage(run, "ablate", rd / "ablate", lambda: stage_ablate(run), res)
    if run.phase == "dev":
        from .ablate_v2 import agreement_check

        _stage(run, "agreement_32_64", rd / "agreement", lambda: agreement_check(run), res)
    from .report_v2 import stage_analyze

    _stage(run, "analyze", rd / "analyze", lambda: stage_analyze(run), res)


def main(argv=None) -> int:
    args = parse(argv)
    root = _root()
    if args.render_prereg:
        from .prereg import write_render

        p = write_render(root)
        print(f"[prereg] wrote {p}")
        return 0
    if args.freeze:
        from .prereg import freeze

        return freeze(root)
    if args.prefetch:
        from .prefetch_v2 import prefetch

        setup_logging(None)
        if args.dev_tiny:
            from .devtiny_v2 import ensure_tiny_models

            ensure_tiny_models()
        return prefetch(load_v2_config("prefetch", dev_tiny=args.dev_tiny), args.dev_tiny)
    if args.verify_smoke:
        from .smoke_v2 import verify

        setup_logging(Path(args.out) / "log.txt")
        return verify(Path(args.out), args.dev_tiny)
    phase = "smoke" if args.smoke else ("dev" if args.dev else "confirm")
    return run_phase(phase, args)


if __name__ == "__main__":
    sys.exit(main())
