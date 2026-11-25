"""Shared v3 helpers: config loading with phase overrides, paths, task lists, the reference setting and the
one-factor-at-a-time cell plan (documents/pilot-v3-plan.md, Section 7).

Phases: prefetch, smoke, dev, freeze, confirm. `--dev-tiny` (developer plumbing test) merges `phases.dev_tiny` on
top of the phase overrides and swaps in tiny random models. The SAE input scaling chosen by the smoke test lives in
config/v3_sae_input_norm.yaml (frozen with the plan) and is merged in as `sae_input_norm`.
"""

from __future__ import annotations

import os
from pathlib import Path

from .utils import CONFIG_DIR, REPO_ROOT, RUNS_DIR, deep_merge, load_yaml
from .v2common import HarnessError, depth_tag, layer_for  # noqa: F401  (re-exported)

V3_CONFIG = CONFIG_DIR / "v3.yaml"
V3_NORM = CONFIG_DIR / "v3_sae_input_norm.yaml"
PREREG3_DIR = REPO_ROOT / "prereg" / "v3"
HYPOTHESES3 = PREREG3_DIR / "hypotheses.yaml"
V3_RUNS = RUNS_DIR / "v3"
TUNING3_PATH = V3_RUNS / "tuning.json"
TASKS = ("truth", "sentiment", "answerability", "toxicity", "ioi")


def load_v3_config(phase: str, dev_tiny: bool = False, root: Path | None = None) -> dict:
    root = Path(root) if root else REPO_ROOT
    base = load_yaml(root / "config" / "v3.yaml")
    phases = base.pop("phases", {}) or {}
    cfg = deep_merge(base, phases.get(phase, {}) or {})
    if dev_tiny:
        cfg = deep_merge(cfg, phases.get("dev_tiny", {}) or {})
        from .devtiny_v3 import dev_tiny_overrides

        cfg = deep_merge(cfg, dev_tiny_overrides(cfg))
    normp = root / "config" / ("v3_sae_input_norm.devtiny.yaml" if dev_tiny else "v3_sae_input_norm.yaml")
    cfg["sae_input_norm"] = load_yaml(normp) if normp.exists() else {}
    cfg["sae_input_norm_file"] = str(normp)
    extra = os.environ.get("PILOT_V3_EXTRA_CONFIG")      # developer tool: YAML merged last, recorded in the manifest
    if extra and phase != "confirm":
        cfg = deep_merge(cfg, load_yaml(Path(extra)))
        cfg["extra_config"] = str(extra)
    cfg["phase"] = phase
    cfg["dev_tiny"] = bool(dev_tiny)
    cfg.setdefault("gates_enforced", True)
    return cfg


def load_hypotheses3(root: Path | None = None) -> dict:
    return load_yaml((Path(root) / "prereg" / "v3" / "hypotheses.yaml") if root else HYPOTHESES3)


def models_in(cfg: dict) -> list[str]:
    only = cfg.get("models_only")
    return [m for m in cfg["models"] if not only or m in only]


def tasks_in(cfg: dict) -> list[str]:
    only = cfg.get("tasks_only")
    return [t for t in TASKS if not only or t in only]


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def ref(cfg: dict) -> dict:
    return dict(cfg["reference"])


def depths_for(cfg: dict, model_id: str) -> list[float]:
    """Depths a model needs: the configured list, restricted by depths_only (dev) or smoke_depths (smoke)."""
    ds = [float(d) for d in cfg["models"][model_id]["depths"]]
    only = cfg.get("depths_only") or cfg.get("smoke_depths")
    if only:
        ds = [d for d in ds if any(abs(d - float(o)) < 1e-9 for o in only)]
    return ds


def same(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-9


def eval_split(cfg: dict) -> str:
    return cfg.get("eval_split", "eval")


# ============================================================================= reference task and cells

def resolve_reference_task(cfg: dict, gates: dict) -> dict:
    """The reference task is `reference.task` unless it fails a gate on either model; then the first task in
    `reference_task_order` that passes on every model (mechanical, frozen with the plan)."""
    r = ref(cfg)
    enforced = bool(cfg.get("gates_enforced", True))
    order = [r["task"]] + [t for t in cfg["reference_task_order"] if t != r["task"]]
    models = list(gates)

    def ok(t):
        return all(t in gates[m]["tasks"] and (gates[m]["tasks"][t]["gate"]["pass"] or not enforced) for m in models)

    for t in order:
        if t in tasks_in(cfg) and ok(t):
            return {"task": t, "changed": t != r["task"], "order": order}
    return {"task": None, "changed": True, "order": order}


def task_ok(cfg: dict, gates: dict, model: str, task: str) -> bool:
    g = gates.get(model, {}).get("tasks", {}).get(task)
    return g is not None and (g["gate"]["pass"] or not cfg.get("gates_enforced", True))


def needed_layers(cfg: dict, model: str, task: str, n_layers: int, ref_task: str) -> list[int]:
    """Layers where a (model, task) needs selection statistics: every depth for the reference task on the reference
    model (depth test), the reference depth otherwise."""
    r = ref(cfg)
    if model == r["model"] and task == ref_task:
        ds = depths_for(cfg, model)
    else:
        ds = [float(r["depth"])]
    return sorted({layer_for(d, n_layers) for d in ds})


def plan_cells(cfg: dict, block: dict, ref_task: str, gates: dict) -> list[dict]:
    r = ref(cfg)
    M, D = r["model"], float(r["depth"])

    def cell(model, task, depth, strategy="mean", kind="grid", **kw):
        return {"model": model, "task": task, "depth": float(depth), "strategy": strategy, "kind": kind,
                "block": block["id"], **kw}

    kind = block["cells"]
    if kind == "reference":
        return [cell(M, ref_task, D, test="D1 D2 D3 D4 D5 D6")]
    if kind == "depth":
        return [cell(M, ref_task, d, test="D5") for d in depths_for(cfg, M) if not same(d, D)]
    if kind == "task":
        return [cell(M, t, D, test="D3") for t in tasks_in(cfg) if t != ref_task and task_ok(cfg, gates, M, t)]
    if kind == "model":
        return [cell(m, ref_task, D, test="D4") for m in models_in(cfg) if m != M and task_ok(cfg, gates, m, ref_task)]
    if kind == "sae_width":
        return [cell(M, ref_task, D, kind="sae_extra", test="D6")]
    if kind == "resample":
        return [cell(M, ref_task, D, strategy="resample", test="S2")]
    if kind == "controls":
        return [cell(M, t, D, kind="controls", test="S1 S3") for t in tasks_in(cfg) if task_ok(cfg, gates, M, t)]
    if kind == "dev":
        return [cell(M, t, D, test="dev") for t in tasks_in(cfg) if task_ok(cfg, gates, M, t)]
    raise ValueError(f"unknown cell kind {kind}")
