"""Shared v2 helpers: config loading with phase overrides, the layer rule, run paths, pairs.

Phases: prefetch, smoke, dev, freeze, confirm. `--dev-tiny` (developer plumbing test) merges
`phases.dev_tiny` on top of the phase overrides and swaps in tiny random models.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from .utils import CONFIG_DIR, REPO_ROOT, RUNS_DIR, deep_merge, load_yaml

V2_CONFIG = CONFIG_DIR / "v2.yaml"
PREREG_DIR = REPO_ROOT / "prereg"
HYPOTHESES = PREREG_DIR / "hypotheses.yaml"
V2_RUNS = RUNS_DIR / "v2"
TUNING_PATH = V2_RUNS / "tuning.json"
PHASES = ("prefetch", "smoke", "dev", "freeze", "confirm")


class HarnessError(RuntimeError):
    """A harness check failed (a bug or an unverifiable setting): stop and report."""


def load_v2_config(phase: str, dev_tiny: bool = False, root: Path | None = None) -> dict:
    base = load_yaml((root / "config" / "v2.yaml") if root else V2_CONFIG)
    phases = base.pop("phases", {}) or {}
    cfg = deep_merge(base, phases.get(phase, {}) or {})
    if dev_tiny:
        cfg = deep_merge(cfg, phases.get("dev_tiny", {}) or {})
        from .devtiny_v2 import dev_tiny_overrides

        cfg = deep_merge(cfg, dev_tiny_overrides(cfg))
    extra = os.environ.get("PILOT_V2_EXTRA_CONFIG")      # developer tool: YAML merged last, recorded in the manifest
    if extra and phase != "confirm":
        cfg = deep_merge(cfg, load_yaml(Path(extra)))
        cfg["extra_config"] = str(extra)
    cfg["phase"] = phase
    cfg["dev_tiny"] = bool(dev_tiny)
    cfg.setdefault("gates_enforced", True)
    return cfg


def load_hypotheses(root: Path | None = None) -> dict:
    return load_yaml((root / "prereg" / "hypotheses.yaml") if root else HYPOTHESES)


def layer_for(depth: float, n_layers: int) -> int:
    """L = round(depth * num_hidden_layers), half rounded up, 0-indexed decoder-layer output."""
    return int(math.floor(depth * n_layers + 0.5))


def models_in(cfg: dict) -> list[str]:
    only = cfg.get("models_only")
    return [m for m in cfg["models"] if not only or m in only]


def depths_for(cfg: dict, model_id: str) -> list[float]:
    ds = [float(d) for d in cfg["models"][model_id]["depths"]]
    only = cfg.get("depths_only")
    if only:
        ds = [d for d in ds if any(abs(d - float(o)) < 1e-9 for o in only)]
    return ds


def tasks_in(cfg: dict) -> list[str]:
    allt = ["truth", "refusal", "sentiment", "answerability", "toxicity"]
    only = cfg.get("tasks_only")
    return [t for t in allt if not only or t in only]


def pair_role(cfg: dict, model_id: str, task: str) -> str | None:
    for role in ("confirmatory", "development"):
        if [model_id, task] in [list(p) for p in cfg["pairs"][role]]:
            return role
    return None


def slug(model_id: str) -> str:
    """Model ids stay logical (e.g. Qwen/Qwen3-1.7B) everywhere, also under --dev-tiny, where only the
    weights (`models.<id>.weights`) and SAE locations are swapped."""
    return model_id.split("/")[-1].lower()


def depth_tag(depth: float) -> str:
    return f"{int(round(depth * 100))}pct"
