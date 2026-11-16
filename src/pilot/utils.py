"""Small shared helpers: config, logging, hashing, JSON I/O, stage markers.

Kept free of torch/pandas imports so the Step 0 check and the prefetcher stay light.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
RUNS_DIR = REPO_ROOT / "runs"
DATA_CACHE = REPO_ROOT / "data_cache"

LOG = logging.getLogger("pilot")


# --------------------------------------------------------------------------- config

def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_pilot_config(smoke: bool) -> dict:
    cfg = load_yaml(CONFIG_DIR / "pilot.yaml")
    if smoke:
        cfg = deep_merge(cfg, load_yaml(CONFIG_DIR / "smoke.yaml"))
    cfg["smoke"] = bool(smoke)
    return cfg


# --------------------------------------------------------------------------- hashing / json

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default, allow_nan=True)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def update_json(path: Path, updates: dict) -> dict:
    data = read_json(path) if Path(path).exists() else {}
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    write_json(path, data)
    return data


# --------------------------------------------------------------------------- logging

class _Flush(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    LOG.setLevel(level)
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = _Flush(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False
    # Route Python warnings (e.g. the sm_121 capability warning) into the log once.
    logging.captureWarnings(True)
    wlog = logging.getLogger("py.warnings")
    wlog.handlers = LOG.handlers[:]
    wlog.propagate = False
    return LOG


# --------------------------------------------------------------------------- misc

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rng_for(seed: int, *keys: Any) -> np.random.Generator:
    """Deterministic generator derived from a base seed and string/int keys."""
    h = hashlib.sha256(("|".join(str(k) for k in (seed,) + keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def stable_int(*keys: Any) -> int:
    h = hashlib.sha256(("|".join(str(k) for k in keys)).encode()).digest()
    return int.from_bytes(h[:4], "little")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0


def fmt_secs(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def fetch_url(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> Path:
    """Download one raw file (no repo clones) into the data cache; reuse if present."""
    import urllib.request

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pilot-ablation/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
            LOG.warning("download failed (%s/%s) %s: %s", attempt + 1, retries, url, e)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"could not download {url}: {last}")


def versions() -> dict:
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version()}
    for p in ("torch", "transformers", "safetensors", "huggingface_hub", "numpy", "pandas",
              "pyarrow", "scipy", "matplotlib", "pyyaml", "tokenizers", "accelerate"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def git_commit() -> str | None:
    import subprocess

    try:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        pass
    return None


def hf_snapshot(repo_id: str, allow_patterns=None, repo_type: str = "model") -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, allow_patterns=allow_patterns, repo_type=repo_type))


MODEL_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json", "*.jinja",
                  "*.model", "*.tiktoken"]


def load_run_file(path: Path):
    """Load a .pt file this pipeline wrote into its own run directory (tensors plus numpy arrays,
    hence weights_only=False). Never use this for downloaded files: the SAE loader uses
    weights_only=True."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)
