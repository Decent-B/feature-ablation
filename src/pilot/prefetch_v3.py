"""`--v3 --prefetch`: download every model, SAE file and dataset the full v3 run needs, and report what is missing.

HF_TOKEN: the Gemma repos are gated. The token is read from the HF_TOKEN environment variable only (huggingface_hub
picks it up); it is never written anywhere. Without it, or if the account has not accepted the Gemma license, the
prefetch stops with a clear message. Models: config, tokenizer and safetensors of both base models. SAEs: for every
layer each model needs, the main file and the fallback / extra file (Gemma Scope 65k + 16k; Qwen-Scope TopK 100 +
TopK 50, plus their config.json). Datasets: v1 raw files plus the SST-2, SQuAD 2.0 and Civil Comments parquet files
pinned to a revision. With --dev-tiny only datasets and tokenizers are fetched.
"""

from __future__ import annotations

import os
import sys
import time

from .utils import LOG, MODEL_PATTERNS, setup_logging
from .v3common import layer_for, load_v3_config, models_in

GATED = ("google/gemma-2-2b",)
N_LAYERS = {"google/gemma-2-2b": 26, "Qwen/Qwen3-1.7B-Base": 28}


class AccessError(RuntimeError):
    pass


def check_token(repos=GATED) -> dict:
    """HF_TOKEN set and accepted for every gated repo; never prints the token."""
    if not os.environ.get("HF_TOKEN"):
        raise AccessError("HF_TOKEN is not set. The Gemma model is gated: create a read token on huggingface.co, accept "
                          "the Gemma license on https://huggingface.co/google/gemma-2-2b with the same account, then "
                          "`export HF_TOKEN=hf_...` before running.")
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    out = {}
    for r in repos:
        try:
            HfApi().auth_check(r)
            out[r] = "access ok"
        except GatedRepoError as e:
            raise AccessError(f"{r}: access denied (HTTP 401/403). Accept the license on https://huggingface.co/{r} "
                              f"with the account that owns HF_TOKEN. ({type(e).__name__})") from None
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403):
                raise AccessError(f"{r}: HTTP {code}: HF_TOKEN is invalid or lacks access to the gated repo.") from None
            raise
    return out


def _model_files(repo: str) -> list[tuple[str, int]]:
    import fnmatch

    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in info.siblings
            if any(fnmatch.fnmatch(s.rfilename, p) for p in MODEL_PATTERNS)]


def manifest(cfg: dict) -> list[dict]:
    from huggingface_hub import HfApi

    from . import sae_v3
    from .gate_v3 import fidelity_depths

    real = load_v3_config("confirm")
    out = []
    api = HfApi()
    for m in models_in(real):
        for f, s in _model_files(m):
            out.append({"kind": "model", "repo": m, "file": f, "size": s, "repo_type": "model"})
        sc = real["models"][m]["sae"]
        n = N_LAYERS.get(m)
        Ls = sorted({layer_for(d, n) for d in fidelity_depths(real, m)})
        repos = {k: sc[k] for k in ("repo", "fallback_repo") if sc.get(k)}
        sizes = {k: {s.rfilename: s.size for s in api.model_info(r, files_metadata=True).siblings}
                 for k, r in repos.items()}
        if sc["format"] == "qwenscope":
            for r in repos.values():
                out.append({"kind": "sae", "repo": r, "file": "config.json", "size": 0, "repo_type": "model"})
        fb = list(sizes["fallback_repo"]) if "fallback_repo" in sizes else None
        for L, rec in sae_v3.resolve(sc, Ls, list(sizes["repo"]), fb).items():
            for slot in ("main", "extra"):
                info = rec[slot]
                if info is None:
                    continue
                k = info["repo_key"]
                out.append({"kind": "sae", "repo": repos[k], "file": info["file"], "size": sizes[k].get(info["file"], 0),
                            "repo_type": "model", "layer": L, "slot": slot})
    for t in ("sentiment", "answerability", "toxicity"):
        d = cfg["data"][t]
        for f in d["files"].values():
            out.append({"kind": "dataset", "repo": d["repo"], "file": f, "size": 0, "repo_type": "dataset",
                        "revision": d.get("revision")})
    return out


def _cached(e: dict) -> bool:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(e["repo"], e["file"], repo_type=e["repo_type"], revision=e.get("revision"))
    return isinstance(p, str)


def _weight(e: dict) -> bool:
    return e["kind"] in ("model", "sae") and not e["file"].endswith(("json", "txt", "jinja", "model"))


def _skip_tiny(e: dict, dev_tiny: bool) -> bool:
    """Under --dev-tiny: weights are replaced by tiny local models, and gated repos are never touched (no token)."""
    return dev_tiny and (_weight(e) or e["repo"] in GATED)


def missing(cfg: dict, dev_tiny: bool) -> dict:
    ents = manifest(cfg)
    need = [e for e in ents if not _skip_tiny(e, dev_tiny)]
    miss = [e for e in need if not _cached(e)]
    skipped = [e for e in ents if e not in need]
    return {"n_files": len(ents), "missing": miss, "skipped_dev_tiny": skipped,
            "bytes_total": sum(e["size"] for e in ents), "bytes_skipped": sum(e["size"] for e in skipped)}


def prefetch(cfg: dict, dev_tiny: bool) -> int:
    from huggingface_hub import hf_hub_download

    from . import data as V1

    t0 = time.time()
    V1._download({"data": {"truth": cfg["data"]["truth"], "refusal": cfg["data"]["refusal"]}})
    if not dev_tiny:
        try:
            LOG.info("prefetch: gated access %s", check_token())
        except AccessError as e:
            LOG.error("prefetch: %s", e)
            return 3
    try:
        ents = manifest(cfg)
    except Exception as e:  # noqa: BLE001
        if dev_tiny:                                   # listing gated repos needs the token; datasets still matter
            LOG.warning("prefetch (dev-tiny): cannot list every repo (%s); fetching datasets only", type(e).__name__)
            ents = [x for x in _datasets_only(cfg)]
        else:
            raise
    total = sum(e["size"] for e in ents)
    LOG.info("prefetch: %d files, %.1f GB in total for the full run", len(ents), total / 1e9)
    for e in ents:
        if _skip_tiny(e, dev_tiny):
            continue
        if _cached(e):
            continue
        LOG.info("prefetch: %s %s/%s (%.2f GB)", e["kind"], e["repo"], e["file"], e["size"] / 1e9)
        hf_hub_download(e["repo"], e["file"], repo_type=e["repo_type"], revision=e.get("revision"))
    miss = [e for e in ents if not _skip_tiny(e, dev_tiny) and not _cached(e)]
    LOG.info("prefetch finished in %.0f s: %d files missing", time.time() - t0, len(miss))
    return 0 if not miss else 1


def _datasets_only(cfg: dict) -> list[dict]:
    out = []
    for t in ("sentiment", "answerability", "toxicity"):
        d = cfg["data"][t]
        for f in d["files"].values():
            out.append({"kind": "dataset", "repo": d["repo"], "file": f, "size": 0, "repo_type": "dataset",
                        "revision": d.get("revision")})
    return out


if __name__ == "__main__":
    setup_logging(None)
    dev = "--dev-tiny" in sys.argv
    sys.exit(prefetch(load_v3_config("prefetch", dev_tiny=dev), dev))
