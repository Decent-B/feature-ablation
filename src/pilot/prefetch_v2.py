"""`--v2 --prefetch`: download every model, SAE file and dataset the full v2 run needs into the shared caches,
and report what (if anything) is still missing. Run by `--v2 --smoke` first.

Models: config, tokenizer and safetensors of both models. SAEs: the file resolved for every depth of every model
(adamkarvonen trainer by width / L0; REINS layer_XX.pt). Datasets: v1 raw files (truth, refusal) plus the SST-2,
SQuAD 2.0 and Civil Comments parquet files pinned to a revision. With --dev-tiny the tiny local models replace
the weights, so only tokenizers, configs and datasets are fetched; the full list is still reported.
"""

from __future__ import annotations

import sys
import time

from .utils import LOG, MODEL_PATTERNS, setup_logging
from .v2common import depths_for, layer_for, load_v2_config, models_in


def _model_files(repo: str) -> list[tuple[str, int]]:
    import fnmatch

    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in info.siblings
            if any(fnmatch.fnmatch(s.rfilename, p) for p in MODEL_PATTERNS)]


def _n_layers(repo: str) -> int:
    import json

    from huggingface_hub import hf_hub_download

    c = json.load(open(hf_hub_download(repo, "config.json")))
    return int((c.get("text_config") or c)["num_hidden_layers"])


def manifest(cfg: dict) -> list[dict]:
    """Every file the full run needs: {kind, repo, file, size, repo_type, revision}."""
    from . import sae_v2

    real = load_v2_config("confirm")                    # the full run's models / SAEs, even under --dev-tiny
    out = []
    for m in models_in(real):
        for f, s in _model_files(m):
            out.append({"kind": "model", "repo": m, "file": f, "size": s, "repo_type": "model"})
        sc = real["models"][m]["sae"]
        n = _n_layers(m)
        Ls = [layer_for(d, n) for d in depths_for(real, m)]
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(sc["repo"])
        sizes = {s.rfilename: s.size for s in HfApi().model_info(sc["repo"], files_metadata=True).siblings}
        for L, info in sae_v2.resolve(sc, m, Ls, files).items():
            if info is None:
                continue
            out.append({"kind": "sae", "repo": sc["repo"], "file": info["file"], "size": sizes.get(info["file"], 0),
                        "repo_type": "model", "layer": L})
            if info.get("config_file"):
                out.append({"kind": "sae", "repo": sc["repo"], "file": info["config_file"], "size": 0, "repo_type": "model"})
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


def missing(cfg: dict, dev_tiny: bool) -> dict:
    ents = manifest(cfg)
    need = [e for e in ents if not (dev_tiny and e["kind"] in ("model", "sae") and not e["file"].endswith(
        ("json", "txt", "jinja")))]
    miss = [e for e in need if not _cached(e)]
    skipped = [e for e in ents if e not in need]
    return {"n_files": len(ents), "missing": miss, "skipped_dev_tiny": skipped,
            "bytes_total": sum(e["size"] for e in ents), "bytes_skipped": sum(e["size"] for e in skipped)}


def prefetch(cfg: dict, dev_tiny: bool) -> int:
    from huggingface_hub import hf_hub_download

    from . import data as V1

    t0 = time.time()
    V1._download({"data": {"truth": cfg["data"]["truth"], "refusal": cfg["data"]["refusal"]}})
    ents = manifest(cfg)
    total = sum(e["size"] for e in ents)
    LOG.info("prefetch: %d files, %.1f GB in total for the full run", len(ents), total / 1e9)
    for e in ents:
        if dev_tiny and e["kind"] in ("model", "sae") and not e["file"].endswith(("json", "txt", "jinja")):
            continue                                     # weights are replaced by tiny local models
        if _cached(e):
            continue
        LOG.info("prefetch: %s %s/%s (%.2f GB)", e["kind"], e["repo"], e["file"], e["size"] / 1e9)
        hf_hub_download(e["repo"], e["file"], repo_type=e["repo_type"], revision=e.get("revision"))
    st = missing(cfg, dev_tiny)
    LOG.info("prefetch finished in %.0f s: %d files missing%s", time.time() - t0, len(st["missing"]),
             f"; {len(st['skipped_dev_tiny'])} weight files ({st['bytes_skipped'] / 1e9:.1f} GB) not fetched under --dev-tiny"
             if dev_tiny else "")
    return 0 if not st["missing"] else 1


if __name__ == "__main__":
    setup_logging(None)
    dev = "--dev-tiny" in sys.argv
    sys.exit(prefetch(load_v2_config("prefetch", dev_tiny=dev), dev))
