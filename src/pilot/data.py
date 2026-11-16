"""Data: download raw files, build splits, apply the chat template, record positions.

Sets and splits (saved as data/<set>_<split>.parquet):
  truth   discovery / validation / eval   label 1 = true statement, 0 = false
  refusal discovery / validation / eval   label 1 = harmful, 0 = harmless
  generic fit (PCA fitting) / cost (KL cost)   held-out harmless Alpaca, disjoint from refusal
Harmful prompt text is stored in the parquet files but never logged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import prompts as P
from .utils import DATA_CACHE, LOG, fetch_url, rng_for, sha256_file

REFUSAL_FILES = [f"{c}_{s}.json" for c in ("harmful", "harmless") for s in ("train", "val", "test")]


def _download(cfg: dict) -> dict:
    files = {}
    t = cfg["data"]["truth"]
    for f in t["files"]:
        files[f"truth/{f}"] = fetch_url(f"{t['base_url']}/{f}", DATA_CACHE / "truth" / f)
    r = cfg["data"]["refusal"]
    for f in REFUSAL_FILES:
        files[f"refusal/{f}"] = fetch_url(f"{r['base_url']}/{f}", DATA_CACHE / "refusal" / f)
    return files


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _balance(df: pd.DataFrame, rng: np.random.Generator, cap: int | None = None) -> pd.DataFrame:
    """Downsample the majority label so both labels are equally frequent (optionally capped)."""
    g = {lab: df[df.label == lab] for lab in (0, 1)}
    m = min(len(g[0]), len(g[1]))
    if cap is not None:
        m = min(m, cap // 2)
    parts = []
    for lab in (0, 1):
        d = g[lab]
        if len(d) > m:
            d = d.iloc[np.sort(rng.choice(len(d), size=m, replace=False))]
        parts.append(d)
    return pd.concat(parts).sort_values("pid").reset_index(drop=True)


def _truth_splits(cfg: dict, files: dict) -> tuple[dict, dict]:
    t = cfg["data"]["truth"]
    rows = []
    info = {}
    for f in t["files"]:
        recs = _read_csv(files[f"truth/{f}"])
        info[f] = len(recs)
        if len(recs) != int(t["expected_rows"]):
            raise RuntimeError(f"{f}: loaded {len(recs)} rows, expected {t['expected_rows']}")
        stem = f.replace(".csv", "")
        for i, r in enumerate(recs):
            rows.append({"pid": f"truth/{stem}/{i:05d}", "statement": r["statement"],
                         "label": int(r["label"]), "group": r["city"]})
    df = pd.DataFrame(rows)
    cities = np.array(sorted(df.group.unique()))
    rng = rng_for(cfg["seed"], "truth-split")
    rng.shuffle(cities)
    fr = t["split"]
    n = len(cities)
    n_disc = int(round(fr["discovery"] * n))
    n_val = int(round(fr["validation"] * n))
    assign = {}
    for i, c in enumerate(cities):
        assign[c] = "discovery" if i < n_disc else ("validation" if i < n_disc + n_val else "eval")
    df["split"] = df.group.map(assign)
    cap = cfg["data"].get("smoke_n")
    out = {}
    for split in ("discovery", "validation", "eval"):
        d = df[df.split == split].copy()
        d = _balance(d, rng_for(cfg["seed"], "truth-balance", split), cap)
        d["text"] = [cfg["tasks"]["truth"]["template"].format(statement=s) for s in d.statement]
        out[split] = d
    # leakage check: a city never appears in two splits
    seen = {}
    for split, d in out.items():
        for c in d.group.unique():
            if c in seen:
                raise RuntimeError(f"city {c} in both {seen[c]} and {split}")
            seen[c] = split
    return out, info


def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _refusal_and_generic(cfg: dict, files: dict, tok, suffix_ids) -> tuple[dict, dict, dict]:
    r = cfg["data"]["refusal"]
    cap = cfg["data"].get("smoke_n")
    max_tok = int(cfg["data"]["max_prompt_tokens"])
    raw = {f.replace(".json", ""): _load_json(files[f"refusal/{f}"]) for f in REFUSAL_FILES}
    dropped = {}

    def pool(name: str) -> pd.DataFrame:
        recs = raw[name]
        d = pd.DataFrame({"pid": [f"{name}/{i:05d}" for i in range(len(recs))],
                          "text": [x["instruction"] for x in recs]})
        d = d.drop_duplicates("text")
        ntok = np.array([len(tok(P.render(tok, t), add_special_tokens=False)["input_ids"]) for t in d.text])
        keep = ntok <= max_tok
        dropped[name] = int((~keep).sum())
        return d[keep].reset_index(drop=True)

    def sample(d: pd.DataFrame, n: int | None, key: str) -> pd.DataFrame:
        if cap is not None:
            n = cap if n is None else min(n, cap)
        if n is None or len(d) <= n:
            return d.copy()
        idx = np.sort(rng_for(cfg["seed"], key).choice(len(d), size=n, replace=False))
        return d.iloc[idx].copy()

    h_train, h_val, h_test = pool("harmful_train"), pool("harmful_val"), pool("harmful_test")
    s_train, s_val, s_test = pool("harmless_train"), pool("harmless_val"), pool("harmless_test")
    disc_h = sample(h_train, r["discovery_harmful"], "disc-harmful")
    disc_s = sample(s_train, r["discovery_harmless"], "disc-harmless")
    eval_h = sample(h_test, r["eval_harmful"], "eval-harmful")
    eval_s = sample(s_test, r["eval_harmless"], "eval-harmless")
    val_h = sample(h_val, None, "val-harmful")
    val_s = sample(s_val, len(val_h), "val-harmless")
    refusal = {}
    for split, (h, s) in {"discovery": (disc_h, disc_s), "validation": (val_h, val_s),
                          "eval": (eval_h, eval_s)}.items():
        h = h.assign(label=1)
        s = s.assign(label=0)
        refusal[split] = pd.concat([h, s]).reset_index(drop=True)
    # generic: held-out harmless Alpaca from harmless_train, disjoint (by text) from every refusal split
    used = set(pd.concat([v.text for v in refusal.values()]))
    used |= set(s_val.text) | set(s_test.text)
    g = s_train[~s_train.text.isin(used)].reset_index(drop=True)
    gc = cfg["data"]["generic"]
    n_fit, n_cost = int(gc["n_fit"]), int(gc["n_cost"])
    idx = rng_for(cfg["seed"], "generic").choice(len(g), size=n_fit + n_cost, replace=False)
    generic = {"fit": g.iloc[np.sort(idx[:n_fit])].assign(label=-1).reset_index(drop=True),
               "cost": g.iloc[np.sort(idx[n_fit:])].assign(label=-1).reset_index(drop=True)}
    info = {"dropped_over_max_tokens": dropped, "pool_sizes": {k: len(v) for k, v in raw.items()}}
    return refusal, generic, info


def build(cfg: dict, tok, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _download(cfg)
    suffix = P.template_suffix(tok)
    if not suffix["has_empty_think_block"]:
        LOG.warning("template suffix has no empty think block: %r", suffix["suffix_text"])
    truth, tinfo = _truth_splits(cfg, files)
    refusal, generic, rinfo = _refusal_and_generic(cfg, files, tok, suffix["suffix_ids"])
    sets = {"truth": truth, "refusal": refusal, "generic": generic}
    summary = {"files": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in files.items()},
               "truth_rows": tinfo, **rinfo, "splits": {}, "positions": {
                   "suffix_text": suffix["suffix_text"], "suffix_ids": suffix["suffix_ids"],
                   "suffix_tokens": suffix["suffix_tokens"], "T": suffix["T"],
                   "intervention_positions": "the last T prompt tokens (all post-instruction template "
                                             "tokens); offset t = column S-T+t under left padding",
                   "discovery_position": "final prompt token (offset T-1), after the empty think block"}}
    for set_name, splits in sets.items():
        for split, d in splits.items():
            ids = [P.encode_checked(tok, t, suffix["suffix_ids"]) for t in d.text]
            d = d.copy()
            d["input_ids"] = ids
            d["n_tokens"] = [len(x) for x in ids]
            keep = [c for c in ("pid", "label", "group", "text", "n_tokens", "input_ids") if c in d.columns]
            d[keep].to_parquet(out_dir / f"{set_name}_{split}.parquet", index=False)
            summary["splits"][f"{set_name}/{split}"] = {
                "n": len(d), "n_pos": int((d.label == 1).sum()), "n_neg": int((d.label == 0).sum()),
                "mean_tokens": float(np.mean(d.n_tokens)), "max_tokens": int(np.max(d.n_tokens))}
    LOG.info("data splits: %s", {k: v["n"] for k, v in summary["splits"].items()})
    return summary


def load(data_dir: Path, set_name: str, split: str) -> pd.DataFrame:
    d = pd.read_parquet(data_dir / f"{set_name}_{split}.parquet")
    d["input_ids"] = [list(map(int, x)) for x in d.input_ids]
    return d
