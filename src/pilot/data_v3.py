"""v3 data: model-independent text splits for the five tasks, the fixed 4-example prefix per task, IOI generation,
dev slices, generic prompts, then per-model tokenization.

Text splits (data/<task>_<split>.parquet: pid, label, query[, group, a_name, b_name]):
  truth          discovery = v1 discovery minus the 4 examples; eval = fixed-seed balanced 500 of the v1 eval split
                 (same subset as v2); dev = balanced 200 of the remaining v1 eval rows
  sentiment      SST-2 (v2 selection); answerability SQuAD 2.0, contexts <= 150 tokens (v2 selection);
  toxicity       Civil Comments (v2 selection)
  ioi            generated: discovery / eval / dev from disjoint name sets (balanced: the answer is A or B)
  generic        fit (PCA pool) and cost (KL prompts): harmless Alpaca instructions as plain text
Example prefix (data/demos.json): per task, 2 examples per class from the discovery split (removed from it),
chosen among the shortest quarter by tokens (answerability: the shortest passages per class), each written as
query + " " + answer, joined by blank lines. label 1 = positive class (true / positive / answerable / toxic /
the answer is the first-mentioned name).

Per model (tokenize): input_ids = prefix_ids + query_ids, with prefix_ids = tokenizer(prefix) including the model's
BOS (Gemma) and query_ids = tokenizer(query) without special tokens; n_demo = len(prefix_ids) is identical for
every prompt of a task, so the prefix's keys and values can be computed once.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as V1
from . import data_v2 as D2
from .utils import LOG, rng_for, sha256_file

SPLITS = {"truth": ("discovery", "eval", "dev"), "sentiment": ("discovery", "eval"),
          "answerability": ("discovery", "eval"), "toxicity": ("discovery", "eval"),
          "ioi": ("discovery", "eval", "dev"), "generic": ("fit", "cost")}


def _cap(cfg: dict, n: int) -> int:
    c = cfg["data"].get("smoke_n")
    return n if c is None else min(n, int(c))


def _ntok(texts, toks) -> np.ndarray:
    return D2._max_tokens(list(texts), toks)


# ============================================================================= tasks from existing data

def _truth(cfg: dict, files: dict) -> dict:
    c1 = {"seed": cfg["seed"], "data": {"truth": cfg["data"]["truth"], "smoke_n": None},
          "tasks": {"truth": {"template": "{statement}"}}}
    splits, info = V1._truth_splits(c1, files)
    seed = cfg["seed"]
    ev_all = splits["eval"]
    ev = D2._subset_balanced(ev_all, int(cfg["data"]["truth"]["n_eval"]), "truth-eval-500", seed)   # v2's subset
    rest = ev_all[~ev_all.pid.isin(ev.pid)]
    dev = D2._subset_balanced(rest, int(cfg["data"]["truth"]["n_dev"]), "truth-dev-200", seed)
    out = {}
    for name, d in (("discovery", splits["discovery"]), ("eval", ev), ("dev", dev)):
        d = d.copy()
        if cfg["data"].get("smoke_n"):
            d = D2._subset_balanced(d, _cap(cfg, len(d)), f"truth-smoke-{name}", seed)
        d["query"] = [cfg["tasks"]["truth"]["template"].format(statement=s) for s in d.statement]
        out[name] = d[["pid", "label", "group", "query"]].reset_index(drop=True)
    return out


def _from_v2(fn, cfg: dict, files: dict, ref_toks, field_map) -> dict:
    """Run a v2 selector with the v3 query template (the v2 code formats `template` into `text`)."""
    res = fn(cfg, files, ref_toks) if ref_toks is not None else fn(cfg, files)
    sets = res[0] if isinstance(res, tuple) else res
    out = {}
    for split, d in sets.items():
        d = d.rename(columns={"text": "query"})
        keep = [c for c in ("pid", "label", "group", "query") if c in d.columns]
        out[split] = d[keep].reset_index(drop=True)
    return out


def _generic(cfg: dict, files: dict, toks) -> dict:
    g = cfg["data"]["generic"]
    recs = json.load(open(files["refusal/harmless_train.json"]))
    d = pd.DataFrame({"pid": [f"harmless_train/{i:05d}" for i in range(len(recs))],
                      "query": [x["instruction"].strip() for x in recs]}).drop_duplicates("query")
    d = d[_ntok(d["query"], toks) <= int(g["max_tokens"])].reset_index(drop=True)
    n_fit, n_cost = int(g["n_fit"]), int(g["n_cost"])
    idx = rng_for(cfg["seed"], "generic-v3").choice(len(d), size=n_fit + n_cost, replace=False)
    return {"fit": d.iloc[np.sort(idx[:n_fit])].assign(label=-1).reset_index(drop=True),
            "cost": d.iloc[np.sort(idx[n_fit:])].assign(label=-1).reset_index(drop=True)}


# ============================================================================= IOI

def ioi_names(cfg: dict, toks) -> dict:
    """Single-token names (with a leading space) in every tokenizer, split into disjoint sets."""
    ic = cfg["ioi"]
    ok = []
    for nm in ic["names"]:
        if all(len(t.encode(" " + nm, add_special_tokens=False)) == 1 for t in toks):
            ok.append(nm)
    rng = rng_for(cfg["seed"], "ioi-names")
    ok = list(np.array(ok)[rng.permutation(len(ok))])
    sp = ic["name_split"]
    n_demo = int(sp["demo"])
    n_disc = int(round(float(sp["discovery"]) * (len(ok) - n_demo)))
    n_dev = int(round(float(sp["dev"]) * (len(ok) - n_demo)))
    sets = {"demo": ok[:n_demo], "discovery": ok[n_demo:n_demo + n_disc],
            "dev": ok[n_demo + n_disc:n_demo + n_disc + n_dev], "eval": ok[n_demo + n_disc + n_dev:]}
    if min(len(v) for v in sets.values()) < 2:
        raise RuntimeError(f"IOI: too few single-token names for disjoint sets: { {k: len(v) for k, v in sets.items()} }")
    return {"sets": sets, "n_single_token": len(ok), "n_listed": len(ic["names"])}


def ioi_split(cfg: dict, names: list[str], n: int, key: str) -> pd.DataFrame:
    """n balanced prompts: label 1 -> S = B (answer A), label 0 -> S = A (answer B)."""
    ic = cfg["ioi"]
    rng = rng_for(cfg["seed"], "ioi", key)
    pairs = [(a, b) for a, b in itertools.permutations(names, 2)]
    rows = []
    for i in range(n):
        lab = int(i % 2 == 0)
        a, b = pairs[rng.integers(len(pairs))]
        tpl = ic["templates"][rng.integers(len(ic["templates"]))]
        place = ic["places"][rng.integers(len(ic["places"]))]
        obj = ic["objects"][rng.integers(len(ic["objects"]))]
        s = b if lab == 1 else a
        rows.append({"pid": f"ioi/{key}/{i:05d}", "label": lab, "query": tpl.format(A=a, B=b, S=s, place=place, object=obj),
                     "a_name": a, "b_name": b, "group": tpl})
    return pd.DataFrame(rows)


# ============================================================================= example prefix

def _answer_word(cfg: dict, task: str, row) -> str:
    if task == "ioi":
        return row["a_name"] if int(row["label"]) == 1 else row["b_name"]
    w = cfg["tasks"][task]["answers"]
    return w[0] if int(row["label"]) == 1 else w[1]


def pick_demos(cfg: dict, task: str, disc: pd.DataFrame, toks) -> tuple[pd.DataFrame, list[str]]:
    """2 examples per class from the discovery split (removed from it); returns (discovery minus demos, demo texts)."""
    dc = cfg["demos"]
    per = int(dc["per_class"])
    n = _ntok(disc["query"], toks)
    picked = []
    for lab in (1, 0):
        idx = np.flatnonzero(disc.label.to_numpy() == lab)
        order = idx[np.argsort(n[idx], kind="stable")]
        if task == "answerability":
            take = order[:per]                                   # the shortest passages
        else:
            pool = order[: max(per, int(np.ceil(float(dc["short_quantile"]) * len(order))))]
            take = np.sort(rng_for(cfg["seed"], "demos", task, lab).choice(pool, size=min(per, len(pool)), replace=False))
        picked += list(take)
    rng = rng_for(cfg["seed"], "demos-order", task)
    picked = list(np.array(picked)[rng.permutation(len(picked))])       # mixed class order
    texts = [disc.iloc[i]["query"] + " " + _answer_word(cfg, task, disc.iloc[i]) + ("." if task == "ioi" else "")
             for i in picked]
    rest = disc.drop(disc.index[picked]).reset_index(drop=True)
    return rest, texts


def build_text(cfg: dict, ref_toks: list, out_dir: Path) -> dict:
    """ref_toks: the tokenizers of every model (length caps use the longest tokenization)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = V1._download({"data": {"truth": cfg["data"]["truth"], "refusal": cfg["data"]["refusal"]}})
    sets = {"truth": _truth(cfg, files)}
    files_all = dict(files)
    sets["sentiment"] = _from_v2(lambda c, f: D2._sentiment(c, f), cfg, files_all, None, None)
    sets["answerability"] = _from_v2(D2._answerability, cfg, files_all, ref_toks, None)
    sets["toxicity"] = _from_v2(D2._toxicity, cfg, files_all, ref_toks, None)
    names = ioi_names(cfg, ref_toks)
    ic = cfg["ioi"]
    sets["ioi"] = {"discovery": ioi_split(cfg, names["sets"]["discovery"], int(ic["n_discovery"]) + 4, "discovery"),
                   "eval": ioi_split(cfg, names["sets"]["eval"], int(ic["n_eval"]), "eval"),
                   "dev": ioi_split(cfg, names["sets"]["dev"], int(ic["n_dev"]), "dev")}
    demo_src = ioi_split(cfg, names["sets"]["demo"], 64, "demo")       # IOI examples use their own names
    sets["generic"] = _generic(cfg, files, ref_toks)
    demos = {}
    for t in ("truth", "sentiment", "answerability", "toxicity", "ioi"):
        src = demo_src if t == "ioi" else sets[t]["discovery"]
        rest, texts = pick_demos(cfg, t, src, ref_toks)
        if t != "ioi":
            sets[t]["discovery"] = rest
        else:
            sets[t]["discovery"] = sets[t]["discovery"].iloc[:int(ic["n_discovery"])].reset_index(drop=True)
        demos[t] = {"examples": texts, "prefix": cfg["demos"]["separator"].join(texts) + cfg["demos"]["separator"]}
    (out_dir / "demos.json").write_text(json.dumps(demos, indent=2) + "\n")
    summary = {"files": {k: {"path": str(v), "sha256": sha256_file(Path(v))} for k, v in files_all.items()},
               "ioi_names": names, "splits": {}, "demos": {t: v["examples"] for t, v in demos.items() if t != "toxicity"}}
    for task, splits in sets.items():
        for split, d in splits.items():
            keep = [c for c in ("pid", "label", "group", "query", "a_name", "b_name") if c in d.columns]
            d = d[keep].reset_index(drop=True)
            d.to_parquet(out_dir / f"{task}_{split}.parquet", index=False)
            summary["splits"][f"{task}/{split}"] = {"n": int(len(d)), "n_pos": int((d.label == 1).sum()),
                                                    "n_neg": int((d.label == 0).sum())}
    LOG.info("v3 text splits: %s", {k: v["n"] for k, v in summary["splits"].items()})
    return summary


def load_text(data_dir: Path, task: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{task}_{split}.parquet")


def load_demos(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "demos.json").read_text())


# ============================================================================= tokenization (per model)

def prefix_ids(tok, prefix: str) -> list[int]:
    """The example prefix with the model's own special start tokens (Gemma: <bos>; Qwen base: none)."""
    return list(tok(prefix, add_special_tokens=True)["input_ids"])


def tokenize(tok, df: pd.DataFrame, prefix: str | None) -> pd.DataFrame:
    d = df.copy()
    if prefix is None:                                   # generic prompts: plain instruction with special start tokens
        d["input_ids"] = [list(tok(q, add_special_tokens=True)["input_ids"]) for q in d["query"]]
        d["n_demo"] = 0
    else:
        p = prefix_ids(tok, prefix)
        d["input_ids"] = [p + list(tok(q, add_special_tokens=False)["input_ids"]) for q in d["query"]]
        d["n_demo"] = len(p)
    if "a_name" in d.columns:                            # IOI: per-example answer tokens (leading space)
        d["a_id"] = [int(tok.encode(" " + a, add_special_tokens=False)[0]) for a in d.a_name]
        d["b_id"] = [int(tok.encode(" " + b, add_special_tokens=False)[0]) for b in d.b_name]
    d["n_tokens"] = [len(x) for x in d.input_ids]
    return d


def load_tok(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["input_ids"] = [list(map(int, x)) for x in d.input_ids]
    return d
