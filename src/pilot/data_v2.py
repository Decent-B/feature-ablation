"""v2 data: model-independent text splits for every task, then per-model tokenization.

Text splits (data/<task>_<split>.parquet: pid, label, text[, group]):
  truth          discovery = v1 discovery; eval = fixed-seed balanced 500 of the v1 eval split
  refusal        discovery = v1 discovery; eval_harmful_pool = harmful_test (all); eval_harmless = 100 of harmless_test
  generic        fit = v1 fit (PCA pool); cost = fixed-seed 100 of v1's 300 cost prompts
  sentiment      SST-2: discovery 1,000 balanced from train; eval 500 balanced from validation
  answerability  SQuAD 2.0, contexts <= 150 tokens (selected, never truncated): 1,000 / 400 balanced
  toxicity       Civil Comments: toxic >= 0.5 vs <= 0.1, comments <= 150 tokens: 1,000 (validation) / 500 (test)
label 1 = positive class (true / harmful / positive / answerable / toxic). Truth, refusal and generic reuse the
v1 split code with v1's seeds, so they match v1 exactly. Harmful and toxic text is stored but never logged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as V1
from . import prompts as P
from .utils import LOG, rng_for, sha256_file


def _v1_cfg(cfg: dict) -> dict:
    """A v1-shaped config so the v1 split code reproduces v1's splits."""
    r = cfg["data"]["refusal"]
    g = cfg["data"]["generic"]
    return {"seed": cfg["seed"],
            "data": {"truth": cfg["data"]["truth"],
                     "refusal": {"base_url": r["base_url"], "discovery_harmful": r["discovery_harmful"],
                                 "discovery_harmless": r["discovery_harmless"],
                                 "eval_harmful": r["eval_harmful_pool"], "eval_harmless": r["n_eval_harmless"]},
                     "generic": {"n_fit": g["n_fit_v1"], "n_cost": g["n_cost_v1"]},
                     "max_prompt_tokens": cfg["data"]["max_prompt_tokens"], "smoke_n": cfg["data"].get("smoke_n")},
            "tasks": {"truth": {"template": cfg["tasks"]["truth"]["template"]}}}


def _subset_balanced(df: pd.DataFrame, n: int, key: str, seed: int) -> pd.DataFrame:
    rng = rng_for(seed, key)
    parts = []
    for lab in (1, 0):
        d = df[df.label == lab]
        m = min(len(d), n // 2)
        parts.append(d.iloc[np.sort(rng.choice(len(d), size=m, replace=False))])
    return pd.concat(parts).sort_values("pid").reset_index(drop=True)


def _hf_parquet(spec: dict, name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(spec["repo"], spec["files"][name], repo_type="dataset", revision=spec.get("revision")))


def _max_tokens(texts: list[str], toks: list) -> np.ndarray:
    out = np.zeros(len(texts), dtype=np.int64)
    for tok in toks:
        n = np.array([len(x) for x in tok(list(texts), add_special_tokens=False)["input_ids"]])
        out = np.maximum(out, n)
    return out


def _cap(cfg: dict, n: int) -> int:
    c = cfg["data"].get("smoke_n")
    return n if c is None else min(n, int(c))


def _sentiment(cfg: dict, files: dict) -> dict:
    s = cfg["data"]["sentiment"]
    out = {}
    for split, src, n in (("discovery", "train", s["n_discovery"]), ("eval", "validation", s["n_eval"])):
        p = _hf_parquet(s, src)
        files[f"sentiment/{src}"] = p
        d = pd.read_parquet(p)
        d = pd.DataFrame({"pid": [f"sst2/{src}/{int(i):06d}" for i in d["idx"]], "label": d["label"].astype(int),
                          "sentence": d["sentence"].str.strip()})
        d = d[d.sentence.str.len() > 0].drop_duplicates("sentence")
        out[split] = _subset_balanced(d, _cap(cfg, n), f"sentiment-{split}", cfg["seed"])
        out[split]["text"] = [cfg["tasks"]["sentiment"]["template"].format(sentence=x) for x in out[split].sentence]
    return out


def _answerability(cfg: dict, files: dict, ref_toks: list) -> tuple[dict, dict]:
    a = cfg["data"]["answerability"]
    out, info = {}, {}
    for split, src, n in (("discovery", "train", a["n_discovery"]), ("eval", "validation", a["n_eval"])):
        p = _hf_parquet(a, src)
        files[f"answerability/{src}"] = p
        d = pd.read_parquet(p)
        ctx = pd.Series(d["context"].unique())
        ntok = pd.Series(_max_tokens(list(ctx), ref_toks), index=ctx.values)
        d = d.assign(ctx_tokens=d["context"].map(ntok))
        short = d[d.ctx_tokens <= int(a["max_context_tokens"])]
        lab = short["answers"].map(lambda x: int(len(x["text"]) > 0))
        short = pd.DataFrame({"pid": "squad_v2/" + src + "/" + short["id"], "label": lab.values,
                              "context": short["context"].str.strip(), "question": short["question"].str.strip(),
                              "group": short["title"]})
        short = short.drop_duplicates("pid")
        info[split] = {"questions_total": int(len(d)), "contexts_total": int(len(ctx)),
                       "questions_with_short_context": int(len(short)),
                       "short_answerable": int(short.label.sum()), "short_unanswerable": int((short.label == 0).sum())}
        out[split] = _subset_balanced(short, _cap(cfg, n), f"answerability-{split}", cfg["seed"])
        out[split]["text"] = [cfg["tasks"]["answerability"]["template"].format(context=c, question=q)
                              for c, q in zip(out[split].context, out[split].question)]
    return out, info


def _toxicity(cfg: dict, files: dict, ref_toks: list) -> tuple[dict, dict]:
    t = cfg["data"]["toxicity"]
    out, info = {}, {}
    for split, src, n in (("discovery", "discovery", t["n_discovery"]), ("eval", "eval", t["n_eval"])):
        p = _hf_parquet(t, src)
        files[f"toxicity/{src}"] = p
        d = pd.read_parquet(p, columns=["text", "toxicity"])
        d = d.assign(pid=[f"civil_comments/{split}/{i:06d}" for i in range(len(d))], text=d["text"].str.strip())
        d = d[d.text.str.len() > 0].drop_duplicates("text")
        tox = d[d.toxicity >= float(t["toxic_min"])].assign(label=1)
        non = d[d.toxicity <= float(t["nontoxic_max"])].assign(label=0)
        rng = rng_for(cfg["seed"], "toxicity-candidates", split)
        need = _cap(cfg, n) // 2
        picked = []
        for g in (tox, non):                          # tokenize shuffled candidates in chunks until enough are short
            perm = rng.permutation(len(g))
            got = []
            for i in range(0, len(perm), 4000):
                c = g.iloc[perm[i:i + 4000]]
                ok = _max_tokens(list(c.text), ref_toks) <= int(t["max_text_tokens"])
                got.append(c[ok])
                if sum(len(x) for x in got) >= 3 * need:
                    break
            picked.append(pd.concat(got))
        pool = pd.concat(picked)[["pid", "label", "text"]].rename(columns={"text": "comment"})
        info[split] = {"rows": int(len(d)), "toxic": int(len(tox)), "nontoxic": int(len(non))}
        out[split] = _subset_balanced(pool, _cap(cfg, n), f"toxicity-{split}", cfg["seed"])
        out[split]["text"] = [cfg["tasks"]["toxicity"]["template"].format(text=x) for x in out[split].comment]
    return out, info


def build_text(cfg: dict, ref_toks: list, out_dir: Path) -> dict:
    """ref_toks[0] is the Qwen3 tokenizer used by v1's length filter; all tokenizers bound context lengths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    c1 = _v1_cfg(cfg)
    files = V1._download(c1)
    truth, tinfo = V1._truth_splits(c1, files)
    suffix = P.template_suffix(ref_toks[0])
    refusal, generic, rinfo = V1._refusal_and_generic(c1, files, ref_toks[0], suffix["suffix_ids"])
    seed = cfg["seed"]
    sets = {"truth": {"discovery": truth["discovery"],
                      "eval": _subset_balanced(truth["eval"], _cap(cfg, int(cfg["data"]["truth"]["n_eval"])),
                                               "truth-eval-500", seed)}}
    rev = refusal["eval"]
    sets["refusal"] = {"discovery": refusal["discovery"],
                       "eval_harmful_pool": rev[rev.label == 1].reset_index(drop=True),
                       "eval_harmless": rev[rev.label == 0].reset_index(drop=True)}
    cost = generic["cost"]
    n_cost = min(len(cost), int(cfg["cost"]["n_generic"]))
    idx = np.sort(rng_for(seed, "generic-cost-v2").choice(len(cost), size=n_cost, replace=False))
    fit = generic["fit"]
    if cfg["data"].get("smoke_n"):
        fit = fit.iloc[: 2 * int(cfg["data"]["smoke_n"])]
    sets["generic"] = {"fit": fit.reset_index(drop=True), "cost": cost.iloc[idx].reset_index(drop=True)}
    files_all = dict(files)
    sets["sentiment"] = _sentiment(cfg, files_all)
    sets["answerability"], ainfo = _answerability(cfg, files_all, ref_toks)
    sets["toxicity"], xinfo = _toxicity(cfg, files_all, ref_toks)
    summary = {"files": {k: {"path": str(v), "sha256": sha256_file(Path(v))} for k, v in files_all.items()},
               "truth_rows": tinfo, "refusal": rinfo, "answerability": ainfo, "toxicity": xinfo, "splits": {}}
    for task, splits in sets.items():
        for split, d in splits.items():
            keep = [c for c in ("pid", "label", "group", "text") if c in d.columns]
            d = d[keep].reset_index(drop=True)
            d.to_parquet(out_dir / f"{task}_{split}.parquet", index=False)
            summary["splits"][f"{task}/{split}"] = {"n": int(len(d)), "n_pos": int((d.label == 1).sum()),
                                                    "n_neg": int((d.label == 0).sum())}
    LOG.info("v2 text splits: %s", {k: v["n"] for k, v in summary["splits"].items()})
    return summary


def load_text(data_dir: Path, task: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{task}_{split}.parquet")


def tokenize(tok, df: pd.DataFrame) -> pd.DataFrame:
    """Chat template with enable_thinking=False; the prompt ends with the template suffix."""
    suffix = P.template_suffix(tok)["suffix_ids"]
    d = df.copy()
    d["input_ids"] = [P.encode_checked(tok, t, suffix) for t in d.text]
    d["n_tokens"] = [len(x) for x in d.input_ids]
    return d


def load_tok(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["input_ids"] = [list(map(int, x)) for x in d.input_ids]
    return d
