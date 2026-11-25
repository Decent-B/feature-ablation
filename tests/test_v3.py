"""v3 unit tests: SAE formulas (both releases), file choice, IOI data, the cached example prefix, the one-factor
cell plan, the dependence tests and the preregistration rendering."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


# ============================================================================= SAEs

def _gs_files(tmp: Path, d=12, n=40, seed=0):
    g = np.random.default_rng(seed)
    W_enc = g.normal(size=(d, n)).astype(np.float32)
    W_dec = g.normal(size=(n, d)).astype(np.float32)
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    b_enc = g.normal(size=n).astype(np.float32)
    b_dec = g.normal(size=d).astype(np.float32)
    thr = np.abs(g.normal(size=n)).astype(np.float32)
    for w, l0s in (("65k", [40, 90, 130]), ("16k", [60, 95, 120])):
        for l0 in l0s:
            p = tmp / "layer_3" / f"width_{w}" / f"average_l0_{l0}" / "params.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, threshold=thr)
    return dict(W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, threshold=thr)


def test_gemmascope_matches_reference_and_l0_choice(tmp_path):
    from pilot import sae_v3 as S

    ref = _gs_files(tmp_path)
    cfg = {"format": "gemmascope", "repo": "x", "width": "65k", "extra_width": "16k", "l0_target": 100,
           "local_dir": str(tmp_path)}
    rec = S.resolve(cfg, [3, 4])
    assert rec[3]["main"]["l0"] == 90 and rec[3]["extra"]["l0"] == 95          # closest to 100, then to 90
    assert rec[4]["main"] is None and rec[4]["extra"] is None
    sae = S.load(cfg, rec[3]["main"], "cpu", expected_d=12)
    x = torch.randn(7, 12) * 2
    t = {k: torch.from_numpy(v) for k, v in ref.items()}
    pre = x @ t["W_enc"] + t["b_enc"]                                           # SAELens JumpReLU, no b_dec subtraction
    f_ref = torch.relu(pre) * (pre > t["threshold"])
    assert torch.allclose(sae.encode(x), f_ref, atol=1e-5)
    assert torch.allclose(sae.decode(f_ref), f_ref @ t["W_dec"] + t["b_dec"], atol=1e-5)
    idx = torch.tensor([3, 0, 39])
    assert torch.allclose(sae.encode(x, idx), f_ref[:, idx], atol=1e-5)
    assert torch.equal(sae.directions(idx), t["W_dec"][idx])
    with pytest.raises(S.SAEError):
        S.load(cfg, rec[3]["main"], "cpu", expected_d=13)


def test_closest_l0_ties_go_lower():
    from pilot.sae_v3 import closest_l0

    assert closest_l0([80, 120], 100) == 80
    assert closest_l0([74, 75, 142], 100) == 75


def _qs_repo(tmp: Path, name: str, k: int, d=10, n=30, base="Qwen/Qwen3-1.7B-Base", seed=1):
    g = torch.Generator().manual_seed(seed)
    sd = {"W_enc": torch.randn(n, d, generator=g), "W_dec": torch.randn(d, n, generator=g),
          "b_enc": torch.randn(n, generator=g), "b_dec": torch.randn(d, generator=g)}
    (tmp / name).mkdir(parents=True, exist_ok=True)
    torch.save(sd, tmp / name / "layer5.sae.pt")
    (tmp / name / "config.json").write_text(json.dumps({"model_type": "topk_sae", "base_model": base, "d_model": d,
                                                         "d_sae": n, "k": k, "hook_point": "resid_post"}))
    return sd


def test_qwenscope_matches_app_py(tmp_path):
    from pilot import sae_v3 as S

    sd = _qs_repo(tmp_path, "main", k=6)
    _qs_repo(tmp_path, "fallback", k=3)
    cfg = {"format": "qwenscope", "repo": "main_repo", "fallback_repo": "fb_repo", "trained_on": "Qwen/Qwen3-1.7B-Base",
           "local_dir": str(tmp_path), "local_repo": "main", "local_fallback_repo": "fallback"}
    rec = S.resolve(cfg, [5, 6])
    assert rec[5]["main"]["repo_key"] == "repo" and rec[5]["extra"]["repo_key"] == "fallback_repo"
    assert rec[6]["main"] is None
    sae = S.load(cfg, rec[5]["main"], "cpu", expected_d=10)
    assert sae.k == 6
    h = torch.randn(9, 10)
    pre = h @ sd["W_enc"].T + sd["b_enc"]                                       # Qwen app.py topk_relu
    r = torch.relu(pre)
    v, i = torch.topk(r, 6, dim=-1)
    f_ref = torch.zeros_like(r).scatter_(-1, i, v)
    assert torch.allclose(sae.encode(h), f_ref, atol=1e-5)
    assert torch.allclose(sae.decode(f_ref), f_ref @ sd["W_dec"].T + sd["b_dec"], atol=1e-5)
    idx = torch.tensor([0, 29, 7])
    assert torch.allclose(sae.encode(h, idx), f_ref[:, idx], atol=1e-6)        # subset read keeps TopK membership
    assert torch.equal(sae.directions(idx), sd["W_dec"][:, idx].T)
    assert int((sae.encode(h) > 0).sum(-1).max()) <= 6
    assert S.load(cfg, rec[5]["extra"], "cpu").k == 3


def test_qwenscope_refuses_other_base_model(tmp_path):
    from pilot import sae_v3 as S

    _qs_repo(tmp_path, "main", k=4, base="Qwen/Qwen3-1.7B")
    cfg = {"format": "qwenscope", "repo": "r", "trained_on": "Qwen/Qwen3-1.7B-Base", "local_dir": str(tmp_path),
           "local_repo": "main"}
    with pytest.raises(S.SAEError):
        S.load(cfg, S.resolve(cfg, [5])[5]["main"], "cpu")


def test_input_norm_rms_d():
    from pilot.sae_v3 import SAE3, set_input_norm

    s = SAE3("topk", torch.randn(8, 4), torch.zeros(8), torch.randn(8, 4), torch.zeros(4), k=2)
    set_input_norm(s, "rms_d", mean_sq_norm=16.0)
    assert abs(s.norm_factor - 2.0) < 1e-9
    set_input_norm(s, "none", None)
    assert s.norm_factor == 1.0


# ============================================================================= IOI

class _Tok:
    def encode(self, s, add_special_tokens=False):
        return [1] if s.strip() in {"Mary", "John", "Anna", "Paul", "Lisa", "Mark", "Kate", "Adam", "Ruth", "Sam",
                                    "Jane", "Carl"} else [1, 2]


def test_ioi_balanced_disjoint_single_token():
    from pilot.data_v3 import ioi_names, ioi_split

    cfg = {"seed": 0, "ioi": {"names": ["Mary", "John", "Anna", "Paul", "Lisa", "Mark", "Kate", "Adam", "Ruth", "Sam",
                                        "Jane", "Carl", "Bartholomew"],
                              "name_split": {"demo": 2, "discovery": 0.4, "dev": 0.2},
                              "templates": ["When {A} and {B} went to the {place}, {S} gave a {object} to"],
                              "places": ["store"], "objects": ["drink"]}}
    nm = ioi_names(cfg, [_Tok()])
    assert "Bartholomew" not in sum(nm["sets"].values(), [])
    sets = [set(v) for v in nm["sets"].values()]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    d = ioi_split(cfg, nm["sets"]["discovery"], 40, "discovery")
    assert d.label.sum() == 20
    for row in d.itertuples():
        ans = row.a_name if row.label == 1 else row.b_name
        s = row.b_name if row.label == 1 else row.a_name
        assert row.query.endswith(f"{s} gave a drink to") and ans != s


# ============================================================================= cached example prefix

def _gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    c = Gemma2Config(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8, query_pre_attn_scalar=8, sliding_window=4096,
                     max_position_embeddings=512, attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
                     attn_implementation="eager")
    m = Gemma2ForCausalLM(c).eval()
    m.requires_grad_(False)
    return m


def _qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    c = Qwen3Config(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    m = Qwen3ForCausalLM(c).eval()
    m.requires_grad_(False)
    m.config._attn_implementation = "sdpa"
    return m


class _MC:
    def __init__(self, model):
        from pilot.models import decoder_layers

        self.model = model
        self.layers = decoder_layers(model)
        self.run = type("R", (), {"device": torch.device("cpu")})()


@pytest.mark.parametrize("make", [_gemma2, _qwen3])
def test_prefix_cache_equals_full_prompt(make):
    from pilot import models as M
    from pilot.engine_v3 import PrefixKV, forward_query

    mc = _MC(make())
    prefix = [2, 11, 12, 13, 14, 15, 16, 17]
    queries = [[20, 21, 22], [30, 31, 32, 33, 34, 35], [40]]
    pkv = PrefixKV(mc, prefix)
    for _ in range(2):                                      # the cache is reusable: nothing accumulates
        qb = M.make_batches(queries, 0, torch.device("cpu"), 8, 10**6)[0]
        fb = M.make_batches([prefix + q for q in queries], 0, torch.device("cpu"), 8, 10**6)[0]
        with torch.inference_mode():
            a = forward_query(mc, pkv, qb)[:, -1].float()
            b = forward_query(mc, None, fb)[:, -1].float()
        order_q, order_f = np.argsort(qb.idx), np.argsort(fb.idx)
        assert torch.allclose(a[order_q], b[order_f], atol=1e-4), (a[order_q] - b[order_f]).abs().max()
    # fast path (layers L+1.. only) with the cached prefix
    cap = M.Capture(mc.layers, 1, None)
    with torch.inference_mode():
        ref = forward_query(mc, pkv, qb)[:, -1].float()
        h = cap.h.clone()
    cap.remove()
    with M.LayerSplit(mc.model, 1) as sp:
        sp.inject.value = h
        with torch.inference_mode():
            fast = forward_query(mc, pkv, qb)[:, -1].float()
    assert torch.allclose(fast, ref, atol=1e-4)
    # gradients flow through the query tokens with the cached prefix
    from pilot.sites import GradCapture

    gc = GradCapture(mc.layers, [1, 2])
    with torch.enable_grad():
        forward_query(mc, pkv, qb)[:, -1, 5].float().sum().backward()
    gc.remove()
    assert gc.g[1].shape == gc.x[1].shape and torch.isfinite(gc.g[1]).all()


# ============================================================================= cell plan

def _cfg():
    from pilot.v3common import load_v3_config

    return load_v3_config("confirm", root=ROOT)


def test_cells_vary_one_factor():
    from pilot.v3common import plan_cells

    cfg = _cfg()
    gates = {m: {"tasks": {t: {"gate": {"pass": True}} for t in ("truth", "sentiment", "answerability", "toxicity", "ioi")}}
             for m in cfg["models"]}
    r = cfg["reference"]
    ref_cell = {"model": r["model"], "task": "truth", "depth": 0.5, "strategy": "mean"}

    def diff(c):
        return {k for k in ref_cell if c[k] != ref_cell[k]}

    for blk in cfg["blocks"]:
        for c in plan_cells(cfg, blk, "truth", gates):
            want = {"B1_reference": set(), "B2_depth": {"depth"}, "B3_task": {"task"}, "B4_model": {"model"},
                    "B5_sae_width": set(), "B6_resample": {"strategy"}, "B7_controls": {"task"} if c["task"] != "truth" else set()}
            assert diff(c) == want[blk["id"]], (blk["id"], c)
    depths = [c["depth"] for c in plan_cells(cfg, {"id": "x", "cells": "depth"}, "truth", gates)]
    assert sorted(depths) == [0.25, 0.4, 0.6, 0.75]


def test_reference_task_fallback():
    from pilot.v3common import resolve_reference_task

    cfg = _cfg()
    ok = {"gate": {"pass": True}}
    bad = {"gate": {"pass": False}}
    gates = {"google/gemma-2-2b": {"tasks": {"truth": bad, "ioi": ok, "sentiment": ok, "toxicity": ok, "answerability": ok}},
             "Qwen/Qwen3-1.7B-Base": {"tasks": {"truth": ok, "ioi": ok, "sentiment": ok, "toxicity": ok, "answerability": ok}}}
    assert resolve_reference_task(cfg, gates)["task"] == "ioi"


# ============================================================================= dependence tests

class _Dat:
    """Minimal stand-in for analysis_v3.Data: measure() reads curves from a dict."""

    def __init__(self, table: dict, B=400, delta=0.05, alpha=0.05, noise=0.005, seed=0):
        self.delta, self.alpha, self.B = delta, alpha, B
        self.mode = "avg"
        g = np.random.default_rng(seed)
        self.table = {k: (v, v + noise * g.standard_normal(B)) for k, v in table.items()}

    def curve(self, cell, basis, rule):
        key = (cell["task"], cell["layer"], cell["model"], basis, rule)
        if rule == "random":
            return {1: (0.0, np.zeros(self.B), 0.0, np.zeros(self.B))}
        if key not in self.table:
            return {}
        v, s = self.table[key]
        return {1: (v, s, 0.0, np.zeros(self.B))}


def _cell(task="truth", layer=13, model="g"):
    return {"model": model, "task": task, "layer": layer, "strategy": "mean"}


def test_difference_and_reversal_tests():
    from pilot import analysis_v3 as A

    base = {("truth", 13, "g", b, r): 0.10 for b in A.BASES for r in A.RULES3}
    d = _Dat(base)
    assert not A.difference_test(d, _cell(), "basis")["holds"]                 # built to fail: all equal
    d = _Dat({**base, ("truth", 13, "g", "dims", "attrib"): 0.30})
    res = A.difference_test(d, _cell(), "basis")
    assert res["holds"] and res["n_reliable"] >= 1
    # reversal: on truth dims/attrib > pca/attrib, on ioi the opposite
    t = {**base, ("truth", 13, "g", "dims", "attrib"): 0.30}
    t.update({("ioi", 13, "g", b, r): 0.10 for b in A.BASES for r in A.RULES3})
    t[("ioi", 13, "g", "pca", "attrib")] = 0.30
    d = _Dat(t)
    rv = A.reversal_test(d, {"truth": _cell("truth"), "ioi": _cell("ioi")})
    assert rv["holds"] and any({r["x"], r["y"]} == {"dims/attrib", "pca/attrib"} for r in rv["reversals"])
    same = {k: v for k, v in t.items() if k[0] == "truth"}
    same.update({("ioi",) + k[1:]: v for k, v in same.items()})
    assert not A.reversal_test(_Dat(same), {"truth": _cell("truth"), "ioi": _cell("ioi")})["holds"]


def test_small_differences_are_not_reliable():
    from pilot import analysis_v3 as A

    base = {("truth", 13, "g", b, r): 0.10 for b in A.BASES for r in A.RULES3}
    base[("truth", 13, "g", "dims", "attrib")] = 0.13                         # below delta = 0.05
    assert not A.difference_test(_Dat(base), _cell(), "basis")["holds"]


def test_label():
    from pilot.analysis_v3 import label

    t = {k: {"holds": h} for k, h in zip(("D1", "D2", "D3", "D4"), (True, True, True, True))}
    assert label(t, {}) == "SUPPORTED"
    t["D4"]["holds"] = False
    assert label(t, {}) == "PARTIALLY SUPPORTED"
    assert label({k: {"holds": False} for k in t}, {}) == "NOT SUPPORTED"
    assert label(t, {"inconclusive": True}) == "INCONCLUSIVE"


# ============================================================================= prereg

def test_prereg_v3_renders():
    from pilot import prereg_v3

    md = prereg_v3.render(ROOT)
    assert "## 4. Dependence tests" in md and "D6" in md and "prereg-v3-frozen" in md
    assert all(f in md for f in prereg_v3.FROZEN)
