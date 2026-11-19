"""v2 analysis on synthetic runs with known answers (C1 supported / C2 reversal / random floor), curve extension
logic, tie-aware Spearman, and the inclusion / substitution rule."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pilot import analysis_v2 as A
from pilot.utils import REPO_ROOT, write_json

N = 200


def _hyp():
    return yaml.safe_load((REPO_ROOT / "prereg" / "hypotheses.yaml").read_text())


def _write_cell(root: Path, model, task, effect: dict, seed=0):
    """effect[(basis, rule)] = damage slope: AUROC drop grows with log cost at that rate."""
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(N // 2), np.zeros(N // 2)].astype(int)
    clean = labels * 2.0 - 1.0 + rng.normal(0, 0.3, N)
    cell = root / "ablate" / "chunks" / model.split("/")[-1].lower() / task / "L14"
    (cell / "S1_mean").mkdir(parents=True, exist_ok=True)
    np.savez(cell / "clean.npz", score=clean.astype(np.float32), aux=np.ones(N, np.float32), labels=labels,
             gen_label=np.zeros(N, bool), eval_pids=np.array([str(i) for i in range(N)]),
             gen_pids=np.array([str(i) for i in range(20)]), refusal=np.array(False))
    base = {"model": model, "task": task, "role": "confirmatory", "depth": 0.5, "layer": 14, "site": "S1",
            "strategy": "mean", "tier": "tier1", "block": "B1", "kind": "grid", "seconds": 0.1, "rel_change": 0.0,
            "curve_flag": "ok", "ext": False, "cap": 2048}
    ks = [1, 5, 20, 50, 100, 200, 500, 1000, 2000]

    def arrays(noise_sd, cost):
        s = clean + rng.normal(0, noise_sd, N)
        return {"score": s.astype(np.float32), "aux": np.ones(N, np.float32),
                "kl9": np.full(20, cost, np.float32) * rng.uniform(0.9, 1.1, 20).astype(np.float32),
                "kl1": np.full(20, cost, np.float32), "rel": np.zeros(20, np.float32)}
    rows, arrs = [], []
    rows.append({**base, "basis": "anchor", "rule": "massmean", "resample": -1, "draw": -1, "k": 1, "k_eff": 1,
                 "cost_kl": 0.05, "cost_kl_v1": 0.05})
    arrs.append(arrays(2.0, 0.05))
    pd.DataFrame(rows).to_parquet(cell / "S1_mean" / "anchor.parquet")
    np.savez(cell / "S1_mean" / "anchor.npz", **{k: np.stack([a[k] for a in arrs]) for k in arrs[0]})
    for basis in ("sae", "pca", "dims"):
        rows, arrs = [], []
        for rule in ("meandiff", "l1probe", "attrib", "random"):
            draws = [(-1, 0), (-1, 1)] if rule == "random" else [(0, -1), (1, -1)]
            for res, dr in draws:
                for k in ks:
                    cost = 0.001 * k
                    sd = effect[(basis, rule)] * np.log1p(k)
                    a = arrays(sd, cost)
                    rows.append({**base, "basis": basis, "rule": rule, "resample": res, "draw": dr, "k": k, "k_eff": k,
                                 "cost_kl": float(a["kl9"].mean()), "cost_kl_v1": cost})
                    arrs.append(a)
        pd.DataFrame(rows).to_parquet(cell / "S1_mean" / f"{basis}.parquet")
        np.savez(cell / "S1_mean" / f"{basis}.npz", **{k: np.stack([a[k] for a in arrs]) for k in arrs[0]})


def _analysis(tmp_path, effects):
    for i, (m, t, eff) in enumerate(effects):
        _write_cell(tmp_path, m, t, eff, seed=i)
    write_json(tmp_path / "manifest.json", {"config": {"analysis": {"bootstrap": 200}, "seed": 0}})
    return A.Analysis(tmp_path, 200, 0, _hyp())


def _eff(attrib=0.6, readout=0.05, random=0.0, overrides=None):
    e = {(b, r): (attrib if r == "attrib" else random if r == "random" else readout)
         for b in ("sae", "pca", "dims") for r in ("meandiff", "l1probe", "attrib", "random")}
    e.update(overrides or {})
    return e


def test_c1_supported_when_attribution_dominates(tmp_path):
    an = _analysis(tmp_path, [("Qwen/Qwen3-1.7B", "sentiment", _eff()), ("Qwen/Qwen3.5-4B", "truth", _eff())])
    pairs = [{"model": "Qwen/Qwen3-1.7B", "task": "sentiment", "role": "confirmatory"},
             {"model": "Qwen/Qwen3.5-4B", "task": "truth", "role": "confirmatory"}]
    res = A.confirmatory(an, pairs)
    assert res["C1"]["supported"], res["C1"]
    assert all(r["stat"] > 0.05 for r in res["C1"]["per_pair"])
    assert res["label"] in ("SUPPORTED", "PARTIALLY SUPPORTED")


def test_c1_not_supported_when_rules_tie(tmp_path):
    an = _analysis(tmp_path, [("Qwen/Qwen3-1.7B", "sentiment", _eff(attrib=0.3, readout=0.3))])
    res = A.confirmatory(an, [{"model": "Qwen/Qwen3-1.7B", "task": "sentiment", "role": "confirmatory"}])
    assert not res["C1"]["supported"]
    assert abs(res["C1"]["per_pair"][0]["stat"]) < 0.05


def test_c2_detects_a_reversal(tmp_path):
    # readout: pca >> dims; attribution: dims >> pca
    ov = {("pca", "meandiff"): 0.6, ("pca", "l1probe"): 0.6, ("dims", "meandiff"): 0.0, ("dims", "l1probe"): 0.0,
          ("pca", "attrib"): 0.0, ("dims", "attrib"): 0.9}
    an = _analysis(tmp_path, [("Qwen/Qwen3.5-4B", "sentiment", _eff(overrides=ov))])
    res = A.confirmatory(an, [{"model": "Qwen/Qwen3.5-4B", "task": "sentiment", "role": "confirmatory"}])
    assert res["C2"]["per_pair"][0]["reversal"], res["C2"]
    assert res["C2"]["supported"]


def test_excess_over_random_removes_the_random_floor(tmp_path):
    an = _analysis(tmp_path, [("Qwen/Qwen3-1.7B", "sentiment", _eff(attrib=0.4, readout=0.4, random=0.4))])
    cell = an.primary_cell("Qwen/Qwen3-1.7B", "sentiment")
    for b in an.bases(cell):
        e = an.E(cell, b, "attrib", 0.1)
        assert abs(e[0]) < 0.06
    d = an.D(cell, "dims", "attrib", 0.1, 0.1)
    assert d[0] > 0.05 and d[2] == 0.0              # real damage, curve reaches c0


def test_spearman_handles_ties():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    Y = np.array([[1.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    r = A.spearman(x, Y)
    assert r[0] == pytest.approx(0.9486833, abs=1e-6)
    assert r[1] == pytest.approx(-1.0)
    assert np.isnan(r[2])


class _FakeRunner:
    L = 0

    def __init__(self, slope):
        self.slope = slope

    def run(self, basis, units, strategy, site, means, reservoir=None):
        k = len(units)
        c = self.slope * k
        return {"score": np.zeros(4, np.float32), "kl9": np.full(3, c, np.float32), "kl1": np.full(3, c, np.float32),
                "rel": np.zeros(3, np.float32), "seconds": 0.0}


def _ablator():
    from pilot.ablate_v2 import Ablator

    ab = Ablator.__new__(Ablator)
    ab.cfg = {"extension": {"factor": 2, "max_steps": None}}
    return ab


def _cell():
    return {"model": "m", "task": "t", "role": "confirmatory", "depth": 0.5, "site": "S1", "strategy": "mean",
            "tier": "tier1", "block": "B", "kind": "grid"}


def test_curve_extension_stops_at_target_and_flags_cap():
    ab = _ablator()
    rows, arrs = [], []
    flag = ab.curve(_FakeRunner(0.001), None, np.arange(5000), 5000, [1, 5, 20, 50, 100, 200, 500], 0.5, _cell(), {}, None,
                    {"basis": "dims", "rule": "random", "resample": -1, "draw": 0}, rows, arrs)
    ks = [r["k_eff"] for r in rows]
    assert ks == [1, 5, 20, 50, 100, 200, 500] and flag == "ok"      # 500 * 0.001 = 0.5 reaches the target
    rows, arrs = [], []
    flag = ab.curve(_FakeRunner(0.0001), None, np.arange(2048), 2048, [1, 5, 20, 50, 100, 200, 500], 0.5, _cell(), {}, None,
                    {"basis": "dims", "rule": "random", "resample": -1, "draw": 0}, rows, arrs)
    assert [r["k_eff"] for r in rows][7:] == [1000, 2000, 2048] and flag == "cap_hit"
    assert all(r["curve_flag"] == "cap_hit" for r in rows)
    rows, arrs = [], []
    flag = ab.curve(_FakeRunner(0.0001), None, np.arange(10**5), 10**5, [1, 500], 0.5, _cell(), {}, None,
                    {"basis": "sae", "rule": "attrib", "resample": 0, "draw": -1}, rows, arrs)
    assert [r["k_eff"] for r in rows] == [1, 500, 1000, 2000, 4000, 8000] and flag == "ok"
    assert rows[-1]["cost_kl"] >= 0.5 > rows[-2]["cost_kl"]


def test_inclusion_substitutes_toxicity_once():
    from pilot.gate_v2 import inclusion
    from pilot.v2common import load_v2_config

    cfg = load_v2_config("confirm")

    def g(ok):
        return {"gate": {"pass": ok, "reasons": [] if ok else ["AUROC low"]}}
    gates = {"Qwen/Qwen3-1.7B": {"tasks": {"truth": g(True), "refusal": g(True), "sentiment": g(True),
                                           "answerability": g(False), "toxicity": g(True)}, "sae": {}},
             "Qwen/Qwen3.5-4B": {"tasks": {"truth": g(False), "refusal": g(True), "sentiment": g(False),
                                           "answerability": g(False), "toxicity": g(True)}, "sae": {}}}
    inc = inclusion(cfg, gates)
    pairs = {(p["model"], p["task"]) for p in inc["pairs"]}
    assert ("Qwen/Qwen3-1.7B", "toxicity") in pairs and ("Qwen/Qwen3-1.7B", "answerability") not in pairs
    assert ("Qwen/Qwen3.5-4B", "toxicity") in pairs                   # replaces sentiment
    exc = {(e["model"], e["task"]) for e in inc["excluded"]}
    assert ("Qwen/Qwen3.5-4B", "answerability") in exc                # toxicity already used once
    assert ("Qwen/Qwen3.5-4B", "truth") in exc                        # pilot task failing on 4B: excluded


def test_analysis_seeding_matches_utils():
    from pilot.utils import rng_for

    assert A._rng(0, "m", "t", "bootstrap-v2").integers(10**9, size=5).tolist() == \
        rng_for(0, "m", "t", "bootstrap-v2").integers(10**9, size=5).tolist()
