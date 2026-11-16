"""`analyze` stage outputs: summary.csv, hypotheses.json, plots and report.md."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import plots as PL
from .analysis import run_analysis
from .utils import LOG, fmt_secs, read_json, write_json

DAMAGE_LABEL = {"truth": "damage: AUROC drop of m(x)", "refusal": "damage: drop in refusal rate (clean-refused harmful)"}


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def _ci(samples, alpha):
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summary_table(res: dict, alpha: float) -> pd.DataFrame:
    rows = []
    for task in ("truth", "refusal"):
        r = res[task]
        for key, v in r["cells_p"].items():
            s, b, rule = key
            lo, hi = _ci(r["cells_b"][key]["D"], alpha)
            g = r["G_p"].get(key)
            gb = r["G_b"].get(key)
            glo, ghi = _ci(gb, alpha) if gb is not None and np.isfinite(gb).any() else (np.nan, np.nan)
            rnd = r["cells_p"].get((s, b, "random"))
            gden = (float(r["anchor_p"][s]["D"][0]) - float(rnd["D"][0])) if (rnd is not None and s in r["anchor_p"]) else np.nan
            rows.append({"task": task, "strategy": s, "basis": b, "rule": rule,
                         "site": "mlp_down_proj_input (different site)" if b == "mlp" else "resid_post",
                         "D_at_cstar": float(v["D"][0]), "D_ci_lo": lo, "D_ci_hi": hi,
                         "G": float(g[0]) if g is not None else np.nan, "G_ci_lo": glo, "G_ci_hi": ghi,
                         "G_denominator": gden,       # D_anchor - D_random; G is undefined when ~0
                         "never_reached_cstar_frac": v["never_frac"], "below_cheapest_point_frac": v["below_frac"],
                         "n_curves": v["n_curves"], "k_at_cstar": v["k_at_cstar"],
                         "bias_shift_at_cstar": r["sec"]["bias_shift"].get(key, np.nan),
                         "coherence_at_cstar": r["sec"]["coherence"].get(key, np.nan),
                         "rel_change_at_cstar": r["sec"]["cost_rel"].get(key, np.nan)})
        for s, a in r["anchor_p"].items():
            lo, hi = _ci(r["anchor_b"][s]["D"], alpha)
            rows.append({"task": task, "strategy": s, "basis": "anchor", "rule": "massmean", "site": "resid_post",
                         "D_at_cstar": float(a["D"][0]), "D_ci_lo": lo, "D_ci_hi": hi, "G": 1.0,
                         "cstar_kl": float(a["cstar"][0])})
        for s, a in r["random_dir_p"].items():
            rows.append({"task": task, "strategy": s, "basis": "anchor", "rule": "random_dir (mean of draws)",
                         "site": "resid_post", "D_at_cstar": float(a["D"][0]), "cstar_kl": float(a["cost"][0])})
    return pd.DataFrame(rows)


def _jsonable_hyp(res: dict) -> dict:
    out = {"decisions": res["decisions"], "overall": res["overall"], "H6": res["H6"]}
    for task in ("truth", "refusal"):
        out[task] = res[task]["hyp"]
    raw = json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

    def clean(o):                                   # undefined statistics -> null (shown as n/a)
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return clean(raw)


def _hyp_rows(res: dict) -> list[tuple]:
    rows = []
    for task in ("truth", "refusal"):
        h = res[task]["hyp"]
        h0 = h["H0"]
        worst = max(h0["random_checks"], key=lambda r: r["ratio"]) if h0["random_checks"] else None
        stat = f"max D_random/D_anchor = {_fmt(worst['ratio'])}" if worst else "n/a"
        if h0.get("anchor_generation"):
            stat += f"; anchor generation drop {_fmt(list(h0['anchor_generation']['drops'].values()))}"
        rows.append((task, "H0", stat, "", "", h0["pass"]))
        for hid in ("H1", "H2", "H3", "H4", "H5"):
            tests = h[hid]["tests"]
            if not tests:
                rows.append((task, hid, "not computable (missing cells)", "", "", False))
                continue
            key = (lambda t: (not t.get("pass", False), t.get("q", 1.0)))
            best = sorted(tests, key=key)[0]
            lab = {k: best[k] for k in ("strategy", "rule", "basis", "pair", "bases", "rules", "cells") if k in best}
            rows.append((task, hid, f"{_fmt(best['stat'])} {lab} ({sum(t.get('pass', False) for t in tests)}/{len(tests)} tests pass)",
                         _fmt(best["ci"]), _fmt(best.get("q")), h[hid]["pass"]))
    for s, v in res["H6"]["by_strategy"].items():
        rows.append(("both", f"H6 ({s})", f"Kendall tau = {_fmt(v['tau'])}", _fmt(v["ci"]), "", v["pass"]))
    return rows


def write_outputs(ctx) -> dict:
    cfg = ctx.cfg
    alpha = float(cfg["thresholds"]["alpha"])
    res = run_analysis(ctx)
    adir = ctx.d("analyze")
    pdir = ctx.d("plots")
    summ = summary_table(res, alpha)
    summ.to_csv(adir / "summary.csv", index=False)
    write_json(adir / "hypotheses.json", _jsonable_hyp(res))
    strategies = list(cfg["strategies"])
    for task in ("truth", "refusal"):
        for s in strategies:
            PL.curves(res[task], s, pdir / f"curves_{task}_{s}.png", DAMAGE_LABEL[task])
        PL.gap_heatmap(res[task], strategies, pdir / f"gap_closure_{task}.png")
    st = pd.read_csv(ctx.run_dir / "select" / "stability.csv")
    PL.stability(st, pdir / "stability.png")
    write_report(ctx, res, summ, st)
    LOG.info("DECISION: %s | truth: %s | refusal: %s | H6 %s", res["overall"], res["decisions"]["truth"]["outcome"],
             res["decisions"]["refusal"]["outcome"], "holds" if res["H6"]["pass"] else "does not hold")
    return {"overall": res["overall"], **{t: res["decisions"][t]["outcome"] for t in ("truth", "refusal")}}


def projection(ctx) -> dict | None:
    """Projected full-run ablation time from the measured ablation throughput."""
    p = ctx.run_dir / "ablate" / "throughput.json"
    if not p.exists():
        return None
    thr = read_json(p)
    from .utils import load_pilot_config

    full = load_pilot_config(False)
    dsum = read_json(ctx.run_dir / "data" / "data_summary.json")["splits"]
    mean_tok = {k: v["mean_tokens"] for k, v in dsum.items()}
    nb = len(full["bases"])
    n_cfg = nb * (3 * full["discovery_resamples"] + full["random_draws"]) * len(full["budgets"]) * len(full["strategies"])
    n_cfg += len(full["strategies"]) * (1 + full["anchors"]["random_dir"])
    gen = full["data"]["generic"]["n_cost"]
    tok_truth = n_cfg * (1496 * mean_tok["truth/eval"] + gen * mean_tok["generic/cost"])
    n_ref = min(500, 572) + 500
    tok_ref = n_cfg * (n_ref * mean_tok["refusal/eval"] + gen * mean_tok["generic/cost"])
    total_tok = tok_truth + tok_ref
    tps = float(thr["throughput_tokens_per_s"])
    params = {m: float(v["approx_nonembedding_params"]) for m, v in full["models"].items()}
    measured_params = ctx.manifest().get("model", {}).get("params", {}).get("non_embedding") or params[ctx.model_id]
    out = {"measured_on": ctx.manifest().get("model", {}).get("device"), "measured_model": ctx.model_id,
           "measured_tokens_per_s": tps, "fast_path": thr.get("fast_path"), "configs_per_task": n_cfg,
           "full_run_prompt_tokens": total_tok, "ablation_hours": {}}
    for m in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-4B"):
        scale = params[m] / float(measured_params)
        out["ablation_hours"][m] = total_tok / tps * scale / 3600
    return out


def write_report(ctx, res: dict, summ: pd.DataFrame, st: pd.DataFrame) -> None:
    man = ctx.manifest()
    gate = ctx.gate
    alpha = float(ctx.cfg["thresholds"]["alpha"])
    L = []
    dec = man.get("decision") or {}
    L.append(f"# Pilot report: basis × selection-rule ablation{' (SMOKE RUN: numbers are meaningless)' if ctx.smoke else ''}")
    L.append("")
    L.append(f"**Model: `{ctx.model_id}`** · Step 0 decision: "
             + (f"`{dec.get('status')}` (decision.json sha256 `{str(dec.get('file_sha256'))[:12]}`, rule hash "
                f"`{str(dec.get('rule_hash'))[:12]}`)" if dec else
                ("not used (DEV tiny random model: plumbing test only)" if man.get("dev_tiny_model")
                 else "not used (smoke run forces Qwen3-1.7B)"))
             + f" · layer {man.get('layer')} of {man.get('model', {}).get('n_layers')} · SAE `{man.get('sae', {}).get('repo')}` "
               f"`{man.get('sae', {}).get('file')}`")
    L.append("")
    L.append("## 1. Decision")
    L.append("")
    under = gate.get("refusal_eval", {}).get("underpowered")
    L.append(f"**DECISION (overall): {res['overall']}** · H6 task consistency: "
             f"{'holds' if res['H6']['pass'] else 'does not hold'} "
             f"({', '.join(f'{s}: tau={_fmt(v['tau'])}' for s, v in res['H6']['by_strategy'].items()) or 'n/a'})")
    L.append("")
    for t in ("truth", "refusal"):
        d = res["decisions"][t]
        note = " (refusal underpowered: see gate)" if t == "refusal" and under else ""
        L.append(f"- **{t}: {d['outcome']}**{note}; hypotheses holding: {', '.join(d['held']) or 'none'}")
        if d["outcome"] == "INVALID HARNESS":
            h0 = res[t]["hyp"]["H0"]
            fails = [f"random/anchor ratio {_fmt(c['ratio'])} for {c['strategy']}/{c['basis']}"
                     for c in h0["random_checks"] if not c["pass"]]
            if h0.get("anchor_generation") and not h0["anchor_generation"]["pass"]:
                fails.append(f"anchor generation drop {h0['anchor_generation']['drops']} < {h0['anchor_generation']['threshold']}")
            L.append(f"  - H0 failed: {'; '.join(fails) or 'see hypotheses.json'}")
    L.append("")
    L.append("Decision rules: H0 fails → INVALID HARNESS; H4 or H5 → OUTCOME 1; H2 and H3 without H4 → OUTCOME 2; "
             "none of H1–H5 → OUTCOME 3; otherwise MIXED. Aggregation over strategies: H1/H4 any, H2/H3 all, H5 across "
             "mean vs resample. Tests pass when the 95% bootstrap CI excludes 0 and BH q ≤ "
             f"{alpha} within the hypothesis family. The final call is a human judgment from this report and the plots.")
    L.append("")
    L.append("## 2. Hypotheses")
    L.append("")
    L.append("| Task | ID | Statistic (best test) | 95% CI | q | Pass |")
    L.append("|---|---|---|---|---|---|")
    for row in _hyp_rows(res):
        L.append("| " + " | ".join(str(x) for x in row[:5]) + f" | {'yes' if row[5] else 'no'} |")
    L.append("")
    L.append("Full statistics, every test and q-values: `analyze/hypotheses.json`. Per-cell D@c*, G and CIs: "
             "`analyze/summary.csv`.")
    L.append("")
    L.append("### D@c* per cell (point estimate [95% CI])")
    L.append("")
    cells = summ[(summ.basis != "anchor")]
    for t in ("truth", "refusal"):
        for s in ctx.cfg["strategies"]:
            sub = cells[(cells.task == t) & (cells.strategy == s)]
            if sub.empty:
                continue
            rules = [r for r in ("meandiff", "l1probe", "attrib", "random") if r in set(sub.rule)]
            L.append(f"**{t} · {s}** (c* = {_fmt(float(summ[(summ.task == t) & (summ.strategy == s) & (summ.rule == 'massmean')].cstar_kl.iloc[0]), 4)} nats; "
                     f"anchor D = {_fmt(float(summ[(summ.task == t) & (summ.strategy == s) & (summ.rule == 'massmean')].D_at_cstar.iloc[0]))})")
            L.append("")
            L.append("| basis | " + " | ".join(rules) + " |")
            L.append("|---|" + "---|" * len(rules))
            for b in [b for b in ("sae", "pca", "dims", "mlp") if b in set(sub.basis)]:
                cellstr = []
                for r in rules:
                    x = sub[(sub.basis == b) & (sub.rule == r)]
                    if x.empty:
                        cellstr.append("")
                        continue
                    x = x.iloc[0]
                    g = "" if r == "random" or not np.isfinite(x.G) else f" G={x.G:.2f}"
                    flag = " ⚑" if x.never_reached_cstar_frac > 0 else ""
                    cellstr.append(f"{x.D_at_cstar:.3f} [{x.D_ci_lo:.3f}, {x.D_ci_hi:.3f}]{g}{flag}")
                L.append(f"| {b}{' (MLP site)' if b == 'mlp' else ''} | " + " | ".join(cellstr) + " |")
            L.append("")
    L.append("⚑ = at least one curve never reached c*; its maximum damage is reported.")
    L.append("")
    L.append("## 3. Plots")
    L.append("")
    for t in ("truth", "refusal"):
        for s in ctx.cfg["strategies"]:
            L.append(f"![{t} {s}](plots/curves_{t}_{s}.png)")
        L.append(f"![gap closure {t}](plots/gap_closure_{t}.png)")
    L.append("")
    L.append("## 4. Gate results and model")
    L.append("")
    L.append(f"- Model `{ctx.model_id}` ({man.get('model', {}).get('params', {}).get('non_embedding', 0) / 1e9:.2f}B "
             f"non-embedding params), {man.get('model', {}).get('dtype')} on {man.get('model', {}).get('device')}. "
             f"Reason: {man.get('model_reason')}")
    sc = gate.get("sae_check", {})
    L.append(f"- SAE sanity at layer {man.get('layer')}: FVE {_fmt(sc.get('fve'))} (min {sc.get('thresholds', {}).get('min_fve')}), "
             f"next-token CE increase {_fmt(sc.get('ce_increase'))} nats (max {sc.get('thresholds', {}).get('max_ce_increase')}), "
             f"L0 {_fmt(sc.get('l0'), 1)}, sink tokens excluded {_fmt(100 * sc.get('sink_fraction', 0), 2)}% → "
             f"{'PASS' if sc.get('pass') else 'FAIL'}"
             + (" (base-trained SAE on the instruct model: this is the transfer check)" if sc.get("base_to_instruct_transfer_check") else ""))
    at = gate.get("answer_tokens", {})
    L.append(f"- Answer tokens {at.get('tokens')} (variant `{at.get('variant')}`; mean mass {at.get('mass_by_variant')})")
    rf = gate.get("refusal", {})
    L.append(f"- Refusal-start tokens {rf.get('start_tokens')}; threshold r > {_fmt(rf.get('threshold'))}; classifier "
             f"agreement with generation labels {_fmt(rf.get('classifier_agreement'))}; damage mode `{rf.get('damage_mode')}`")
    re_ = gate.get("refusal_eval", {})
    L.append(f"- Refusal eval: {re_.get('n_refused_harmful')} of {re_.get('n_harmful')} harmful prompts refused by the clean "
             f"model (need ≥ {re_.get('min_refused_harmful')}); harmless refusal rate {_fmt(re_.get('refusal_rate_harmless'))} "
             f"(max {re_.get('max_refuse_harmless')}) → {'UNDERPOWERED' if re_.get('underpowered') else 'adequately powered'}")
    cm = read_json(ctx.run_dir / "cache" / "clean_metrics.json")
    L.append(f"- Clean baselines: truth eval AUROC {_fmt(cm['truth']['auroc'])}, accuracy {_fmt(cm['truth']['accuracy_at_0'])}, "
             f"coherence {_fmt(cm['truth']['coherence'])}; refusal AUROC of r(x) {_fmt(cm['refusal']['auroc_r'])}")
    chk = read_json(ctx.run_dir / "ablate" / "checks.json")
    idc = {f"{t}/{b}": _fmt(v["max_abs_logit_change"], 4) for t, d in chk["identity"].items() for b, v in d.items()}
    L.append(f"- Identity check (v = r(x), max |Δlogit|): {idc}")
    L.append(f"- Fast path: { {t: v.get('enabled') for t, v in chk['fast_path'].items()} }")
    gpath = ctx.run_dir / "ablate" / "generation_checks.csv"
    if gpath.exists() and gpath.stat().st_size > 1:
        g = pd.read_csv(gpath)
        L.append("")
        L.append("Generation checks (greedy, substring refusal labels, harmful eval prompts the clean model refuses):")
        L.append("")
        L.append("| strategy | configuration | k | n | refusal rate | drop vs clean |")
        L.append("|---|---|---|---|---|---|")
        for r in g.itertuples():
            L.append(f"| {r.strategy if isinstance(r.strategy, str) else ''} | {r.label} | {r.k} | {r.n_prompts} | "
                     f"{r.refusal_rate_gen:.3f} | {r.drop:.3f} |")
    L.append("")
    L.append("## 5. Selection stability")
    L.append("")
    if st.empty:
        L.append("Not computed: needs ≥ 2 discovery resamples.")
    else:
        ks = sorted(st.k.unique())
        ks = [k for k in ks if k in (1, 10, 100, 500)] or ks
        piv = st[st.k.isin(ks)].pivot_table(index=["task", "basis", "rule"], columns="k", values="jaccard_mean")
        L.append("| task | basis | rule | " + " | ".join(f"k={k}" for k in piv.columns) + " |")
        L.append("|---|---|---|" + "---|" * len(piv.columns))
        for idx, row in piv.iterrows():
            L.append("| " + " | ".join(map(str, idx)) + " | " + " | ".join(_fmt(float(v)) for v in row.values) + " |")
    L.append("")
    L.append("![stability](plots/stability.png)")
    L.append("")
    L.append("## 6. Caveats triggered during this run")
    L.append("")
    cav = []
    if ctx.smoke:
        cav.append("Smoke run: tiny splits and budgets; statistics are not interpretable.")
    if man.get("env", {}).get("gpu_only_checks_untested"):
        cav.append("No GPU: run on CPU in fp32; GPU-only paths (cu130 build, capability (12,1), bf16 numerics) untested.")
    if man.get("dev_tiny_model"):
        cav.append("DEV tiny random model: behavioural checks relaxed; numbers are meaningless.")
    nev = summ[(summ.get("never_reached_cstar_frac", pd.Series(dtype=float)) > 0)]
    if len(nev):
        cav.append(f"{len(nev)} cells have curves that never reach c* (maximum damage reported, flagged ⚑).")
    sel = ctx.run_dir / "select" / "l1_diagnostics.json"
    if sel.exists():
        dg = read_json(sel)
        if any(v.get("prefiltered") for v in dg.values()):
            cav.append(f"l1probe on the SAE basis was prefiltered to the top {ctx.cfg['l1probe']['sae_prefilter']} latents by |d|.")
        unc = sum(v["unconverged_steps"] for v in dg.values())
        if unc:
            cav.append(f"l1probe: {unc} regularisation-path steps hit max_iter without converging (see select/l1_diagnostics.json).")
    if rf.get("damage_mode") == "generation":
        cav.append("Refusal classifier agreement < threshold: refusal damage uses generation labels for all configurations.")
    if under:
        cav.append("Refusal results are underpowered (too few clean refusals or too many harmless refusals).")
    sink = man.get("sink", {})
    if sink:
        cav.append(f"Attention-sink tokens (norm > {ctx.cfg['sink_norm_multiple']}× median) excluded from SAE/PCA/discovery "
                   f"statistics: {_fmt(100 * sink.get('fraction_excluded_pool', 0), 3)}% of pooled tokens.")
    if not all(v.get("enabled") for v in chk["fast_path"].values()):
        cav.append("Fast path disabled for at least one task (equivalence test failed or off); full forward passes used.")
    if ctx.cfg["anchors"].get("das"):
        cav.append("DAS skyline requested but not implemented in this build.")
    cav.append("Interventions are applied on the post-instruction template positions only (also during generation checks).")
    L += [f"- {c}" for c in cav]
    L.append("")
    L.append("## 7. Runtime per stage")
    L.append("")
    L.append("| stage | wall time | peak torch alloc (GB) | min MemAvailable (GB) | peak used (GB, /proc/meminfo) |")
    L.append("|---|---|---|---|---|")
    for s, v in man.get("stages", {}).items():
        mem = v.get("memory", {})
        L.append(f"| {s} | {fmt_secs(v.get('seconds', 0))} | {mem.get('torch_max_memory_allocated_gb')} | "
                 f"{mem.get('min_mem_available_gb')} | {mem.get('peak_mem_used_gb')} |")
    pj = projection(ctx)
    if pj:
        L.append("")
        L.append(f"**Measured ablation throughput:** {pj['measured_tokens_per_s']:.0f} prompt tokens/s on "
                 f"{pj['measured_on']} with `{pj['measured_model']}` (fast path {pj['fast_path']}). Projected full-run "
                 f"ablation stage ({pj['configs_per_task']} configs/task, {pj['full_run_prompt_tokens'] / 1e6:.0f}M prompt "
                 "tokens): " + ", ".join(f"`{m}` ≈ {h:.1f} h" for m, h in pj["ablation_hours"].items())
                 + ". Trust this over the spec's table; a CPU measurement does not transfer to the GB10.")
        write_json(ctx.run_dir / "analyze" / "projection.json", pj)
        LOG.info("PROJECTION (measured %.0f prompt tokens/s on %s): full-run ablation stage %s", pj["measured_tokens_per_s"],
                 pj["measured_on"], ", ".join(f"{m} ~ {h:.1f} h" for m, h in pj["ablation_hours"].items()))
    (ctx.run_dir / "report.md").write_text("\n".join(L) + "\n")
