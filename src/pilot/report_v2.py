"""v2 `analyze` stage: confirmatory analysis first (analyze/confirmatory.json), then secondary, development and
exploratory outputs, plots, and report_v2.md (confirm / smoke) or dev_report.md (dev)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis_v2 as A
from . import plots_v2 as PV
from .utils import LOG, fmt_secs, read_json, write_json


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    return f"{x:.{nd}f}"


def _wait_workers(run_dir: Path, timeout: float = 3600):
    p = run_dir / "analyze_worker.pids"
    if not p.exists():
        return
    t0 = time.time()
    for pid in [int(x) for x in p.read_text().split()]:
        while time.time() - t0 < timeout:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            time.sleep(2)


def _pairs_as(pairs, role):
    return [{**p, "role": "confirmatory"} for p in pairs if p["role"] == role]


def stage_analyze(run) -> dict:
    run_dir = run.run_dir
    _wait_workers(run_dir)
    cfg = run.cfg
    hyp = A.load_hyp(run_dir)
    B = A.bootstrap_B(run_dir)
    an = A.Analysis(run_dir, B, int(cfg["seed"]), hyp)
    inc = read_json(run_dir / "inclusion.json")
    pairs = inc["pairs"]
    adir = run.d("analyze")
    # 1. preregistered confirmatory analysis first
    conf = A.confirmatory(an, pairs)
    write_json(adir / "confirmatory.json", A.clean_json(conf))
    if conf["C1"]["n_pairs"]:
        LOG.info("CENTRAL CLAIM: %s (C1 %s, C2 %s, C3 %s)", conf["label"], conf["C1"]["supported"],
                 conf["C2"]["supported"], conf["C3"]["supported"])
    else:
        LOG.info("no confirmatory pairs in this run (development pairs only): no central-claim label")
    # 2. everything else
    sec = A.secondary(an, pairs)
    write_json(adir / "secondary.json", A.clean_json(sec))
    dev = A.confirmatory(an, _pairs_as(pairs, "development"))
    write_json(adir / "development.json", A.clean_json({**dev, "note": "DEVELOPMENT pairs: never counted toward C1-C3"}))
    ex = run.d("analyze", "exploratory")
    A.exploratory_cells(an).to_csv(ex / "cells_exploratory.csv", index=False)
    pdir = run.d("plots")
    figs = []
    for p in pairs:
        cell = an.primary_cell(p["model"], p["task"])
        if cell:
            name = f"curves_{p['model'].split('/')[-1]}_{p['task']}.png"
            lab = "damage: AUROC drop" if p["task"] != "refusal" else "damage: drop in generation refusal rate"
            if PV.curves(an, cell, pdir / name, lab, f"{p['model']} · {p['task']} ({p['role']}) · 50% depth, S1, mean"):
                figs.append(name)
    for m in sorted({p["model"] for p in pairs}):
        mp = [p for p in pairs if p["model"] == m]
        cells = [an.primary_cell(p["model"], p["task"]) for p in mp]
        keep = [(c, f"{p['task']} ({p['role'][:3]})") for c, p in zip(cells, mp) if c]
        name = f"E_heatmap_{m.split('/')[-1]}.png"
        if PV.e_heatmap(an, [k[0] for k in keep], [k[1] for k in keep], pdir / name):
            figs.append(name)
        name = f"delta_by_depth_{m.split('/')[-1]}.png"
        if PV.delta_by_depth(sec["S1_depth"], m, pdir / name):
            figs.append(name)
    if PV.c3_scatter(an, [p for p in pairs if p["role"] == "confirmatory"] or pairs, pdir / "c3_scatter.png"):
        figs.append("c3_scatter.png")
    name = "dev_report.md" if run.phase == "dev" else "report_v2.md"
    text = dev_report(run, an, conf, dev, sec, figs) if run.phase == "dev" else report(run, an, conf, dev, sec, inc, figs)
    (run_dir / name).write_text(text)
    return {"label": conf["label"], "report": name}


# ============================================================================= tables

def _h0_lookup(sec: dict) -> dict:
    return {(r["model"], r["task"], r["layer"], r["site"], r["strategy"]): r.get("status", "")
            for r in sec["S5_anchors_h0"]}


def _c_tables(an, conf: dict, sec: dict, label: str) -> list[str]:
    L = []
    h0 = _h0_lookup(sec)
    c1, c2, c3 = conf["C1"], conf["C2"], conf["C3"]
    L.append(f"| ID | Claim | Result | Statistic | Supported |")
    L.append("|---|---|---|---|---|")
    L.append(f"| C1 (primary) | Causal selection beats readout selection | {c1['n_supported']}/{c1['n_pairs']} {label} pairs "
             f"supported (need {c1['required']}; per model {c1['per_model_supported']}) | mean Delta per pair, see below | "
             f"{'**yes**' if c1['supported'] else 'no'} |")
    L.append(f"| C2 (primary) | Basis rankings depend on the selection rule | reversal in {c2['fraction_with_reversal']:.0%} of pairs "
             f"(need {c2['required_fraction']:.0%}) | CI-separated reversals | {'**yes**' if c2['supported'] else 'no'} |")
    L.append(f"| C3 (secondary) | Readout strength only weakly predicts causal effect | pooled rho {_f(c3.get('pooled_rho'))} "
             f"95% CI {_f(c3.get('ci'))} | need rho <= 0.3 and CI upper < 0.5 | {'**yes**' if c3['supported'] else 'no'} |")
    L.append("")
    L.append(f"Per {label} pair (primary setting: 50% depth, S1, mean ablation; c0 = {an.c0} nats; delta = {an.delta}):")
    L.append("")
    L.append("| model | task | bases | mean Delta [95% CI] | q | Delta per basis | C1 | C2 reversal | H0 (caveat only) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rev = {(p["model"], p["task"]): p["reversal"] for p in c2["per_pair"]}
    for r in c1["per_pair"]:
        cell = an.primary_cell(r["model"], r["task"])
        db = "; ".join(f"{b} {_f(v['stat'])} {_f(v['ci'])}" + (" ⚑" if v.get("never_reached_frac") else "")
                       for b, v in r["delta_b"].items())
        L.append(f"| {r['model']} | {r['task']}{' (' + r['note'] + ')' if r.get('note') else ''} | {', '.join(r['bases'])} | "
                 f"{_f(r['stat'])} {_f(r['ci'])} | {_f(r.get('q'))} | {db} | {'yes' if r['supported'] else 'no'} | "
                 f"{'yes' if rev.get((r['model'], r['task'])) else 'no'} | {h0.get(cell, 'n/a') if cell else 'n/a'} |")
    if conf["excluded_pairs"]:
        L.append("")
        L.append("Not analysed: " + "; ".join(f"{e['model']}/{e['task']} ({e['reason']})" for e in conf["excluded_pairs"]))
    L.append("")
    L.append("⚑ = at least one curve never reached c0 (its maximum damage is used). C2 reversals tested per basis pair and "
             "readout rule; details in the JSON.")
    return L


def _secondary_md(sec: dict, roles=("confirmatory", "development")) -> list[str]:
    L = ["### S1 Depth: Delta by depth (S1, mean)", "", "| model | task | role | depth (layer) | mean Delta [CI] | per basis |",
         "|---|---|---|---|---|---|"]
    for r in sorted(sec["S1_depth"], key=lambda r: (r["model"], r["task"], r["depth"])):
        if r["role"] in roles:
            L.append(f"| {r['model']} | {r['task']} | {r['role']} | {r['depth']:.2f} ({r['layer']}) | {_f(r['delta_mean'])} "
                     f"{_f(r['ci'])} | " + "; ".join(f"{b} {_f(v['stat'])}" for b, v in r["delta_b"].items()) + " |")
    L += ["", "### S2 Site: refusal Delta under S2 vs S1 (50% depth, mean)", ""]
    sec = {**sec, "S2_site": [r for r in sec["S2_site"] if r["role"] in roles],
           "S3_strategy": [r for r in sec["S3_strategy"] if r["role"] in roles],
           "S4": {**sec["S4"], "a_shuffled": [r for r in sec["S4"]["a_shuffled"] if r["role"] in roles],
                  "b_cross_task": [r for r in sec["S4"]["b_cross_task"] if r["role"] in roles],
                  "c_attribution": sec["S4"]["c_attribution"] if "confirmatory" in roles
                  else sec["S4"].get("c_attribution_development")}}
    for r in sec["S2_site"]:
        L.append(f"- {r['model']} ({r['role']}): S1 {_f(r['delta_S1'])} {_f(r['ci_S1'])}, S2 {_f(r['delta_S2'])} "
                 f"{_f(r['ci_S2'])}; S2 − S1 = {_f(r['diff_S2_minus_S1'])} {_f(r['diff_ci'])}")
    if not sec["S2_site"]:
        L.append("- not available (block not run or refusal excluded)")
    L += ["", "### S3 Strategy: refusal, mean vs resample (50% depth)", ""]
    for r in sec["S3_strategy"]:
        L.append(f"- {r['model']} ({r['role']}): Kendall tau over {r['n_cells']} basis x rule cells = {_f(r['kendall_tau'])} "
                 f"{_f(r['ci'])}")
    if not sec["S3_strategy"]:
        L.append("- not available")
    s4 = sec["S4"]
    L += ["", "### S4 Circularity controls", "", "(a) attribution with shuffled labels, E(c0) (should be within ±delta of 0):", ""]
    for r in s4["a_shuffled"]:
        L.append(f"- {r['model']}/{r['task']}/{r['basis']} ({r['role']}): {_f(r['E'])} {_f(r['ci'])} -> "
                 f"{'within' if r['within_delta'] else 'OUTSIDE'} ±delta")
    L += ["", "(b) cross-task specificity, E_B(units attributed for A) vs E_B(units attributed for B):", ""]
    for r in s4["b_cross_task"]:
        L.append(f"- {r['model']}: A={r['source_task_A']} -> B={r['task_B']}, {r['basis']}, k={r['k']}: {_f(r['E_B_units_A'])} vs "
                 f"{_f(r['E_B_units_B'])} -> {'specific' if r['specific'] else 'NOT specific'} (diff CI {_f(r['diff_ci'])})")
    ca = s4.get("c_attribution") or {}
    L += ["", f"(c) attribution accuracy (Spearman of −summed attribution vs group damage, pooled): {_f(ca.get('pooled_rho'))} "
              f"{_f(ca.get('ci'))}", ""]
    return L


def _anchors_md(sec: dict) -> list[str]:
    L = ["| model | task | role | layer | site | strategy | anchor D | c* | random_dir D | H0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sec["S5_anchors_h0"]:
        L.append(f"| {r['model']} | {r['task']} | {r['role']} | {r['layer']} | {r['site']} | {r['strategy']} | "
                 f"{_f(r.get('anchor_D'))} | {_f(r.get('cstar'), 4)} | {_f(r.get('random_dir_D'))} | {r.get('status')} |")
    return L


def _runtime_md(run) -> list[str]:
    man = run.manifest()
    L = ["| stage | wall time | peak torch alloc (GB) | min MemAvailable (GB) | peak used (GB) |", "|---|---|---|---|---|"]
    for s, v in man.get("stages", {}).items():
        m = v.get("memory", {})
        L.append(f"| {s} | {fmt_secs(v.get('seconds', 0))} | {m.get('torch_max_memory_allocated_gb')} | "
                 f"{m.get('min_mem_available_gb')} | {m.get('peak_mem_used_gb')} |")
    bp = run.run_dir / "ablate" / "blocks.json"
    if bp.exists():
        L += ["", "| block | status | wall time | projected |", "|---|---|---|---|"]
        for b, v in read_json(bp)["blocks"].items():
            L.append(f"| {b} | {v.get('status')} | {fmt_secs(v.get('seconds', 0))} | {fmt_secs(v.get('projected_seconds', 0))} |")
    env = man.get("env", {})
    L += ["", f"Hardware: {env.get('device', {}).get('name')} (capability {env.get('device', {}).get('capability')}), "
              f"{env.get('platform', {}).get('machine')}, torch {env.get('torch', {}).get('version')}."]
    return L


def _inclusion_md(inc: dict, run) -> list[str]:
    L = [f"Gates {'enforced' if inc['enforced'] else 'NOT enforced (smoke)'}.", ""]
    L.append("| model | task | role | gate | note |")
    L.append("|---|---|---|---|---|")
    for p in inc["pairs"]:
        L.append(f"| {p['model']} | {p['task']} | {p['role']} | {'pass' if p['gate_pass'] else 'FAIL'} | {p.get('note') or ''} |")
    for e in inc["excluded"]:
        L.append(f"| {e['model']} | {e['task']} | {e['role']} | EXCLUDED | {'; '.join(e['reasons'])} |")
    for s in inc["substitutions"]:
        L.append(f"| {s['model']} | {s['substitute']} | substitute | pass | replaces {s['failed']} ({'; '.join(s['reasons'])}) |")
    L += ["", "SAE inclusion per (model, layer): FVE >= 0.70 and next-token loss +<= 10%:", ""]
    for m, d in inc["sae"].items():
        L.append(f"- {m}: " + "; ".join(f"L{L_} {'included' if v['included'] else 'EXCLUDED'} (FVE {_f(v['fve'])}, loss "
                                         f"{_f(None if v['loss_rel_increase'] is None else 100 * v['loss_rel_increase'], 1)}%)"
                                         + ("" if v["included"] else f" — {v['reason']}") for L_, v in d.items()))
    L += [""] + _sae_setting_md(inc)
    return L


def _sae_setting_md(inc: dict) -> list[str]:
    """SAE layer / input-scaling setting used (from each release's model card)."""
    L = ["SAE setting used (from each release's model card; sources in config/v2.yaml):", ""]
    for m, s in (inc.get("sae_setting") or {}).items():
        L.append(f"- {m}: file named N used at layer N (offset {s['layer_offset']}), input scaling `{s['input_norm']}`.")
    return L


def report(run, an, conf, dev, sec, inc, figs) -> str:
    man = run.manifest()
    smoke = run.phase == "smoke"
    L = [f"# Pilot v2 report{' (SMOKE RUN: tiny data, numbers are meaningless)' if smoke else ''}"
         + (" — DEV TINY random models (plumbing only)" if run.dev_tiny else ""), ""]
    L.append(f"## 1. Central claim: **{conf['label']}**")
    L.append("")
    L += _c_tables(an, conf, sec, "confirmatory")
    L += ["", "## 2. Deviations from the preregistration and skipped blocks", ""]
    devs = man.get("deviations") or []
    L += [f"- DEVIATION ({d['timestamp']}): {d['reason']} — changed files: {', '.join(d['changed_files']) or 'none'}" for d in devs] \
        or ["- no deviations recorded"]
    bp = run.run_dir / "ablate" / "blocks.json"
    if bp.exists():
        for b, v in read_json(bp)["blocks"].items():
            if v.get("status") != "done":
                L.append(f"- block {b}: {v.get('status')}")
    L += ["", "## 3. Inclusion decisions", ""] + _inclusion_md(inc, run)
    L += ["", "## 4. Secondary analyses (S1–S5)", ""] + _secondary_md(sec, ("confirmatory",))
    L += ["", "## 5. Development pairs (Qwen3-1.7B truth and refusal) — DEVELOPMENT, never counted toward C1–C3", ""]
    if dev["C1"]["per_pair"]:
        L += _c_tables(an, dev, sec, "development")
        L += [""] + _secondary_md(sec, ("development",))
    else:
        L.append("No development-pair data.")
    L += ["", "## 6. Anchors and H0 per cell (H0 never changes a verdict)", ""] + _anchors_md(sec)
    L += ["", "## 7. Exploratory analyses (labelled exploratory)", "",
          "- `analyze/exploratory/cells_exploratory.csv`: D and E at c = 0.02, 0.1, 0.5 and D@c* for every cell, basis and rule.",
          ""]
    L += [f"![{f}](plots/{f})" for f in figs]
    L += ["", "## 8. Hardware, runtime per stage and peak memory", ""] + _runtime_md(run)
    return "\n".join(L) + "\n"


def dev_report(run, an, conf, dev, sec, figs) -> str:
    L = ["# Development run report (Qwen3-1.7B truth + refusal, 50% depth)", "",
         "Development data only: nothing here enters the confirmatory analysis. Use it to check the harness, the cost "
         "range and the CI widths before freezing the preregistration.", ""]
    L += ["## Development-pair statistics (same tables as the confirmatory report)", ""] + _c_tables(an, dev, sec, "development")
    L += ["", "## Anchors and H0", ""] + _anchors_md(sec)
    L += ["", "## Cost ranges per curve family (KL over 9 positions)", "", "| task | site | strategy | basis | rule | min cost | max cost | curves capped below target |",
          "|---|---|---|---|---|---|---|---|"]
    df = an.df.assign(cost=an.cost_p)
    for (t, s, st, b, r), g in df[df.basis != "anchor"].groupby(["task", "site", "strategy", "basis", "rule"]):
        capped = g[g.curve_flag == "cap_hit"].groupby(["resample", "draw"]).ngroups if "curve_flag" in g.columns else 0
        L.append(f"| {t} | {s} | {st} | {b} | {r} | {g.cost.min():.4f} | {g.cost.max():.4f} | {capped} |")
    grid = an.df[(an.df.kind == "grid") & (an.df.basis != "anchor")]
    if len(grid):
        n_ext = int(grid.ext.astype(bool).sum())
        n_lad = int((~grid.ext.astype(bool)).sum())
        capped = grid[grid.curve_flag == "cap_hit"].drop_duplicates(["task", "site", "strategy", "basis", "rule", "resample", "draw"])
        cap_txt = ", ".join(f"{b}/{r} {n}" for (b, r), n in capped.groupby(["basis", "rule"]).size().items()) or "none"
        L += ["", f"Curve extension added {n_ext} points to {n_lad} ladder points ({n_ext / max(n_lad, 1):.0%}); the confirm "
                  f"projection assumes {run.cfg['budget']['extension_factor']:.0%} (config budget.extension_factor) and the "
                  f"time guard switches to the measured rate once the run has one. Curves capped below target: {cap_txt}."]
    ag = run.run_dir / "ablate" / "agreement_32_64.json"
    L += ["", "## Refusal labels: 32 vs 64 generated tokens", ""]
    if ag.exists():
        for k, v in read_json(ag).items():
            L.append(f"- {k}: agreement {_f(v['agreement'])} on {v['n']} prompts (refusal rate {_f(v['rate_32'])} at 32 tokens, "
                     f"{_f(v['rate_64'])} at 64)")
    else:
        L.append("- not computed")
    L += ["", "## CI widths at c0 (compare with delta = 0.05)", ""]
    widths = []
    for r in dev["C1"]["per_pair"]:
        widths.append(r["ci"][1] - r["ci"][0])
        L.append(f"- {r['model']}/{r['task']}: mean-Delta CI width {_f(r['ci'][1] - r['ci'][0])}; per basis "
                 + ", ".join(f"{b} {_f(v['ci'][1] - v['ci'][0])}" for b, v in r["delta_b"].items()))
    if widths and max(widths) > 2 * an.delta:
        L.append(f"- **CI half-widths exceed delta = {an.delta}: consider raising the test-set sizes before freezing.**")
    L += ["", "## Throughput", ""]
    res = an.df
    for (m, t), g in res.groupby(["model", "task"]):
        L.append(f"- {m}/{t}: {len(g)} interventions, {g.seconds.mean():.2f} s each on average")
    L += [""] + [f"![{f}](plots/{f})" for f in figs]
    L += ["", "## Runtime", ""] + _runtime_md(run)
    return "\n".join(L) + "\n"
