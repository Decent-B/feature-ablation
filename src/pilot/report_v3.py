"""v3 analyze stage and reports.

confirm / smoke: analyze/{tests.json, secondary.json, methods.json} and report_v3.md (headline label first, then
the D1-D6 table, deviations and skipped blocks, inclusion, secondary analyses, anchors, runtime).
dev: dev_report.md with harness health only (gates, coherence, SAE fidelity, cost ranges, CI widths, anchors' cost,
flags). It withholds every Ē value and every test result, so all tasks stay confirmatory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .analysis_v3 import analyze, clean_json
from .utils import fmt_secs, read_json, write_json


def _f(x, nd=3):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def stage_analyze(run) -> dict:
    out = run.d("analyze")
    res = clean_json(analyze(run.run_dir))
    inc = read_json(run.run_dir / "inclusion.json")
    if run.phase == "dev":
        meth = [{k: v for k, v in m.items() if k not in ("E_bar", "ci")} for m in res.get("methods", [])]
        write_json(out / "dev.json", {"methods_ci_width_and_cost": meth, "anchors_cost": [
            {k: a[k] for k in ("model", "task", "layer", "massmean_cost", "random_dir_cost")} for a in res.get("anchors", [])]})
        (run.run_dir / "dev_report.md").write_text(dev_report(run, res, inc, meth))
        return {"report": "dev_report.md"}
    write_json(out / "tests.json", {k: res.get(k) for k in ("label", "tests", "delta", "alpha", "B", "cells", "inconclusive")})
    write_json(out / "secondary.json", res.get("secondary", {}))
    write_json(out / "methods.json", res.get("methods", []))
    (run.run_dir / "report_v3.md").write_text(report(run, res, inc))
    return {"label": res.get("label"), "tests": {k: v.get("holds") for k, v in (res.get("tests") or {}).items()}}


# ============================================================================= sections

def _tests_md(res: dict) -> list[str]:
    t = res["tests"]
    names = {"D1": "Basis", "D2": "Selection rule", "D3": "Task", "D4": "Model", "D5": "Depth", "D6": "SAE width"}
    L = ["| Test | Factor | Holds | Reliable comparisons | Largest change |", "|---|---|---|---|---|"]
    for d in ("D1", "D2", "D3", "D4", "D5", "D6"):
        v = t.get(d, {})
        if not v.get("computable", True):
            L.append(f"| {d} | {names[d]} | not computable | — | — |")
            continue
        big = v.get("largest") or v.get("largest_span") or {}
        bigs = (f"{big.get('x')} vs {big.get('y')}: {_f(big.get('point', big.get('span')))}" if big else "—")
        L.append(f"| {d} | {names[d]} | **{'yes' if v.get('holds') else 'no'}** | {v.get('n_reliable', 0)} of "
                 f"{v.get('n_comparisons', 0)} | {bigs} |")
    for sub in ("D4_sae_only", "D4_pca_dims"):
        v = t.get(sub, {})
        L.append(f"| {sub} | Model ({'SAE basis only' if 'sae' in sub else 'PCA + raw dims only'}) | "
                 f"{'yes' if v.get('holds') else 'no'} | {v.get('n_reliable', 0)} of {v.get('n_comparisons', 0)} | — |")
    return L


def _reversals_md(res: dict) -> list[str]:
    L = []
    for d in ("D3", "D4", "D5"):
        rv = (res["tests"].get(d) or {}).get("reversals") or []
        if rv:
            L += [f"**{d}** ({len(rv)} reversals; first 5):", ""]
            for r in rv[:5]:
                L.append(f"- {r['x']} beats {r['y']} at {r['x_better_at']} ({_f(r['diff_at_x_level'])}), and loses at "
                         f"{r['y_better_at']} ({_f(r['diff_at_y_level'])})")
            L.append("")
    return L


def _methods_md(res: dict) -> list[str]:
    L = ["| Model | Task | Layer | Strategy | Basis | Rule | Ē | 95% CI | Mean cost | Broken budgets |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in res.get("methods", []):
        L.append(f"| {m['model'].split('/')[-1]} | {m['task']} | {m['layer']} | {m['strategy']} | {m['basis']} | "
                 f"{m['rule']} | {_f(m['E_bar'])} | [{_f(m['ci'][0])}, {_f(m['ci'][1])}] | {_f(m['mean_cost'], 4)} | "
                 f"{m['broken_budgets']} |")
    return L


def _inclusion_md(inc: dict) -> list[str]:
    L = [f"Gates enforced: {inc['enforced']}. Reference task: **{inc.get('reference_task')}**"
         + (" (changed by the preregistered fallback rule)" if inc.get("reference_task_changed") else "") + ".", "",
         "| Model | Task | Clean AUROC | Coherence | Gate |", "|---|---|---|---|---|"]
    for m, ts in inc["baselines"].items():
        for t, v in ts.items():
            b = v.get("baseline") or {}
            L.append(f"| {m} | {t} | {_f(b.get('auroc'))} | {_f(b.get('coherence'))} | "
                     f"{'pass' if v['gate']['pass'] else 'FAIL ' + '; '.join(v['gate']['reasons'])} |")
    L += ["", "| Model | Layer | SAE used | Main FVE | Main loss | Main scaling | Layer index | Extra (16k / TopK 50) FVE |"
          " Extra loss |", "|---|---|---|---|---|---|---|---|---|"]
    for m, Ls in inc["sae"].items():
        for Lr, v in Ls.items():
            mn, ex = v["main"], v["extra"]
            L.append(f"| {m.split('/')[-1]} | {Lr} | {v['used'] or 'none (basis dropped)'} | {_f(mn.get('fve'))} | "
                     f"{_f(100 * mn['ce_rel_increase'] if mn.get('ce_rel_increase') is not None else None, 1)}% | "
                     f"{mn.get('input_norm')} | {'ok' if mn.get('layer_index_ok') else 'CHECK'} | {_f(ex.get('fve'))} | "
                     f"{_f(100 * ex['ce_rel_increase'] if ex.get('ce_rel_increase') is not None else None, 1)}% |")
    return L


def _secondary_md(sec: dict) -> list[str]:
    L = []
    s1 = sec.get("S1_readout_vs_causal") or {}
    if s1.get("pooled"):
        p = s1["pooled"]
        L += [f"- **S1 Readout vs causal effect:** pooled Spearman ρ = {_f(p['rho'])} [{_f(p['ci'][0])}, {_f(p['ci'][1])}]"
              f" (v2 bar: ρ ≤ 0.3 and upper CI < 0.5); attribution accuracy ρ = {_f(p['rho_attribution'])}."]
    s2 = sec.get("S2_strategy")
    if s2:
        k = s2.get("kendall_mean_vs_resample") or {}
        L.append(f"- **S2 Strategy (resample):** D1 {'holds' if s2['D1_resample']['holds'] else 'does not hold'}, D2 "
                 f"{'holds' if s2['D2_resample']['holds'] else 'does not hold'}; Kendall τ mean vs resample = "
                 f"{_f(k.get('tau'))} [{_f((k.get('ci') or [None, None])[0])}, {_f((k.get('ci') or [None, None])[1])}].")
    ct = sec.get("S3_cross_task") or []
    if ct:
        L.append(f"- **S3 Cross-task:** {sum(c['specific'] for c in ct)} of {len(ct)} (target, source, basis, k) "
                 "comparisons task-specific (cross-task damage < 0.5 × own-task damage).")
    for k, v in (sec.get("S4_single_budgets") or {}).items():
        L.append(f"- **S4 Single budget k = {k}:** " + ", ".join(f"{d} {'yes' if h else 'no'}" for d, h in v.items()))
    s5 = sec.get("S5_cost_matched")
    if s5:
        L.append(f"- **S5 Cost-matched (c = {s5['cost']}):** " + ", ".join(
            f"{d} {'yes' if h else 'no'} ({s5['n_comparisons'][d]} comparisons)" for d, h in s5["holds"].items()))
    return L


def _anchors_md(sec: dict) -> list[str]:
    L = ["| Model | Task | Layer | Strategy | massmean damage | massmean cost | random_dir damage | random_dir cost |",
         "|---|---|---|---|---|---|---|---|"]
    for a in sec.get("S7_anchors") or []:
        L.append(f"| {a['model'].split('/')[-1]} | {a['task']} | {a['layer']} | {a['strategy']} | {_f(a['massmean_damage'])} | "
                 f"{_f(a['massmean_cost'], 4)} | {_f(a['random_dir_damage'])} | {_f(a['random_dir_cost'], 4)} |")
    return L


def _runtime_md(run) -> list[str]:
    man = run.manifest()
    L = ["| Stage | Wall time |", "|---|---|"]
    for k, v in (man.get("stages") or {}).items():
        L.append(f"| {k} | {fmt_secs(v.get('seconds', 0))} |")
    bp = run.run_dir / "ablate" / "blocks.json"
    if bp.exists():
        L += ["", "| Block | Status | Wall time | Projected |", "|---|---|---|---|"]
        for b, v in read_json(bp)["blocks"].items():
            L.append(f"| {b} | {v.get('status')} | {fmt_secs(v['seconds']) if v.get('seconds') else '—'} | "
                     f"{fmt_secs(v['projected_seconds']) if v.get('projected_seconds') else '—'} |")
    return L


def _resources_md(run) -> list[str]:
    p = run.run_dir / "resources.json"
    ev = read_json(p).get("events", []) if p.exists() else []
    if not ev:
        return ["No waiting or batch changes: memory was free throughout."]
    L = ["Other jobs' memory use changed the batch sizes (speed only; results are unaffected):", "",
         "| Time | Stage | Event | Free memory (GB) | Batch factor | Detail |", "|---|---|---|---|---|---|"]
    for e in ev:
        det = e.get("reason") or e.get("note") or (f"waited {e['waited_s']} s" if "waited_s" in e else "") or \
            (f"headroom {e.get('headroom_gb')} GB" if "headroom_gb" in e else "")
        L.append(f"| {e['time']} | {e['stage']} | {e['kind']} | {e['mem_available_gb']} | x{e['factor']:g} | {det} |")
    return L


def _deviations_md(run) -> list[str]:
    devs = run.manifest().get("deviations") or []
    L = []
    if devs:
        L += ["**Deviations from the frozen preregistration:**", ""] + [f"- {d['timestamp']}: {d['reason']} "
                                                                       f"(changed: {', '.join(d['changed_files']) or 'none'})"
                                                                       for d in devs] + [""]
    bp = run.run_dir / "ablate" / "blocks.json"
    if bp.exists():
        sk = [b for b, v in read_json(bp)["blocks"].items() if str(v.get("status", "")).startswith(("skipped", "dropped"))]
        if sk:
            L += [f"**Blocks not run:** {', '.join(sk)} (time budget or preregistered drop list; speed only).", ""]
    return L or ["No deviations; every block ran.", ""]


def report(run, res: dict, inc: dict) -> str:
    L = ["# Pilot v3 report", "",
         f"Run `{run.run_dir.name}` ({run.phase}{', dev-tiny plumbing test' if run.dev_tiny else ''}).", "",
         f"## Headline: **{res.get('label')}**", "",
         "Feature causality depends on the basis (D1), the selection rule (D2), the task (D3) and the model (D4); "
         "depth (D5) and SAE width (D6) are reported beside it. A test holds if at least one comparison is reliable "
         f"(|difference| ≥ δ = {res.get('delta')}, 95% CI excludes 0, BH q ≤ α = {res.get('alpha')}); D3–D5 need a "
         "reliable reversal of a method pair between two levels.", ""]
    L += _tests_md(res) + [""] + _reversals_md(res)
    L += ["## Deviations and skipped blocks", ""] + _deviations_md(run)
    L += ["## Inclusion", ""] + _inclusion_md(inc) + [""]
    L += ["## Secondary analyses", ""] + _secondary_md(res.get("secondary", {})) + [""]
    L += ["## Ē per method (damage beyond random, averaged over the budget ladder)", ""] + _methods_md(res) + [""]
    L += ["## Anchors (S7)", ""] + _anchors_md(res.get("secondary", {})) + [""]
    L += ["## Runtime", ""] + _runtime_md(run) + [""]
    L += ["## Competing memory use", ""] + _resources_md(run) + [""]
    return "\n".join(L) + "\n"


def dev_report(run, res: dict, inc: dict, meth: list[dict]) -> str:
    L = ["# Pilot v3 development run (harness health only)", "",
         f"Run `{run.run_dir.name}`. This report withholds every Ē value and every test result.", "",
         "## Gates and SAE fidelity", ""] + _inclusion_md(inc) + ["", "## CI widths and costs per method", "",
                                                                  "| Task | Layer | Basis | Rule | CI width | Mean cost | Max cost | Broken budgets |",
                                                                  "|---|---|---|---|---|---|---|---|"]
    for m in meth:
        L.append(f"| {m['task']} | {m['layer']} | {m['basis']} | {m['rule']} | {_f(m['ci_width'])} | "
                 f"{_f(m['mean_cost'], 4)} | {_f(m['max_cost'], 4)} | {m['broken_budgets']} |")
    widths = [m["ci_width"] for m in meth if m.get("ci_width") is not None]
    if widths:
        L += ["", f"Median CI width {_f(float(np.median(widths)))} (δ = 0.05): "
              + ("narrow enough to resolve δ-sized differences." if np.median(widths) / 2 < 0.05 else
                 "wider than δ; consider larger test sets before freezing.")]
    L += ["", "## Runtime", ""] + _runtime_md(run) + [""]
    L += ["## Competing memory use", ""] + _resources_md(run) + [""]
    chk = run.run_dir / "ablate" / "checks.json"
    if chk.exists():
        L += ["## Fast-path checks", "", "```", json.dumps(read_json(chk), indent=1)[:3000], "```", ""]
    return "\n".join(L) + "\n"
