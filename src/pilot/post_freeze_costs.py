"""POST-FREEZE, EXPLORATORY: the preregistered statistics re-read at the secondary costs (0.02 and 0.5 nats).

Why: the preregistration fixes c0 = 0.1 for every test and lists 0.02 and 0.5 only as secondary costs, which the
frozen report shows as a per-cell CSV (analyze/exploratory/cells_exploratory.csv). This script asks what the C1/C2
statistics, the depth trend and the scrambled-label control look like at those costs. It reuses the frozen code
unchanged (analysis_v2.confirmatory / delta_b / E) and only swaps the cost the frozen code reads (Analysis.c0).
c0 = 0.1 reproduces analyze/confirmatory.json exactly. Nothing here changes the verdict; it must be labelled
post-freeze and exploratory.

Usage: .venv/bin/python -m pilot.post_freeze_costs <confirm run dir> [figure dir]
Writes <run dir>/analyze/post_freeze/costs.json and figures (to [figure dir] if given, else next to the JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from . import analysis_v2 as A  # noqa: E402
from .plots import BASIS_STYLE, DIVERGING, INK, INK2, RANDOM, SURFACE, _fig  # noqa: E402
from .plots_v2 import RULE_COLOR  # noqa: E402
from .v2common import load_v2_config  # noqa: E402

COSTS = (0.02, 0.1, 0.5)
RULES3 = ("meandiff", "l1probe", "attrib")
COST_COLOR = {0.02: "#8fb8e8", 0.1: "#2a78d6", 0.5: "#0d3c78"}


def _short(model: str) -> str:
    return model.split("/")[-1]


# ----------------------------------------------------------------------------- numbers

def at_cost(an: A.Analysis, pairs: list[dict], c: float) -> dict:
    an.c0 = c                                                   # the only change: the cost the frozen code reads
    conf = A.confirmatory(an, pairs)
    dev = A.confirmatory(an, [{**p, "role": "confirmatory"} for p in pairs if p["role"] == "development"])
    depth, shuffled = [], []
    for p in pairs:
        m, t = p["model"], p["task"]
        for cell in [x for x in an.cells() if x[0] == m and x[1] == t and x[3] == "S1" and x[4] == "mean"]:
            db = an.delta_b(cell)
            if db:
                sm = np.mean([v["samples"] for v in db.values()], axis=0)
                depth.append({"model": m, "task": t, "role": p["role"], "layer": cell[2],
                              "depth": float(an.cell_rows(cell).depth.iloc[0]),
                              "delta_mean": float(np.mean([v["stat"] for v in db.values()])), "ci": an.ci(sm),
                              "delta_b": {b: {"stat": v["stat"], "ci": v["ci"], "never_reached_frac": v["never_frac"]}
                                          for b, v in db.items()}})
        prim = an.primary_cell(m, t)
        if prim:
            for b in an.bases(prim):
                e = an.E(prim, b, "attrib_shuffled", c)
                if e is not None:
                    shuffled.append({"model": m, "task": t, "role": p["role"], "basis": b, "E": e[0], "ci": an.ci(e[1]),
                                     "within_delta": bool(abs(e[0]) <= an.delta)})
    out = {"cost": c,
           "C1": conf["C1"], "C2": conf["C2"],
           "label_C1_C2": "SUPPORTED" if (conf["C1"]["supported"] and conf["C2"]["supported"]) else
           ("PARTIALLY SUPPORTED" if (conf["C1"]["supported"] or conf["C2"]["supported"]) else "NOT SUPPORTED"),
           "development_C1": dev["C1"], "development_C2": dev["C2"],
           "S1_depth": depth, "S4a_shuffled": shuffled}
    an.c0 = 0.1
    return out


# ----------------------------------------------------------------------------- figures

def heatmap_grid(an: A.Analysis, pairs: list[dict], model: str, path: Path) -> None:
    mp = [p for p in pairs if p["model"] == model and an.primary_cell(p["model"], p["task"])]
    cells = [an.primary_cell(p["model"], p["task"]) for p in mp]
    nb = max(len(an.bases(c)) for c in cells)
    fig, axes = plt.subplots(len(COSTS), len(cells), figsize=(2.9 * len(cells) + 1.4, (0.55 * nb + 1.0) * len(COSTS)),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, c in enumerate(COSTS):
        mats = [np.array([[(an.E(cell, b, r, c) or (np.nan,))[0] for r in RULES3] for b in an.bases(cell)]).reshape(-1, 3)
                for cell in cells]
        finite = [np.abs(M[np.isfinite(M)]).max() for M in mats if np.isfinite(M).any()]
        lim = max([0.05] + finite)                              # one colour scale per cost row
        im = None
        for j, (cell, p, M) in enumerate(zip(cells, mp, mats)):
            ax = axes[i][j]
            bs = an.bases(cell)
            im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            for a in range(len(bs)):
                for b in range(3):
                    v = M[a, b]
                    ax.text(b, a, "n/a" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color=INK)
            ax.set_xticks(range(3), RULES3 if i == len(COSTS) - 1 else [""] * 3, fontsize=7, color=INK2)
            ax.set_yticks(range(len(bs)), bs, fontsize=8, color=INK2)
            title = f"{p['task']} ({p['role'][:3]})" if i == 0 else ""
            ax.set_title(title, fontsize=8, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"cost {c}", fontsize=9, color=INK, labelpad=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
        cb = fig.colorbar(im, ax=axes[i].tolist(), shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle(f"{_short(model)}: excess damage E at three costs (50% depth, S1, mean; colour scale per row)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def delta_by_cost(res: dict, pairs: list[dict], model: str, path: Path) -> None:
    mp = [p for p in pairs if p["model"] == model]
    fig, axes = _fig(3.3 * len(mp) + 1.4, 3.3, len(mp))
    for ax, p in zip(axes, mp):
        key = "C1" if p["role"] == "confirmatory" else "development_C1"
        recs = []
        for c in COSTS:
            r = next((x for x in res[str(c)][key]["per_pair"] if x["model"] == model and x["task"] == p["task"]), None)
            recs.append(r)
        if not all(recs):
            continue
        for b in ("sae", "pca", "dims"):
            if b in recs[0]["delta_b"]:
                ls, mk = BASIS_STYLE[b]
                ax.plot(COSTS, [r["delta_b"][b]["stat"] for r in recs], ls=ls, marker=mk, lw=1.5,
                        color=INK2 if b != "sae" else "#2a78d6", label=b)
        ys = np.array([r["stat"] for r in recs])
        lo = np.array([r["ci"][0] for r in recs])
        hi = np.array([r["ci"][1] for r in recs])
        ax.fill_between(COSTS, lo, hi, color=INK, alpha=0.10, lw=0, label="mean, 95% CI")
        ax.plot(COSTS, ys, color=INK, lw=2.2, marker="o", label="mean (C1 statistic)")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axhline(0.05, color="#c0392b", lw=0.8, ls=(0, (4, 2)), label="delta = 0.05")
        ax.set_xscale("log")
        ax.set_xticks(COSTS, [str(c) for c in COSTS], fontsize=7)
        ax.minorticks_off()
        ax.set_title(f"{p['task']} ({p['role'][:3]})", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("cost", fontsize=8, color=INK2)
    axes[0].set_ylabel("Delta (attrib - best readout)", fontsize=8, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center right", fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle(f"{_short(model)}: causal advantage Delta at three costs (50% depth, S1, mean)", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1 - 1.5 / fig.get_figwidth(), 0.92))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def delta_by_depth_costs(res: dict, pairs: list[dict], model: str, path: Path) -> None:
    tasks = [p["task"] for p in pairs if p["model"] == model]
    fig, axes = _fig(3.3 * len(tasks) + 1.4, 3.2, len(tasks))
    for ax, t in zip(axes, tasks):
        for c in COSTS:
            rs = sorted([r for r in res[str(c)]["S1_depth"] if r["model"] == model and r["task"] == t],
                        key=lambda r: r["depth"])
            if rs:
                ax.plot([r["depth"] for r in rs], [r["delta_mean"] for r in rs], marker="o", lw=2, color=COST_COLOR[c],
                        label=f"cost {c}")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(t, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("depth", fontsize=8, color=INK2)
    axes[0].set_ylabel("Delta, mean over bases", fontsize=8, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center right", fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle(f"{_short(model)}: Delta by depth at three costs (S1, mean)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1 - 1.2 / fig.get_figwidth(), 0.92))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def curves_costs(an: A.Analysis, cell, path: Path, damage_label: str, title: str) -> None:
    """plots_v2.curves with vertical lines at all three costs."""
    g = an.cell_rows(cell)
    bases = an.bases(cell)
    fig, axes = _fig(4.0 * len(bases) + 1.8, 3.6, len(bases))
    a = an.anchor(cell)
    for ax, b in zip(axes, bases):
        ls, mk = BASIS_STYLE[b]
        for rule in ("meandiff", "l1probe", "attrib", "attrib_shuffled", "random"):
            sub = g[(g.basis == b) & (g.rule == rule)]
            if sub.empty:
                continue
            pts = sub.assign(c=an.cost_p[sub.index], d=an.dmg_p[sub.index]).groupby("k_eff")[["c", "d"]].mean() \
                .sort_values("c")
            col = RANDOM if rule == "random" else RULE_COLOR[rule]
            ax.plot(pts.c.clip(lower=1e-6), pts.d, ls="-" if rule == "random" else ls, marker=mk, ms=4,
                    lw=1.4 if rule == "random" else 2, color=col, label=rule, markeredgecolor=SURFACE,
                    markeredgewidth=0.6)
        if a:
            ax.plot([max(a["c_p"], 1e-6)], [a["D_p"]], marker="*", ms=12, ls="none", color=INK,
                    label="massmean anchor (c*)")
        for c in COSTS:
            ax.axvline(c, color="#c0392b", lw=1.0 if c == 0.1 else 0.7, alpha=1.0 if c == 0.1 else 0.55,
                       ls=(0, (4, 2)) if c == 0.1 else (0, (1, 2)), label=f"cost {c}" + (" (c0)" if c == 0.1 else ""))
        ax.set_xscale("log")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(b, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("cost: KL over 9 positions (generic prompts)", fontsize=8, color=INK2)
    axes[0].set_ylabel(damage_label, fontsize=9, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center right", fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1 - 1.7 / fig.get_figwidth(), 0.93))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main(run_dir: Path, fig_dir: Path | None = None) -> Path:
    hyp = yaml.safe_load(open(run_dir / "prereg_snapshot" / "hypotheses.yaml"))
    cfg = load_v2_config("confirm")
    an = A.Analysis(run_dir, A.bootstrap_B(run_dir), int(cfg["seed"]), hyp)
    pairs = json.loads((run_dir / "inclusion.json").read_text())["pairs"]
    res = {str(c): at_cost(an, pairs, c) for c in COSTS}

    frozen = json.loads((run_dir / "analyze" / "confirmatory.json").read_text())   # c0 = 0.1 must match the frozen run
    got = {(r["model"], r["task"]): r["stat"] for r in res["0.1"]["C1"]["per_pair"]}
    for r in frozen["C1"]["per_pair"]:
        assert abs(got[(r["model"], r["task"])] - r["stat"]) < 1e-9, ("mismatch with frozen C1", r["model"], r["task"])

    out = run_dir / "analyze" / "post_freeze" / "costs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(A.clean_json({"note": "POST-FREEZE, EXPLORATORY (see src/pilot/post_freeze_costs.py)",
                                            "bootstrap": an.B, "by_cost": res}), indent=1))
    fig_dir = fig_dir or out.parent
    fig_dir.mkdir(parents=True, exist_ok=True)
    for m in sorted({p["model"] for p in pairs}):
        s = _short(m)
        heatmap_grid(an, pairs, m, fig_dir / f"E_heatmap_costs_{s}.png")
        delta_by_cost(res, pairs, m, fig_dir / f"delta_by_cost_{s}.png")
        delta_by_depth_costs(res, pairs, m, fig_dir / f"delta_by_depth_costs_{s}.png")
    cell = an.primary_cell("Qwen/Qwen3.5-4B", "sentiment")
    if cell:
        curves_costs(an, cell, fig_dir / "curves_costs_Qwen3.5-4B_sentiment.png", "damage: AUROC drop",
                     "Qwen/Qwen3.5-4B · sentiment (confirmatory) · 50% depth, S1, mean")
    for c in COSTS:
        r = res[str(c)]
        print(f"cost {c}: C1 {r['C1']['n_supported']}/{r['C1']['n_pairs']} (need {r['C1']['required']}), "
              f"C2 {r['C2']['fraction_with_reversal']:.2f} (need {r['C2']['required_fraction']}), {r['label_C1_C2']}")
    print(f"-> {out}; figures in {fig_dir}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
