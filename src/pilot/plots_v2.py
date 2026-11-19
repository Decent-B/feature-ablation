"""v2 figures (matplotlib, Agg), same visual language as v1: colour = selection rule (CVD-safe slots), panel /
marker = basis, random = neutral gray, anchors = black. Exact numbers are in analyze/*.json and exploratory CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .plots import BASIS_STYLE, DIVERGING, GRID, INK, INK2, RANDOM, RULE_COLOR, SURFACE, _fig, _style  # noqa: E402

RULE_COLOR = {**RULE_COLOR, "attrib_shuffled": "#9b59b6"}


def curves(an, cell, path: Path, damage_label: str, title: str):
    g = an.cell_rows(cell)
    bases = an.bases(cell)
    if not bases:
        return False
    fig, axes = _fig(4.0 * len(bases) + 1.8, 3.6, len(bases))
    a = an.anchor(cell)
    for ax, b in zip(axes, bases):
        ls, mk = BASIS_STYLE[b]
        for rule in ("meandiff", "l1probe", "attrib", "attrib_shuffled", "random"):
            sub = g[(g.basis == b) & (g.rule == rule)]
            if sub.empty:
                continue
            pts = sub.assign(c=an.cost_p[sub.index], d=an.dmg_p[sub.index]).groupby("k_eff")[["c", "d"]].mean().sort_values("c")
            col = RANDOM if rule == "random" else RULE_COLOR[rule]
            ax.plot(pts.c.clip(lower=1e-6), pts.d, ls="-" if rule == "random" else ls, marker=mk, ms=4, lw=1.4 if rule == "random" else 2,
                    color=col, label=rule, markeredgecolor=SURFACE, markeredgewidth=0.6)
        if a:
            ax.plot([max(a["c_p"], 1e-6)], [a["D_p"]], marker="*", ms=12, ls="none", color=INK, label="massmean anchor (c*)")
            ax.axvline(max(a["c_p"], 1e-6), color=INK2, lw=0.8, ls=(0, (2, 3)))
        ax.axvline(an.c0, color="#c0392b", lw=0.8, ls=(0, (4, 2)), label=f"c0 = {an.c0}")
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
    return True


def e_heatmap(an, cells: list[tuple], labels: list[str], path: Path):
    if not cells:
        return False
    fig, axes = plt.subplots(1, len(cells), figsize=(3.0 * len(cells) + 1.0, 2.9), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    im = None
    mats = [np.array([[(an.E(c, b, r, an.c0) or (np.nan,))[0] for r in ("meandiff", "l1probe", "attrib")]
                      for b in an.bases(c)]).reshape(-1, 3) for c in cells]
    finite = [np.abs(m[np.isfinite(m)]).max() for m in mats if np.isfinite(m).any()]
    lim = max([0.05] + finite)                          # one colour scale for every panel
    for ax, cell, lab, M in zip(axes[0], cells, labels, mats):
        bs = an.bases(cell)
        im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(bs)):
            for j in range(3):
                v = M[i, j]
                ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(3), ["meandiff", "l1probe", "attrib"], fontsize=7, color=INK2)
        ax.set_yticks(range(len(bs)), bs, fontsize=8, color=INK2)
        ax.set_title(lab, fontsize=8, color=INK, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    if im is not None:
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        cb.set_label(f"E(c0={an.c0}): damage in excess of random", fontsize=7, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.savefig(path, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return True


def delta_by_depth(rows: list[dict], model: str, path: Path):
    rows = [r for r in rows if r["model"] == model]
    if not rows:
        return False
    tasks = sorted({r["task"] for r in rows})
    fig, axes = _fig(3.6 * len(tasks) + 1.4, 3.2, len(tasks))
    for ax, t in zip(axes, tasks):
        rs = sorted([r for r in rows if r["task"] == t], key=lambda r: r["depth"])
        for b in ("sae", "pca", "dims"):
            xs = [r["depth"] for r in rs if b in r["delta_b"]]
            ys = [r["delta_b"][b]["stat"] for r in rs if b in r["delta_b"]]
            if xs:
                ls, mk = BASIS_STYLE[b]
                ax.plot(xs, ys, ls=ls, marker=mk, color=INK2 if b != "sae" else "#2a78d6", label=b, lw=1.6)
        ax.plot([r["depth"] for r in rs], [r["delta_mean"] for r in rs], color=INK, lw=2.2, marker="o", label="mean")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{t}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("depth", fontsize=8, color=INK2)
    axes[0].set_ylabel("Delta at c0 (attrib - best readout)", fontsize=8, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center right", fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle(f"{model}: Delta by depth (S1, mean)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1 - 1.2 / fig.get_figwidth(), 0.92))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return True


def c3_scatter(an, pairs: list[dict], path: Path):
    pts = []
    for p in pairs:
        cell = an.primary_cell(p["model"], p["task"])
        if not cell:
            continue
        g = an.cell_rows(cell)
        g = g[g.rule == "c3_group"]
        for b, gb in g.groupby("basis"):
            pts.append((f"{p['model'].split('/')[-1]}/{p['task']}/{b}", gb.strength.to_numpy(float), an.dmg_p[gb.index]))
    if not pts:
        return False
    fig, axes = _fig(5.2, 3.6, 1)
    ax = axes[0]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#9b59b6", "#c0392b", "#16a085", "#7f8c8d"]
    for i, (lab, x, y) in enumerate(pts):
        ax.scatter(x, y, s=14, color=cols[i % len(cols)], label=lab, alpha=0.85)
    ax.set_xlabel("group readout strength (mean |Cohen's d|)", fontsize=8, color=INK2)
    ax.set_ylabel("group damage", fontsize=8, color=INK2)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("C3: readout strength vs damage (20-unit groups, primary setting)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return True


__all__ = ["curves", "e_heatmap", "delta_by_depth", "c3_scatter", "GRID", "_style"]
