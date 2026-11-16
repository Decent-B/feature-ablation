"""Static PNG figures for report.md (matplotlib, Agg).

Colour carries the selection rule (first three slots of the validated reference palette,
all-pairs CVD-safe); basis is carried by panel, marker and line style; the random rule
is neutral gray and the anchors are black markers, so identity never rests on colour
alone. Exact numbers are in analyze/summary.csv (the table view).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
RULE_COLOR = {"meandiff": "#2a78d6", "l1probe": "#eb6834", "attrib": "#1baf7a"}
RANDOM = "#8c8b86"
BASIS_STYLE = {"sae": ("-", "o"), "pca": ("--", "s"), "dims": (":", "^"), "mlp": ("-.", "D")}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#d03b3b", "#f0efec", "#2a78d6"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(w, h, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        _style(ax)
    return fig, axes[0]


def curves(res_task: dict, strategy: str, path: Path, damage_label: str):
    td = res_task["td"]
    cf = td.configs.copy()
    cf["cost"] = res_task["cost_p"]
    cf["damage"] = res_task["dmg_p"]
    cf = cf[cf.strategy == strategy]
    bases = [b for b in ("sae", "pca", "dims", "mlp") if b in set(cf.basis)]
    fig, axes = _fig(4.2 * max(1, len(bases)) + 1.6, 3.8, max(1, len(bases)))
    anc = cf[cf.rule == "massmean"]
    rdir = cf[cf.rule == "random_dir"]
    for ax, b in zip(axes, bases):
        sub = cf[cf.basis == b]
        ls, mk = BASIS_STYLE[b]
        for rule in ("meandiff", "l1probe", "attrib", "random"):
            g = sub[sub.rule == rule]
            if g.empty:
                continue
            m = g.groupby("k")[["cost", "damage"]].mean().sort_values("cost")
            col = RANDOM if rule == "random" else RULE_COLOR[rule]
            ax.plot(m.cost.clip(lower=1e-8), m.damage, ls=ls if rule != "random" else "-", marker=mk, ms=5,
                    lw=2 if rule != "random" else 1.4, color=col, label=rule, markeredgecolor=SURFACE, markeredgewidth=0.8)
        if not anc.empty:
            ax.plot(anc.cost.clip(lower=1e-8), anc.damage, marker="*", ms=13, ls="none", color=INK, label="massmean anchor (c*)")
            ax.axvline(float(anc.cost.iloc[0]), color=INK2, lw=0.8, ls=(0, (2, 3)))
        if not rdir.empty:
            ax.plot(rdir.cost.clip(lower=1e-8), rdir.damage, marker="x", ms=6, ls="none", color=INK2, label="random_dir anchors")
        ax.set_xscale("log")
        ax.axhline(0, color=INK2, lw=0.6)
        title = f"{b}" + (" (MLP site, not comparable)" if b == "mlp" else "")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("cost: mean KL(clean || ablated) on generic prompts", fontsize=8, color=INK2)
    axes[0].set_ylabel(damage_label, fontsize=9, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(f"{td.task} · {strategy}-ablation · damage vs cost (curves averaged over resamples/draws)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1 - 1.5 / fig.get_figwidth(), 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def gap_heatmap(res_task: dict, strategies: list[str], path: Path):
    G = res_task["G_p"]
    bases = [b for b in ("sae", "pca", "dims", "mlp") if any(k[1] == b for k in G)]
    rules = [r for r in ("meandiff", "l1probe", "attrib") if any(k[2] == r for k in G)]
    strategies = [s for s in strategies if any(k[0] == s for k in G)]
    fig, axes = plt.subplots(1, max(1, len(strategies)), figsize=(3.6 * max(1, len(strategies)) + 1.2, 3.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    im = None
    for ax, s in zip(axes[0], strategies):
        M = np.array([[G.get((s, b, r), [np.nan])[0] for r in rules] for b in bases], dtype=float)
        im = ax.imshow(np.clip(M, -1, 2), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
        for i in range(len(bases)):
            for j in range(len(rules)):
                v = M[i, j]
                ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color=INK if (not np.isfinite(v) or abs(v) < 0.6) else "#ffffff")
        ax.set_xticks(range(len(rules)), rules, fontsize=8, color=INK2)
        ax.set_yticks(range(len(bases)), bases, fontsize=8, color=INK2)
        ax.set_title(f"{s}-ablation", fontsize=10, color=INK, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    if im is not None:
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        cb.set_label("gap closure G (0 = random rule, 1 = anchor)", fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle(f"{res_task['task']} · gap closure at c*", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def stability(st: pd.DataFrame, path: Path):
    tasks = sorted(st.task.unique()) if not st.empty else ["truth", "refusal"]
    fig, axes = _fig(4.6 * len(tasks) + 1.4, 3.4, len(tasks))
    if st.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "selection stability needs >= 2 discovery resamples", ha="center", va="center",
                    transform=ax.transAxes, color=INK2, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        for ax, t in zip(axes, tasks):
            sub = st[st.task == t]
            for (b, r), g in sub.groupby(["basis", "rule"]):
                ls, mk = BASIS_STYLE.get(b, ("-", "o"))
                g = g.sort_values("k")
                ax.plot(g.k, g.jaccard_mean, ls=ls, marker=mk, ms=5, lw=2, color=RULE_COLOR.get(r, RANDOM),
                        label=f"{b} / {r}", markeredgecolor=SURFACE, markeredgewidth=0.8)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_title(t, fontsize=10, color=INK, loc="left")
            ax.set_xlabel("budget k", fontsize=8, color=INK2)
        axes[0].set_ylabel("top-k Jaccard across discovery resamples", fontsize=8, color=INK2)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="center right", fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle("selection stability (colour = rule, line style/marker = basis)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1 - 1.3 / fig.get_figwidth(), 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
