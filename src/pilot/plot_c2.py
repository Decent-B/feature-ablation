"""Figure for the results write-up: every C2 test of the frozen confirmatory analysis as one point.

x = readout difference  E_readout(basis A) - E_readout(basis B)
y = causal difference   E_attrib(basis A)  - E_attrib(basis B)
both at c0, with 95% CIs as error bars. A flip needs the two CIs on opposite sides of zero, i.e. the point
(with its error bars) inside the upper-left or lower-right quadrant, plus BH q <= alpha.
Reads analyze/confirmatory.json only (no recomputation; the frozen code is not touched).

Usage: .venv/bin/python -m pilot.plot_c2 <confirm run dir> <output png>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .plots import GRID, INK, INK2, RULE_COLOR, SURFACE  # noqa: E402

NAME = {"sae": "SAE", "pca": "PCA", "dims": "dims"}
MARK = {("sae", "pca"): "s", ("sae", "dims"): "^", ("pca", "dims"): "o"}


def main(run_dir: Path, out: Path) -> None:
    c2 = json.loads((run_dir / "analyze" / "confirmatory.json").read_text())["C2"]
    pp = c2["per_pair"]
    tests = [t for p in pp for t in p["tests"]]
    xs = [v for t in tests for v in t["d_readout_ci"]] + [0]
    ys = [v for t in tests for v in t["d_attrib_ci"]] + [0]
    padx, pady = 0.1 * (max(xs) - min(xs)), 0.1 * (max(ys) - min(ys))
    xl, yl = (min(xs) - padx, max(xs) + padx), (min(ys) - pady, max(ys) + pady)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, p in zip(axes.flat, pp):
        ax.set_facecolor(SURFACE)
        ax.fill_between([0, xl[1]], yl[0], 0, color="#1baf7a", alpha=0.07, lw=0)   # flip: A wins readout, B wins attrib
        ax.fill_between([xl[0], 0], 0, yl[1], color="#1baf7a", alpha=0.07, lw=0)   # flip: B wins readout, A wins attrib
        ax.axhline(0, color=INK2, lw=0.7)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.grid(color=GRID, lw=0.5)
        for t in p["tests"]:
            x, y = t["d_readout"], t["d_attrib"]
            col = RULE_COLOR[t["readout_rule"]]
            ax.errorbar(x, y, xerr=[[x - t["d_readout_ci"][0]], [t["d_readout_ci"][1] - x]],
                        yerr=[[y - t["d_attrib_ci"][0]], [t["d_attrib_ci"][1] - y]], fmt="none", ecolor=col, elinewidth=1,
                        alpha=0.6, capsize=0)
            ax.plot(x, y, MARK[tuple(t["bases"])], ms=9 if t["pass"] else 7, color=col if t["pass"] else SURFACE,
                    markeredgecolor=col, markeredgewidth=1.6, zorder=3)
        ax.set_xticks([-0.1, -0.05, 0, 0.05])
        ax.tick_params(labelsize=8, colors=INK2)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        flip = "flip" if p["reversal"] else "no flip"
        ax.set_title(f"{p['model'].split('/')[-1]} · {p['task']}: {flip}", fontsize=9, color=INK, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("readout difference: E(A) − E(B) under readout rule", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("causal difference: E(A) − E(B) under attrib", fontsize=8, color=INK2)
    hs = [plt.Line2D([], [], ls="none", marker="o", color=RULE_COLOR[r], label=f"readout rule: {r}") for r in
          ("meandiff", "l1probe")]
    hs += [plt.Line2D([], [], ls="none", marker=m, markerfacecolor=SURFACE, markeredgecolor=INK2,
                      label=f"A−B = {NAME[a]}−{NAME[b]}") for (a, b), m in MARK.items()]
    hs += [plt.Line2D([], [], ls="none", marker="o", color=INK2, label="filled = passes (flip and q ≤ 0.05)"),
           plt.Line2D([], [], ls="none", marker="o", markerfacecolor=SURFACE, markeredgecolor=INK2, label="hollow = fails"),
           plt.Rectangle((0, 0), 1, 1, color="#1baf7a", alpha=0.15, label="flip regions")]
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("C2: every test at c0 = 0.1 (50% depth, S1, mean). Bars: 95% CIs",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"{len(tests)} tests -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
