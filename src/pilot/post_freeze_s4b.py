"""POST-FREEZE analysis: S4b cross-task control, recomputed from a finished confirmatory run.

Why: in the frozen analysis code (analysis_v2._cross_task) the task's own attribution rows are selected
with `g.resample == 0`. On a pandas DataFrame `g.resample` is the built-in resample() method, not the
column, so the comparison is always False, no own-task rows are found and S4b comes out empty.
This script repeats the frozen computation exactly, with `g["resample"]` instead. The frozen file is not
edited, so the result is a post-freeze analysis and must be labelled as such.

Usage: .venv/bin/python -m pilot.post_freeze_s4b <confirm run dir>
Writes <run dir>/analyze/post_freeze/s4b_cross_task.json (one record per cross-task intervention).

Fields: E_B_units_A = excess damage on task B from units picked (attrib) for task A;
E_B_units_B = excess damage on B from B's own attrib units at the same k; both against random picking at
the same cost and basis. "specific" (frozen rule) = E_B_units_A < 0.5 * E_B_units_B; diff_ci is the 95% CI
of E_B_units_A - 0.5 * E_B_units_B.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from . import analysis_v2 as A
from .v2common import load_v2_config, pair_role


def cross_task(an: A.Analysis, cell, basis: str, role: str | None) -> list[dict]:
    g = an.cell_rows(cell)
    x = g[(g.basis == basis) & (g.rule == "cross_task")]
    rnd = g[(g.basis == basis) & (g.rule == "random")]
    own = g[(g.basis == basis) & (g.rule == "attrib") & (g["resample"] == 0)]      # the one-word fix
    out = []
    if x.empty or rnd.empty:
        return out
    for i in x.index:
        k = int(x.loc[i, "k_eff"])
        o = own[own.k_eff == k]
        if o.empty:
            continue
        j = o.index[0]

        def excess(ix):
            dr = an.D(cell, basis, "random", float(an.cost_p[ix]), an.cost_b[ix])
            return an.dmg_p[ix] - dr[0], an.dmg_b[ix] - dr[1]
        ex_p, ex_b = excess(i)
        own_p, own_b = excess(j)
        out.append({"model": cell[0], "task_B": cell[1], "source_task_A": str(x.loc[i, "source_task"]),
                    "basis": basis, "role": role, "layer": int(cell[2]), "k": k,
                    "E_B_units_A": float(ex_p), "E_B_units_B": float(own_p),
                    "specific": bool(ex_p < 0.5 * own_p), "diff_ci": [float(v) for v in an.ci(ex_b - 0.5 * own_b)]})
    return out


def main(run_dir: Path) -> Path:
    hyp = yaml.safe_load(open(run_dir / "prereg_snapshot" / "hypotheses.yaml"))
    cfg = load_v2_config("confirm")
    an = A.Analysis(run_dir, A.bootstrap_B(run_dir), int(cfg["seed"]), hyp)
    rows = []
    for cell in an.cells():
        if (an.cell_rows(cell).rule == "cross_task").any():
            for b in an.bases(cell):
                rows += cross_task(an, cell, b, pair_role(cfg, cell[0], cell[1]))
    out = run_dir / "analyze" / "post_freeze" / "s4b_cross_task.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": "POST-FREEZE analysis (see src/pilot/post_freeze_s4b.py)",
                               "bootstrap": an.B, "rows": rows}, indent=1))
    print(f"{len(rows)} cross-task comparisons -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
