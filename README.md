# Basis × selection-rule ablation pilot

This pilot ablates units chosen by 3 bases (SAE latents, PCA components, raw residual dims) × 3 selection
rules (mean difference, L1 probe, attribution) plus a random rule, under 2 ablation strategies (mean,
resample), on 2 tasks (truth, refusal). It then compares methods on damage-vs-cost curves. The design
is in `pilot-implementation-plan.md`.

## Prerequisites

- One NVIDIA GB10 (aarch64, CUDA 13.0 driver), about 30 GB of free disk, and internet access.
- No gated models. A Hugging Face token is **optional**; it only avoids rate limits
  (`export HF_TOKEN=...` or `hf auth login`).
- Nothing else. The script installs `uv` if it's missing, creates `.venv` (Python 3.12), installs
  PyTorch from the CUDA 13.0 index (`https://download.pytorch.org/whl/cu130`), and downloads the
  model, SAEs and datasets.
- Recommended once per boot, not done by the script: `sudo swapoff -a`. GPU memory is shared with
  the OS, so swapping can freeze the machine instead of failing cleanly.

## Run it

```bash
bash run_pilot.sh
```

That single command:

1. Runs the Step 0 truth check if `runs/check/decision.json` does not exist yet.
2. Loads the model the check chose (Qwen3-1.7B on PASS, Qwen3.5-4B on FAIL). This choice is binding.
3. Runs all eight stages and writes `runs/<run_id>/report.md`.

To run only the Step 0 check:

```bash
bash run_pilot.sh --check            # writes runs/check/decision.json and check_report.md
```

The check thresholds and the PASS/FAIL model mapping live in `config/check.yaml`, which is preregistered.
If you edit that file after a decision exists, rerunning the check is refused unless you pass
`--force`, and the override is recorded in `decision.json`. The pilot refuses to start if
`check.yaml` no longer matches the hash stored in the decision.

## Smoke test

```bash
bash run_pilot.sh --smoke            # under 10 min on the GB10
```

The smoke test runs every stage on a tiny configuration (Qwen3-1.7B forced, 16k SAE, 32 examples per
split, budgets 1/10/100). It then runs the acceptance checks:

- `pytest`
- identity ablation leaves logits unchanged
- SAE sanity check
- no NaN or Inf values
- every expected output exists
- `--resume` finishes in under 30 s
- `--check --smoke` writes only `decision_smoke.json`

The results go to `runs/<run_id>/smoke_checks.json`. The smoke test also measures throughput and
prints a projected full-run time for both models. Trust that projection over the table in the plan.

## Expected runtime on the GB10

| Run | Time |
|---|---|
| `--check` | 4–12 min the first time (install plus a 3.4 GB download), then about 1 min |
| `--smoke` | under 10 min after install |
| Full pilot, Qwen3-1.7B | about 2.5–5 h with the fast path (on by default), 4–8 h with full forward passes |
| Full pilot, Qwen3.5-4B | about 4–8 h with the fast path, 9–18 h without |

The ablation stage dominates the runtime. The cheapest ways to cut it, in `config/pilot.yaml`:

- `discovery_resamples: 2`
- fewer `budgets`
- `data.generic.n_cost: 200`

## Where outputs land

Each run gets its own directory, `runs/<YYYYmmdd-HHMMSS>-<pilot|smoke>-<model>/`:

| Path | Contents |
|---|---|
| `report.md` | **Read this first.** Decision line, hypothesis table, plots, gate, stability, caveats, runtime |
| `manifest.json` | Model, Step 0 decision hash, resolved layer and SAE files, positions, versions, seeds, hardware, per-stage time and memory |
| `log.txt` | Full log (prompt ids only; harmful prompt text and generations are never printed) |
| `gate/gate.json` | Format check, answer tokens, refusal-start tokens and threshold, SAE sanity check |
| `select/rankings.parquet`, `select/stability.csv` | Unit rankings; top-k Jaccard across discovery resamples |
| `ablate/results.parquet`, `ablate/per_example/` | One row per configuration; per-example scores for the bootstrap |
| `ablate/generation_checks.csv` | Generation-based refusal rates for the anchors and the best and worst cell per basis |
| `analyze/summary.csv`, `analyze/hypotheses.json` | D@c\*, gap closure G and CIs per cell; H0–H6 and X1 with CIs and q-values |
| `plots/*.png` | Damage-vs-cost curves, gap-closure heatmaps, stability |

The Step 0 files are in `runs/check/`.

## Resuming

Each stage writes a `DONE` marker when it finishes. If a run stops (a crash, Ctrl-C, or the memory
watchdog aborting a stage when `MemAvailable` drops below 16 GB), continue it with:

```bash
bash run_pilot.sh --resume                 # continues the latest run of the same kind
bash run_pilot.sh --resume --out runs/<id> # continues a specific run
```

Finished stages are skipped. The ablation stage also resumes chunk by chunk, one chunk per
task × strategy × basis.

## Reading `report.md`

1. **Decision line.** This is `OUTCOME 1: interaction`, `OUTCOME 2: selection beats basis`,
   `OUTCOME 3: uniform`, `MIXED` (with the hypotheses that held), or `INVALID HARNESS`. There is one line
   per task and one overall, and H6 (whether the two tasks agree) is printed beside it. If the result is
   `INVALID HARNESS`, stop and fix the harness; nothing else in the report is interpretable.
2. **Hypothesis table.** For H0–H6, each row gives the best test, its 95% CI and its BH q-value. A test
   passes when its CI excludes 0 and q ≤ 0.05. Every test is listed in `analyze/hypotheses.json`.
3. **D@c\*.** Damage at the reference cost c\*, the KL cost of the mass-mean anchor. G is gap closure:
   0 means the cell is no better than the random rule in that basis, 1 means it matches the anchor. A ⚑
   marks a curve that never reached c\*; its maximum damage is shown instead.
4. **Gate.** Which model ran and why, how well the SAE reconstructs at the chosen layer, the refusal
   classifier, and whether refusal is underpowered (fewer than 150 refused harmful prompts).
5. **Caveats and runtime.** Check these before trusting a result.

The decision line is a suggestion computed from preregistered rules. The final call is yours, made from
the report and the plots.

## Layout

```
run_pilot.sh            the only command
config/pilot.yaml       full-run settings (model is NOT set here; Step 0 decides)
config/smoke.yaml       --smoke overrides
config/check.yaml       preregistered Step 0 rule (hashed into decision.json)
src/pilot/              check, env_check, data, prompts, models, sae, bases, selection,
                        pipeline (gate/cache/bases/select), ablate, metrics, analysis,
                        plots, report, cli, memory, smoke, prefetch, devtiny
tests/                  ablation identity, basis round-trips, planted-unit recovery, statistics
```

---

## v2 (pilot-v2-amendment.md)

v2 changes where and how deep the interventions go, adds tasks (sentiment, answerability; toxicity as a
substitute) and a second model (Qwen3.5-4B), extends every curve to the comparison cost, and splits the work
into a development run and a preregistered confirmatory run. v1 modes and outputs are unchanged; the v1 run
`20261116-180218-pilot-qwen3-1.7b/` is archived as "v1, superseded" and is never read by v2.

### One command

```bash
bash run_v2.sh            # every step below, in the background (you can close the terminal)
bash run_v2.sh --status   # progress, and the summary when it finishes or stops
bash run_v2.sh --stop     # stop; finished work is kept
```

`run_v2.sh` runs the commands below in order, with the preregistration finalised as drafted (it sets
`status: final`, renders, commits and freezes). It stops at the first error, including a failed smoke check,
and says why; after fixing the cause, `bash run_v2.sh` continues from that step. Only the downloads are retried
(3 tries). Log: `runs/v2/pipeline/run_v2.log`; summary: `runs/v2/pipeline/SUMMARY.txt`.

### Order of commands

```bash
bash run_pilot.sh --v2 --smoke           # 1. prefetch (~19 GB) + every stage on tiny data + checks + projections
bash run_pilot.sh --v2 --dev             # 2. development run -> runs/v2/dev/<id>/dev_report.md
#    3. read dev_report.md; edit prereg/hypotheses.yaml (set status: final, authors, dropped_blocks if any)
bash run_pilot.sh --v2 --render-prereg   # 4. regenerate prereg/PREREGISTRATION.md from the YAML
git add -A && git commit -m "Final v2 preregistration"
bash run_pilot.sh --v2 --freeze          # 5. hash + commit + tag prereg-v2-frozen
bash run_pilot.sh --v2 --confirm         # 6. confirmatory run -> runs/v2/confirm/<id>/report_v2.md
```

`--v2 --prefetch` downloads everything on its own (the smoke test runs it first). Add `--resume` to continue
any v2 run after a crash or Ctrl-C; finished stages and finished ablation chunks are skipped.

### What the freeze does and why

The confirmatory run should test hypotheses on data that were not used to form them, with the analysis fixed
in advance. `--freeze`:

1. refuses if tracked files have uncommitted changes, or if `PREREGISTRATION.md` is out of date with the YAML;
2. writes `prereg/FROZEN.json`: the SHA-256 of `prereg/hypotheses.yaml`, `prereg/PREREGISTRATION.md`,
   `config/v2.yaml`, `src/pilot/analysis_v2.py` and `src/pilot/stats.py`, plus the commit and a UTC timestamp;
3. commits `FROZEN.json` and tags the commit `prereg-v2-frozen`;
4. reminds you that uploading `PREREGISTRATION.md` to OSF or AsPredicted adds an independent timestamp (the
   script never uploads anything).

`--confirm` refuses to start unless `FROZEN.json` and the tag exist and every hash matches. If you must change
a frozen file afterwards, run `--confirm --deviation "<reason>"`: the reason, time and changed files go into
`prereg/deviations.md` and to the top of the report. The run plan (block priority and any blocks dropped
because the smoke projection says they will not fit) is part of `hypotheses.yaml`, so it is frozen too.

### Expected runtime on the GB10

| Phase | Time |
|---|---|
| `--v2 --prefetch` | network-bound: ~19 GB (both models, 3 + 5 SAE files, datasets) |
| `--v2 --smoke` | under 25 min after the prefetch |
| `--v2 --dev` | ~0.6 h |
| `--v2 --confirm` | ~8–9 h; the time guard stops before any block that would push dev + confirm past 12 h |

The smoke test measures throughput, auto-tunes batch sizes (written to `runs/v2/tuning.json`, read by `--dev`
and `--confirm`) and prints projected times per block. Trust that projection over this table. If it lists
blocks that will not fit, decide what to drop and put them under `run_plan.dropped_blocks` before freezing.

### Blocks (priority order, frozen)

1. `B1_primary`: inclusion gates, then 50% depth, site S1, mean ablation, with anchors, for every included pair
   (everything C1 and C2 need).
2. `B2_controls`: shuffled-label attribution, cross-task units, C3 groups.
3. `B3_tier2_25_75`: 25% and 75% depth, both models.
4. `B4_refusal_s2_resample`: refusal at site S2, and refusal with resample ablation (50% depth).
5. `B5_tier2_40_60`: 40% and 60% depth (Qwen3.5-4B only).

Within a block, confirmatory pairs run before development pairs. Skipping depends only on speed.

### Reading `report_v2.md`

1. **Central-claim label** (`SUPPORTED`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`), the C1–C3 table, and one row per
   confirmatory pair. The main number is Δ: damage of attribution-selected units minus the better readout rule,
   both measured as damage in excess of random selection at a cost of 0.1 nats. The H0 column is a caveat only;
   it never changes a verdict.
2. **Deviations** from the preregistration and blocks skipped by the time budget.
3. **Inclusion**: which (model, task) pairs passed their gates, substitutions, and where the SAE basis was
   dropped because its reconstruction was too poor.
4. **Secondary analyses** S1–S5 (depth, site, strategy, circularity controls, anchors).
5. **Development pairs** (Qwen3-1.7B truth and refusal), same tables, never counted toward C1–C3.
6. **Anchors and H0** per cell, 7. **exploratory** files, 8. **runtime and memory**.

Machine-readable results: `analyze/confirmatory.json` (written first), `analyze/secondary.json`,
`analyze/development.json`, `analyze/exploratory/`, and `ablate/results.parquet` (one row per intervention).

### v2 outputs

`runs/v2/<phase>/<run_id>/`: `inclusion.json`, `models/<model>/{gate,prep,data}/` (gates, SAE fidelity,
rankings, means, reservoirs), `ablate/chunks/` (per-example results), `ablate/results.parquet`,
`ablate/blocks.json` (block status and timing), `analyze/`, `plots/`, `report_v2.md` (or `dev_report.md`),
`smoke_report.md` and `smoke_checks.json` (smoke only), `manifest.json`, `log.txt`.

### If something stops

- **Gated DeltaNet kernels.** On the GPU, `run_pilot.sh` tries to install `flash-linear-attention` (Triton) for
  Qwen3.5-4B. The gate logs which implementation loaded. If the kernels misbehave, uninstall them
  (`uv pip uninstall --python .venv/bin/python flash-linear-attention fla-core`) and set `PILOT_V2_FLA=0`; the
  model then uses the slower PyTorch path and the projection accounts for it.
- **Developer plumbing test** (no downloads of weights): `bash run_pilot.sh --v2 --smoke --dev-tiny` runs the
  whole v2 pipeline on tiny random versions of both architectures. Its numbers are meaningless.
