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
