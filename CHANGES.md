# CHANGES

This lists every file created or modified to implement `pilot-implementation-plan.md`, then every
deviation from that spec and the reason for it. The plan itself was not modified.

## Files created

| File | Purpose |
|---|---|
| `run_pilot.sh` | The only command. Sets up uv/.venv (Py 3.12), installs torch from cu130 (CPU only for `--smoke`), prefetches the model while torch installs, wraps Python in a systemd `MemoryMax` scope, dispatches `--check/--smoke/--resume/--out/--force` |
| `README.md` | Run tutorial: prerequisites, the command, smoke test, runtime, outputs, resuming, reading `report.md` |
| `CHANGES.md` | This file |
| `requirements-check.txt` | Pinned minimal dependencies for Step 0 (transformers, safetensors, huggingface_hub, numpy, pyyaml) |
| `requirements.txt` | Pinned full pilot dependencies (adds pandas, pyarrow, matplotlib, pytest) |
| `config/check.yaml` | Preregistered Step 0 rule: check model, dataset URL and row count, template, thresholds, PASS/FAIL mapping |
| `config/pilot.yaml` | Full-run configuration: models and SAE releases, factors, budgets, data, gate, thresholds, memory |
| `config/smoke.yaml` | `--smoke` overrides (1.7B forced, 16k SAE, 32 per split, k ∈ {1,10,100}, 1 resample/draw, 50 bootstraps) |
| `src/pilot/__init__.py` | Package marker |
| `src/pilot/utils.py` | Config loading and merging, hashing, JSON I/O, logging, seeds, raw-file download, versions, git commit |
| `src/pilot/stats.py` | Numpy-only statistics: tie-aware weighted AUROC, paired bootstrap counts, BH, Kendall τ-b, D@c\* interpolation |
| `src/pilot/check.py` | Step 0 truth check and decision rule; `decision.json` / `decision_smoke.json`; hash guard; `--force` override record |
| `src/pilot/prompts.py` | Chat template with `enable_thinking=False`, template suffix (intervention positions), answer-token variants |
| `src/pilot/models.py` | Model loading (SDPA), left-padded batches with mask-derived positions, ablation hook, layer-split fast path, gradients, generation |
| `src/pilot/memory.py` | `/proc/meminfo` watchdog thread (aborts the current stage) and memory telemetry |
| `src/pilot/env_check.py` | `env` stage: GPU/capability, bf16 matmul, cu130 build, CUDA 12 wheel detection, HF access, hardware record |
| `src/pilot/data.py` | Pinned raw downloads, city-grouped truth splits, refusal/generic splits, templating, position records |
| `src/pilot/sae.py` | One SAE interface for both releases; repo listing to resolve layer and trainer; JumpReLU with the fixed threshold |
| `src/pilot/bases.py` | `sae` / `pca` / `dims` / `mlp` / direction bases sharing one read/write interface; PCA fit; masked per-offset means |
| `src/pilot/selection.py` | meandiff (Cohen's d), L1 logistic path (warm-started FISTA), attribution contributions, random, Jaccard |
| `src/pilot/metrics.py` | m(x), r(x), KL, refusal labelling, refusal-start set, threshold calibration, per-config and bootstrap damage |
| `src/pilot/pipeline.py` | Run context plus the `data`, `gate`, `cache`, `bases`, `select` stages |
| `src/pilot/ablate.py` | `ablate` stage: config enumeration, resumable chunks, fast-path equivalence test, identity check, generation checks |
| `src/pilot/analysis.py` | Curves, D@c\*, G, paired bootstrap, H0–H6, X1, per-task and overall decision |
| `src/pilot/plots.py` | Damage-vs-cost curves, gap-closure heatmaps, stability plot (validated colour-blind-safe palette) |
| `src/pilot/report.py` | `analyze` stage outputs: `summary.csv`, `hypotheses.json`, plots, `report.md`, throughput projection |
| `src/pilot/cli.py` | Stage orchestration: run ids, `--resume`, DONE markers, manifest, watchdog, INVALID/abort handling |
| `src/pilot/smoke.py` | Automated smoke acceptance checks; writes `smoke_checks.json` |
| `src/pilot/prefetch.py` | Background model download (stdlib + huggingface_hub only) started by `run_pilot.sh` |
| `src/pilot/devtiny.py` | Developer plumbing test: tiny random Qwen3 plus random SAE, no download (`--dev-tiny`) |
| `tests/conftest.py` | Test fixtures: tiny random Qwen3 and a toy SAE |
| `tests/test_ablation.py` | v = r(x) identity, SAE error term kept, hook positions, batch invariance, fast path vs full forward |
| `tests/test_bases.py` | PCA/dims orthonormal, read/write round-trips, JumpReLU batch independence, masked means |
| `tests/test_selection.py` | Planted unit recovered by meandiff, l1probe (incl. prefilter), attrib; random; AUROC/bootstrap/interpolation/BH/τ |
| `tests/test_check.py` | The pilot refuses missing, INVALID or hash-mismatched decisions; smoke paths never alias `decision.json` |

Generated at run time (not source): `.venv/`, `data_cache/` (raw datasets), `runs/` (all outputs).

## Deviations from the spec, and why

**Layout and stage order**

1. **Repository root.** The spec's `pilot/` top-level folder is the repository root here, so the
   command is `bash run_pilot.sh` from this directory.
2. **Extra modules.** The spec's suggested layout gains these modules:
   - `check`, `prompts`, `stats`, `memory`, `pipeline`, `smoke`, `prefetch`, `devtiny` and `utils`;
   - `pipeline.py` holds the gate/cache/bases/select stages, so no single module is huge;
   - `tests/test_check.py` and `tests/conftest.py`.
3. **Stage order** is `env, data, gate, cache, bases, select, ablate, analyze`. `data` comes before
   `gate` because the gate's format check and refusal calibration use the discovery split. The Step 0
   decision is read and verified before any stage runs (running the check first if it is missing),
   and it is recorded in `manifest.json`.

**Model and runtime**

4. **CPU path uses fp32.** bf16 is emulated, and very slow, on CPUs without AVX512-BF16/AMX. The GPU
   uses bf16 as specified.
5. **No `device_map` when loading.** It would require `accelerate`. The model is loaded, then moved
   to the device.
6. **Memory limits on smaller machines.** On machines with less than about 116 GB, the watchdog
   threshold scales to 12.5% of MemTotal (16/128) and the systemd `MemoryMax` to 85% of MemTotal.
   This matters only for the CPU smoke check; on the GB10 the values are the spec's 16 GB and 100G.

**Data and metrics**

7. **Generic prompts.** The 400 generic prompts are 100 for PCA fitting plus 300 for the KL cost. This
   reconciles "Generic: 400 prompts" with the runtime section's "300 generic".
8. **Smoke refusal splits** use 32 prompts *per class*, harmful and harmless, matching how the refusal
   splits are defined per class.
9. **The refused set** is defined by clean greedy-generation labels (32 tokens) on the harmful eval
   prompts. Refusal damage is (classifier refusal rate, clean) − (classifier refusal rate, ablated) on
   that set. If the classifier fails calibration, generation with the same token budget replaces it
   for every configuration. If the clean model refuses none of the harmful eval prompts, the gate
   stops with INVALID HARNESS, because damage would be undefined.
10. **Truth coherence only.** Coherence is defined for truth only, so refusal rows have no coherence
    value.
11. **Mean values for `attrib`** and for both strategies come from the full discovery set. The 80%
    resamples change only the rankings.
12. **Resample sources** use one fixed random discovery example per eval or generic prompt, shared
    across all configurations (common random numbers), so differences between cells are paired.

**Curves, statistics and hypotheses**

13. **Below the cheapest budget**, the curve is joined linearly to the origin (k = 0: zero cost, zero
    damage). The spec covers only curves that never *reach* c\*: those report their maximum and are
    flagged.
14. **Test pass rule.** A test passes when its 95% bootstrap CI excludes 0 *and* its BH q ≤ 0.05 within
    the hypothesis family. H2 and H3 are one-sided (CI lower bound > 0). For H4 and H5 a swap needs both
    differences CI-separated; its p-value is the larger of the two.
15. **Aggregation over strategies** (unspecified in the plan):
    - H1 and H4: any strategy.
    - H2 and H3: all strategies.
    - H5: across mean vs resample.
    - H6: τ ≥ 0.6 under every strategy.
    - Overall decision: the shared outcome if both tasks agree, otherwise `MIXED (tasks disagree: ...)`.
16. **H0.** Truth has no generation-confirmed positive control, so its H0 is the random-floor check
    (random D@c\* ≤ 25% of anchor damage, with anchor damage > 0) in every basis and strategy. Refusal
    H0 also needs the anchor's generation drop to be ≥ 0.5 under each strategy.
17. **The MLP arm** is reported everywhere with a "different site" label. It is excluded from H1–H6,
    and the `zero` strategy is diagnostic only.
18. **G is undefined (n/a)** when the anchor's damage equals the random floor. `summary.csv` carries
    `G_denominator`, and `hypotheses.json` writes undefined statistics as `null`.

**Selection and SAE**

19. **L1 path.** Twenty λ values run geometrically from about 0.7·λ_max to 1e-3·λ_max, with warm-started
    FISTA (`max_iter` 1000, relative `tol` 1e-4; float32 on GPU because GB10 FP64 throughput is low).
    Units that never enter are ranked after those that do, by |coef| and then |d|.
20. **`rankings.parquet`** stores the top 2,000 units per ranking (`rank_store`), more than the largest
    budget. It also has an explicit `draw` column; `resample` is −1 for random-rule rows.
21. **SAE normalisation.** `dictionary_learning` folds its activation normalisation into the saved
    biases and threshold (`scale_biases`), and REINS' runtime applies its SAE to raw residuals. So no
    normalisation is applied unless a checkpoint carries an explicit `norm_factor`. The gate also
    reports a diagnostic FVE under global RMS normalisation; it never switches automatically.
22. **SAE sanity check.** The spec's "small next-token loss increase" is set to ≤ 0.5 nats (configurable),
    together with FVE ≥ 0.70 (`dictionary_learning`'s definition), with attention sinks excluded.

**Scope and smoke tooling**

23. **Generation checks** use the first of the 10 `random_dir` directions. Each basis's best and worst
    cell are picked per strategy, at the budget whose cost is nearest c\*, on discovery resample 0.
    Interventions apply only to the template positions during generation (the prefill pass).
24. **DAS** (`anchors.das`) is not implemented. Setting it logs a warning and appears in the report's
    caveats.
25. **Smoke acceptance runs inside the smoke command.** `bash run_pilot.sh --smoke` also runs pytest, a
    timed `--resume`, and `--check --smoke`, then writes `smoke_checks.json`. `--check --smoke` writes
    `decision_smoke.json` (plus `check_report_smoke.md` and `log_smoke.txt`), never `decision.json`.
26. **Developer additions**, not in the spec and hidden from `--help`:
    - `--dev-tiny` runs a tiny random Qwen3 and a random SAE with behavioural checks relaxed and
      logged; with `--check` it writes only `decision_dev.json`.
    - `PILOT_EXTRA_CONFIG=<yaml>` merges an extra config last, for example to enable the `mlp` arm in
      a test.
27. **Pinned versions.** Package versions are pinned for determinism. torch is not pinned: it comes
    from the cu130 index, and its version is recorded in `manifest.json`.

## Verification status

- `pytest`: 30 tests pass.
- **Smoke test on this machine.** The development machine is x86_64 WSL2 with no GPU, and its network
  was too slow to fetch the 4 GB Qwen3-1.7B weights in reasonable time. As instructed, the end-to-end
  smoke test therefore ran on the tiny random model: `bash run_pilot.sh --dev-tiny`.
  - Every stage ran, as did a resume after a failure, partial resume of one stage, the optional MLP
    arm with the `zero` strategy, and the Step 0 check (fresh, refused after a rule change, `--force`
    recorded).
  - All applicable acceptance checks pass: pytest, identity check, no NaN/Inf, outputs present,
    decision line, `--resume` in under 2 s, watchdog running, peak memory logged.
  - The result is `INVALID HARNESS`, as it should be for random weights.
- **Not yet verified on a real model:**
  - Qwen3-1.7B behaviour: format check, refusal calibration, SAE fidelity at layer 14, and the
    throughput projection.
  - Every GPU-only path: cu130 build, capability (12, 1), bf16 numerics, Triton.
  - The Qwen3.5-4B / REINS path; its loader follows REINS' own runtime but no checkpoint was downloaded.

  Run `bash run_pilot.sh --smoke` on the GB10 before the full run.
