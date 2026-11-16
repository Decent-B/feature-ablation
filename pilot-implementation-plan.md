# Pilot Implementation Spec: Basis × Selection-Rule Ablation (for Claude Code)


## Purpose

The pilot decides which of three papers the full study becomes, at a small fraction of the full grid's cost. It ablates units chosen by 3 unit bases × 3 selection rules × 2 ablation strategies, on 2 tasks, in one ungated Qwen instruct model at one residual-stream layer. Methods are compared on damage-vs-off-target-cost curves.

| Outcome | Pattern in the pilot results | Paper it leads to |
| --- | --- | --- |
| 1. Interaction | The ranking of bases changes with selection rule (H4) or with ablation strategy (H5) | Method rankings depend on selector or strategy; scale up to more tasks and a second model |
| 2. Selection beats basis | Attribution beats readout rules in every basis (H2), spread across rules exceeds spread across bases (H3), no H4 | "Selection, not basis"; per-unit readout-vs-causal analysis becomes the centerpiece |
| 3. Uniform | No CI-separated differences on H1–H5 | Confirmation result; workshop paper |

The code computes H0–H6 and prints a suggested outcome. The final call is a human judgment from the report and plots. A short pre-pilot check (Step 0) first decides which model the pilot uses.

## Instructions to Claude Code

Build a Python package plus one Bash entry point, `run_pilot.sh`, that runs the whole pilot end to end on a single GPU server. The user will run nothing else.

1. **One script.** `bash run_pilot.sh` creates or reuses a virtual environment, installs dependencies, downloads models, SAEs and datasets, runs every stage, and writes the report. Flags: `--check` (Step 0 only), `--smoke` (tiny run, under 10 minutes), `--out DIR`, `--resume`. There is no model flag: the model always comes from the Step 0 decision. Everything else lives in `config/pilot.yaml`.
2. **Target machine.** The GB10 in the Target machine section below (CUDA 13.0, aarch64, 128 GB unified memory). Follow its install and memory-safety settings; do not assume x86 or CUDA 12 wheels. Fail early with a clear message if `torch.cuda.is_available()` is false.
3. **Resumable.** Each stage writes its outputs and a `DONE` marker under `runs/<run_id>/`. Re-running with `--resume` skips finished stages. Log to `runs/<run_id>/log.txt` and to stdout.
4. **Deterministic.** Fixed seeds in the config; record package versions, git commit, model and SAE identifiers in `runs/<run_id>/manifest.json`.
5. **README.md.** Write a short run tutorial: prerequisites (no gated models; a Hugging Face token is optional and only avoids rate limits), the one command, the smoke-test command, expected runtime, where outputs land, how to resume, and how to read `report.md`.
6. **Report all changes.** When finished, list every file you created or modified with a one-line purpose each, plus any deviation from this spec and why. Also write that list to `CHANGES.md`.
7. **Smoke test.** Run `bash run_pilot.sh --smoke` yourself before finishing and fix every failure. If the machine you are on has no GPU, run the smoke test on CPU with Qwen3-1.7B and say so; the CPU path exists only for this check.
8. **Ask, don't guess, when blocked.** If a model or SAE release named here cannot be found, stop and report the closest available alternative instead of silently substituting.

## Target machine

The pilot runs on one NVIDIA GB10 box with CUDA 13.0 on aarch64. Its 128 GB of memory is shared by CPU and GPU, which shapes both the install and memory safety.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GB10 (Blackwell), compute capability 12.1 (sm\_121) |
| Driver | 580.126.09 |
| CUDA version supported by the driver | 13.0 |
| OS / kernel | Ubuntu, Linux 6.14.0-1015-nvidia, aarch64 |
| Memory | 128 GB unified LPDDR5x at 273 GB/s; `nvidia-smi` reports GPU memory usage as Not Supported |
| GPU settings | Persistence mode On, compute mode Default, MIG not available |
| Host | `gx10-ecb4` |

Values come from `nvidia-smi` and `uname -a` on 2026-11-16. Compute capability is not shown by `nvidia-smi`; the `env` stage confirms it.

**Settings these specs require:**

1. **PyTorch from the CUDA 13.0 index.** Install `torch` from `https://download.pytorch.org/whl/cu130` (aarch64 wheels exist) into a Python 3.12 virtual environment. Never install a CUDA 12 build of any package: only CUDA 13 runtime libraries exist here, so such packages fail with `libcudart.so.12` import errors. PyTorch warns that capability 12.1 is outside its supported range; per this [DGX Spark setup guide](https://github.com/natolambert/dgx-spark-setup), sm\_120 and sm\_121 are binary compatible and the warning is harmless. Log it once.
2. **Attention: SDPA only.** Load models with `attn_implementation="sdpa"` and do not install `flash-attn`.
3. **Triton.** Export `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` in `run_pilot.sh` before Python starts; Triton kernels, including any fused Qwen3.5 Gated DeltaNet kernels, need it on this GPU. Do not use `torch.compile`.
4. **Memory safety.** GPU allocations draw from the same pool as the OS, so a runaway allocation can freeze the machine instead of raising a CUDA OOM. Run the Python process inside `systemd-run --user --scope -p MemoryMax=100G -p MemorySwapMax=0` when that works; otherwise continue with a warning. Also run a watchdog thread that reads `MemAvailable` from `/proc/meminfo` every second and aborts the current stage cleanly if it drops below 16 GB, keeping finished stages for `--resume`. The README recommends, but the script never runs, `sudo swapoff -a`.
5. **Memory telemetry.** Do not query `nvidia-smi` for memory. Log peak usage from `torch.cuda.max_memory_allocated()` and `/proc/meminfo`, and size batches from those.
6. **Environment record.** The `env` stage writes driver version, CUDA version, kernel, hostname, torch version and CUDA build, and device capability to `manifest.json`, plus the raw `nvidia-smi` output.

## Step 0: pre-pilot check

Measure how well **Qwen/Qwen3-1.7B** answers "True or False" about statements in the Geometry-of-Truth `cities` dataset, on GPU. Apply the preregistered decision rule below to decide whether the upcoming ablation pilot uses 1.7B or falls back to **Qwen/Qwen3.5-4B**. This check only runs 1.7B; a failing result is itself the decision to fall back, and 4B is never downloaded or run here.

**Command.** `bash run_pilot.sh --check` writes `runs/check/decision.json` and `runs/check/check_report.md`, then exits. The decision is binding: a later `bash run_pilot.sh` loads exactly the model it names (see Model choice). If the file is missing, the pilot runs the check first.

**What it runs:**

1. Download only `Qwen/Qwen3-1.7B` (config, tokenizer, safetensors) and `cities.csv` (1,496 statements) from the geometry-of-truth repo. No SAEs, no other datasets, no 4B.
2. Build each prompt with the pilot's truth template and `enable_thinking=False`; the user turn is `Answer with True or False only.\nStatement: {statement}`.
3. Format check: greedy-decode 8 statements for 8 tokens each and assert every response starts with True or False, not a thinking block.
4. One forward pass per statement, logits at the first response position only. m(x) = logit(True) − logit(False), using the answer-token variant (with or without a leading space) that carries more probability mass on average; log both variants.
5. Compute AUROC of m(x) vs label, accuracy at the fixed threshold m(x) = 0, coherence (mean P(True) + P(False)), label balance, and 95% bootstrap CIs from 1,000 resamples.

**Preregistered decision rule.** Fixed in `config/check.yaml` before the run; its SHA-256 hash is written into `decision.json`.

| Condition | Threshold |
| --- | --- |
| AUROC of m(x) vs label | ≥ 0.90 |
| Accuracy, sign of m(x) at threshold 0 (never tuned) | ≥ 0.80 |
| Coherence: mean probability on the two answer tokens | ≥ 0.90 |

- **PASS** (all three, on point estimates) → `"pilot_model": "Qwen/Qwen3-1.7B"`.
- **FAIL** (any one) → `"pilot_model": "Qwen/Qwen3.5-4B"`, listing which conditions failed.
- **INVALID** → no decision and a non-zero exit: the format check failed, the two answer tokens are identical, or the dataset did not load all 1,496 rows. This signals a harness bug, not a weak model, and must never trigger the fallback.

Rerunning with a changed `config/check.yaml` after a decision exists is refused unless `--force` is passed, and the override is recorded in `decision.json`.

**Making it fast:**

- Install only what the check needs, with `uv`: torch from the cu130 index, transformers, safetensors, huggingface\_hub, numpy, pyyaml. The pilot's heavier dependencies are added later to the same environment, so nothing is repeated.
- Install `huggingface_hub` first and start the model download in the background while PyTorch installs.
- Download with `allow_patterns` limited to config, tokenizer and `*.safetensors` files, into the shared Hugging Face cache the pilot reuses.
- Fetch `cities.csv` as one raw file instead of cloning the repo.
- Load in bf16 straight onto the GPU with SDPA attention, under `torch.inference_mode()`.
- Batch 256 prompts sorted by length, left-padded, with `logits_to_keep=1`.
- Compute AUROC with the rank-sum formula and a vectorized NumPy bootstrap; no scikit-learn.
- Skip everything else the pilot needs: refusal data, SAEs, PCA, activation caching, refusal calibration.

**Expected time on the GB10.** Compute is trivial: about 1,500 prompts × \~45 tokens ≈ 70k tokens, roughly 2 × 10¹⁴ FLOP. Setup dominates.

| Step | First run | Later runs |
| --- | --- | --- |
| Environment install with `uv`, overlapped with the download | 2–6 min | Skipped |
| Model download (\~3.4 GB) | 1–6 min, network-bound | Cached |
| Load the model onto the GPU | 10–30 s | 10–30 s |
| Format check + 1,496 forward passes | Under 20 s | Under 20 s |
| Metrics, bootstrap, report | Under 5 s | Under 5 s |
| **Total** | **About 4–12 min** | **About 1 min** |

**Smoke variant.** `bash run_pilot.sh --check --smoke` runs the same code on 64 statements and writes `decision_smoke.json`, never `decision.json`, so it cannot be mistaken for the real decision.

## Model configuration

The Step 0 decision fixes the model: Qwen3-1.7B if it passes the truth check, Qwen3.5-4B if it fails. Both models are Apache-2.0 and ungated, and both have public residual-stream SAEs at the layer the pilot needs.

| Setting | Small (check passes) | Large (check fails) |
| --- | --- | --- |
| HF model id | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3.5-4B` |
| Layers / d\_model | 28 / 2048 (read from config) | 32 / 2560 (read from config) |
| Architecture | Standard Qwen3 decoder | Hybrid Gated DeltaNet + attention |
| SAE release | [adamkarvonen/qwen3-1.7b-saes](https://huggingface.co/adamkarvonen/qwen3-1.7b-saes) (MIT) | [Carlos4869/REINS-SAE](https://huggingface.co/Carlos4869/REINS-SAE), folder `qwen3.5-4b-base/` (Apache-2.0) |
| SAE trained on | This post-trained model; 90% FineWeb, 10% LMSYS-Chat | The base model; Pile + BeaverTails + PKU-SafeRLHF |
| SAE type | BatchTopK, run as JumpReLU with its fixed threshold | BatchTopK, k = 128 |
| SAE layers available | 25%, 50%, 75% depth | All 32 layers |
| Width / L0 | 65k, L0 80 (`trainer_2`); 16k, L0 80 (`trainer_0`) for smoke | 20,480 |
| Intervention layer | The 50%-depth SAE layer | Layer nearest 50% depth (about 16) |
| Precision | bf16 model, fp32 SAE math and statistics | Same |

**Contingency outside the plan.** If the check picks 4B but its SAE fails the sanity check, or Qwen3.5 does not run on the GB10, the documented replacement is `Qwen/Qwen3-8B` with [adamkarvonen/qwen3-8b-saes](https://huggingface.co/adamkarvonen/qwen3-8b-saes), the same SAE recipe as the 1.7B release. Switching is a deliberate change to the preregistered plan: edit the FAIL model in `config/check.yaml`, rerun `bash run_pilot.sh --check --force`, and the change is recorded in `decision.json`. The pilot never switches models on its own. It costs about twice the 4B runtime.

**Resolving SAE files.** List each repo's files instead of hardcoding paths. The adamkarvonen releases store `ae.pt` plus `config.json` per layer and trainer; pick the middle of the three layers. REINS stores `layer_XX.pt` per layer. Implement both formats behind one SAE interface (encode, decode, decoder directions, threshold), apply each release's own input normalization (REINS uses global RMS normalization), and record the resolved files in the manifest.

**Hook point.** Use the residual stream after decoder layer L (`resid_post`). Verify the hook matches the SAE's training site: on chat-formatted prompts, reconstruction should explain at least 70% of variance with a small next-token loss increase. For the REINS SAE this is also the base-to-instruct transfer check. If it fails, stop with `INVALID HARNESS` and point to the contingency above.

**Attention-sink tokens.** Qwen models occasionally produce residual norms 100–1,000× the median, and the adamkarvonen SAEs were trained with tokens above 10× the median filtered out. Exclude such tokens from SAE statistics, the PCA fit and discovery statistics, and log the fraction excluded.

**Thinking mode.** Always call `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` so the first response token is the answer. In stage 0, greedily decode 8 prompts per task and assert the output starts with an answer, not a thinking block.

**Model choice.** Decided once by the Step 0 check above, on the truth task only, and binding for the pilot. The pilot loads exactly the model named in `runs/check/decision.json` and never re-runs the gate. Before loading, it verifies that the decision is PASS or FAIL rather than INVALID and that its rule hash matches the current `config/check.yaml`, then records the decision file's hash in `manifest.json` and names the model and decision at the top of `report.md`. A missing decision file makes the pilot run the check first; an INVALID or mismatched one stops the pilot.

Refusal is not a selection criterion, because small Qwen3 models refuse harmful prompts less often: [He et al. (2026)](https://arxiv.org/abs/2606.26620) report unsteered refusal on Arditi-style harmful prompts of 41.7% for Qwen3-1.7B and 76% for Qwen3-4B. Instead, refusal damage is measured only on harmful eval prompts the clean model refuses. The run requires at least 150 such prompts and a harmless refusal rate ≤ 0.10; if not, it logs a warning and reports refusal results as underpowered.

Refusal thresholds live in `config/pilot.yaml`; the truth-check thresholds and the PASS/FAIL model mapping live in `config/check.yaml`.

**Optional MLP-neuron arm** (off by default): MLP hidden activations at the same layer L. Label it as a different site in every output, since it is not comparable as a basis alone.

## Tasks and data

Two behavioral tasks plus one generic prompt set for measuring collateral cost. Every task is run through the model's chat template so the model actually produces the behavior being measured.

| Set | Source | Prompt (user turn) | Behavioral score |
| --- | --- | --- | --- |
| Truth | [geometry-of-truth](https://github.com/saprmarks/geometry-of-truth) `cities.csv` + `neg_cities.csv` (1,496 rows each) | `Answer with True or False only.\nStatement: {statement}` | m(x) = logit(True) − logit(False) at the first response position |
| Refusal | [refusal\_direction](https://github.com/andyrdt/refusal_direction) `dataset/splits/harmful_*.json` and `harmless_*.json` | The instruction as given | r(x) = log P(refusal-start tokens) − log P(all other tokens) at the first response position |
| Generic (cost) | Held-out harmless Alpaca instructions from the same repo, disjoint from refusal splits | The instruction as given | Next-token distribution at the first response position, for KL |

**Answer tokens.** Resolve the first token id of `True` and `False` with the tokenizer and assert they differ. Coherence = mean probability mass on those two tokens.

**Refusal-start tokens.** In stage 0, greedily generate 32 tokens for 200 clean harmful and 200 clean harmless discovery prompts. Label refusals by substring match (`I cannot`, `I can't`, `I'm sorry`, `I am unable`, `As an AI`, and model-specific variants seen in stage 0; keep the list in config). Take the set of first tokens covering ≥ 90% of refusals as the refusal-start set and save it. The first-token score r(x) is the cheap metric used everywhere; generation-based refusal rate is computed only for the anchors and the best and worst configuration per basis (see Metrics).

**Splits.**

- Truth: split by city name so a city never appears in two splits (its affirmative and negated statements stay together). Discovery 40%, validation 10%, eval 50%. Balance labels in each split.
- Refusal: discovery = 300 harmful (from `harmful_train`) + 300 harmless. Eval = up to 500 harmful (from `harmful_test`) + 500 harmless; refusal damage uses only the harmful eval prompts the clean model refuses. Validation = `harmful_val` + an equal harmless sample.
- Generic: 400 prompts.
- Smoke test: 32 examples per split and set.

**Positions.** Discovery statistics are computed at the final prompt token: the last token of the assistant prefix that the chat template produces, after the empty thinking block that `enable_thinking=False` inserts. Interventions are applied at all post-instruction template tokens (everything after the user content), using per-position-offset statistics for mean values. Record the exact token positions in the manifest.

**Filtering.** Report metrics on all eval examples. Also report truth metrics on the subset the clean model answers correctly; this subset is secondary.

## Experimental factors

The grid is 3 bases × 3 selection rules × 2 strategies × 9 budgets × 2 tasks, plus random and direction anchors. Every unit in every basis is ablated through one shared formula, so differences cannot come from inconsistent definitions of "ablate".

Each unit i has a read value r\_i(x) and a write direction w\_i in residual space. Ablating a set S of units to target values v\_i:

```latex
x' = x + \sum_{i \in S} \big(v_i - r_i(x)\big)\, w_i
```

**Bases** (all at the residual stream after layer L):

| Basis key | Unit | r\_i(x) | w\_i | Notes |
| --- | --- | --- | --- | --- |
| `sae` | Public SAE latent for the chosen model | SAE encoder activation | Decoder row, in raw activation scale | Add deltas to x; never replace x by its reconstruction, so the error term is preserved |
| `pca` | Principal component | u\_i · (x − μ) | u\_i (orthonormal) | Fit on \~20k token activations from generic and discovery prompts, all positions |
| `dims` | Raw residual coordinate | x\_i | e\_i | The "neuron-like" baseline |
| `mlp` (optional) | MLP hidden unit at layer L | Input to `down_proj` | Column of `down_proj` | Hook on `down_proj` input; reported as a different site |

**Selection rules** (computed on the discovery split, per task and basis, at the final prompt token):

1. `meandiff`: Cohen's d of r\_i between classes; rank by |d|.
2. `l1probe`: L1 logistic regression on standardized r. Fit along a path of 20 regularization strengths from strong to weak; rank units by the step at which they first become nonzero, ties broken by |coefficient| at the weakest step. For `sae`, prefilter to the top 8,192 latents by |d| and log that this was done.
3. `attrib`: predicted damage from first-order attribution, A\_i = E\_x\[(v\_i − r\_i(x)) · ∂S/∂r\_i\], summed over intervention positions. S(x) = y·m(x) for truth and y·r(x) for refusal, with y = +1/−1 by class. Compute ∂S/∂x at layer L with one backward pass per example, then ∂S/∂r\_i = w\_i · ∂S/∂x for all units at once. Rank by most negative A\_i. Use mean-ablation values for v\_i under both strategies.
4. `random`: uniform over units, 3 draws per k. For `sae`, sample only latents that fire on ≥ 1% of discovery tokens so dead latents do not make the floor trivially zero.

Repeat every rule on 3 resampled discovery sets (80% subsamples, fixed seeds) and report top-k Jaccard overlap across resamples.

**Ablation strategies:**

- `mean`: v\_i = mean of r\_i over the pooled, class-balanced discovery set, computed per position offset.
- `resample`: v\_i = r\_i(x̃) for one random pooled discovery example x̃ at the same position offset. Resample all selected units jointly from the same x̃. Seeded.
- `zero` (config flag, off by default): diagnostic only.

**Budgets:** k ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500}. Units are ablated jointly as a set.

**Anchors:**

- `massmean`: u = normalize(μ₁ − μ₀) from discovery activations at the final token; projection ablation x' = x + (v − u·x)u under each strategy. For refusal this is the positive control; for truth it is the reference for gap closure.
- `random_dir`: 10 random unit directions with the same projection ablation, as the floor for the direction anchor.
- `das` (config flag, off by default): a trained rank-1 skyline, if time allows.

## Metrics, hypotheses and decision rules

Each configuration is scored as one point (cost, damage) per budget k; the points form a curve, and curves are compared at a fixed reference cost.

**Damage (y-axis):**

- Truth (primary): AUROC drop of m(x) vs label, clean minus ablated. Secondary: flip rate, accuracy, signed bias shift E\[m\_ablated − m\_clean\] per class, coherence.
- Refusal (primary): drop in refusal rate on the harmful eval prompts the clean model refuses, using the first-token classifier (r(x) above a threshold calibrated in stage 0 against generation labels). Secondary: refusal rate on harmless prompts, AUROC of r(x).
- Refusal uses a rate rather than AUROC because suppressing refusal is a bias shift by design; AUROC could stay high while refusal collapses.

**Cost (x-axis):**

- Primary: mean KL(p\_clean ‖ p\_ablated) at the first response position on generic prompts, intervening at the same template positions.
- Secondary: mean relative residual change ‖x' − x‖ / ‖x‖ at intervention positions.

**Summaries:**

- D@c\*: damage interpolated (linear in log cost, monotone envelope) at c\* = the `massmean` anchor's cost for that task and strategy. If a curve never reaches c\*, report its maximum and flag it.
- Gap closure G = (D@c\* − D\_random@c\*) / (D\_anchor − D\_random@c\*), with D\_random from the `random` rule in the same basis.
- Uncertainty: 1,000 bootstrap resamples of eval examples, averaged over the 3 discovery resamples. Benjamini–Hochberg correction within each hypothesis family; report q-values.
- Generation check: for both anchors and for each basis's best and worst cell, generate 48 greedy tokens on 200 harmful eval prompts and report the substring-based refusal rate. If the first-token classifier agrees with generation labels on < 90% of stage-0 prompts, warn and switch refusal damage to generation for all configurations.

**Hypotheses**, computed automatically on D@c\* per task and strategy (thresholds in config):

| ID | Hypothesis | Computed test | Default threshold |
| --- | --- | --- | --- |
| H0 | Harness works | Anchor drops harmful refusal rate (generation-confirmed); random rule damage at c\* is small relative to the anchor | Drop ≥ 0.5; random ≤ 25% of anchor damage |
| H1 | Bases differ | For some rule, the largest between-basis difference in D@c\* | 95% CI excludes 0 |
| H2 | Attribution beats readout | In every basis, D@c\*(attrib) − max(meandiff, l1probe) | CI excludes 0 in all 3 bases |
| H3 | Selection dominates basis | Mean spread across rules minus mean spread across bases; G per rule and basis | Difference CI > 0; G(attrib) ≥ 0.7 and G(readout) ≤ 0.4 in every basis |
| H4 | Basis × rule interaction | A pair of bases swaps order between two rules | Swapped difference CI-separated under both rules |
| H5 | Strategy interaction | A pair of (basis, rule) cells swaps order between mean and resample; Kendall τ across strategies | One CI-separated reversal |
| H6 | Task consistency | Kendall τ of the 9 cells' D@c\* between truth and refusal | τ ≥ 0.6 |
| X1 | Bias vs discrimination (exploratory) | Truth: bias shift per unit AUROC drop, per cell | Report only |

**Decision logic** printed at the top of `report.md` (per task, then overall):

1. H0 fails → `INVALID HARNESS`: stop interpreting; list what failed.
2. H4 or H5 holds → `OUTCOME 1: interaction`.
3. H2 and H3 hold, H4 does not → `OUTCOME 2: selection beats basis`.
4. None of H1–H5 holds → `OUTCOME 3: uniform`.
5. Anything else → `MIXED`, listing which hypotheses held (e.g. H1 + H2 without H3 = two main effects).

Always print H6 beside the outcome, since two tasks that disagree weaken any outcome.

## Repository layout and pipeline

A small Python package driven by `run_pilot.sh`, with eight resumable stages. Suggested layout (Claude Code may adjust and must report changes):

```
pilot/
  run_pilot.sh            # the only command the user runs
  README.md               # run tutorial
  CHANGES.md              # every file created/modified, one line each
  requirements.txt
  config/pilot.yaml       # full run
  config/smoke.yaml       # overrides for --smoke
  src/pilot/
    cli.py                # stage orchestration, --resume, run_id
    env_check.py          # GPU, aarch64 torch build, HF token, versions
    models.py             # load model, resolve layer, hooks
    data.py               # download, chat templates, splits, positions
    bases.py              # sae / pca / dims / mlp: read(), write dirs, fit
    selection.py          # meandiff, l1probe, attrib, random
    ablate.py             # unified ablation hook + strategies
    metrics.py            # damage, cost, coherence, refusal classifier
    analysis.py           # curves, D@c*, G, bootstrap, H0-H6, decision
    plots.py
    report.py
  tests/
    test_ablation.py      # v_i = r_i(x) leaves x unchanged; sae keeps error term
    test_bases.py         # pca/dims orthonormal; read/write round-trip
    test_selection.py     # toy data with a planted unit is recovered by each rule
```

**Config sketch** (`config/pilot.yaml`):

```yaml
model: auto            # auto | 1.7b | 4b | 8b
layer_depth: 0.5
sae: {small: {width: 65k, l0: 80}, large: {width: 20480}}
bases: [sae, pca, dims]          # add mlp to enable the optional arm
rules: [meandiff, l1probe, attrib, random]
strategies: [mean, resample]     # add zero for the diagnostic
budgets: [1, 2, 5, 10, 20, 50, 100, 200, 500]
discovery_resamples: 3
random_draws: 3
bootstrap: 1000
batch_size: 64
seed: 0
gate: {min_refused_harmful: 150, refuse_harmless: 0.10}
thresholds: {h0_drop: 0.5, h0_random_frac: 0.25, g_attrib: 0.7, g_readout: 0.4, h6_tau: 0.6}
```

**Stages** (each writes outputs plus a `DONE` marker in `runs/<run_id>/<stage>/`):

1. `env`: check GPU, torch build, HF access; write `manifest.json`.
2. `gate`: read the Step 0 decision (running the check first if it is missing) and load that model; calibrate refusal-start tokens and the first-token threshold; SAE sanity check at the resolved layer.
3. `data`: download, split, apply chat templates, record intervention positions.
4. `cache`: clean forward passes; cache layer-L activations at intervention positions, clean scores m(x) and r(x), and clean next-token log-probabilities on generic prompts.
5. `bases`: fit PCA, load SAE, compute per-unit, per-position mean values.
6. `select`: rankings for every task × basis × rule × discovery resample (one backward pass per discovery example for `attrib`).
7. `ablate`: for every configuration and k, a full forward pass with the ablation hook at layer L on eval and generic prompts; save per-example scores. Then the generation checks.
8. `analyze`: curves, D@c\*, G, bootstrap CIs, H0–H6, decision; write plots and `report.md`.

Use full forward passes with a hook rather than restarting from layer L; prompts are short and this avoids reimplementing each model's attention masking and Qwen3.5's recurrent layers.

## Smoke test

`bash run_pilot.sh --smoke` runs every stage on a tiny configuration in under 10 minutes on the GB10, and Claude Code must run it and fix all failures before finishing.

| Setting | Smoke value |
| --- | --- |
| Model | Qwen3-1.7B forced; the Step 0 decision is neither read nor written |
| SAE | 16k width, L0 80 (trainer\_0) |
| Examples | 32 per split per set |
| Factors | All bases, rules and strategies |
| Budgets | k ∈ {1, 10, 100} |
| Resamples | 1 discovery resample, 1 random draw, 50 bootstrap samples |
| Generation check | 8 prompts × 16 tokens |

The smoke run passes only if all of these hold:

- `pytest -q` passes (ablation identity, basis round-trip, planted-unit recovery).
- Identity check on the real model: ablating with v\_i = r\_i(x) changes first-response logits by less than 1e-2 (bf16 tolerance).
- SAE sanity check passes at the resolved layer.
- No NaN or Inf in any saved metric.
- Every expected file in Outputs exists and `report.md` contains a decision line (its content is meaningless at this size).
- `--resume` after the run completes finishes in under 30 seconds without recomputing.
- The `env` check reports a cu130 torch build and device capability (12, 1), the memory watchdog is running, and peak memory from `/proc/meminfo` is logged.
- `bash run_pilot.sh --check --smoke` also passes and writes only `decision_smoke.json`.

If no GPU is available where Claude Code runs, run the same smoke test on CPU with Qwen3-1.7B, say so in the summary, and leave GPU-only paths untested but flagged.

## Outputs and report

Everything lands in `runs/<run_id>/`; `report.md` is the file the user reads first, and it opens with the decision line.

| Path | Contents |
| --- | --- |
| `manifest.json` | Model, resolved layer, SAE id, positions, package versions, git commit, seeds, wall time and peak memory per stage, and the hardware record from Target machine |
| `log.txt` | Full log |
| `gate/gate.json` | Baseline metrics, chosen model and why, refusal-start tokens, classifier agreement |
| `select/rankings.parquet` | task, basis, rule, resample, unit, rank, score |
| `select/stability.csv` | Top-k Jaccard overlap across discovery resamples |
| `ablate/results.parquet` | One row per task × basis × rule × strategy × k × resample × draw: damage, cost, coherence, bias shift |
| `ablate/per_example/` | Per-example scores for bootstrapping |
| `ablate/generation_checks.csv` | Generation-based refusal rates for anchors and best/worst cells |
| `analyze/summary.csv` | D@c\*, G and CIs per cell |
| `analyze/hypotheses.json` | H0–H6 and X1: statistic, CI, q-value, pass/fail |
| `plots/curves_<task>_<strategy>.png` | Damage vs cost (log x), 9 cells + random + anchor point |
| `plots/gap_closure_<task>.png` | G as a basis × rule heatmap, one panel per strategy |
| `plots/stability.png` | Selection stability per rule and basis |
| `report.md` | Human-readable summary (below) |

**`report.md` order:**

1. Decision line per task and overall (`OUTCOME 1/2/3`, `MIXED` or `INVALID HARNESS`), with H6.
2. Hypothesis table: ID, statistic, CI, q-value, pass/fail.
3. Damage-vs-cost plots and gap-closure heatmaps.
4. Gate results and which model ran.
5. Selection stability.
6. Caveats triggered during the run (e.g. curves that never reached c\*, SAE prefiltering, classifier fallback).
7. Runtime per stage.

## Runtime estimate on GB10

Expect about 4–8 hours end to end with Qwen3-1.7B (2.5–5 hours with the optional fast path) and about 9–18 hours with Qwen3.5-4B, dominated by the ablation stage. The smoke test runs in under 10 minutes.

**Workload.** About 674 configurations per task: 3 bases × (3 rules × 3 discovery resamples + 3 random draws) × 9 budgets × 2 strategies, plus 26 anchor runs. Each truth configuration evaluates about 1,500 eval + 300 generic prompts; each refusal configuration up to 1,000 + 300, fewer once harmful prompts are filtered to those refused at baseline. That is about 2.1 million prompt forwards, roughly 125 million tokens at about 60 tokens per prompt.

**Compute per token.** Logits are computed only at the answer position, so cost is about 2 × non-embedding parameters: about 2.8 GFLOP per token for Qwen3-1.7B (about 1.4B non-embedding parameters) and roughly 6.5–7 GFLOP for Qwen3.5-4B (about 3.3B). Read exact counts from the loaded models. The ablation stage is about 3.5 × 10¹⁷ FLOP for 1.7B and about 8 × 10¹⁷ for 4B.

**Throughput assumption.** NVIDIA lists the GB10 with [128 GB unified memory at 273 GB/s](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) but gives no dense BF16 figure; the DGX Spark setup guide's training code models the GB10 at about 104 TFLOPS peak. This estimate assumes 20–40 TFLOPS effective for batched Hugging Face prefill; short prompts at batch 64–128 are compute-bound, and every model here fits well under the 100 GB memory cap. For Qwen3.5-4B it also assumes the fused Gated DeltaNet kernels load on aarch64; the pure-PyTorch fallback can be several times slower.

| Stage | Qwen3-1.7B | Qwen3.5-4B |
| --- | --- | --- |
| Install and downloads (model \~3.4 GB / \~8–9 GB; one SAE layer \~0.5–1.1 GB / \~0.4 GB) | 10–20 min | 20–40 min |
| Gate, data, cache, bases | 15–30 min | 20–40 min |
| Selection (attribution passes, L1 paths on CPU) | 15–45 min | 20–45 min |
| Ablation, full forward passes | 3–6 h | 7.5–15 h |
| Ablation with prefix reuse (optional fast path) | 1.5–3 h | 4–8 h |
| Analysis and bootstrap | 20–40 min | 20–40 min |
| **Total, full forward** | **4–8 h** | **9–18 h** |

The Qwen3-8B contingency (about 7B non-embedding parameters) would need about 15–30 hours of ablation with full forward passes, or 8–15 hours with the fast path.

**Required optimizations:** compute logits only at the answer position (`logits_to_keep=1` or equivalent), sort prompts by length to minimize padding, and batch at 64–128.

**Optional fast path:** run layers 0..L once per prompt, cache the full-sequence hidden states leaving layer L, and run only layers L+1..end per configuration. This also works for the hybrid model, because every layer consumes the whole sequence. Enable it only if an equivalence test passes (max |Δlogit| vs full forward < 1e-2 on 64 prompts); otherwise fall back to full forward and log why.

**Measured projection:** the smoke test must time throughput (tokens/s for the ablation forward) and write a projected full-run time for the small and large models to the console and `report.md`. Trust that number over this table.

If a run is too long, the cheapest cuts are 2 discovery resamples instead of 3 (−33% of the ablation stage), 7 budgets instead of 9, and 200 generic prompts instead of 300.

## Acceptance criteria and known pitfalls

The work is done when the smoke test passes on the GB10, a full run produces `report.md` with a decision line, and `CHANGES.md` lists every file touched.

**Acceptance checklist:**

- [ ] `bash run_pilot.sh --smoke` passes every check in the Smoke test section.
- [ ] `bash run_pilot.sh --check`, then `bash run_pilot.sh`, runs to completion; the pilot uses the model named in `decision.json`, and `report.md` opens with that model and the decision line.
- [ ] `--resume` skips finished stages.
- [ ] No hardcoded layer numbers or SAE ids; both are resolved and recorded in `manifest.json`.
- [ ] Unit tests cover the ablation identity, basis round-trips and planted-unit recovery.
- [ ] README.md has the run tutorial; CHANGES.md lists all files with one-line purposes and any deviations from this spec.

**Known pitfalls to handle explicitly:**

- **aarch64 PyTorch.** The GB10 is ARM64 with a new Blackwell GPU. In the `env` stage, check that CUDA is available, print the device capability, and run a small bf16 matmul. Expect capability (12, 1). A libcudart.so.12 import error means a CUDA 12 wheel slipped in: stop and name the package. If the cu130 wheels fail, suggest NVIDIA's NGC PyTorch container.
- **Qwen3.5 hybrid layers.** Qwen3.5-4B needs a `transformers` version that supports it, and its Gated DeltaNet layers may use optional fused-kernel packages. Triton-based kernels work once TRITON\_PTXAS\_PATH is set; compiled CUDA extensions are usable only if built for CUDA 13 on aarch64, so skip any CUDA 12 build. In `env`, log which implementation loads; if it is the pure-PyTorch fallback, warn and recompute the runtime projection.
- **Base-trained SAE on a post-trained model.** The REINS SAE was trained on Qwen3.5-4B-Base. Report its explained variance and loss increase on chat prompts in `report.md`, and stop with `INVALID HARNESS` if it fails the sanity check; switching to Qwen3-8B is a manual plan change, never automatic.
- **SAE thresholds and scaling.** Run the adamkarvonen BatchTopK SAEs with their fixed per-SAE threshold (JumpReLU-style), never batch top-k, so activations do not depend on batch composition. Apply each release's own input normalization and derive decoder directions in raw activation space.
- **Attention sinks.** Exclude tokens with residual norm above 10× the median from SAE, PCA and discovery statistics, matching how the 1.7B SAEs were trained.
- **Thinking mode.** Pass `enable_thinking=False` to every chat-template call; a missing flag silently shifts the answer position.
- **Answer tokenization.** `True`/`False` may tokenize differently with and without a leading space at the response start. Pick the variant with higher clean probability and log it.
- **Numerics.** Compute ablation deltas in fp32, then cast back to bf16.
- **Leakage.** Resample values come from the discovery split only, never from eval.
- **L1 path speed.** On the SAE basis, use warm starts along the regularization path, cap iterations, and log convergence warnings.
- **Harmful text.** Refusal prompts contain harmful content. Log prompt ids, not prompt text, and never print generations from harmful prompts to stdout.

## Sources

- [adamkarvonen/qwen3-1.7b-saes](https://huggingface.co/adamkarvonen/qwen3-1.7b-saes) (SAE layers, widths, L0, training data, attention-sink filtering) and the [Qwen SAEs collection](https://huggingface.co/collections/adamkarvonen/qwen-saes) (8B release)
- [REINS-SAE](https://huggingface.co/Carlos4869/REINS-SAE) (Qwen3.5-4B-Base SAEs: layers, width, k, normalization, license)
- [He et al. (2026), Qwen3-Instruct SAE paper](https://arxiv.org/abs/2606.26620) (unsteered refusal rates for Qwen3-1.7B and Qwen3-4B)
- [geometry-of-truth](https://github.com/saprmarks/geometry-of-truth) (truth datasets)
- [refusal\_direction](https://github.com/andyrdt/refusal_direction) (harmful and harmless splits)
- [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) (GB10 memory and bandwidth)
- [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup) (GB10 sm\_121, cu130 aarch64 wheels, harmless capability warning, SDPA, TRITON\_PTXAS\_PATH, unified-memory OOM behavior)
