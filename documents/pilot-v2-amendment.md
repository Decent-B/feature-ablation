# Pilot v2 Amendment: Implementation Spec (for Claude Code)

Revised to finish the development and confirmatory runs within 12 GPU hours on the GB10 (Section 17). Qwen3-1.7B runs three depths; Qwen3.5-4B runs five.

This amends `pilot-implementation-plan.md` (v1). Everything in v1 still applies unless this document changes it: target-machine settings, GB10 install rules, memory safety, resumable stages, manifest, unit tests and pitfalls. Read v1 first, then this.

## 1. Why v2 exists

The v1 pilot (run `20261116-180218-pilot-qwen3-1.7b`) returned `INVALID HARNESS`. Intervening only on the 9 formatting tokens after the user's message left the truth behaviour unbreakable, and the refusal anchor fell just short of its bar. v2 changes where and how deep the interventions go, adds tasks and a second model, makes every curve reach the comparison cost, and splits the work into a development run and a preregistered confirmatory run.

## 2. Amendment log

| ID | Change | v1 | v2 | Reason |
| --- | --- | --- | --- | --- |
| A1 | Intervention sites | 9 template tokens | S1: all prompt tokens. S2: all prompt tokens + every generated token | Truth information lives on the statement's own words; refusal literature also intervenes on generated tokens |
| A2 | Depths | 50% | Qwen3-1.7B: 25, 50, 75%. Qwen3.5-4B: 25, 40, 50, 60, 75% | Test whether conclusions depend on layer; 1.7B's SAE release covers only its three depths |
| A3 | Tasks | Truth, refusal | + sentiment, answerability (toxicity as a preregistered substitute) | Pilot tasks have been looked at; new tasks provide confirmatory evidence |
| A4 | Models | Qwen3-1.7B (Step 0) | + Qwen3.5-4B as a replication model | Replication on an unseen model and architecture. Step 0's binding rule applied to v1 only |
| A5 | Curve extension | Budgets stop at 500 | Every curve continues until it reaches the comparison cost | Pilot random curves never reached c* in two bases |
| A6 | Mean / resample values | Per position offset | Position-pooled, per token role | Prompts differ in length; generated tokens have no fixed slots |
| A7 | Primary metric and H0 | Anchor-normalized (G); H0 gates the verdict | Damage in excess of random at fixed costs; H0 and anchors reported, never gating | Anchors no longer decide conclusions; the random floor stays inside the metric |
| A8 | Refusal damage and cost | First-token score; KL at the first response position | Generation-based refusal rate; KL over the first response position + 8 teacher-forced tokens | S2 acts on generated tokens, which first-token measures cannot see |
| A9 | Process | One run | Smoke → development run → freeze preregistration → confirmatory run | Test hypotheses on data not used to form them |
| A10 | Run time | — | 12-hour GPU budget: smaller test sets, a leaner grid and a preregistered priority order (Sections 12 and 17) | Fit the development and confirmatory runs into one overnight window |

Resample ablation keeps drawing source values from both classes. Copying only from same-class prompts (considered after the pilot) would preserve the class information that ablation is meant to remove, so it is not adopted.

## 3. Instructions to Claude Code

1. **Extend the existing repo.** `run_pilot.sh` remains the single entry point. Add the modes `--v2 --prefetch`, `--v2 --smoke`, `--v2 --dev`, `--v2 --freeze` and `--v2 --confirm`; `--v2 --smoke` runs the prefetch first. All v2 settings live in `config/v2.yaml`. v1 modes and outputs stay untouched and reproducible.
2. **Separate outputs.** v2 runs write to `runs/v2/<phase>/<run_id>/`. Never modify or delete v1 run directories.
3. **README.md.** Add a v2 section: the order of commands, what the freeze step does and why, expected runtime per phase, and how to read `report_v2.md`.
4. **Report all changes.** List every file created or modified with a one-line purpose, plus any deviation from this spec and why, in your final message and in a v2 section of `CHANGES.md`.
5. **Smoke test.** Run `bash run_pilot.sh --v2 --smoke` yourself and fix every failure. Do not run `--dev`, `--freeze` or `--confirm`; the user runs them.
6. **Draft the preregistration.** Write `prereg/hypotheses.yaml` and `prereg/PREREGISTRATION.md` from Section 13. The user reviews, edits and freezes them.
7. **Ask, don't guess.** If a dataset is gated, an SAE file layout or layer indexing is unclear, or a model fails to load, stop and report the closest alternative instead of substituting silently.

## 4. Run phases

| Phase | Command | What it runs | Who |
| --- | --- | --- | --- |
| Smoke | `--v2 --smoke` | Prefetches every model, SAE and dataset, then runs every stage on tiny data for both models; tunes batch size, measures throughput and projects runtimes against the 12-hour budget | Claude Code |
| Development | `--v2 --dev` | Qwen3-1.7B, truth and refusal, 50% depth, both sites, reduced Tier 1 (Section 12); writes `dev_report.md` | User |
| Review | none | The user reads the dev report and edits the draft preregistration if needed | User |
| Freeze | `--v2 --freeze` | Hashes and commits the preregistration and analysis code, then tags the commit | User |
| Confirmatory | `--v2 --confirm` | Inclusion gates, then the grid for both models and all tasks in priority order within the 12-hour budget (Section 17); confirmatory analysis runs first | User |

- The dev run may change harness settings that involve no hypothesis (generation length, batch sizes, bug fixes). Its data never enter the confirmatory analysis.
- `--confirm` refuses to start unless a valid freeze exists (Section 13).

## 5. Models and depths

| Setting | Qwen3-1.7B | Qwen3.5-4B |
| --- | --- | --- |
| HF model id | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3.5-4B` |
| Layers / d_model | 28 / 2048 | 32 / 2560 |
| SAE release | `adamkarvonen/qwen3-1.7b-saes`, 65k, L0 80 (`trainer_2`) | `Carlos4869/REINS-SAE`, folder `qwen3.5-4b-base/`, width 20,480 |
| SAE depths available | 25%, 50%, 75% | Every layer |
| Depths run | 25, 50, 75% | 25, 40, 50, 60, 75% |
| Layers | 7, 14, 21 | 8, 13, 16, 19, 24 |

**Layer rule.** L = round(depth × `num_hidden_layers`), on the 0-indexed decoder-layer output (`resid_post`). Read `num_hidden_layers` from the model config. The pilot's layer 14 for Qwen3-1.7B matches this rule. Confirm each SAE file's layer indexing (0- or 1-based) with its reconstruction check, not from the filename.

**Depths per model.** Qwen3-1.7B runs only the three depths its SAE release covers, so all three bases are available at every depth it runs. Qwen3.5-4B runs all five.

**SAE inclusion rule** (preregistered, mechanical). The SAE basis is included at a (model, depth) only if, on chat-formatted prompts, its reconstruction explains at least 70% of variance and next-token loss rises by at most 10% (relative) when the state is replaced by its reconstruction. Otherwise that cell runs without the SAE basis and the report says so. This replaces v1's stop-the-run behaviour.

## 6. Tasks

| Task | Status | Data | User turn | Answer tokens | Score |
| --- | --- | --- | --- | --- | --- |
| Truth | Pilot task | v1 `cities` + `neg_cities` splits | v1 template | True / False | m(x) = logit(True) − logit(False) |
| Refusal | Pilot task | v1 `refusal_direction` splits | The instruction as given | — | Generation-based refusal (32 greedy tokens); first-token r(x) secondary |
| Sentiment | New | SST-2 (`stanfordnlp/sst2`): discovery 1,000 balanced from train; eval 500 balanced from validation | `Is the sentiment of this review positive or negative? Answer with Positive or Negative only.\nReview: {sentence}` | Positive / Negative | logit(Positive) − logit(Negative) |
| Answerability | New | SQuAD 2.0 (`rajpurkar/squad_v2`), contexts ≤ 150 tokens: discovery 1,000 balanced from train; eval 400 balanced from validation | `Read the passage and decide whether the question can be answered from it. Answer with Yes or No only.\nPassage: {context}\nQuestion: {question}` | Yes / No | logit(Yes) − logit(No); answerable = positive class |
| Toxicity | Substitute only | Civil Comments (`google/civil_comments`): toxic ≥ 0.5, non-toxic ≤ 0.1; 1,000 discovery / 500 eval, balanced | `Is the following comment toxic? Answer with Yes or No only.\nComment: {text}` | Yes / No | logit(Yes) − logit(No) |

Verify that every dataset id exists and is ungated before use. Resolve answer-token variants (with and without a leading space) as in v1.

**Eval sizes.** Truth: a fixed-seed, balanced 500-statement subset of the v1 eval split. Refusal: up to 200 harmful eval prompts the clean model refuses, plus 100 harmless. Generic cost set: 100 of v1's 300 prompts.

**Task inclusion gate** (preregistered, mechanical; separate from H0). A (model, task) pair enters the analysis only if the clean model passes:

- Single-token tasks: AUROC ≥ 0.85 and coherence ≥ 0.9.
- Refusal: at least 150 harmful eval prompts refused (generation-labelled) and harmless refusal ≤ 0.10.

If sentiment or answerability fails on a model, toxicity replaces it on that model. If a pilot task fails on Qwen3.5-4B, it is excluded and reported.

**Development vs confirmatory pairs.**

- Development (reported, never counted toward C1–C3): (Qwen3-1.7B, truth) and (Qwen3-1.7B, refusal).
- Confirmatory: (Qwen3-1.7B, sentiment), (Qwen3-1.7B, answerability), and all four tasks on Qwen3.5-4B. That is six pairs if all pass their gates.

## 7. Intervention sites and positions

**Token roles.**

- `prompt`: every prompt token, including the user's content, the chat-template tokens and the empty thinking block.
- `generated`: every token the model produces in its reply.
- `excluded`: the first token and any attention-sink token (residual norm above 10× the median at layer L). Never intervened on and never used in statistics. Log counts per prompt.

**Sites.**

| Site | Intervened tokens | Used for |
| --- | --- | --- |
| S1 | All prompt tokens | All tasks (primary site) |
| S2 | All prompt tokens + every generated token | Refusal only |

Single-token tasks are scored at the first response position, which the model computes from the prompt alone. S2 is therefore identical to S1 for them by construction and is not run.

**Generation under each site** (refusal damage and generation checks):

- S1: the hook is active during prefill on prompt tokens and inactive while decoding.
- S2: the hook is also active at every decoding step on the new token, using generated-role values. Edited states flow into the KV cache as usual.
- Greedy decoding, 32 new tokens, refusal labelled with v1's substring rules. The dev run compares 32- vs 64-token labels on 100 prompts and reports their agreement.

**Selection statistics.**

- A unit's readout value for an example is its mean over that example's non-excluded prompt tokens. `meandiff`, `l1probe` and the `massmean` anchor use these per-example means.
- `attrib`: A_i = E_x[ Σ_t (v_i − r_i(x_t)) · ∂S/∂r_i(x_t) ], summed over the example's prompt tokens. S = y·m(x) for single-token tasks and y·r(x) (first-token refusal score) for refusal, with y = ±1 by class. v_i is the prompt-role pooled mean.
- One backward pass per discovery example gives ∂S/∂x at all five depths at once, with hooks on all five layers.
- Known limitation, to state in the paper: selection uses prompt tokens for both sites, and refusal attribution uses the first-token score although refusal damage is generation-based.

## 8. Ablation values: position-pooled means and resampling

The formula is unchanged from v1, applied at every intervened token:

```latex
x' = x + \sum_{i \in S} \big(v_i - r_i(x)\big)\, w_i
```

- **mean**: v_i is the unit's average over all non-excluded tokens of the same role across the class-balanced discovery prompts. Prompt-role means come from the discovery prompts; generated-role means come from the clean greedy replies (32 tokens) to those prompts. There is one value per unit, per role, per (model, task, depth), and no per-position values.
- **resample**: for each intervened token, copy r_i from a random token of the same role in a random discovery prompt drawn from both classes. All selected units are copied jointly from the same source token. Seeded.
- Keep a per-role reservoir of source token vectors (for example 20,000 prompt-role and 5,000 generated-role tokens per model, task and depth) instead of storing all activations.

## 9. Budgets and curve extension

**Ladders.** Tier 1: k ∈ {1, 5, 20, 50, 100, 200, 500}. Tier 2: k ∈ {1, 5, 20, 100, 200, 500}.

**Extension rule** (applies to every rule, including random). After the ladder, if a curve's largest cost is below c_target = max(c*, 0.5), keep doubling k (1,000, 2,000, …) until the cost reaches c_target or the cap. Caps:

- `dims` and `pca`: all d_model units.
- `sae` random: all latents firing on at least 1% of discovery tokens.
- `sae` meandiff and attrib: all ranked latents.
- `l1probe`: its 8,192 prefiltered latents.

A curve that hits its cap below a needed cost is flagged; its maximum damage is used there and listed in the report.

## 10. Metrics

**Damage** (per task):

- Single-token tasks: AUROC drop of the task score against the label.
- Refusal: drop in generation-based refusal rate on the clean-refused harmful prompts. Harmless refusal rate and first-token AUROC are secondary.

**Cost (A8).** Mean KL(p_clean ‖ p_ablated) over the first response position plus 8 teacher-forced continuation positions, on the 100 generic prompts. Generate each generic prompt's clean 8-token continuation once. Then score prompt + continuation in one forward pass with the intervention applied per site (S1: prompt tokens only; S2: also the continuation tokens). Also log v1's first-position KL for comparability.

**Primary metric (A7): damage in excess of random at fixed costs.**

E_{r,b}(c) = D_{r,b}(c) − D_{random,b}(c), within the same model, task, depth, site, strategy and basis.

- D(c) is interpolated linearly in log cost on the curve's monotone envelope.
- c₀ = 0.1 nats is the primary cost; 0.02 and 0.5 are secondary.
- 95% CIs from 1,000 bootstrap resamples of eval examples, averaged over discovery resamples. Benjamini–Hochberg correction within each hypothesis family.

**Secondary.** c* (the anchor's cost), D@c* and gap closure G. G is computed only where the anchor's damage exceeds random by at least 0.1 at c*; otherwise it is reported as "not computed".

## 11. Anchors and H0

Anchors work as in v1, adapted to v2: the `massmean` direction comes from per-example prompt-token means, and its projection is ablated at every intervened token. `random_dir` uses 5 directions in Tier 1 and 3 in Tier 2.

H0 is computed per (model, task, depth, site, strategy) with v1's bars: refusal anchor drop ≥ 0.5 in generated replies, and random ≤ 25% of the anchor's damage at c*. For single-token tasks the anchor's AUROC drop is reported descriptively.

**H0 never changes a verdict, a label or the decision logic.** The report shows H0 status beside every result as a caveat column.

## 12. Design tiers

| Tier | Depths | Strategies | Discovery resamples | Random draws | Budgets | Anchors |
| --- | --- | --- | --- | --- | --- | --- |
| Tier 1 | 50% | mean; resample for refusal only | 2 | 2 | 7 + extension | massmean + 5 random_dir per strategy |
| Tier 2 | Qwen3-1.7B: 25, 75%. Qwen3.5-4B: 25, 40, 60, 75% | mean | 1 | 2 | 6 + extension | massmean + 3 random_dir |
| Controls | 50%, S1 | mean | 1 | — | See below | — |
| Dev run | 50% (1.7B truth and refusal) | mean; resample for refusal only | 1 | 2 | 7 + extension | As Tier 1 |

Site S2 (refusal) runs in Tier 1 and the dev run with mean ablation only. Tier 2 uses site S1 only.

**Controls** (support secondary analysis S4):

- `attrib_shuffled`: attribution computed with shuffled class labels, in all 3 bases, on the Tier 1 ladder.
- Cross-task: the top-k attribution units (k = 50 and 200) chosen for task A are ablated on task B, for the pairs truth ↔ refusal and sentiment ↔ answerability (or toxicity), on each model.
- C3 groups: per basis, split units into 5 quintiles of readout strength (|Cohen's d|; for `sae`, among latents firing on ≥ 1% of tokens). Draw 4 random groups of 20 units per quintile, ablate each group, and record damage, cost and the group's summed attribution.

**Size.** Tier 1 ≈ 200 interventions per (model, task, site, strategy). Tier 2 ≈ 110 per (model, task, depth). Controls ≈ 87 per (model, task). Totals: about 2,400 interventions on Qwen3-1.7B and 3,300 on Qwen3.5-4B.

## 13. Preregistration

**What is frozen:**

- `prereg/hypotheses.yaml` (machine-readable) and `prereg/PREREGISTRATION.md` (human-readable, generated from the YAML plus the rationale).
- `config/v2.yaml` and the analysis code (`src/pilot/analysis_v2.py` and the metrics code it calls).
- The run plan: the block priority order and any blocks the smoke projection says will not fit in the 12-hour budget (Section 17).

**Freeze** (`--v2 --freeze`, run by the user):

1. Refuse if the git working tree has uncommitted changes.
2. Compute the SHA-256 of every frozen file and write `prereg/FROZEN.json` with the hashes, the git commit and a UTC timestamp.
3. Commit `FROZEN.json` and tag the commit `prereg-v2-frozen`.
4. Print a reminder that uploading `PREREGISTRATION.md` to OSF or AsPredicted adds an independent timestamp. The script never uploads anything.

**Enforcement** (`--v2 --confirm`):

- Refuse to start unless `FROZEN.json` and the tag exist and every hash matches.
- Any change after the freeze requires `--deviation "<reason>"`. The reason, timestamp and changed files are appended to `prereg/deviations.md` and listed at the top of the report.
- Run the preregistered confirmatory analysis first and write `analyze/confirmatory.json` before any other analysis. Everything else is labelled exploratory.

### Draft hypotheses (the user reviews before freezing)

**Notation.** Primary setting = 50% depth, site S1, mean ablation. For each basis b, Δ_b = E_attrib,b(c₀) − max(E_meandiff,b(c₀), E_l1probe,b(c₀)). Smallest effect of interest δ = 0.05 (in AUROC or refusal-rate units).

| ID | Claim | Test | Supported if |
| --- | --- | --- | --- |
| C1 (primary) | Causal selection beats readout selection | Per confirmatory pair: the mean of Δ_b over included bases | CI lower bound > 0, point estimate ≥ δ, and no basis whose Δ_b CI lies entirely below 0. Overall: supported in ≥ 75% of included confirmatory pairs (rounded up) and in at least one pair per model |
| C2 (primary) | Basis rankings depend on the selection rule | Per pair: bases b1, b2 and a readout rule r exist with E_r,b1 − E_r,b2 > 0 and E_attrib,b1 − E_attrib,b2 < 0, both CI-separated | A reversal is present in ≥ 50% of included confirmatory pairs |
| C3 (secondary) | Readout strength only weakly predicts causal effect | Spearman ρ between group readout strength and group damage (20 groups per basis), pooled by mean Fisher z over bases and pairs | Pooled ρ ≤ 0.3 with 95% CI upper bound < 0.5 |

**Secondary descriptive analyses** (reported with CIs, not part of the central claim):

- **S1 Depth:** Δ by depth for each basis (three depths on Qwen3-1.7B, five on Qwen3.5-4B).
- **S2 Site:** refusal Δ under S2 vs S1, at 50% depth.
- **S3 Strategy:** Kendall τ of the 9 basis × rule cells between mean and resample, for refusal at 50% depth.
- **S4 Circularity controls:**
  - (a) `attrib_shuffled` E(c₀) lies within ±δ of 0.
  - (b) Cross-task specificity: E_B(units attributed for A) < 0.5 × E_B(units attributed for B) at equal k.
  - (c) Attribution accuracy: Spearman between summed attribution and observed damage across the C3 groups.
- **S5 Anchors and H0** per cell.

**Central-claim label** (printed first in the report):

- `SUPPORTED`: C1 and C2 both supported.
- `PARTIALLY SUPPORTED`: exactly one supported.
- `NOT SUPPORTED`: neither supported.

C3 and S4 are reported beside the label as strengthening or weakening evidence. H0 never changes the label.

**Inclusion rules frozen with the hypotheses:** the task gate and toxicity substitution (Section 6), the SAE inclusion rule (Section 5), and the development-pair list.

## 14. What gets rerun

| Item | Status |
| --- | --- |
| Environment, Step 0 decision, cached Qwen3-1.7B weights | Reused |
| v1 code modules | Reused and extended; v1 behaviour preserved under v1 flags |
| v1 run outputs | Archived as "v1, superseded"; never used in v2 analysis |
| Qwen3-1.7B truth and refusal | Rerun at its three depths (refusal at both sites at 50%), as development pairs |
| SAE files | adamkarvonen layers 7 and 21 added; REINS layers at all five depths |
| Qwen3.5-4B, sentiment, answerability | New |

## 15. Outputs and report

New files in `runs/v2/confirm/<run_id>/`:

| Path | Contents |
| --- | --- |
| `inclusion.json` | Baselines per (model, task), SAE fidelity per (model, depth), substitutions and exclusions |
| `ablate/results.parquet` | One row per intervention: model, task, depth, site, strategy, basis, rule, resample, draw, k, damage, cost (v2 and v1 KL), coherence, flags |
| `analyze/confirmatory.json` | C1–C3: statistic, CI, q-value, supported or not, per pair and overall |
| `analyze/secondary.json` | S1–S5 |
| `analyze/exploratory/` | Anything else, labelled exploratory |
| `plots/` | Primary-setting curves per (model, task), E heatmaps, Δ by depth, C3 scatter |
| `report_v2.md` | See below |

**`report_v2.md` order:**

1. Central-claim label, the C1–C3 table, and a per-pair table with an H0 caveat column.
2. Deviations from the preregistration and blocks skipped by the time budget, if any.
3. Inclusion decisions.
4. Secondary analyses S1–S5.
5. Development pairs (Qwen3-1.7B truth and refusal), same tables, labelled development.
6. Anchors and H0 per cell.
7. Exploratory analyses, labelled.
8. Hardware, runtime per stage and peak memory.

## 16. Smoke test (v2 additions)

`bash run_pilot.sh --v2 --smoke` passes only if all of these hold:

- All v1 smoke checks still pass.
- Both models load, and Qwen3.5-4B logs which Gated DeltaNet implementation loaded.
- The SAE fidelity check runs at one depth per model.
- The identity check passes at both sites, including during decoding: with v_i = r_i(x), generated tokens are unchanged under S2.
- Teacher-forced cost is exactly 0 for a no-op intervention.
- Position-pooled means and the resample reservoir are built for both roles.
- Curve extension triggers on a deliberately short ladder and stops at the target cost.
- `--confirm` refuses to run without a freeze, and refuses after a frozen file is edited (tested on a temporary copy).
- Prefetch has downloaded every model, SAE file and dataset the full run needs.
- Batch size is auto-tuned to the largest that fits under the 100 GB memory cap.
- Throughput is measured per (model, task), and projected times per block for `--dev` and `--confirm` are written to the console and `smoke_report.md`, together with any blocks that would not fit in the 12-hour budget.

Smoke settings: 16 examples per split, the 50% depth plus one Tier 2 depth, budgets {1, 10}, 1 discovery resample, 50 bootstrap samples, 8 generation prompts × 16 tokens. Target: under 25 minutes on the GB10, excluding the prefetch.

## 17. Run-time budget and estimate on the GB10

**Budget.** The development and confirmatory runs together must finish within 12 hours of GPU wall time, excluding the prefetch and the smoke test. The plan below needs about 9 hours, leaving about 3 hours of headroom.

**Priority order** (frozen with the preregistration). `--confirm` runs blocks in this order:

1. Inclusion gates, then the primary setting (50% depth, site S1, mean ablation) with anchors for every included pair. This is everything C1 and C2 need.
2. Controls and C3 groups.
3. Tier 2 at 25% and 75% depth, both models.
4. Refusal at site S2, and refusal with resample ablation (50% depth).
5. Tier 2 at 40% and 60% depth (Qwen3.5-4B only).

**Two guards.**

- **Before freezing:** the smoke test projects each block's time from measured throughput. If the total exceeds 12 hours, it lists the lowest-priority blocks that will not fit. The user decides what to drop or shrink before freezing, and the resulting drop list is frozen.
- **During the run:** `--confirm` keeps a running projection. If finishing the next block would exceed 12 hours, it stops, analyses the completed blocks, and lists the skipped ones in the report. Skipping depends only on speed, never on results.

**Required speedups** (no effect on results):

- Download every model, SAE file and dataset in `--v2 --prefetch`.
- Solve the L1 paths with a GPU proximal-gradient solver in torch, not scikit-learn.
- Auto-tune the batch size in the smoke test (largest that fits under the 100 GB cap).
- Run CPU work (bootstraps, plots) in a separate process while the GPU continues with the next block.
- Keep v1's speedups: logits only at the answer position, prompts sorted by length, prefix reuse.

**Per-intervention time.** Calibrated from the v1 pilot, which ran 1,340 interventions in 1 h 32 min on Qwen3-1.7B (about 19k tokens/s with prefix reuse). Qwen3.5-4B is assumed about 2.4× slower per token, provided its fused Gated DeltaNet kernels load on aarch64. Each figure includes the 100-prompt cost pass and about 0.3 s of fixed overhead.

| Task (eval size) | Qwen3-1.7B | Qwen3.5-4B |
| --- | --- | --- |
| Truth (500) | ~1.8 s | ~3.8 s |
| Sentiment (500) | ~1.9 s | ~4.1 s |
| Answerability (400, contexts ≤ 150 tokens) | ~3.7 s | ~8.5 s |
| Refusal (≤ 300 prompts, 32 generated tokens) | ~2.7 s | ~5.2 s |

| Phase | Time |
| --- | --- |
| Development run | ~0.6 h |
| Baselines, caching, PCA (both models) | ~0.75 h |
| Selection (attribution passes, GPU L1 paths) | ~0.5 h |
| Ablations, Qwen3-1.7B (~2,400 interventions) | ~1.7 h |
| Ablations, Qwen3.5-4B (~3,300 interventions) | ~4.9 h |
| Analysis and plots (mostly overlapped with GPU work) | ~0.5 h |
| **Total** | **~9 h** |

**Worst case.** If Qwen3.5-4B falls back to the pure-PyTorch Gated DeltaNet implementation and runs about 3× slower, blocks 1 and 2 for both models still finish in roughly 8.5 hours. The run-time guard then trims blocks 3–5.

**What the 12-hour plan gives up** compared with the earlier, roughly 30-hour version:

- Confidence intervals about 40% wider, from halving the test sets, partly offset by comparing methods on the same examples. The dev run reports the actual widths; if they exceed δ = 0.05, raise the test-set sizes before freezing.
- Coarser curves near c₀ (7 budgets instead of 9).
- The mean-vs-resample comparison on refusal only, and the S2 site comparison at 50% depth only.
- Selection stability from 2 discovery resamples instead of 3.

## 18. Acceptance criteria and pitfalls

**Acceptance checklist:**

- [ ] `--v2 --smoke` passes every check in Section 16 on the GB10.
- [ ] `--v2 --dev` produces `dev_report.md` with anchors, H0 status, cost ranges and the 32- vs 64-token agreement.
- [ ] `--v2 --freeze` and `--v2 --confirm` enforce Section 13.
- [ ] `--v2 --confirm` follows the priority order, stops within the 12-hour budget, and the report lists any skipped blocks.
- [ ] `report_v2.md` opens with the central-claim label and the C1–C3 table; development pairs and exploratory results are clearly separated.
- [ ] No hardcoded layer numbers or SAE paths; all are resolved and recorded in the manifest.
- [ ] README v2 section and CHANGES.md v2 section are complete.

**Pitfalls** (in addition to v1's):

- **Sink tokens.** Intervening on attention-sink tokens wrecks attention patterns. Excluding them is mandatory, not an optimization.
- **Padding.** With batched prompts of different lengths, intervene only on real tokens; padding is never intervened on or pooled.
- **KV cache under S2.** The hook must edit each newly generated token's state at layer L before later layers write it to the cache. The decoding identity check verifies this.
- **REINS details.** Verify layer indexing and input normalization by reconstruction, not by filename.
- **SQuAD 2.0.** Its unanswerable questions are adversarial, so Qwen3-1.7B may fail the task gate. That is what the toxicity substitute is for; do not loosen the gate.
- **Long prompts.** Cap answerability contexts at 150 tokens when building the data, by selecting short passages, never by truncating mid-passage.
- **Harmful text.** Keep harmful prompts and generations out of stdout, logs and `dev_report.md` (v1 rule).
