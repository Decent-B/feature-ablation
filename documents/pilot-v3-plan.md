# Pilot v3 Plan: Does feature causality depend on basis, selection rule, task and model? Gemma 2 2B and Qwen3-1.7B base models (for Claude Code)

Status: draft for review, 2026-11-25.

This plan builds on `pilot-v2-amendment.md` (v2), which builds on `pilot-implementation-plan.md` (v1).
Everything in v2 still applies unless this document changes it. That includes the ablation formula,
position-pooled values, metrics, anchors, freeze and enforcement, memory safety and resumable stages.

## 1. Goal

**Headline to test:** feature causality depends on the basis, the selection rule, the task and the model.

"Feature causality" means how much switching off the selected units damages a behaviour, beyond what
switching off random units in the same basis does. The plan tests six factors:

- **Basis and selection rule** define the method. The test asks: does changing one of them
  reliably change the measured effect?
- **Task, model and depth** define the setting. The test asks: does the method that looks best
  change when the setting changes? This is the stronger form of "it depends": a conclusion drawn in
  one setting does not carry over to another.
- **SAE width** (Gemma only): does a 16k-feature SAE give different results from a 65k one at the
  same layer?

**One factor at a time.** Everything is held at one **reference setting**: Gemma 2 2B, the truth task,
50% depth, mean ablation and the 65k SAE. Each test changes only its own factor; everything else stays
at the reference. For example, the depth test runs five depths on Gemma and truth only. This keeps the
run short, and any change in a test can only come from the factor it varies.

v3 **does not test C1 or C2** from v2. The preregistered claims are the six dependence tests D1–D6 in
Section 8. Like any preregistered test, they can fail, and Section 8 states in advance what counts as
support.

**Why these models.** Both are base models whose developers published SAEs for every layer, trained
on that exact model:

- **Gemma 2 2B:** Google's Gemma Scope.
- **Qwen3-1.7B-Base:** Qwen's Qwen-Scope.

All five tasks work without a chat model (Section 6).

## 2. Change log (v2 → v3)

| ID | Change | v2 | v3 | Reason |
| --- | --- | --- | --- | --- |
| G1 | Models | Qwen3-1.7B, Qwen3.5-4B (chat) | `google/gemma-2-2b` (main) + `Qwen/Qwen3-1.7B-Base` (model test only), both **base** | Each has official SAEs at every layer |
| G2 | Design | Full grid over models, tasks and depths | **One factor at a time** around a reference setting (Section 7) | Each test needs only its own factor varied; far fewer runs |
| G3 | Depths | 3 (1.7B), 5 (4B) | 25, 40, 50, 60, 75% for the depth test; 50% everywhere else | Depth gets many levels only where it is the tested factor |
| G4 | SAE | 65k (1.7B, trained on the post-trained model); REINS (4B, excluded) | Gemma Scope 65k, plus 16k for the width test and as the fallback; Qwen-Scope 32k (TopK 100; TopK 50 fallback) | Each SAE matches its model, and SAE width can be tested |
| G5 | Tasks | Truth, refusal, sentiment, answerability (+ toxicity as a substitute) | Truth, sentiment, answerability, toxicity, **IOI** | Refusal needs a chat model. IOI (indirect object identification) is a standard base-model task |
| G6 | Prompts | Chat template + instruction | Plain text with a fixed 4-example prefix, identical on both models | Base models need examples to answer in format |
| G7 | Sites | S1 (prompt) + S2 (generated tokens, refusal) | S1 only: every token after the example prefix | No task generates text any more |
| G8 | Budgets | 7-point ladder, then doubling until a cost target | Fixed 6-point ladder (1, 5, 20, 100, 200, 500), **no doubling** | Simpler and faster |
| G9 | Comparison point | Fixed cost (c₀ = 0.1 nats) | **Fixed unit budgets**, averaged over the ladder; cost reported beside every result | Without doubling, most curves never reach a fixed cost (Section 7) |
| G10 | Repeats | 2 discovery resamples (Tier 1), 2 random draws | 1 discovery resample, 1 random draw | Shorter run |
| G11 | Ablation strategy | Mean everywhere; resample for refusal at 50% | Mean everywhere; resample at the reference setting | Check that results do not depend on how a unit is switched off |
| G12 | Hypotheses | C1–C3 | Dependence tests D1–D6; C1 and C2 dropped; C3 becomes secondary | The new headline |
| G13 | Controls | `attrib_shuffled`; one cross-task partner; C3 groups of 20 | No `attrib_shuffled`; cross-task over all task pairs; readout groups of 100 | Scrambled-label rule removed; v2's 20-unit groups barely did anything |
| G14 | Dev run | Development pairs excluded from the claims | Gemma dev run on held-out dev slices; its report withholds results | All tasks stay confirmatory |
| G15 | v2 fixes | — | SAE scaling chosen by measurement before the freeze; cross-task summary tested; budget overrides logged as deviations | The three process gaps v2 had |
| G16 | Access | Open models | Gated Gemma model, read with `HF_TOKEN` from the environment | Gemma requires accepting Google's license |

## 3. Models and Hugging Face access

| Setting | Gemma 2 2B (main model) | Qwen3-1.7B-Base (model test only) |
| --- | --- | --- |
| HF model id | `google/gemma-2-2b` (base, not `-it`) | `Qwen/Qwen3-1.7B-Base` (not the post-trained `Qwen/Qwen3-1.7B` used in v2) |
| Layers / d_model | 26 / 2,304 | 28 / 2,048 |
| Dtype / attention | bf16 / `eager` (Section 13) | bf16 / `sdpa` |
| Prompt format | Plain text (Section 6), BOS first | The same plain text (its tokenizer adds no BOS) |
| SAE | Gemma Scope, trained on `google/gemma-2-2b` | Qwen-Scope, trained on `Qwen/Qwen3-1.7B-Base` |
| Gated | Yes | No |
| Runs | Every test | Only the reference task at 50% depth, for D4 |

**Caveat for the model test (D4).** Each model is used with its own official SAE family, and the two
differ:

- Gemma Scope: JumpReLU, 65k latents, about 28× d_model.
- Qwen-Scope: TopK, 32k latents, 16× d_model.

Both are set to about 100 active features per token. So in the `sae` basis, "model" also includes
"which SAE release". The `pca` and `dims` bases have no such difference. Two things address this:

- D4 is reported separately for the `sae` basis and for `pca` + `dims` (Section 8).
- D6 measures how much SAE width alone matters, on Gemma.

**Access with `HF_TOKEN` (required for Gemma).** The Gemma model repos are gated. The Qwen model and
both SAE repos are public, but use the token for every download anyway.

- Read the token only from the `HF_TOKEN` environment variable (`huggingface_hub` picks it up
  automatically). Never hardcode it, print it, write it to logs, the manifest or the report, or commit it.
- The Hugging Face account behind the token must have accepted the Gemma license on the model page.
- `run_v3.sh` and `--v3 --prefetch` check for the token first. They stop with a clear message if
  `HF_TOKEN` is unset, or if Hugging Face returns 401/403 (wrong token, or license not accepted).
- The user sets it once before starting, for example `export HF_TOKEN=hf_...` in the shell or in `~/.bashrc`.

## 4. Which layers have SAEs

Checked on Hugging Face on 2026-11-25.

**Gemma 2B:**

| Release | Model | Base or chat | Layers (residual stream) | Use in v3 |
| --- | --- | --- | --- | --- |
| `google/gemma-scope-2b-pt-res` (Gemma Scope) | Gemma 2 2B | Base | **Every layer, 0–25.** 16k and 65k widths at every layer; 1M at layers 5, 12, 19 | **Yes** |
| Official chat-model SAEs for Gemma 2 2B | — | Chat | None. Gemma Scope's only chat-model SAEs are for 9B (`google/gemma-scope-9b-it-res`) | — |
| Community SAEs for Gemma 2 2B | Gemma 2 2B | Chat | Single layers (8, 13 or 14). One repo (`weijie210/gemma-scope-2b-it-pt-res`) has a file per layer but no model card | No |
| Gemma Scope 2 (`google/gemma-scope-2-*`) | Gemma 3, 270M–27B | Base and chat | Every layer at 16k and 262k widths; 65k and 1M at four layers | No (another model family) |
| `jbloom/Gemma-2b-*-Residual-Stream-SAEs` | Gemma 1 2B | Base / chat | 0, 6, 10, 12, 17 / only 12 | No: too few depths |

**Qwen3-1.7B:**

| Release | Model | Layers (residual stream) | Use in v3 |
| --- | --- | --- | --- |
| `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100` (Qwen-Scope) | Qwen3-1.7B-Base | **Every layer, 0–27.** 32k, TopK 100 | **Yes** |
| `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (Qwen-Scope) | Qwen3-1.7B-Base | Every layer, 0–27. 32k, TopK 50 | Fallback |
| `adamkarvonen/qwen3-1.7b-saes` (used in v2) | Qwen3-1.7B (post-trained) | Layers 7, 14, 21 | No: trained on the post-trained model |

Qwen-Scope's license restricts use to scientific research, which covers this project.

## 5. Depths and SAE files

**Layer rule** (unchanged): L = round(depth × number of layers), rounding halves up, on the 0-indexed
decoder-layer output (`resid_post`).

**Gemma 2 2B** (all five depths are used by the depth test; 50% is used by everything else):

| Depth | Layer | 65k file (expected) | 16k file (fallback; width test at 50%) |
| --- | --- | --- | --- |
| 25% | 7 | `layer_7/width_65k/average_l0_107` | `layer_7/width_16k/average_l0_137` |
| 40% | 10 | `layer_10/width_65k/average_l0_128` | `layer_10/width_16k/average_l0_166` |
| **50% (reference)** | 13 | `layer_13/width_65k/average_l0_75` | `layer_13/width_16k/average_l0_83` |
| 60% | 16 | `layer_16/width_65k/average_l0_128` | `layer_16/width_16k/average_l0_154` |
| 75% | 20 | `layer_20/width_65k/average_l0_114` | `layer_20/width_16k/average_l0_139` |

**Qwen3-1.7B-Base** (50% only): layer 14, `layer14.sae.pt` from the TopK 100 repo, with the TopK 50
repo as fallback.

**Gemma Scope files:**

- Width 65k, at the sparsity closest to L0 = 100 (Gemma Scope's "canonical" convention; ties go to the lower L0).
- Width 16k, at the sparsity closest to the chosen 65k file's L0. This way the width comparison (D6)
  is not also a sparsity comparison. About 0.3 GB each.
- Each file is `params.npz` with `W_enc`, `W_dec`, `b_enc`, `b_dec` and `threshold` (JumpReLU).
  A 65k file is about 1.2 GB.

**Qwen-Scope files:**

- Each `layer{n}.sae.pt` is a dict with `W_enc` (32768 × 2048), `W_dec` (2048 × 32768), `b_enc` and `b_dec`.
- Encode as the model card does: keep the top k of x·W_encᵀ + b_enc (k from the repo's
  `config.json`), and set the rest to 0. Decode: x̂ = W_dec·f + b_dec.
- About 0.5 GB each.

The harness resolves every file from the repo listing and records it in the manifest.

**Inclusion rule** (preregistered; same bars as v2), per model and depth:

1. Try the main file (Gemma 65k; Qwen TopK 100). It is included if, on the task prompts (first token
   excluded), it explains at least 70% of variance and next-token loss rises by at most 10% when the
   state is replaced by its reconstruction.
2. If it fails, try the fallback (Gemma 16k; Qwen TopK 50) with the same bars.
3. If both fail, that cell runs without the SAE basis, and the report says so.

**For D6**, the Gemma 16k file at 50% must pass the same bars on its own. If the 65k file fails there
and the 16k file stands in as the main SAE, D6 is not computed.

**Verify by measurement, not by documentation** (the v2 lesson):

- **Input scaling:** the smoke test measures FVE with `none` and with `rms_d` at every layer used. It
  writes the better option per layer into `config/v3.yaml` before the freeze. (Both model cards feed
  raw activations.)
- **Layer indexing:** each file for layer N should match the output of decoder block N. Confirm it by
  comparing reconstruction at layers N−1, N and N+1.

## 6. Tasks and prompts

**Prompt format (identical on both models).** Every prompt is a fixed **4-example prefix** followed
by the query:

- Each example is the query template filled in, plus its answer, separated by a blank line. There are
  2 examples per class, drawn once (seeded) from the discovery split and removed from it.
- Examples are kept short. For answerability, use the four shortest passages.
- The prefix is identical for every prompt of a task. It is never intervened on, and never used in
  statistics (role `demo`).
- Because the prefix never changes, its keys and values are computed once per model and reused (Section 11).

**Score and damage (every task):**

- Score = logit(positive answer) − logit(negative answer) at the last position.
- Coherence = the probability on the two answers together.
- Damage = AUROC drop, so every task is measured on the same scale.

| Task | Query template (the answer follows) | Answers (positive first) | Data | Eval size |
| --- | --- | --- | --- | --- |
| Truth (**reference**) | `Statement: {statement}\nTrue or false?\nAnswer:` | ` True` / ` False` | v1 `cities` + `neg_cities` | 500 |
| Sentiment | `Review: {sentence}\nPositive or negative?\nAnswer:` | ` Positive` / ` Negative` | SST-2 | 500 |
| Answerability | `Passage: {context}\nQuestion: {question}\nCan the question be answered from the passage? Yes or no?\nAnswer:` | ` Yes` / ` No` | SQuAD 2.0, contexts ≤ 150 tokens | 400 |
| Toxicity | `Comment: {text}\nToxic? Yes or no?\nAnswer:` | ` Yes` / ` No` | Civil Comments (toxic ≥ 0.5, non-toxic ≤ 0.1), ≤ 150 tokens | 500 |
| IOI | `When {A} and {B} went to the {place}, {S} gave a {object} to` | ` {A}` / ` {B}` | Generated (below) | 500 |

Data sources, revisions and sizes are as in `config/v2.yaml`. The 150-token caps use the longer of the
two tokenizers. Answer tokens are resolved per tokenizer; after `Answer:` they carry a leading space.

**Reference task: truth.** It has been used since v1, its prompts are short (cheap to run at five
depths), and it is expected to pass the gate on both models. If it fails the gate on either model, the
reference becomes the first task in this order that passes on both: IOI, sentiment, toxicity,
answerability. This rule is frozen with the plan.

**IOI (indirect object identification), the replacement for refusal:**

- **What it is:** the model must name who receives the object. In "When Mary and John went to the
  store, John gave a drink to", the answer is " Mary". It is a standard interpretability task on base
  models, and base models of this size do it.
- **Templates:** about 15 sentence frames in this style, written into `config/v3.yaml`, with lists of
  places and objects.
- **Names:**
  - Common first names that are a single token (with a leading space) in both tokenizers.
  - The name list is split into separate sets for the example prefix, discovery, dev and eval, so eval
    names are never seen during selection.
- **Classes:** A is the first-mentioned name and B the second. The repeated name S is B in half the
  prompts (positive class: the answer is A) and A in the other half (the answer is B).
- **Sizes:** 1,000 discovery, 500 eval, 200 dev. Generated locally with a fixed seed; nothing to download.

**Task gate** (unchanged bars): AUROC ≥ 0.85 and coherence ≥ 0.9.

- On Gemma, all five tasks are gated; a failing task is excluded from the task test (D3) and reported.
- On Qwen, only the reference task is gated.
- The label is `INCONCLUSIVE` if no task passes on both models, or fewer than 3 tasks pass on Gemma.

**Dev slices** (disjoint from the eval sets; used only by the Gemma dev run):

- Truth: 200 statements from the v1 eval split, outside the 500-statement eval subset.
- IOI: 200 prompts built from the dev name set.

**Cost prompts.** The same 100 generic prompts as v2, as plain text. The cost is the KL over the next
position plus 8 teacher-forced positions of the clean model's greedy continuation (as in v2). The
intervention covers prompt tokens only.

## 7. What is run, and how methods are compared

**Reference setting:** Gemma 2 2B, truth, 50% depth (layer 13), mean ablation, 65k SAE.

**Methods.** Every cell measures the same **method set**:

- 3 bases (`sae`, `pca`, `dims`) × 3 selection rules (readout: `meandiff`, `l1probe`; causal: `attrib`).
- Plus `random` in each basis as the floor.

Several methods are needed because D3–D5 ask whether the *best* method changes.

**One factor at a time.** Each test varies only its own factor:

| Test | Varies | Levels | Everything else |
| --- | --- | --- | --- |
| D1 | Basis | `sae`, `pca`, `dims` | Reference; the rule is held fixed in each comparison |
| D2 | Selection rule | `meandiff`, `l1probe`, `attrib` | Reference; the basis is held fixed in each comparison |
| D3 | Task | Truth, sentiment, answerability, toxicity, IOI | Reference (Gemma, 50%, mean, 65k) |
| D4 | Model | Gemma 2 2B, Qwen3-1.7B-Base | Reference (truth, 50%, mean) |
| D5 | Depth | 25, 40, 50, 60, 75% | Reference (Gemma, truth, mean, 65k) |
| D6 | SAE width | 65k, 16k | Reference |
| S2 (secondary) | Ablation strategy | Mean, resample | Reference |

D1 and D2 both use the reference cell. In total there are **10 cells**:

- the reference cell;
- 4 more tasks (D3);
- 1 Qwen cell (D4);
- 4 more depths (D5).

D6 adds 16k SAE curves to the reference cell, and S2 repeats the reference cell with resample ablation.

**Inside each cell:**

- 1 discovery resample for selection, 1 random draw, and the budget ladder k = 1, 5, 20, 100, 200,
  500 units (no doubling beyond 500).
- Anchors: massmean + 3 random_dir.
- That gives 12 curves (3 bases × [3 rules + 1 random]), about 76 interventions per cell.
- The 16k SAE (D6) adds 4 curves, about 24 interventions.

**Selection statistics** are as in v2, computed over query tokens:

- A unit's value for an example is its mean over that example's query tokens.
- `attrib` uses S = y · score, where y = ±1 by class.

**The two strategies** (formulas as in v2, Section 8):

- **Mean:** each selected unit is set to its average over discovery query tokens.
- **Resample:** each intervened token copies the selected units' values, jointly, from a random query
  token of a random discovery prompt drawn from both classes. Seeded.
- Both strategies use the same selected units. `random` is re-run under each strategy.

**How methods are compared: fixed unit budgets.**

- E_{b,r}(k) = D_{b,r}(k) − D_{b,random}(k): damage with k units picked by rule r in basis b, minus
  damage with k random units in the same basis.
- The primary number for each method is **Ē**: E averaged over the 6 budgets. The ladder is
  roughly log-spaced, so this is close to an area under the curve on a log scale.
- Collateral cost is reported beside every Ē. Budgets where a method costs more than 1 nat are
  flagged, because the model is broadly broken there.

**Why not fixed cost, as in v2?** Without doubling, most curves stop before they reach a fixed cost.
In v2's data, cut at 500 units:

- Only 25% of method cells had both their own curve and the random curve reach 0.1 nats.
- 69% reached 0.02 nats.

The trade-off: at a fixed unit count, a method can look strong partly because it does more collateral
damage. That is why cost is shown beside every result, and a cost-matched check at 0.02 nats is kept
as a secondary analysis (S5).

**Controls:**

- Cross-task (Gemma, 50%): for every ordered pair of tasks (20 pairs), the top-k attribution units
  (k = 50 and 200) picked for task A in the D3 cells are ablated on task B.
- Readout groups (reference cell): per basis, 5 readout-strength bands × 4 random groups of 100 units.

## 8. Hypotheses (draft; the user reviews them before freezing)

**Notation:**

- A **method** is a (basis, rule) pair: 9 methods per cell (6 if the SAE is excluded).
- Ē is measured with mean ablation.
- Smallest effect of interest δ = 0.05, in AUROC units.
- 95% CIs come from 1,000 bootstrap resamples of eval examples, paired across methods that share examples.
- Benjamini–Hochberg correction is applied across the comparisons within each test.

**Two kinds of evidence:**

- **Reliable difference:** two values differ by at least δ, and the corrected CI of the difference excludes 0.
- **Reliable reversal:** for two methods m1 and m2, m1 beats m2 at one level of a factor and m2 beats
  m1 at another, each by a reliable difference.

| ID | In plain words | What is compared (everything else at the reference) | Holds if |
| --- | --- | --- | --- |
| D1 Basis | Changing the basis changes the effect | The 3 bases, with the rule held fixed | Under at least one rule, two bases differ reliably |
| D2 Rule | Changing the selection rule changes the effect | The 3 rules, with the basis held fixed | In at least one basis, two rules differ reliably |
| D3 Task | The best method on one task is not the best on another | The 9 methods across the 5 tasks | A reliable reversal between two tasks |
| D4 Model | The best method on Gemma is not the best on Qwen | The 9 methods on the two models | A reliable reversal between the models |
| D5 Depth | The best method at one depth is not the best at another | The 9 methods across the 5 depths | A reliable reversal between two depths |
| D6 SAE width | Changing only the SAE's width (16k vs 65k, similar sparsity) changes the effect | The 3 rules in the SAE basis, at both widths | Under at least one rule, the two widths differ reliably |

Each test is supported if it holds.

**Headline label** (printed first in the report):

| Label | Condition |
| --- | --- |
| `SUPPORTED` | D1, D2, D3 and D4 all hold |
| `PARTIALLY SUPPORTED` | One to three of them hold (the report names which) |
| `NOT SUPPORTED` | None holds |
| `INCONCLUSIVE` | No task passes on both models, or fewer than 3 tasks pass on Gemma (Section 6) |

- D5 (depth) and D6 (SAE width) are reported directly under the label as additional factors.
- D4 is also reported separately:
  - for the `sae` basis alone, where the two models' SAE releases differ;
  - for `pca` + `dims` alone, where they don't (Section 3).
- If D6 holds, part of an SAE-only model difference could come from width (65k vs 32k).
- H0 and anchors never change the label.

**Secondary analyses** (reported with CIs; they never change the label):

- **S1 Readout vs causal effect** (v2's C3): Spearman ρ between a group's readout strength and its
  damage, over the readout groups in the reference cell, pooled over bases. v2's bars are reported:
  ρ ≤ 0.3, CI upper bound < 0.5.
- **S2 Strategy:**
  - D1 and D2 recomputed under resample ablation.
  - Kendall τ of the method ranking between mean and resample, in the reference cell.
- **S3 Cross-task specificity:** all task pairs. Also attribution accuracy: summed attribution vs
  measured damage, over the readout groups.
- **S4 Single budgets:** D1–D6 recomputed at single budgets k = 20, 100 and 500 instead of Ē.
- **S5 Cost-matched check:** D1–D6 recomputed with E at a fixed cost of 0.02 nats, where both curves reach it.
- **S6 Size per factor:** how far Ē moves when each factor changes (the largest pairwise change, with
  CI). All tasks use AUROC, so the factors can be compared side by side.
- **S7 Anchors and H0:** massmean and random_dir damage and cost per cell, descriptive.

## 9. Run plan

**One command: `bash run_v3.sh`** (modelled on `run_v2.sh`). It checks `HF_TOKEN`, then runs
prefetch → smoke → dev → freeze → confirm. It stops at the first error and says why. Running it again
resumes where it stopped. Individual steps are also available as
`run_pilot.sh --v3 --prefetch | --smoke | --dev | --freeze | --confirm`. Outputs go to
`runs/v3/<phase>/<run_id>/`, and the report is `report_v3.md`. Freezing tags `prereg-v3-frozen`.

**Dev run:**

- Gemma only: truth and IOI at 50%, on the dev slices.
- Its report shows harness health only: gates, coherence, cost ranges, anchors, CI widths and flags.
- It **withholds Ē and the D-test results**, so every task stays confirmatory.

**Confirmatory blocks, in priority order:**

| Block | What | Feeds |
| --- | --- | --- |
| Prep | Task gates, SAE checks, caching, PCA, attribution, L1 paths | — |
| B1 | Reference cell, with anchors | D1, D2 (and the reference level of D3–D6) |
| B2 | Depth cells: 25, 40, 60, 75% | D5 |
| B3 | Task cells: sentiment, answerability, toxicity, IOI | D3 |
| B4 | Model cell: Qwen3-1.7B-Base | D4 |
| B5 | SAE width: the 16k SAE in the reference cell | D6 |
| B6 | Resample ablation in the reference cell | S2 |
| B7 | Controls: cross-task, readout groups | S1, S3 |

**Time guard:**

- The budget is 4 GPU hours for dev + confirm, frozen with the plan.
- Before each block, `--confirm` projects the time. If the block would not fit, it stops there and
  analyses what is done. Skipping depends only on speed, never on results.
- Any change to the budget after the freeze, including through an environment variable, is written to
  `prereg/deviations.md` and shown at the top of the report. v2 missed this.

## 10. Run-time estimate on the GB10

**Basis:**

- v2 measured the time per intervention on the GB10 for each task on Qwen3-1.7B, at full eval sizes
  (each time includes the 100-prompt cost pass).
- Qwen3-1.7B-Base has the same architecture, so those times apply to it directly.
- Gemma 2 2B does about 1.45× the compute per token (2.0B vs 1.4B non-embedding parameters) and needs
  eager attention. So the estimate uses **1.5×**, with a plausible range of 1.3–2.0×.
- Attending to the cached example prefix adds about 10% on both models.
- IOI prompts are about as short as truth prompts.
- The smoke test measures the real figures before the freeze.

| Task (eval size) | Qwen3-1.7B-Base (v2 measured → v3 estimate) | Gemma 2 2B, estimated |
| --- | --- | --- |
| Truth (500) | 2.2 s → ~2.4 s | ~3.6 s |
| Sentiment (500) | 3.8 s → ~4.2 s | ~6.3 s |
| Answerability (400) | 6.4 s → ~7.0 s | ~10.6 s |
| Toxicity (500) | not run → ~5.3 s | ~7.9 s |
| IOI (500) | new → ~2.0 s | ~3.0 s |

| Step | Interventions | Time (Gemma at 1.5×) |
| --- | --- | --- |
| Prefetch (~23 GB: Gemma ~10.5 GB (stored in fp32), 10 Gemma SAE files ~7.5 GB, Qwen ~3.4 GB, 2 Qwen SAE files ~1.1 GB; IOI generated locally) | — | ~5 min |
| Smoke test (both models) | — | ~20 min |
| Dev run (Gemma, truth + IOI, dev slices) | ~150 | ~0.15 h |
| Freeze | — | < 1 min |
| Confirm: prep | — | ~0.35 h |
| B1: reference cell | ~76 | ~0.1 h |
| B2: 4 depth cells | ~300 | ~0.3 h |
| B3: 4 task cells | ~300 | ~0.6 h |
| B4: Qwen cell | ~76 | ~0.05 h |
| B5: 16k SAE | ~24 | ~0.03 h |
| B6: resample | ~76 | ~0.1 h |
| B7: controls | ~180 | ~0.3 h |
| Final analysis | — | ~0.2 h |
| **Total** | **~1,040** (confirm) | **~2.5 h** |

**Range:** about 2.2 h if Gemma is 1.3× Qwen3-1.7B, and about 3.2 h if it is 2.0×. Everything fits
inside the 4-hour guard in both cases. All six D-tests are done after about 2 h.

**What this design gives up:**

- Each test rests on one setting. It can show that a factor *can* change the result, but not how
  often or how widely that happens.
- No interactions, for example whether depth matters more on some tasks than others.
- Coarser curves (6 budgets) and a noisier random baseline (1 draw), so the CIs are wider than v2's.

## 11. Implementation notes for Claude Code

- **Extend, don't fork.** Reuse the v2 engine. v1 and v2 modes and outputs stay reproducible and untouched.
- **New files:**
  - `config/v3.yaml` (including the reference setting, the cell list, IOI templates, name sets, places and objects)
  - `prereg/v3/hypotheses.yaml` and `prereg/v3/PREREGISTRATION.md` (drafts for the user)
  - `run_v3.sh`
  - a v3 section in `README.md` and in `CHANGES.md`
- **Cells:** build the cell list from the reference setting plus one varied factor per test (Section 7).
  Do not run a full grid.
- **Prompts:**
  - Plain-text builder with the 4-example prefix and token roles `demo` / `query`. No chat template on either model.
  - IOI generator: seeded, balanced, with disjoint name sets and a single-token check in both tokenizers.
- **Cached example prefix.**
  - Compute the prefix's keys and values once per (model, task), clean, and reuse them for every batch,
    including inside the fast path, which reuses layers 0..L.
  - Without this, the prefix would make each intervention about 2–3× slower.
- **Grid:**
  - Turn curve extension off (no doubling past the ladder).
  - Remove `attrib_shuffled`, site S2 and everything that generates replies (refusal labels, generated-role values).
  - In the reference cell, add the 16k SAE as an extra basis (mean ablation only) for D6.
- **Model loading (`models.py`):** make `attn_implementation` a per-model setting: `eager` for Gemma
  2, `sdpa` for Qwen.
- **SAE format `gemmascope` (`sae_v2.py`):**
  - Load `params.npz` and apply JumpReLU: pre = x·W_enc + b_enc; f = pre · (pre > threshold).
  - Check that the SAE was trained on `google/gemma-2-2b`.
  - Resolve the 65k file with L0 closest to 100, and the 16k file with L0 closest to the 65k file's.
- **SAE format `qwenscope` (`sae_v2.py`):**
  - Load `layer{n}.sae.pt`. Encode with TopK(k) of x·W_encᵀ + b_enc, with k from `config.json`.
    Decode with W_dec·f + b_dec.
  - Check `base_model: Qwen/Qwen3-1.7B-Base`.
  - Add the TopK 50 fallback.
- **Prefetch:**
  - The `HF_TOKEN` check, with fail-fast messages for 401/403.
  - Both models.
  - Gemma's 65k and 16k files at the five depth-test layers.
  - Qwen's main and fallback files at layer 14.
- **Analysis:**
  - Ē (fixed-budget average).
  - The D1–D6 tests, with the reliable-difference and reliable-reversal helpers.
  - The headline label.
  - D4 also on `sae` alone and on `pca` + `dims` alone.
  - S1–S7.
  - The dev report withholds Ē and the D-tests.
  - Budget overrides are logged as deviations.
- **Tests:**
  - Load `gemmascope` and `qwenscope` from tiny synthetic files.
  - The IOI generator gives balanced classes, disjoint name sets and single-token names.
  - Scores with the cached prefix equal scores from the full prompt.
  - The cell list varies exactly one factor per test.
  - D1–D6 and the label on synthetic results, including a case built to fail each test.
  - The cross-task summary is non-empty.
  - Freeze enforcement with the v3 prereg paths.
- **No hardcoded layers or paths.** Resolve them from the config and the repo listing, and record them in the manifest.

## 12. Smoke test (v3 additions)

`--v3 --smoke` passes only if all of these hold:

- `HF_TOKEN` is set. The gated Gemma model downloads and loads (26 layers, d = 2,304, eager
  attention, bf16), and so does Qwen3-1.7B-Base (28 layers, d = 2,048).
- SAE fidelity is measured at every layer used (five on Gemma, one on Qwen), for the main and
  fallback files and both input-scaling options, and the chosen setting is written to `config/v3.yaml`.
- The layer-indexing check (N−1, N, N+1) picks N at every layer used.
- Scores with the cached example prefix match scores from the full prompt, on both models (tolerance 1e-2).
- The identity check passes: setting units to their own values changes nothing.
- A no-op intervention has zero cost.
- No curve goes past 500 units.
- Resample ablation runs in the reference cell.
- Coherence and AUROC are reported for every gated task, so a failing gate shows up before the freeze.
- Batch size is auto-tuned under the 100 GB cap.
- Throughput is measured per model and task. Projected block times are written against the 4-hour
  budget, listing any blocks that would not fit.
- `--confirm` refuses without a valid v3 freeze.

## 13. Pitfalls

**Gemma 2:**

- **Gated weights.** Use `HF_TOKEN` from the environment only. The license must be accepted on the account.
- **Use the base model.** The id is `google/gemma-2-2b`, not `google/gemma-2-2b-it`.
- **Attention must be `eager`.** Gemma 2 caps its attention scores (soft-capping). Some `sdpa` paths
  skip this silently and give wrong outputs.
- **bf16, never fp16.** Gemma 2 overflows in fp16.
- **BOS is required and has a huge activation norm.** Always prepend it. It falls inside the example
  prefix, so it is never intervened on, and it is excluded from SAE fidelity statistics.
- **Final-logit soft-capping (at 30)** compresses logit differences. Scores use the model's actual
  logits, and AUROC is unaffected.

**Qwen3-1.7B-Base:**

- **Use the base model** (`Qwen/Qwen3-1.7B-Base`). v2's `Qwen/Qwen3-1.7B` is the post-trained
  model, and Qwen-Scope was not trained on it.
- **No BOS token.** The first token still falls inside the example prefix, so it is never intervened on.

**Both models:**

- **Answer tokens carry a leading space** (" True", " Mary"). Resolve them per tokenizer, and check
  that IOI names are single tokens in both.
- **Base-model formatting.** Without the example prefix, a base model spreads probability over
  formatting tokens and fails the coherence gate. Do not drop the prefix to save time.

## 14. Out of scope

- A full grid, and interactions between factors. Each test varies one factor at a time around the reference setting.
- Refusal, and any task that needs a chat model.
- C1 and C2 from v2, and the scrambled-label control (`attrib_shuffled`).
- Cost-matched comparison as the primary measure. Without doubling it is not possible for most cells;
  it is kept as a secondary check (S5).
- The 90% depth.
- More than two models, for example Gemma 3 with Gemma Scope 2, or Qwen3.5 with Qwen-Scope.
