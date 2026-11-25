# Feature-ablation pilot v3: does feature causality depend on basis, selection rule, task and model?

Status: **draft** · drafted 2026-11-25

Generated from `prereg/v3/hypotheses.yaml` by `bash run_pilot.sh --v3 --render-prereg`; do not edit by hand.

## 1. Headline

Feature causality (how much switching off selected units damages a behaviour, beyond switching off random units in the same basis) depends on the basis, the selection rule, the task and the model. Depth and SAE width are tested beside the headline.

## 2. Design

- Reference setting: google/gemma-2-2b, task truth, 50% depth, mean ablation, Gemma Scope 65k SAE. Each test varies only its own factor; everything else stays at the reference.
- Reference task fallback (if it fails a gate on either model): truth → ioi → sentiment → toxicity → answerability.
- Models: google/gemma-2-2b (base), Qwen/Qwen3-1.7B-Base (model test only). Bases: sae, pca, dims. Rules: meandiff, l1probe, attrib, random (random is the floor).
- Budgets: 1, 5, 20, 100, 200, 500 units; no doubling.
- Measure: Ē (damage beyond random, averaged over the budget ladder). E(k) = D_rule(k) - D_random(k) in the same cell and basis at the same number of units k; Ē is the mean of E over the budgets present in both curves. Collateral cost is reported beside every Ē; budgets costing more than `broken_cost` nats are flagged.
- Damage: AUROC drop of logit(positive answer) - logit(negative answer) on the eval split. Cost: mean KL(clean || ablated) over the next position + 8 teacher-forced positions of 100 generic prompts.
- δ = 0.05; α = 0.05; 1000 bootstrap resamples. A difference is reliable if |point| >= delta, its 95% percentile bootstrap CI excludes 0, and its Benjamini-Hochberg q-value within the test is <= alpha. A reversal is a method pair whose difference is reliably positive at one level of the factor and reliably negative at another.

## 3. Inclusion rules (mechanical)

- task_gate: clean AUROC >= 0.85 and coherence >= 0.9 on the eval split; a failing task is excluded (no substitute)
- reference_task: the reference task must pass on both models; otherwise the first task in reference_task_order that does
- sae: FVE >= 0.70 and next-token loss increase <= 10% per (model, layer); main file first (Gemma 65k at L0 closest to 100; Qwen-Scope TopK 100), then the fallback (Gemma 16k at the L0 closest to the 65k file's; Qwen-Scope TopK 50); otherwise the layer runs without the SAE basis. Input scaling (none or rms_d) as measured by the smoke test (config/v3_sae_input_norm.yaml).
- inconclusive: no task passes on both models, or fewer than 3 tasks pass on Gemma

## 4. Dependence tests

| ID | Claim | Compared | Holds if |
|---|---|---|---|
| D1 | Changing the basis changes the effect | the 3 bases in the reference cell, with the rule held fixed (3 rules x 3 pairs) | under at least one rule, two bases differ reliably |
| D2 | Changing the selection rule changes the effect | the 3 rules in the reference cell, with the basis held fixed | in at least one basis, two rules differ reliably |
| D3 | The best method on one task is not the best on another | the 9 methods on the 5 tasks (Gemma, 50% depth) | a reliable reversal of a method pair between two tasks |
| D4 | The best method on Gemma is not the best on Qwen | the 9 methods on the two models (reference task, 50% depth); also reported for the SAE basis alone and for PCA + raw dims alone | a reliable reversal of a method pair between the models |
| D5 | The best method at one depth is not the best at another | the 9 methods at 25, 40, 50, 60 and 75% depth (Gemma, reference task) | a reliable reversal of a method pair between two depths |
| D6 | Changing only the SAE's width changes the effect | Gemma Scope 65k vs 16k at matched sparsity, each rule, reference cell | under at least one rule, the two widths differ reliably |

## 5. Headline label

- `SUPPORTED`: D1, D2, D3 and D4 all hold
- `PARTIALLY SUPPORTED`: one to three of D1-D4 hold (the report names which)
- `NOT SUPPORTED`: none of D1-D4 holds
- `INCONCLUSIVE`: per the inclusion rules
- `note`: D5 and D6 are reported directly under the label. Anchors and H0 never change the label.

## 6. Secondary analyses (never change the label)

- S1: readout strength vs damage over the readout groups (reference cell), Spearman rho pooled over bases (v2 bars reported)
- S2: D1 and D2 under resample ablation; Kendall tau of the method ranking between mean and resample (reference cell)
- S3: cross-task specificity over all task pairs at k = 50 and 200; attribution accuracy over the readout groups
- S4: D1-D6 recomputed at single budgets k = 20, 100 and 500 instead of Ē
- S5: D1-D6 recomputed with E at a fixed cost of 0.02 nats, where both curves reach it
- S6: size per factor: the largest change each factor causes
- S7: anchors (massmean, random_dir): damage and cost per cell

## 7. Run plan (frozen)

- Priority order: B1_reference → B2_depth → B3_task → B4_model → B5_sae_width → B6_resample → B7_controls.
- Dropped before the run: none.
- Time guard: 4 GPU hours for dev + confirm (config/v3.yaml budget.total_hours); before each block the projected time is added to the elapsed time and the run stops if it would exceed the budget (speed only). Any budget override after the freeze needs --deviation and is listed at the top of the report..

## 8. Known limitations

- Each test rests on one setting: it shows that a factor can change the result, not how widely that happens.
- No interactions between factors (one factor at a time).
- Comparisons use equal unit counts; a method can look strong partly by doing more collateral damage (cost is reported; S5 checks a fixed cost).
- The two models come with different SAE releases (JumpReLU 65k vs TopK 32k), so the SAE-basis part of D4 mixes model and SAE release; D6 measures width alone on Gemma.
- The bootstrap covers which eval examples were used, not which discovery examples picked the units (one selection per cell).
- Qwen3-1.7B-Base runs only the reference task at 50% depth.

## 9. Frozen files

SHA-256 hashes are recorded in `prereg/v3/FROZEN.json` at the commit tagged `prereg-v3-frozen`:

- `prereg/v3/hypotheses.yaml`
- `prereg/v3/PREREGISTRATION.md`
- `config/v3.yaml`
- `config/v3_sae_input_norm.yaml`
- `src/pilot/analysis_v3.py`
- `src/pilot/analysis_v2.py`
- `src/pilot/stats.py`
