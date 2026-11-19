# The feature-ablation pilot, explained

This document explains, in plain language, what the pilot experiment is for, how it was designed,
and what the first full run found. No prior knowledge of the project is assumed. The technical
plan is in `pilot-implementation-plan.md`, and the raw run output is in
`20261116-180218-pilot-qwen3-1.7b/` (start with its `report.md`).

## The short version

- **The question.** When researchers try to find the parts of a language model responsible for a
  behaviour, does it matter more *which kind of parts* they look at, or *how they pick* the parts?
- **What we did.** We switched off parts of a small open model (Qwen3-1.7B) in 670 different ways for
  each of two behaviours: judging whether a statement is true, and refusing harmful requests. For
  each way we measured how much it broke the behaviour and how much collateral damage it caused (unwanted side effects - how much its prediction for the first word of each reply changes after the intervention - KL divergence).
- **The official verdict is INVALID HARNESS.** The experiment includes built-in sanity checks, and two
  of them failed, so by the rules we set in advance the main results must not be interpreted yet.
  The software worked correctly; the failing checks point to design choices that need revisiting.
- **What failed.**
  - **Truth:** our reference intervention for the truth behaviour did nothing at all.
  - **Refusal:** the reference intervention cut refusals a lot, but slightly less than the bar we set,
    and switching off *random* parts also did more damage than the rules allow.
- **What we saw anyway (unofficial).** These patterns are suggestive only; they guide the redesign
  and are not findings yet.
  - **How you pick the parts matters a great deal.** To break refusal, we chose which parts of the
    model to switch off in two ways:
    - **By correlation:** pick the parts whose values differ most between harmful and harmless requests.
    - **By predicted effect:** estimate how much refusal would drop if each part were switched off,
      and pick the parts with the biggest predicted drop.

    Allowed the same amount of collateral damage, picking by effect cut the refusal rate by 70
    percentage points; picking by correlation cut it by only 35–39. Same model, same kind of parts,
    same number of side effects: only the picking method changed, and the result roughly doubled.
    (Section 2 walks through this example.)
  - **Which kind of part looks "best" depends on how you pick.**
    - Picking by correlation, parts defined as PCA axes did better than the model's raw numbers
      (45–48 vs. 35–39 points).
    - Picking by effect, the order flipped: raw numbers did better than PCA axes (70 vs. 52 points).

    So a claim like "kind of part A beats kind of part B" is not a fixed fact; it depends on the
    picking method. This is the "interaction" outcome the pilot was designed to detect.
  - **The two behaviours disagree.** The methods that worked best for refusal were not the best for
    the truth behaviour, where almost nothing worked at all.

---

## 1. Why this pilot exists

### The background, in plain terms

A language model processes text one token (roughly, a word piece) at a time. At every layer it keeps
a list of numbers for each token, its internal "state". In this model the list has 2,048 numbers. In
the field this list is called the **residual stream**.

Interpretability research tries to find which pieces of that internal state cause a given behaviour.
A standard test is **ablation**: you switch off, or neutralise, some pieces and see whether the
behaviour breaks. If breaking a small set of pieces breaks the behaviour, those pieces matter for it.

Two choices have to be made before any ablation:

1. **What counts as a "piece"** (we call these *units*). You can use the raw 2,048 numbers directly,
   re-express them along new axes, or use a learned dictionary of "features". We call this choice the
   **basis**.
2. **How you choose which pieces to switch off.** We call this the **selection rule**.

Papers often compare bases, for example "SAE features vs. raw neurons". They usually fix one
selection rule while doing so, so it is unclear whether the winner is the better basis or just
better suited to that rule.

### The question the pilot answers

The pilot tests every combination of 3 bases × 3 selection rules, under 2 ways of "switching off",
on 2 behaviours. It is small and cheap (about 1.5 hours on one machine). Its purpose is to decide
which of three papers the full study should become:

| If the pilot shows... | ...the full study becomes |
|---|---|
| **Outcome 1, interaction.** Which basis wins depends on the selection rule or the way of switching off | "Method rankings depend on the method." Scale up to more behaviours and a second model. |
| **Outcome 2, selection beats basis.** How you pick units matters much more than which basis you use | "Selection, not basis." The paper centres on readout vs. causal selection. |
| **Outcome 3, uniform.** No reliable differences anywhere | A confirmation result, suitable for a workshop paper. |

There is also a fourth possibility, **INVALID HARNESS**. It means the experiment's own sanity
checks failed, so none of the above can be concluded yet. This run landed here.

---

## 2. How the experiment is built

### The model

We used **Qwen3-1.7B**, a small open chat model (28 layers, 1.7 billion parameters).

A quick pre-check, called "Step 0", tested whether it is good enough at the truth task: it must
answer "True or False" about statements like "The city of Lodz is in Poland." reliably. It passed
all three bars. Had it failed, the plan would have switched to a larger model (Qwen3.5-4B).

### Where we intervene

- **Layer:** layer 14, the middle of the model. A public, pre-trained dictionary of features (the
  SAE, below) exists for exactly this layer.
- **Tokens:** only the 9 fixed formatting tokens that the chat format inserts between the user's
  message and the model's reply. These mark "the user is done, now the assistant speaks". The model's
  first word of the answer is predicted at the last of them.

**What those 9 tokens are.** A message to the model is not fed in as plain text. It is wrapped in a
fixed chat format. This is exactly what the model sees for one truth prompt:

```
<|im_start|>user
Answer with True or False only.
Statement: The city of Lodz is in Poland.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The model reads text as **tokens**, roughly word pieces. Everything after "Poland." is 9 fixed
tokens added by the chat format:

| # | Token | What it means |
|---|---|---|
| 1 | `<\|im_end\|>` | "The user's message ends here" |
| 2 | newline | |
| 3 | `<\|im_start\|>` | "A new turn starts" |
| 4 | `assistant` | "...and it's the assistant's turn" |
| 5 | newline | |
| 6–9 | `<think>`, blank lines, `</think>`, blank lines | An empty "thinking" block. This model can think before answering; we turn that off, which leaves this empty placeholder. |

**Why the answer is read at the last one.** A language model works by predicting the next token at
each position. At the 9th token, the last blank line, its prediction *is* the first word of its
reply: "True" or "False" here, or "I" (as in "I can't help with that") for refusals. All our
scores are read at that point.

**Why we intervene only on these 9 tokens:**

- They are identical in every prompt, so "position 3" means the same thing everywhere. That makes it
  easy to compute a typical value for each position and to compare prompts.
- Earlier research found that a model's decision to refuse is largely made on these tokens after
  the user's message.
- We never edit the user's own words.

**The downside, which showed up in the results.** The facts that decide truth ("Lodz", "Poland")
sit on the user's own words, which we never touch. Later layers can look back at those words and
work the answer out again. This is the most likely reason our interventions barely affected the
truth behaviour (see Results).

### The two behaviours (tasks)

| Task | What the model sees | How we measure the behaviour |
|---|---|---|
| **Truth** | "Answer with True or False only. Statement: The city of X is in Y." (about 3,000 statements, half of them negated: "...is not in Y") | How well the model's preference for "True" over "False" separates true from false statements. The score is AUROC: 1.0 is perfect, 0.5 is guessing. The untouched model scores **0.89**. |
| **Refusal** | Harmful requests (e.g. asking for dangerous instructions) and ordinary harmless requests | How often the model refuses. Untouched, it refused **212 of 500** harmful requests (42%) and almost no harmless ones (0.6%). Damage is measured on those 212 refused prompts. |

A third set of 300 **ordinary prompts** (everyday instructions) is used to measure collateral damage.

Data is split so that units are *chosen* on one set of prompts (the "discovery" set) and *tested*
on a separate set (the "eval" set). For truth, a city never appears in both sets.

### What we vary

Think of the model's internal state as a panel of wires. The experiment varies which wires exist,
how we choose which to cut, what we put in their place, and how many we cut.

**1. Basis: what the "wires" are**

| Basis | Plain description |
|---|---|
| `dims` | The model's own 2,048 numbers, as they are. |
| `pca` | The same information re-expressed along 2,048 new axes, ordered from "varies the most" to "varies the least" across ordinary text. |
| `sae` | A separately trained dictionary of 65,536 "features" (a *sparse autoencoder*, SAE). Each feature is meant to capture one recognisable concept, and only a few dozen are active at a time. This is a public one trained by another researcher for this model and layer. |

**2. Selection rule: how we pick which units to switch off**

| Rule | Plain description | Type |
|---|---|---|
| `meandiff` | Pick the units whose values differ most between the two groups (true vs. false, harmful vs. harmless). | "Readout": finds units that *carry information* |
| `l1probe` | Train a simple classifier that is forced to rely on as few units as possible, and pick the units it reaches for first. | "Readout" |
| `attrib` | Use the model's gradients to *predict* how much switching off each unit would hurt the behaviour, and pick the most damaging. | "Causal": finds units that *drive the behaviour* |
| `random` | Pick units at random. This is the floor: any real method should beat it. | Baseline |

The main contrast is readout vs. causal. A unit can carry information about a behaviour without
being what the model uses to produce it.

**A worked example: picking 50 units to break refusal.** Suppose we want to switch off 50 of the
model's 2,048 raw numbers to stop it refusing harmful requests. We have a few hundred harmful and
harmless example requests to learn from.

- **Readout picking (`meandiff`).** Run all the examples through the model and record each of the
  2,048 numbers. Suppose number #731 averages 4.0 on harmful requests and 0.5 on harmless ones, while
  number #12 is about 1.0 on both. Number #731 clearly "knows" whether a request is harmful, so it
  ranks high and #12 ranks low. Take the 50 numbers with the biggest harmful-vs-harmless difference.
- **Causal picking (`attrib`).** For each example, ask the model (using its gradients, a cheap
  mathematical shortcut) how much its tendency to refuse would change if each number were set to its
  typical value. Suppose number #731 would barely change the refusal, because the model doesn't
  actually use it to decide, but number #1502 would cut it sharply. Then #1502 ranks high even if
  it differs only modestly between harmful and harmless requests. Take the 50 numbers with the
  biggest predicted drop.

An everyday analogy: to stop a fire alarm, readout picking cuts the wires that are hottest when
the alarm rings, while causal picking cuts the wires that actually carry the signal to the bell. The
hottest wires are often near the action without being the ones that matter.

**3. Strategy: what we put in place of a switched-off unit**

- **Mean ablation:** replace the unit's value with its average value over many prompts. This
  removes the prompt-specific signal while keeping a "typical" value.
- **Resample ablation:** replace the unit's value with the value it had on a different, randomly
  chosen prompt.

**4. Budget: how many units**

We switch off 1, 2, 5, 10, 20, 50, 100, 200 or 500 units at a time.

**Totals.** Each selection was repeated on 3 random subsets of the discovery data, to see whether
the same units get picked. That makes 670 different interventions per task.

### The reference interventions ("anchors")

To judge whether a method did "a lot" or "a little", we compare it with two references:

- **Mass-mean anchor (positive control).** A single direction in the model's state: the difference
  between the average state for one group and for the other (for example harmful minus harmless).
  We remove the model's position along that one direction. Earlier research found that this kind of
  removal strongly suppresses refusal, so it should clearly work there.
- **Random directions (negative control).** The same operation along 10 random directions, which
  should do nothing.

### How a method is scored: damage vs. cost

Breaking a behaviour is easy if you are allowed to wreck the whole model. So every intervention is
scored on two axes:

- **Damage:** how much the target behaviour got worse. For truth, the drop in AUROC; for refusal, the
  drop in refusal rate.
- **Cost:** how much the model's predictions on the 300 ordinary prompts changed. This is measured
  as KL divergence. In plain terms, 0 means "no change at all", about 0.01 is barely noticeable, and
  about 0.5 or more means ordinary predictions moved noticeably.

Each method traces a curve as the budget grows from 1 to 500 units: more units, more damage, more
cost. Methods are compared **at the same cost**, namely the cost of the mass-mean anchor (called
**c\***). The number reported for each method is its damage at that cost, called **D@c\***.

From this we also compute **gap closure (G)** for each method:

- G = 0 means no better than picking units at random in the same basis.
- G = 1 means as damaging as the mass-mean anchor.
- G above 1 means even more damaging than the anchor at the same cost.

Every number comes with a 95% confidence range, estimated by re-running the analysis on 1,000
resampled versions of the test data. Because many comparisons are made at once, a correction for
multiple testing is applied.

---

## 3. The hypotheses

All of these were written down before the run, together with the rules for turning them into a verdict.

| ID | In plain words | What would count as "yes" |
|---|---|---|
| **H0** | **The experiment itself works.** The positive control clearly breaks refusal, and random selection does little. | The mass-mean anchor cuts the refusal rate (in actual generated replies) by at least 0.5, **and** random selection does at most 25% of the anchor's damage, in every basis. |
| H1 | The bases give different results. | For at least one rule, two bases differ reliably. |
| H2 | Causal selection beats readout selection. | `attrib` beats the better of `meandiff`/`l1probe` in **all three** bases. |
| H3 | The choice of rule matters more than the choice of basis. | Results spread more across rules than across bases, and `attrib` gets near the anchor (G ≥ 0.7) while readout rules stay far from it (G ≤ 0.4) in every basis. |
| H4 | Basis and rule interact. | Two bases swap places depending on the rule (basis A beats B under one rule, B beats A under another), reliably. |
| H5 | Basis/rule and switching-off strategy interact. | Two methods swap places between mean and resample ablation. |
| H6 | The two behaviours agree. | The ranking of the 9 basis × rule combinations is similar for truth and refusal (rank correlation ≥ 0.6). |

**Decision rules:**

1. If H0 fails, the verdict is **INVALID HARNESS**: stop and fix the experiment.
2. If H4 or H5 holds, the verdict is **Outcome 1, interaction**.
3. If H2 and H3 hold without H4, the verdict is **Outcome 2, selection beats basis**.
4. If none of H1–H5 holds, the verdict is **Outcome 3, uniform**.
5. Anything else is **MIXED**.

---

## 4. Results

### The verdict: INVALID HARNESS (both behaviours)

H0, the "does the experiment itself work" check, failed for both tasks. By our own rules we cannot
draw the main conclusions from this run. This is not a software bug. The model, data and
measurements all behaved correctly (see "What worked" below). The failure says that some design
choices do not give us a usable yardstick yet.

### Why H0 failed for truth: the reference did nothing

Removing the mass-mean "truth direction" did not hurt the model's truth judgments at all. The
AUROC went from 0.892 to 0.896, a tiny *improvement*. Only 0.3% of answers flipped. Since the
reference does no damage, "random does at most 25% of the reference's damage" cannot be satisfied,
and gap-closure scores for truth are meaningless (values like −6.6).

The reference also set the comparison cost c\* very low (0.005, barely noticeable), so almost every
method was compared at a cost where nothing happens:

![Truth: damage vs cost, mean ablation](20261116-180218-pilot-qwen3-1.7b/plots/curves_truth_mean.png)

*How to read this.* There is one panel per basis. The horizontal axis is cost (log scale) and the
vertical axis is damage. Each coloured line is a selection rule and grey is random. The black star
is the mass-mean anchor, and the dotted line marks its cost c\*. Almost every line stays flat at
zero. Only the attribution rule (green) does real damage, and only at much higher cost: up to a 0.26
AUROC drop when switching off 500 raw dimensions, at about 100 times the anchor's cost.

**The most likely reason** is the place we intervene. We only edit the 9 formatting tokens after
the statement. The facts that decide truth (which city, which country) live on the statement's own
words, which we never touch. Later layers can look back at those words and recompute the answer.
A single "truth direction" read off the last token appears to be a *summary* the model can rebuild,
not the place where the judgment is made.

### Why H0 failed for refusal: close, but under the bar, and random did too much

The refusal reference **does** work. In actual generated replies to the 200 harmful prompts the
model normally refuses, removing the "refusal direction" lowered the refusal rate from 98.5% to
56% (mean ablation) and to 59.5% (resample ablation). Random directions did nothing: 97–97.5%. But
the pre-set bar was a drop of at least 0.5, and the measured drops were **0.43 and 0.39**.

Second, **random selection did more than 25% of the reference's damage** in 5 of the 6 basis ×
strategy combinations (28% to 48%). Only SAE with mean ablation passed (12%). Two things drive this:

- **The reference is "expensive".** Its cost is 0.51 under mean ablation and 1.59 under resample,
  both clearly noticeable changes to ordinary predictions. With that much disruption allowed, even
  switching off hundreds of random units lowers refusal by 13–21 percentage points. In the PCA and
  raw-dimension bases, random selection never even reached that cost with 500 units; its maximum
  damage was used instead.
- **The refusal signal is spread across many units.** The difference between harmful and harmless
  requests is not confined to a few units, so switching off hundreds of random ones catches some of
  it under either strategy. Random selection looks worse under resample ablation mainly because that
  reference is about three times as expensive, which gives random picks three times the room.
  Compared at the same cost, random selection does about the same damage under both strategies for
  PCA axes and raw dimensions. For SAE features it does slightly more under resample, but the
  amounts stay small (for example 0.04 vs. 0.02 at a cost of 0.2).

![Refusal: damage vs cost, resample ablation](20261116-180218-pilot-qwen3-1.7b/plots/curves_refusal_resample.png)

*How to read this.* Under resample ablation the reference (black star) sits far to the right, at a
cost of about 1.6. By that point the grey random line has already climbed to about 0.2 of damage in
every basis. In the SAE panel, random selection with 500 units eventually catches up with the
selection rules, though only at costs beyond the reference. Because the reference allows so much
collateral damage, even random picks break refusal noticeably at that level. That makes the
"random does at most 25% of the reference's damage" check hard to pass, and it narrows the gap
between random picking and good methods.

### What worked

The parts of the experiment that check the machinery all passed:

- The Step 0 truth check passed, so the small model was the right choice.
- The SAE reconstructs the model's state well at layer 14: it explains 86% of the variation, and the
  model's predictions are not degraded when the state is replaced by the SAE's reconstruction.
- The refusal behaviour matches the published literature. Other researchers report that this model
  refuses about 42% of these harmful prompts, which is exactly what we measured.
- The fast "first word of the reply" refusal detector agreed with full generated replies 97.5% of the
  time.
- The plumbing checks passed:
  - Switching a unit "off" to its own current value changes nothing (exactly zero difference).
  - The speed-up trick of reusing the first half of the model matches a full run exactly.
- The whole run took about 1 hour 35 minutes on the GB10 machine, far less than the 4–8 hours
  originally estimated.

### What we saw anyway (unofficial, because H0 failed)

These patterns are suggestive only. They should guide the redesign, not be reported as findings.

**For refusal, every method can do real damage, and the picking rule matters a lot.**

![Refusal: damage vs cost, mean ablation](20261116-180218-pilot-qwen3-1.7b/plots/curves_refusal_mean.png)

Damage at the reference cost, refusal task, mean ablation (the reference itself scores 0.47):

| Basis | meandiff | l1probe | attrib (causal) | random |
|---|---|---|---|---|
| SAE features | 0.26 | 0.18 | **0.42** | 0.06 |
| PCA axes | 0.45 | 0.48 | **0.52** | 0.14 |
| Raw dims | 0.35 | 0.39 | **0.70** | 0.13 |

The same results as gap closure (0 = random level, 1 = reference level):

![Refusal: gap closure](20261116-180218-pilot-qwen3-1.7b/plots/gap_closure_refusal.png)

In plain terms:

- **The causal rule (`attrib`) is the best rule in every basis**, and by a wide margin with raw
  dimensions (0.70 vs. 0.35–0.39). Units that are merely *different* between harmful and harmless
  prompts are weaker levers than units predicted to *drive* the behaviour.
- **SAE features chosen by readout rules are the weakest non-random option** (G of 0.31–0.50). The
  "interpretable features" that best distinguish harmful from harmless are not the ones that most
  control refusal.
- **In the PCA basis, the rule hardly matters**: all three land near the reference (G ≈ 1).
- **For truth**, the readout rules did essentially nothing in every basis. Only `attrib` produced any
  damage (0.033 with SAE features at the reference cost).

**Hypothesis by hypothesis**, as if H0 had passed:

| Hypothesis | Result | Plain meaning |
|---|---|---|
| H1, bases differ | **Yes**, both tasks | E.g. for refusal, with the `l1probe` rule, PCA beats SAE by 0.29. |
| H2, causal beats readout in every basis | **No**, both tasks | True for SAE and raw dims, but not for PCA, where all rules tie. |
| H3, rule matters more than basis | **No**, both tasks | The spread across rules was not larger than the spread across bases. |
| H4, basis and rule interact | **Yes**, both tasks | For refusal, PCA beats raw dims under readout rules (by 0.09–0.10), but raw dims beat PCA under `attrib` (by 0.18). For truth the swap is real but tiny (0.002 vs. 0.019). |
| H5, strategy interacts | **No**, both tasks | Mean and resample ablation rank the 9 methods almost identically (rank correlation 0.94). The way of switching off barely affects which method wins. |
| H6, the two tasks agree | **No** | Rank correlation of 0.22–0.33, below the 0.6 bar. Truth and refusal favour different methods, partly because almost nothing works on truth at the reference cost. |

Had H0 passed, the rules would have returned **Outcome 1, interaction**, for both tasks, driven by H4.
Because the two tasks disagree (H6), even that would have been a weak result.

---

## 5. What this means and what to decide next

The run did its job as a pilot: it showed that the yardstick needs fixing before the full study.
Each fix below changes the pre-registered plan, so each should be decided deliberately and
written down as a change.

1. **Truth needs an intervention site where it can be broken.** Options: also intervene on the
   statement's own words (all positions), or accept that truth cannot be tested with interventions
   on the formatting tokens alone and replace it with another behaviour.
2. **The refusal reference fell just short (0.39–0.43 vs. 0.5).** Published work that gets large
   effects also intervenes on every token the model generates, not only on the formatting tokens. We
   could do the same, or revisit the 0.5 bar. Lowering a bar after seeing the data must be reported
   openly.
3. **The random baseline needs a fairer comparison point.** Options:
   - Let random selection switch off more units (for example 1,000 or 2,000), so its curve actually
     reaches the reference's cost instead of being represented by its maximum.
   - Compare all methods at a lower cost that every method reaches, so the comparison is not
     dominated by heavy collateral damage.
   - Report "random selection still does some damage" as a finding: it shows the refusal signal is
     spread widely across the model's internal state.

   One tempting fix does not work: copying resampled values only from prompts of the same kind
   (harmful from harmful). That would keep the "harmful" signal in place, so the reference and every
   good method would stop reducing refusal too, not just random selection.
4. **Then rerun.** A full run takes about 1.5 hours on the GB10, so iterating is cheap.

---

## 6. Run facts

| Item | Value |
|---|---|
| Model | Qwen/Qwen3-1.7B (chosen by the Step 0 check, which passed) |
| Intervention site | Layer 14 of 28, the 9 formatting tokens after the user's message |
| SAE | `adamkarvonen/qwen3-1.7b-saes`, layer 14, 65,536 features |
| Truth data | 1,196 statements to choose units, 1,496 to test (half true, half false) |
| Refusal data | 260 harmful + 300 harmless to choose units; 500 + 500 to test |
| Ordinary prompts | 100 for fitting the PCA axes, 300 for measuring cost |
| Interventions | 670 per task, 1,340 in total |
| Machine and time | NVIDIA GB10; about 1 h 35 min in total, 1 h 32 min of it for the ablations |

**Run history.** The first attempt stopped after a few minutes because one plumbing check (switching
a unit "off" to its own value must change nothing) reported small differences. These came from
rounding noise: the check compared numbers computed in differently sized batches, not a real error.
Pull request #1, "Fix identity check to reuse cache-stage batches", made the check compare like with
like. The run then resumed from where it stopped. The fix touched only the check, not any result.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| Ablation | Switching off (neutralising) part of a model's internal state to see what breaks. |
| Anchor | A reference intervention used as a yardstick. |
| AUROC | How well a score separates two groups: 1.0 is perfect, 0.5 is chance. |
| Basis | The way the internal state is cut into units: raw dims, PCA axes, or SAE features. |
| c\* | The reference cost: the collateral damage caused by the mass-mean anchor. |
| Confidence interval (CI) | The range the true value probably lies in (95% here). "The CI excludes 0" means the effect is reliably not zero. |
| D@c\* | A method's damage when it is allowed exactly the reference cost c\*. |
| Gap closure (G) | Where a method falls between random (0) and the reference (1). |
| Harness | The experimental machinery: data, measurements, controls and checks. |
| KL divergence | How much the model's predictions changed; used here as collateral-damage "cost". |
| Layer | One of the model's 28 processing steps. |
| Mass-mean direction | The difference between the average internal state for one group and for another. |
| PCA | A way to re-express data along axes ordered by how much they vary. |
| Readout vs. causal | Units that *carry information* about a behaviour vs. units that *produce* it. |
| Residual stream | The model's running internal state: 2,048 numbers per token per layer. |
| SAE (sparse autoencoder) | A learned dictionary that rewrites the internal state as a few active "features" out of many. |
| Unit | One piece of the internal state that can be switched off: a raw dimension, a PCA axis, or an SAE feature. |

## 8. Where to find things

| File | Contents |
|---|---|
| `pilot-implementation-plan.md` | The full technical plan (written as instructions for the code) |
| `20261116-180218-pilot-qwen3-1.7b/report.md` | The machine-generated report, with all numbers and plots |
| `20261116-180218-pilot-qwen3-1.7b/analyze/summary.csv` | Damage, gap closure and confidence ranges for every method |
| `20261116-180218-pilot-qwen3-1.7b/analyze/hypotheses.json` | Every statistical test behind H0–H6 |
| `20261116-180218-pilot-qwen3-1.7b/ablate/generation_checks.csv` | Refusal rates measured on actual generated replies |
| `README.md` | How to run the pilot yourself |
