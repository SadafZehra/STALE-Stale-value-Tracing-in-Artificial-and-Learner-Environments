# STALE

### **S**tale-value **T**racing in **A**rtificial and **L**earner **E**nvironments

*Working-memory limits of code language models and novice programmers: behaviour, mechanisms, cognitive modelling, and personalised hints.*

> **Notebook:** `Variable_Tracing_Working_Memory.ipynb`
> **Type:** Research prototype (behavioural + mechanistic interpretability + cognitive modelling + educational simulation)
> **Status:** All learner results are **simulated**; LLM results are from real model runs (Qwen2.5-Coder 0.5B and 1.5B).

---

## Abstract

Tracing how variable values change through a program is a core skill in programming education and a basic test of whether a language model actually *executes* code or merely pattern-matches it. STALE studies this with one controlled benchmark and four connected analyses. (1) We generate 400 straight-line Python programs that systematically vary the number of variables, the number of re-assignments of a queried variable, and the *kind* of re-assignment (plain, augmented, tuple swap, loop), and measure two code LMs by forced-choice scoring over meaningful distractors. (2) We classify every error as a **stale-value** error (an earlier value of the queried variable), an **other-variable** error, or other. (3) We look inside the smaller model with layer-wise linear probes and activation patching. (4) We compare model errors to an **ACT-R-style declarative-memory model** with a single capacity parameter *W*, and then use that same model to estimate a simulated student's capacity from a few answers and to choose personalised hints.

The main empirical finding is that both models are near ceiling when a variable is never re-assigned (90 to 96%) but collapse to roughly 10 to 26% as soon as it is re-assigned once, with **stale-value errors as the dominant failure** (about 50 to 75% of answers in the re-assignment conditions). Probing shows that the *initial* (stale) value is more linearly decodable than the *current* value in later layers, and activation patching shows that the answer is causally carried by the initial-value token in early/middle layers, with almost no causal contribution from the last re-assignment position.

---

## Research Questions

| ID | Question | Where answered |
|----|----------|----------------|
| **RQ1** | How does tracing accuracy of code LMs depend on the number of variables, the number of re-assignments, and the type of re-assignment? | Behavioural evaluation |
| **RQ2** | Are errors random, or systematically *stale* (outdated) values, as in human binding/working-memory failures? | Error taxonomy |
| **RQ3** | Do solved demonstrations (few-shot) reduce stale-value errors? | Few-shot test |
| **RQ4** | Where in the network is the value of a variable represented, and which activations causally determine the answer? | Probing, activation patching |
| **RQ5** | Can a limited-capacity declarative-memory account (ACT-R style) describe LLM error patterns? | ACT-R fit |
| **RQ6** | Can the same account estimate a learner's capacity from very few answers and support personalised hint selection? | Few-shot capacity estimation, hint policy |

---

## Pipeline Overview

```
 Controlled program generator (vars x re-assignments x pattern)
                       │
                       ▼
        400 programs with executed ground truth
   (answer, stale values, other-variable values, read/write events)
                       │
        ┌──────────────┼───────────────────────────┐
        ▼              ▼                           ▼
  Behaviour of     Mechanisms (0.5B)        ACT-R memory model
  0.5B and 1.5B    • layer-wise probes      • capacity W
  • accuracy       • activation patching    • fit W to LLM errors
  • stale errors                            • simulate students
  • few-shot                                • few-shot W estimation
                                            • personalised hints
```

---

## Method

### 1. Benchmark

Programs are straight-line Python with a queried **target** variable and distractor statements on other variables. Each program is *executed statement by statement* to log reads, writes, and every value the target takes.

| Factor | Levels |
|--------|--------|
| Variables in memory | 2, 3, 4, 5 |
| Re-assignments of the target | 0, 1, 2, 3, 4 |
| Re-assignment pattern | `plain` (`x = x + y`), `augmented` (`x += y`), `swap` (`x, y = y, x`), `loop` (`for i in range(n): x = x + i`) |
| Repetitions | 5 per cell (20 for the no-re-assignment cells) |

**Total: 400 items** (80 with no re-assignment; 80 for each of re-assignment levels 1 to 4). A separate pool of 40 programs supplies few-shot demonstrations and is never used as test items.

**Error taxonomy.** The correct answer is the final value of the target. A *stale* value is an earlier value of the target that differs from the answer. *Other-variable* values are initial or final values of other variables. Remaining candidates are random numbers.

### 2. Models and scoring

| Role | Model |
|------|-------|
| "Small" (also used for mechanistic analysis) | `Qwen/Qwen2.5-Coder-0.5B` |
| "Large" | `Qwen/Qwen2.5-Coder-1.5B` |

Prompts end with `# Answer: <target> =`. Each model scores every candidate value (log-probability of ` <number>`), and the highest-scoring candidate is its answer. This is a **forced-choice** design, so accuracy is not comparable to free-generation accuracy, and chance depends on the number of candidates.

### 3. Few-shot test

A fixed subset of 160 items with at least one re-assignment is evaluated with 0, 2, and 4 solved demonstrations (same demonstrations for both models). Zero- versus four-shot accuracy is compared with an exact McNemar (binomial) test.

### 4. Mechanistic analysis (small model)

- **Linear probes.** Last-token hidden states at all 25 layers (embeddings + 24 blocks) predict (a) the current value, (b) the stale initial value, and (c) a shuffled-label control, using ridge regression with 5-fold cross-validated R². Only items where the initial value differs from the answer are used (290 items).
- **Activation patching.** Clean/corrupted program pairs differ only in the target's initial digit. A clean activation at one of three positions (*initial value*, *last re-assignment*, *final token*) is patched into the corrupted run at one layer at a time, and we report the fraction of the clean-vs-corrupted logit difference that is restored. Pairs are restricted to plain/augmented patterns with 1 to 2 re-assignments, and pairs the model cannot tell apart are skipped.

### 5. ACT-R-style memory model

Every assignment creates or strengthens a (variable, value) chunk, and every read rehearses the variable's current chunk. Activation combines base-level decay, a capacity-scaled spreading-activation cue (*W*) for the queried variable and its most recent value, a partial-matching penalty for chunks of other variables, and a retrieval-failure threshold. Retrieval is Boltzmann-distributed and yields probabilities of **correct / stale / other** outcomes.

Fixed parameters: decay 0.5, noise 0.4, threshold -1.0, mismatch 1.5, line time 2.0 s. Capacity *W* is the single free parameter (grid 0.0 to 3.0, step 0.1).

### 6. Learner simulation and hints

- **Simulated students (n = 200):** novices with *W* ~ N(0.6, 0.2) and experienced programmers with *W* ~ N(1.8, 0.3). Answers are sampled from the ACT-R model.
- **Few-shot capacity estimation:** Bayesian posterior over *W* (flat prior on the grid) from K = 1, 3, 5, 10 answers, compared with a population-mean baseline.
- **Hint policy:** three actions: `none`, `recall_reminder` (cue boost 0.6, cost 0.05), and `trace_table` (assumed success 0.97, cost 0.30). The personalised policy maximises expected utility (success minus effort) under the posterior over *W*.

---

## Results

All numbers are from the saved notebook run (Colab GPU). Intervals are 95% bootstrap CIs (1,000 resamples) unless stated.

### RQ1 and RQ2: Behaviour and stale-value errors (zero-shot)

**Overall outcome distribution (all 400 items):**

| Model | Correct | Stale | Other variable | Other |
|-------|--------:|------:|---------------:|------:|
| Small (0.5B) | 0.288 | **0.530** | 0.152 | 0.030 |
| Large (1.5B) | 0.322 | **0.418** | 0.212 | 0.048 |

**Accuracy by number of re-assignments:**

| Re-assignments | Small acc. | Small stale rate | Large acc. | Large stale rate |
|---------------:|-----------:|-----------------:|-----------:|-----------------:|
| 0 | **0.962** | 0.000 | **0.900** | 0.000 |
| 1 | 0.100 | 0.750 | 0.262 | 0.488 |
| 2 | 0.138 | 0.638 | 0.175 | 0.500 |
| 3 | 0.138 | 0.625 | 0.175 | 0.575 |
| 4 | 0.100 | 0.638 | 0.100 | 0.525 |

**Accuracy by re-assignment pattern** (re-assignment items only, plus the no-re-assignment baseline):

| Pattern | Small | Large |
|---------|------:|------:|
| none | 0.962 | 0.900 |
| plain | 0.088 | 0.238 |
| augmented | 0.062 | 0.075 |
| loop | 0.025 | 0.100 |
| swap | 0.300 | 0.300 |

**Take-aways**

- A **cliff, not a slope**: a single re-assignment drops accuracy from about 0.9 to 0.1 to 0.26. Additional re-assignments change little.
- Errors are **systematically stale** rather than random, which matches the signature of interference/binding failures in human working memory.
- The larger model is better on the first re-assignment, but not clearly different at 3 to 4 re-assignments (overlapping CIs). Scale alone does not remove the problem here.
- Augmented assignment and loops are hardest. Tuple swap is comparatively easier for both models.

### RQ3: Few-shot demonstrations (160 items with re-assignment)

| Model | Shots | Accuracy | Stale rate |
|-------|------:|---------:|-----------:|
| Small | 0 | 0.138 | 0.662 |
| Small | 2 | 0.131 | 0.494 |
| Small | 4 | 0.100 | 0.525 |
| Large | 0 | 0.200 | 0.506 |
| Large | 2 | 0.238 | 0.381 |
| Large | 4 | 0.250 | 0.344 |

Exact McNemar (0 vs. 4 shots): small, 14 items lost vs. 8 gained (p = 0.286); large, 9 lost vs. 17 gained (p = 0.169). **Neither effect is statistically significant.** Demonstrations seem to shift errors *away from stale values* (especially for the large model) without reliably improving accuracy.

### RQ4: Where is the value represented? (small model)

**Probing (selected layers, cross-validated R²):**

| Layer | Current value | Initial (stale) value | Shuffled control |
|------:|--------------:|----------------------:|-----------------:|
| 0 | -0.033 | -0.006 | -0.007 |
| 5 | 0.321 | 0.307 | -0.009 |
| 10 | 0.397 | 0.534 | -0.010 |
| 17 | 0.428 | 0.779 | -0.008 |
| 19 | 0.443 | **0.808** | -0.008 |
| 20 | **0.462** | 0.799 | -0.008 |
| 24 | 0.394 | 0.786 | -0.062 |

- The shuffled control stays near 0, so the probes are not trivially overfitting.
- From about layer 5 onward the **initial (stale) value is more decodable than the current value**, and the gap widens in later layers (about 0.80 vs. 0.46 at layers 19 to 20). The final-token representation is dominated by the *old* value.

**Activation patching (20 usable pairs; fraction of clean preference restored):**

| Layers | Initial-value token | Last re-assignment token | Final token |
|-------:|--------------------:|-------------------------:|------------:|
| 0 to 7 | about 0.98 to 1.02 | about 0.00 | about 0.00 |
| 8 to 15 | about 0.90 to 0.91 | about 0.00 to 0.01 | up to 0.015 |
| 16 to 21 | 0.03 to 0.10 | about 0.00 | 0.024 to 0.036 |
| 22 to 23 | about 0.00 | about 0.00 | 0.028 |

- Patching the **initial-value token** restores nearly the whole clean answer up to layer 15, then drops sharply at layer 16, consistent with information being moved out of that position into later positions around layers 15 to 16.
- Patching the **last re-assignment** position restores essentially nothing at any layer, and the final token restores at most about 3.6%. In this setting the model's answer depends on the *initial* value, but there is **no sign that the updated value is stored at the position of the last re-assignment**. This is consistent with the stale-error behaviour, although it does not by itself establish a full circuit (see Limitations).

### RQ5: ACT-R capacity account

Best-fitting capacity (maximum likelihood on zero-shot outcomes):

| Model | Best W | Log-likelihood | r (predicted vs. observed accuracy across re-assignment levels) |
|-------|-------:|---------------:|-----------------------------------------------------------------|
| Small | 0.1 | -385.2 | 0.793 |
| Large | 0.1 | -439.4 | 0.861 |

Both LLMs are best described by a **very low effective capacity** (near the bottom of the grid), and the model reproduces the qualitative decline of accuracy with the number of re-assignments (r = 0.79 to 0.86 over five levels). It does **not** capture the sharp cliff between 0 and 1 re-assignments observed in the LLMs, since the model predicts a smoother decline.

### RQ6: Few-shot capacity estimation and personalised hints (simulated students)

**Capacity estimation (posterior mean of W):**

| K answers | MAE of W | Novice vs. experienced accuracy | MAE of population-mean baseline |
|----------:|---------:|--------------------------------:|--------------------------------:|
| 1 | 0.502 | 0.690 | 0.621 |
| 3 | 0.368 | 0.805 | 0.621 |
| 5 | 0.354 | 0.855 | 0.621 |
| 10 | 0.292 | **0.935** | 0.621 |

**Hint policies** (expected utility = success minus effort; 60 students x 5 held-out items):

| Policy | Utility | 95% CI |
|--------|--------:|--------|
| Always none | 0.669 | 0.643 to 0.696 |
| Always recall reminder | 0.778 | 0.761 to 0.795 |
| Always trace table | 0.670 | 0.670 to 0.670 |
| **Personalised (K = 5)** | **0.787** | 0.774 to 0.801 |
| Oracle (true W) | 0.813 | 0.802 to 0.826 |

By group: novices gain most from personalisation (none 0.454, personalised 0.690, oracle 0.715), while experienced students are already served well by no hint or a light reminder (none 0.856, personalised 0.872, oracle 0.899).

The personalised policy beats "no hint" and "trace table" clearly, but its advantage over **always giving a recall reminder** is small and the CIs overlap, so we do not claim a reliable gain over that strong baseline.

---

## Limitations and Threats to Validity

These matter for any claims made from this prototype:

1. **Learner results are circular.** Simulated students are generated *from* the same ACT-R model that is then used to estimate their capacity and choose hints, so good recovery (MAE 0.29, 93.5% group accuracy) is expected by construction. No human data has been collected yet.
2. **Hint utilities are assumed.** Costs and effects (e.g. trace table success 0.97, cost 0.30; reminder boost 0.6) are hand-set. The "trace table" policy has a constant utility by design.
3. **ACT-R is not fitted end to end.** Only *W* is estimated; decay, noise, threshold, and mismatch penalty are fixed by hand. The best *W* (0.1) sits near the lower edge of the grid, and the model misses the 0 to 1 re-assignment cliff.
4. **Forced-choice scoring.** Models choose among a candidate set that includes stale and other-variable values. Stale errors are therefore partly a function of which distractors exist; free-generation behaviour may differ.
5. **Small models, one family.** Only Qwen2.5-Coder 0.5B and 1.5B base models were tested. Results may not transfer to larger or instruction-tuned models, or to chain-of-thought prompting, which is likely to change tracing behaviour substantially.
6. **Limited mechanistic evidence.** Probes use 290 items with ridge regression on single-digit targets (correlational). Patching uses only **20 usable pairs** (of 40 attempted), restricted to plain/augmented patterns with 1 to 2 re-assignments, and only three positions. It does not identify attention heads or MLP circuits.
7. **Few-shot effects are not significant** (McNemar p = 0.29 and 0.17), so no conclusion about demonstration effects should be drawn beyond a trend in stale-error rates.
8. **Narrow task.** Straight-line integer programs with single-character variable names; no functions, objects, aliasing, or control flow beyond a small loop pattern.
9. **Single seed** (`SEED = 7`) for generation and sampling; bootstrap CIs reflect item sampling, not seed variability.

---

## Suggested Extensions

- **Human study.** Administer the benchmark to novice and experienced programmers to test whether stale-value errors, the swap/augmented/loop difficulty ordering, and the capacity parameter generalise (the notebook already accepts real data, see below).
- **Stronger causal analysis.** Path patching, attention-head and MLP attribution, and logit lens to locate where the initial value is "copied" and why the updated value is not.
- **Mitigations.** Chain-of-thought / trace-table prompting, fine-tuning on execution traces, and comparison with instruction-tuned models.
- **Better cognitive model.** Fit all ACT-R parameters jointly, compare with alternative accounts (e.g. interference-based or Bayesian forgetting), and model response times for human data.
- **Adaptive tutoring.** Replace assumed hint utilities with measured effects from a classroom experiment, and test the personalised policy prospectively.

---

## Reproducibility

### Requirements

- Python 3.9+, GPU recommended (the two zero-shot/few-shot evaluation passes took about 1,320 s and 1,335 s on a Colab GPU)
- Packages (auto-installed by the first cell if missing): `transformers`, `accelerate`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `torch`
- Internet access on first run to download the two Hugging Face models

### Run

1. Open `Variable_Tracing_Working_Memory.ipynb` in Jupyter, VS Code, or Google Colab.
2. Run all cells from top to bottom.
3. Outputs are written to `tracing_work/outputs/`.

### Key configuration

| Parameter | Default | Meaning |
|-----------|--------:|---------|
| `SEED` | 7 | Global seed |
| `MODELS` | 0.5B and 1.5B | Models compared |
| `MECH_MODEL` | `small` | Model used for probes and patching |
| `N_VARS_GRID` | `[2, 3, 4, 5]` | Variables per program |
| `N_REBIND_GRID` | `[0, 1, 2, 3, 4]` | Re-assignments of the target |
| `PATTERNS` | plain, augmented, swap, loop | Re-assignment types |
| `REPS` | 5 | Programs per design cell |
| `SHOT_GRID` | `[0, 2, 4]` | Few-shot demonstration counts |
| `N_FEWSHOT_ITEMS` | 160 | Items in the few-shot test |
| `N_PATCH_PAIRS` | 40 | Max clean/corrupted pairs |
| `N_BOOT` | 1000 | Bootstrap resamples |

### Using real classroom data

Place `tracing_work/student_tracing.csv` with columns:

| Column | Meaning |
|--------|---------|
| `student_id` | Student identifier |
| `item_id` | ID of a benchmark item (see `tracing_items.csv`) |
| `outcome` | `correct`, `stale`, or `other` |

The notebook then estimates each student's capacity *W* and saves `real_student_capacity.csv`.

### Output files (`tracing_work/outputs/`)

| File | Content |
|------|---------|
| `tracing_items.csv` | The 400-item benchmark (reusable as a quiz) |
| `llm_tracing_predictions.csv` | Every model prediction and outcome type |
| `accuracy_by_rebind.csv`, `accuracy_by_vars.csv`, `accuracy_by_pattern.csv` | Zero-shot accuracy tables |
| `fewshot_effect.csv` | Few-shot accuracy and stale-rate table |
| `probes_by_layer.csv` | Layer-wise probe R² |
| `activation_patching.csv` | Recovered share per layer and position |
| `actr_fit_to_llms.csv` | Best capacity per LLM |
| `fewshot_capacity_estimation.csv` | Student capacity estimation |
| `hint_policy_simulation.csv`, `hint_policy_summary.csv` | Hint policy results |
| `summary.json` | Headline numbers |
| `fig1` to `fig7` (`.png`) | Figures for all analyses |

---

## Suggested Citation

```
STALE: Stale-value Tracing in Artificial and Learner Environments.
Working-memory limits of code language models and novice programmers.
[Your Name], [Year]. Code and notebook: Variable_Tracing_Working_Memory.ipynb
```

## License

Add your preferred license here (e.g. MIT).
