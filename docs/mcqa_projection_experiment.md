# MCQA Experiment: Discriminability, Steering, and Projection on the Diff-in-Means Line

## 1. Motivation

This experiment is the MCQA analogue of `projection_steerability_experiment.md`, with two goals:

1. **Experiment 1 — Cross-layer link (train-side → test-side).** At each layer, compute the
   training-data discriminability `d'` on the MCQA pairs, and the test-set steering accuracy
   when we add the unnormalized DiffMean vector with factor `α = 1`. Plot both as a function
   of layer and test whether they are linearly correlated across layers.

2. **Experiment 2a — Sanity check: does the diff-in-means line encode the behavior?**
   For each of 50 held-out MCQA test prompts: run the model unsteered, extract the activation
   at the greedy-chosen answer letter, project that activation onto the diff-in-means line
   using `κ_a` (Braun thesis, §4.6), and plot `κ_a` (x) vs the baseline behavior score
   `P(matching)` (y). If the direction is meaningful, the two should be strongly correlated.

Everything runs on MCQA data only — both training and test come from the A/B dataset files
under `datasets/raw/<behavior>/dataset.json` and `datasets/test/<behavior>/test_dataset_ab.json`.

---

## 2. Setup

- **Model:** configurable (default `google/gemma-2-9b-it`).
- **Training data:** `datasets/raw/<behavior>/dataset.json`. Randomly subsampled (with a
  fixed seed) to `--max_examples` (default **300**), since raw MCQA sizes vary from 340
  (`corrigible-neutral-HHH`) to 20 184 (`sycophancy`).
- **Test data:** `datasets/test/<behavior>/test_dataset_ab.json` (50 MCQA A/B prompts per
  behavior).
- **Layers:** configurable range, default `10-32` (23 layers on Gemma-2-9b-it).
- **Steering factor:** `α = 1` (unnormalized DiffMean — we add the raw `μ⁺ − μ⁻` vector).

---

## 3. Mathematical Formalization

### 3.1 Training phase — per layer `l`

For each contrastive pair (`question`, `answer_matching`, `answer_not_matching`) we
tokenize both `question + answer_matching` and `question + answer_not_matching` through
the chat template (answer placed in the assistant message), then extract the hidden state
at the **answer-letter token** position (offset auto-detected, typically −4 on Gemma
chat models). This follows `train_caa_mcqa.py` and the original CAA paper (Rimsky et al.).

- Positive centroid: `μ⁺ = mean(act_letter | matching answer)`
- Negative centroid: `μ⁻ = mean(act_letter | non-matching answer)`
- Steering vector: `v = μ⁺ − μ⁻`, unit vector `v̂ = v / ‖v‖`
- Overall mean: `μ = (μ⁺ + μ⁻) / 2`
- Half-gap: `h = (μ⁺ · v̂ − μ⁻ · v̂) / 2 = ‖v‖ / 2`

### 3.2 Discriminability `d'` (training data only)

Projecting each per-example training activation onto `v̂` gives two 1-D distributions
(positive and negative). Their signal-to-noise ratio is

    d' = |mean(p⁺) − mean(p⁻)| / sqrt( (var(p⁺) + var(p⁻)) / 2 )

High `d'` means the two answer classes are cleanly separable along the DiffMean axis.

### 3.3 Test activation and κ_a

Following Braun, §4.6, we parameterize the diff-in-means line by `κ ∈ ℝ`:

    doml(κ) = (κ/2) · v + μ                          (so κ=+1 → μ⁺, κ=−1 → μ⁻, κ=0 → μ)

Given any test activation `a ∈ ℝᵈ`, its projection onto the line corresponds to

    κ_a = 2 · (a − μ) · v / ‖v‖²

`κ_a > 0` → the activation sits on the positive side of the midpoint; `κ_a > 1` → it
projects beyond the positive training centroid.

### 3.4 Test-time protocol — per prompt, per layer

For each MCQA test prompt:

1. **Score both candidates** (unsteered).
   Tokenize the full sequence `question + " (A)"` and `question + " (B)"` through the chat
   template. For each, sum the log-probs of the *answer tokens only* (conditioned on the
   prompt prefix). Let `ℓ_A = log P(" (A)" | prompt)`, `ℓ_B = log P(" (B)" | prompt)`.

2. **Baseline behavior score.**
   `P(matching) = softmax([ℓ_match, ℓ_notmatch])[0]`  (continuous, in `[0, 1]`).
   `greedy_match = 1` iff `ℓ_match > ℓ_notmatch`.

3. **Greedy answer-letter activation.**
   Let `g ∈ {A, B}` be the greedy choice. Re-use the forward pass for candidate `g` and
   extract the hidden state at the answer-letter position at every target layer `l`.
   This is the "baseline answer token" activation used for projection.

4. **Projection.**
   For each layer `l`, compute `κ_a` from the layer-`l` saved `μ⁺, μ⁻, v`.

5. **Steered score (factor α = 1).** Per layer `l`: add a forward hook at
   `model.layers[l].output` that adds `α · v` to hidden states at positions
   `≥ prefix_length` (same convention as `sweep_layers_open_ended.py`). Re-score both
   candidates; compute `P_steered(matching)`.

Per-layer aggregates over the 50 prompts:

| Quantity | Meaning |
|---|---|
| `dprime` | Training discriminability (from §3.2) |
| `baseline_acc_mean` | Mean `P(matching)` at α=0 |
| `baseline_greedy_acc` | Fraction with `greedy_match = 1` at α=0 |
| `steered_acc_mean` | Mean `P(matching)` at α=1 |
| `delta_acc` | `steered_acc_mean − baseline_acc_mean` |
| `proj_spearman_rho` | Spearman(`κ_a`, baseline `P(matching)`), N=50 |
| `proj_pearson_r` | Pearson(`κ_a`, baseline `P(matching)`), N=50 |

### 3.5 Hypotheses

- **Exp 1 (train-side predictor).** Across layers,
  `Pearson(d', steered_acc_mean) > 0` and significant — layers with cleaner training
  separation steer better at α = 1.
- **Exp 1' (test-side predictor, κ_a analogue).** Across layers, the per-layer projection
  quality `ρ_l = Spearman(κ_a, baseline_score)_l` also tracks steering success:
  `Pearson(ρ_l, steered_acc_mean) > 0` and significant. This is the MCQA analogue of the
  `ρ_l` predictor from `projection_steerability_experiment.md`.
- **Exp 2a (within-layer sanity check).** For each layer, `Pearson(κ_a, baseline_score) > 0`
  and significant for at least the layers where `d'` is high, confirming that (1) the
  direction generalizes train → test and (2) the diff-in-means line is actually
  behavior-aligned.

---

## 4. Outputs

### Training script (`mcqa_train_diffmean_sweep.py`)

```
<output_dir>/
  layer_{N}/
    DiffMean_weight.pt         # [1, hidden_size]
    DiffMean_bias.pt           # zeros([1])
    mu_pos.pt                  # [hidden_size]
    mu_neg.pt                  # [hidden_size]
    separability.json          # {dprime, auroc, norm}
    config.json                # {model, layer, seed, n_examples, answer_token_offset, ...}
  training_stats.json          # per-layer dprime, auroc, norm (sweep-level)
  sweep_config.json            # full args used
```

### Test script (`mcqa_test_steer_and_project.py`)

```
<sweep_dir>/mcqa_analysis/
  per_prompt_results.csv       # one row per (layer, prompt_idx): baseline_p_match, steered_p_match,
                               #                                  kappa_a, greedy_match_baseline, greedy_match_steered
  per_layer_summary.csv        # one row per layer with all aggregate metrics
  cross_layer_correlation.csv  # Pearson/Spearman between each predictor (d', κ_a→P_base ρ,
                               # κ_a→P_base r) and each test-side metric, across layers
  summary.json
  plots/
    dprime_vs_steering_by_layer.png         # Exp 1: two lines (d', steered_acc) vs layer + correlation annotation
    kappa_rho_vs_steering_by_layer.png      # Exp 1': two lines (ρ(κ_a, P_base), steered_acc) vs layer + corr
    combined_predictors_vs_steering.png     # z-scored d' and κ_a ρ overlaid with steered acc
    projection_rho_by_layer.png             # Per-layer Spearman/Pearson(κ_a, baseline_score) across layers
    projection_scatter_layer_{N}.png        # Exp 2a scatter plot, one per layer
```

---

## 5. Running

```bash
# Phase 1 — train across all 23 layers (one multi-layer forward pass per training example).
uv run python axbench/scripts/mcqa_train_diffmean_sweep.py \
    --behavior corrigible-neutral-HHH \
    --model_name google/gemma-2-9b-it \
    --layers 10-32 \
    --max_examples 300 \
    --seed 42 \
    --output_dir results/mcqa_sweep/gemma-2-9b-it/corrigible-neutral-HHH

# Phase 2 — test: steered accuracy at α=1 per layer + projection scatter.
uv run python axbench/scripts/mcqa_test_steer_and_project.py \
    --behavior corrigible-neutral-HHH \
    --model_name google/gemma-2-9b-it \
    --sweep_dir results/mcqa_sweep/gemma-2-9b-it/corrigible-neutral-HHH \
    --steering_factor 1.0
```
