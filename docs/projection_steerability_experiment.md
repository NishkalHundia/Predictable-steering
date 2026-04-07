# Test-Time Projection Analysis: Does the Steering Direction Predict Behavior and Steerability?

## 1. Motivation

Activation steering via DiffMean computes a steering vector **v** from contrastive training examples and adds a scaled version of it to a model's hidden states at inference time to modulate a target behavior. A layer sweep evaluates many layers and steering factors to find the best configuration.

But this raises a natural question: **does the steering direction carry information about test prompts before any steering is applied?**

If a test prompt's unsteered activations already project strongly toward the "positive behavior" end of the steering vector, we'd expect two things:

1. Its **baseline (unsteered) generation** should already exhibit the target behavior more strongly.
2. When steered, it should be **easier to push further** in that direction (or at least achieve a higher absolute score), since it's already "aligned" with the steering direction.

This experiment tests both claims.

---

## 2. Setup

### Model and Behaviors

- **Model:** Gemma-2-9B-IT (`google/gemma-2-9b-it`)
- **Layers analyzed:** 10 through 32 (23 layers), covering middle to final transformer blocks
- **Behaviors tested:**
  - `corrigible-neutral-HHH` (scale: -5 to +5)
  - `hallucination` (scale: 0 to 5)
  - `myopic-reward` (scale: -5 to +5)
  - `survival-instinct` (scale: -5 to +5)

### Prerequisites

The experiment requires a completed layer sweep (`sweep_layers_open_ended.py`) which produces:
- A DiffMean steering vector per layer
- Per-prompt, per-factor evaluation results scored by an LM judge
- Per-prompt fluency scores from an axbench judge
- Training separability metrics (d') per layer

### Test Prompts

Each behavior has 32 test prompts. For each prompt, we have:
- The **unsteered generation** (steering factor = 0) and its judge-assigned behavior score
- **Steered generations** at factors {0.5, 1, 2, 5, 8, 10, 15, 20} and their scores
- **Fluency scores** for each generation

---

## 3. Mathematical Formalization

### 3.1 Training Phase: Computing the Steering Vector

Given a set of contrastive training pairs {(x_i, y_i^+, y_i^-)}:

- **Positive centroid:** mu_pos = mean of response-token activations for all y_i^+ at layer l
- **Negative centroid:** mu_neg = mean of response-token activations for all y_i^- at layer l
- **Steering vector:** v = mu_pos - mu_neg
- **Unit steering vector:** v_hat = v / ||v||

The d' (discriminability index) measures how well the training distributions are separated along v_hat:

```
d' = (mu_pos . v_hat - mu_neg . v_hat) / sqrt( (sigma_pos^2 + sigma_neg^2) / 2 )
```

where sigma_pos and sigma_neg are the standard deviations of the positive and negative projections onto v_hat.

### 3.2 Centroid Projections and Normalization

Project the training centroids onto the steering direction:

```
c+ = mu_pos . v_hat       (positive centroid projection)
c- = mu_neg . v_hat       (negative centroid projection)
```

By construction, c+ > c- (since v = mu_pos - mu_neg, and v_hat points from negative toward positive).

Define a normalization that maps these centroids to a standard scale:

```
m = (c+ + c-) / 2         (midpoint)
h = (c+ - c-) / 2         (half-range)
```

The **normalized projection** of any vector a is:

```
p_tilde = (a . v_hat - m) / h
```

Under this transform:
- c+ maps to **+1**
- c- maps to **-1**
- The midpoint maps to **0**

This gives us an interpretable scale: a normalized projection of +2 means the point is as far beyond the positive centroid as the gap between the two centroids.

### 3.3 Test-Time Projection

For test prompt x_j with unsteered generation y_j:

1. Run a forward pass on the concatenation [x_j, y_j]
2. Extract the hidden state at layer l for the **response tokens only** (excluding the prompt)
3. Average across response token positions to get **mean_act_j**
4. Compute the raw projection: p_j = mean_act_j . v_hat
5. Normalize: p_tilde_j = (p_j - m) / h

### 3.4 Behavior Score

The **baseline score** b_j is the LM judge's score for the unsteered generation y_j on a behavior-specific rubric (e.g., -5 to +5 for corrigibility, 0 to 5 for hallucination). Higher scores indicate more of the target behavior.

### 3.5 Per-Prompt Steerability

For each prompt j at layer l, we define per-prompt steerability metrics using **fluency-filtered** results (only counting steering factors where prompt j's generation has fluency_score >= 1.0):

- **best_steered_score_j:** max behavior_score across all fluency-valid positive steering factors
- **delta_j:** best_steered_score_j - b_j (improvement from steering)
- **best_factor_j:** the steering factor that achieved best_steered_score_j

---

## 4. Hypotheses

### Hypothesis 1 (H1): Projection Predicts Baseline Behavior

> A test prompt whose unsteered activation projects closer to the positive training centroid should receive a higher baseline behavior score.

Formally:

```
rho(p_tilde, baseline_score) >> 0
```

where rho is Spearman's rank correlation, computed per layer across all test prompts.

**Intuition:** If the steering vector captures a meaningful behavioral axis, then prompts that naturally sit closer to the "positive behavior" end should already exhibit that behavior more in their unsteered output.

### Hypothesis 2 (H2): Projection Predicts Steerability

> Prompts with higher projection should achieve higher absolute steered scores.

Tested two ways:

**A. Within-layer per-prompt correlation:**
```
rho(p_tilde_j, best_steered_score_j) > 0
```

**B. Group comparison (median split):**
Split prompts into "high projection" (p_tilde >= median) and "low projection" (p_tilde < median), then compare their steerability via Mann-Whitney U test and Cohen's d effect size.

### Expected Relationship Between H1 and H2

If both hypotheses hold, there is a natural consequence for **delta** (improvement from steering):

```
rho(p_tilde, delta) < 0    (expected ceiling effect)
```

**Why?** High-projection prompts already have high baselines, leaving less room for improvement. A prompt at baseline +4 can only improve by +1 on a [-5, +5] scale, while a prompt at baseline -3 can improve by +8. So delta should be negatively correlated with projection even when the absolute steered score is positively correlated. This is not a failure of the hypothesis -- it is a predicted consequence of it.

---

## 5. Methodology

### 5.1 Phase 1: Projection vs Baseline Score

**Script:** `projection_vs_steering_factor.py`

For each behavior:

1. Load the model and compute mean response-token activations for each test prompt at each layer
2. Load training contrastive data, compute activations for positive and negative examples
3. Project all activations onto the per-layer steering vector
4. Compute centroids (c+, c-) from training projections
5. Normalize all projections to the centroid scale (c+ -> +1, c- -> -1)
6. Correlate normalized_projection vs baseline_score per layer (Spearman and Pearson)

**Outputs:** Per-layer scatter plots (projection vs score), distribution plots (train vs test), correlation-across-layers plot, and CSVs with all raw data.

### 5.2 Phase 2: Projection vs Per-Prompt Steerability

**Script:** `projection_steerability_link.py`

For each behavior:

1. Load normalized projections from Phase 1
2. For each prompt at each layer, load eval_results.parquet and eval_axbench/eval_results.parquet
3. **Fluency filter:** For each (prompt, factor) pair, keep only if fluency_score >= 1.0
4. Compute per-prompt steerability: best_steered_score, delta, best_factor
5. **Within-layer correlation:** rho(p_tilde, best_steered_score) and rho(p_tilde, delta) per layer
6. **Group comparison:** Median-split prompts by p_tilde, Mann-Whitney U on steerability, Cohen's d
7. **Supplementary:** Aggregate per-layer steerability (with min_valid >= 25 filter for factor validity) and cross-layer correlations with d'

### 5.3 Filtering

Two levels of filtering are applied:

| Filter | Level | Condition | Purpose |
|--------|-------|-----------|---------|
| **Fluency filter** | Per (prompt, factor) | fluency_score >= 1.0 | Exclude gibberish generations |
| **Min-valid filter** | Per factor (aggregate only) | n_valid_prompts >= 25 | Ensure aggregate means are reliable |

The fluency filter is applied to both per-prompt steerability (Phase 2) and aggregate steerability metrics. The min-valid filter is only used for aggregate metrics (the per-prompt analysis doesn't need it since each data point is a single prompt).

---

## 6. Results

### 6.1 Phase 1: Projection vs Baseline Score

#### corrigible-neutral-HHH

**All 23 layers show significant positive correlation.** Spearman rho ranges from +0.455 (layer 10) to +0.789 (layer 21), all at p < 0.05. This is the strongest and most consistent result: the steering vector direction captures behavioral variance in unsteered test outputs at every layer.

| Layer range | Typical rho | Significance |
|-------------|-------------|--------------|
| 10-13       | +0.45 to +0.67 | All p < 0.01 |
| 14-20       | +0.59 to +0.78 | All p < 1e-3 |
| 21-27       | +0.66 to +0.79 | All p < 1e-4 |
| 28-32       | +0.60 to +0.72 | All p < 1e-3 |

#### hallucination, myopic-reward, survival-instinct

**Weak or non-significant correlations at all layers.** The projection does not meaningfully predict baseline behavior scores for these three behaviors. The DiffMean direction separates training examples (d' > 3 for all) but this separation doesn't generalize to test-time behavioral variance.

### 6.2 Phase 2: Projection vs Per-Prompt Steerability

#### corrigible-neutral-HHH

**12 of 23 layers show significant rho(projection, best_steered_score) at p < 0.05.** The relationship strengthens at later layers (23-32), with rho values from +0.39 to +0.64.

The group comparison confirms this: Cohen's d is consistently positive across all layers, reaching +1.10 at layer 31. High-projection prompts achieve higher absolute steered scores than low-projection prompts.

**Delta is consistently negatively correlated** (rho from -0.22 to -0.77, significant at 21 of 23 layers). This confirms the predicted ceiling effect: high-projection prompts start high and have less room to improve.

Selected layers:

| Layer | rho(proj, best_steered) | rho(proj, delta) | Cohen's d |
|-------|------------------------|-------------------|-----------|
| 15    | +0.388*  | -0.536*  | +0.87 |
| 16    | +0.406*  | -0.757*  | +0.73 |
| 19    | +0.329   | -0.773*  | +0.66 |
| 24    | +0.582*  | -0.402*  | +0.91 |
| 27    | +0.638*  | -0.679*  | +0.99 |
| 30    | +0.638*  | -0.441*  | +1.04 |
| 31    | +0.632*  | -0.656*  | +1.10 |

\* = p < 0.05

#### hallucination

**2 of 23 layers significant** for rho(projection, best_steered_score): layers 24 (+0.641) and 25 (+0.436). Layer 24 shows a very large group effect (Cohen's d = +1.57). This suggests the hallucination steering direction captures test-time information only at late layers.

At the **aggregate level**, d' positively predicts layer steerability (rho = +0.54 to +0.58, p < 0.01).

#### myopic-reward

**0 of 23 layers significant** for rho(projection, best_steered_score). Correlations are weak and directionless. The DiffMean direction does not predict per-prompt steerability.

Notably, at the aggregate level, **d' negatively correlates with steerability** (rho = -0.69 to -0.72, p < 0.001). Layers where training examples are most separable actually steer *worse*. This suggests the DiffMean vector for myopic-reward may capture a training-specific feature that does not generalize.

#### survival-instinct

**0 of 23 layers significant** for rho(projection, best_steered_score). No per-prompt relationship.

However, at the aggregate level, **d' positively predicts steerability** (rho = +0.65, p < 0.001). Training separability does predict which layers steer well, even though projection does not predict which prompts within a layer steer well.

### 6.3 Summary Table

| Behavior | H1: proj -> baseline | H2: proj -> steerability | d' -> steerability |
|----------|---------------------|-------------------------|-------------------|
| **corrigible-neutral-HHH** | 23/23 layers sig (rho up to +0.79) | 12/23 layers sig (rho up to +0.64) | Non-significant |
| **hallucination** | ~0/23 layers sig | 2/23 layers sig (late layers) | Positive (rho ~ +0.55) |
| **myopic-reward** | ~0/23 layers sig | 0/23 layers sig | **Negative** (rho ~ -0.70) |
| **survival-instinct** | ~0/23 layers sig | 0/23 layers sig | Positive (rho ~ +0.65) |

---

## 7. Interpretation

### 7.1 Why Does It Work for Corrigibility But Not Others?

The DiffMean vector is computed as the difference of means between positive and negative training activations. For this to predict test-time behavior, two conditions must hold:

1. **The training contrast must isolate a behaviorally meaningful direction.** The positive and negative examples must differ primarily on the target behavior, not on confounds (topic, style, length).

2. **The test prompts must vary along that same direction.** If all test prompts cluster in one region of activation space (e.g., all naturally non-corrigible), there would be no variance to predict.

For **corrigible-neutral-HHH**, both conditions appear to hold strongly. The contrastive training set captures a genuine behavioral axis, and test prompts vary meaningfully along it.

For the other behaviors, the DiffMean direction likely captures **confounded features** from training that don't generalize. The training d' values are high (3.0-4.3 for all behaviors), meaning training examples *are* separable, but the separating direction may not correspond to the target behavior in novel contexts.

### 7.2 The Ceiling Effect Is Informative

The consistent negative correlation between projection and delta for corrigible-neutral-HHH is not a nuisance -- it carries information:

- It means the **score scale has a ceiling** relative to where high-projection prompts already sit
- It means steering is most impactful for **low-projection prompts** (those far from the positive centroid)
- It implies **adaptive steering factors** could be useful: low-projection prompts may benefit from larger steering factors

### 7.3 Different Predictors for Different Purposes

The results reveal two distinct prediction regimes:

**Per-prompt prediction (projection):** Answers "which specific prompts will steer well?" Only works when the steering direction captures genuine behavioral variance at test time (corrigible-neutral-HHH, partially hallucination).

**Per-layer prediction (d'):** Answers "which layers steer well on average?" Works for survival-instinct and hallucination where aggregate separability predicts aggregate steerability, even without prompt-level prediction.

**Neither works for myopic-reward**, where d' actually anti-correlates with steerability, suggesting the DiffMean direction for this behavior captures something other than the target concept.

---

## 8. Reproducing the Experiment

### Phase 1: Projection vs Baseline Score

```bash
uv run python axbench/scripts/projection_vs_steering_factor.py \
    --behavior corrigible-neutral-HHH \
    --model_name google/gemma-2-9b-it \
    --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep \
    --train_dataset_path datasets/generated/gemma-2-9b-it/corrigible-neutral-HHH/train_contrastive.json
```

Requires GPU (runs model inference to extract activations). Outputs go to `{sweep_dir}/projection_analysis/`.

### Phase 2: Projection vs Steerability

```bash
uv run python axbench/scripts/projection_steerability_link.py \
    --behavior corrigible-neutral-HHH \
    --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep
```

No GPU needed (reads existing eval data). Outputs go to `{sweep_dir}/steerability_link/`.

### Running All Four Behaviors

```bash
for beh in hallucination corrigible-neutral-HHH myopic-reward survival-instinct; do
  uv run python axbench/scripts/projection_vs_steering_factor.py \
      --behavior "$beh" \
      --model_name google/gemma-2-9b-it \
      --sweep_dir "results/gemma-2-9b-it/${beh}-sweep" \
      --train_dataset_path "datasets/generated/gemma-2-9b-it/${beh}/train_contrastive.json"
done

for beh in hallucination corrigible-neutral-HHH myopic-reward survival-instinct; do
  uv run python axbench/scripts/projection_steerability_link.py \
      --behavior "$beh" \
      --sweep_dir "results/gemma-2-9b-it/${beh}-sweep"
done
```

---

## 9. Output Files Reference

### Phase 1 (`projection_analysis/`)

| File | Description |
|------|-------------|
| `test_projections.csv` | Per (prompt, layer): raw projection, normalized projection, baseline score |
| `train_projections.csv` | Per (training example, layer): raw projection, normalized projection, label |
| `centroids.csv` | Per layer: c+, c-, midpoint, half-range |
| `correlations.csv` | Per layer: Spearman rho, Pearson r, p-values, sample size |
| `analysis_summary.json` | Full summary including all parameters and centroid values |
| `plots/scatter_layer_{N}.png` | Scatter of normalized projection vs baseline score |
| `plots/distribution_layer_{N}.png` | Overlaid histograms of train-pos, train-neg, and test projections |
| `plots/correlation_across_layers.png` | Spearman rho across layers with significance markers |

### Phase 2 (`steerability_link/`)

| File | Description |
|------|-------------|
| `per_prompt_steerability.csv` | Per (prompt, layer): projection, baseline, best steered, delta, group |
| `within_layer_results.csv` | Per layer: rho(proj, best_steered), rho(proj, delta), Mann-Whitney U, Cohen's d |
| `layer_metrics.csv` | Per layer: d', aggregate steerability, projection-baseline rho |
| `cross_layer_correlations.csv` | Meta-correlations of rho/d' vs steerability across layers (supplementary) |
| `summary.json` | Full summary with parameters and results |
| `plots/within_layer_rho.png` | Per-prompt rho across layers with significance markers |
| `plots/group_comparison.png` | Cohen's d bar chart for high vs low projection groups |
| `plots/scatter_layer_{N}.png` | Scatter of projection vs steerability for top-K layers |
| `plots/metrics_by_layer.png` | Multi-panel overview: rho, d', and delta across layers |
