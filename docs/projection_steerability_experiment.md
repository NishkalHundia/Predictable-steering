# Test-Time Projection Analysis: Does the Steering Direction Predict Behavior and Steerability?

## 1. Motivation

Activation steering via DiffMean computes a steering vector **v** from contrastive training examples and adds a scaled version of it to a model's hidden states at inference time to modulate a target behavior. A layer sweep evaluates many layers and steering factors to find the best configuration.

This raises a natural question: **does the quality of the projection-behavior alignment at a given layer predict how well steering works at that layer?**

If the steering vector at layer l truly captures a meaningful behavioral axis — evidenced by the fact that unsteered test prompts which project closer to the positive training centroid already exhibit the target behavior more — then we'd expect layer l to also be a layer where actively applying that vector as a steering intervention is more effective.

This experiment tests that link across layers.

---

## 2. Setup

- **Model:** Gemma-2-9B-IT
- **Layers analyzed:** 10 through 32 (23 layers), covering middle to final transformer blocks
- **Behaviors:** `corrigible-neutral-HHH`, `hallucination`, `myopic-reward`, `survival-instinct`
- **Test set:** 32 prompts per behavior, each with an unsteered generation and steered generations at multiple factors, all scored by an LM judge

---

## 3. Mathematical Formalization

### 3.1 Training Phase: Computing the Steering Vector

Given a set of contrastive training pairs {(x_i, y_i^+, y_i^-)}:

- **Positive centroid:** mu_pos = mean of response-token activations for all y_i^+ at layer l
- **Negative centroid:** mu_neg = mean of response-token activations for all y_i^- at layer l
- **Steering vector:** v = mu_pos - mu_neg
- **Unit steering vector:** v_hat = v / ||v||

### 3.2 Training Separability: d'

The **d' (discriminability index)** measures how well-separated the positive and negative training distributions are when projected onto v_hat. It is the signal-to-noise ratio of the centroid gap:

```
d' = (mu_pos . v_hat - mu_neg . v_hat) / sqrt( (sigma_pos^2 + sigma_neg^2) / 2 )
```

where sigma_pos and sigma_neg are the standard deviations of the positive and negative training projections onto v_hat. The numerator is the gap between centroids in the steering direction; the denominator is the pooled within-class spread. A high d' means the training examples are cleanly linearly separable along v_hat; a low d' means the two distributions heavily overlap even in the direction the vector points.

Crucially, d' is computed entirely from **training data** — it says nothing directly about test prompts or whether the direction is behaviorally meaningful in deployment. It is a per-layer quantity, so we can ask whether layers with higher d' also tend to steer better.

### 3.3 Centroid Normalization (per layer)

Project training centroids onto the steering direction:

```
c+ = mu_pos . v_hat
c- = mu_neg . v_hat
```

Define:

```
m = (c+ + c-) / 2         (midpoint)
h = (c+ - c-) / 2         (half-range)
```

The **normalized projection** of any vector a onto layer l's steering direction is:

```
p_tilde = (a . v_hat - m) / h
```

Under this transform, c+ -> +1, c- -> -1, midpoint -> 0. A value of +2 means the point sits as far beyond the positive centroid as the centroid gap itself.

### 3.3 Test-Time Projection

For test prompt x_j with unsteered generation y_j at layer l:

1. Forward pass on [x_j, y_j]; extract hidden states at the response tokens only
2. Average across response token positions -> mean_act_j
3. Raw projection: p_j = mean_act_j . v_hat
4. Normalized: p_tilde_j = (p_j - m) / h

### 3.4 Projection-Baseline Correlation (per layer)

For each layer l, compute the Spearman rank correlation across all test prompts:

```
rho_l = Spearman( p_tilde_j, baseline_score_j )    over j = 1..32
```

where baseline_score_j is the LM judge's score for prompt j's **unsteered** generation.

A high rho_l means the steering vector at layer l genuinely encodes behavioral variance: prompts that already "sit" closer to the positive training centroid also behave more like the target behavior without any intervention.

### 3.5 Normalized Steering Delta (per layer)

For each layer l, compute per-prompt improvement first, normalize by each prompt's own headroom, then average:

```
delta_j          = best_steered_score_j - baseline_score_j
headroom_j       = scale_max - baseline_score_j
norm_delta_j     = delta_j / headroom_j

normalized_delta_l = mean( norm_delta_j )    over j = 1..32
```

Normalizing per-prompt before averaging is important: a prompt with baseline +4 on a [-5, +5] scale only has 1 point of headroom, while a prompt with baseline -3 has 8 points. Using an aggregate headroom denominator would confound the steering signal with the baseline distribution. Per-prompt normalization gives each prompt equal weight regardless of where it started. A value of 1.0 means every prompt reached the ceiling.

---

## 4. Hypothesis

> **Layers where the projection better predicts baseline behavior (higher rho_l) should also be layers where steering is more effective (higher normalized_delta_l).**

Formally:

```
Spearman( rho_l, normalized_delta_l )  >>  0    over l = 1..23 layers
```

**Intuition:** If rho_l is high, the DiffMean vector at layer l has found a real behavioral axis — the direction genuinely encodes the target concept in the geometry of that layer's representations. Applying that vector as a steering intervention should then work well, because you're actually pushing along a meaningful semantic direction. If rho_l is near zero, the vector is pointing at something that doesn't cleanly correspond to the behavior, and steering along it is unlikely to reliably produce the target behavior.

---

## 5. Results

### 5.1 Does rho_l predict normalized_delta_l?

**No — for none of the four behaviors does the cross-layer correlation reach significance.**

| Behavior | rho(proj_baseline_rho, normalized_delta) | p-value |
|----------|------------------------------------------|---------|
| corrigible-neutral-HHH | -0.235 | 0.280 |
| hallucination | +0.363 | 0.089 |
| myopic-reward | -0.161 | 0.463 |
| survival-instinct | +0.239 | 0.272 |

For corrigible-neutral-HHH the correlation is even slightly negative, despite rho_l being strong and consistent across all layers (0.455 to 0.789). The likely reason: both proj_baseline_rho and normalized_delta have little variance across layers for this behavior — steerability is uniformly high (65–97%) regardless of layer, so there's nothing to predict.

### 5.2 Does d' predict normalized_delta_l?

**Yes — and the results are striking and behavior-dependent.**

| Behavior | rho(d', normalized_delta) | p-value | Direction |
|----------|--------------------------|---------|-----------|
| corrigible-neutral-HHH | +0.373 | 0.080 | Positive (borderline) |
| hallucination | +0.476 | 0.022* | Positive |
| myopic-reward | **-0.748** | <0.001* | **Strongly negative** |
| survival-instinct | +0.626 | 0.001* | Positive |

For hallucination and survival-instinct, d' is a strong positive predictor of steerability — layers where training examples are more separable also produce better steering. For myopic-reward, the relationship inverts completely: the more separable the training distributions at a layer, the *worse* that layer steers (rho = -0.748, p < 0.001). d' also correlates strongly with other steerability metrics beyond normalized_delta:

**hallucination:** d' → auc_positive rho = +0.680 (p < 0.001), d' → efficiency rho = +0.631 (p = 0.001)

**survival-instinct:** d' → auc_positive rho = +0.722 (p < 0.001), d' → score_f5 rho = +0.686 (p < 0.001)

**myopic-reward:** d' → agg_best_score rho = -0.718 (p < 0.001), d' → auc_positive rho = -0.622 (p = 0.002)

### 5.3 Summary

| Behavior | rho_l (proj → baseline) | rho_l → steerability | d' → steerability |
|----------|-------------------------|----------------------|-------------------|
| **corrigible-neutral-HHH** | Strong, 0.45–0.79 across all 23 layers | Not significant (−0.24) | Borderline positive (+0.37) |
| **hallucination** | Near zero most layers; spike at layers 24–25 | Not significant (+0.36) | Positive (+0.48)* |
| **myopic-reward** | Near zero | Not significant (−0.16) | **Strongly negative (−0.75)*** |
| **survival-instinct** | Near zero | Not significant (+0.24) | Strongly positive (+0.63)* |

---

## 6. Interpretation

### rho_l Does Not Predict Steerability Across Layers

The main hypothesis is not supported. Even for corrigible-neutral-HHH — where rho_l is strong (0.45–0.79) and significant at every layer — it does not predict which layers steer well. The reason appears to be a floor/ceiling effect: for corrigible-neutral-HHH, every layer steers well (normalized_delta 65–97%), leaving no variance for rho_l to predict. When both variables are uniformly high, cross-layer correlation becomes uninformative.

For the other three behaviors, rho_l is near zero everywhere, so again there's no signal to correlate.

### d' Is the Better Layer-Level Predictor — When It Works

d' (training separability) predicts steerability significantly for hallucination and survival-instinct, and borderline for corrigible-neutral-HHH. This suggests that for most behaviors, how cleanly the training examples separate along the steering direction at a given layer is a useful proxy for how well that layer will steer.

### myopic-reward: A Confounded Direction

The strongly negative d'-steerability correlation for myopic-reward (rho = -0.748) is the most informative result. The layers where training examples are most linearly separable are the layers where steering works *worst*. This is consistent with the DiffMean vector overfitting to a spurious feature in the training contrastive pairs — some superficial difference between positive and negative examples that gets more pronounced at certain layers but has nothing to do with myopic reward behavior. The more "confident" the separator, the more off-target the steering direction.

### The Role of rho_l Revisited

rho_l remains useful as a test-time diagnostic: for corrigible-neutral-HHH, the fact that it is high and significant at every layer confirms the steering vector genuinely encodes the behavioral axis. But high rho_l at the layer level does not, by itself, mean that layer steers better than other layers — it just means the direction is meaningful. Whether that translates into steerability depends on other factors (model depth, representation geometry) that d' partially captures.
