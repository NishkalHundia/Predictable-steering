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

The d' (discriminability index) measures training separability along v_hat:

```
d' = (mu_pos . v_hat - mu_neg . v_hat) / sqrt( (sigma_pos^2 + sigma_neg^2) / 2 )
```

### 3.2 Centroid Normalization

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

### corrigible-neutral-HHH

**Strong positive relationship.** rho_l ranges from +0.455 to +0.789 across layers, all significant. Layers with the highest projection-baseline alignment (layers 21-27) also tend to produce the best steering outcomes.

### hallucination

**Weak projection-baseline correlation overall** (near zero rho_l at most layers), but a notable exception at layers 24-25 where rho spikes and steering effectiveness also peaks. The relationship holds locally where it exists.

### myopic-reward

**No projection-baseline correlation at any layer.** Correspondingly, d' (training separability) anti-correlates with steerability (rho ~ -0.70): the layers where training examples are most linearly separable actually steer *worse*. The DiffMean direction appears to capture a training-specific confound that doesn't generalize.

### survival-instinct

**No projection-baseline correlation.** But d' does positively predict aggregate steerability (rho ~ +0.65), meaning training separability is informative even when projection-to-behavior alignment is not.

### Summary

| Behavior | rho_l -> baseline (H1) | rho_l -> normalized_delta |
|----------|------------------------|--------------------------|
| **corrigible-neutral-HHH** | Strong (up to +0.79, 23/23 layers sig) | Positive |
| **hallucination** | Weak overall, spike at layers 24-25 | Locally positive |
| **myopic-reward** | None | None (d' negative) |
| **survival-instinct** | None | None (but d' positive) |

---

## 6. Interpretation

### Why Does Alignment Predict Steerability?

When rho_l is high, the DiffMean vector at layer l has successfully isolated a direction in activation space that encodes genuine behavioral variance across prompts — not just a direction that separates training examples. Applying that same vector as a steering intervention then moves activations along an axis the model "uses" for that behavior, producing reliable behavioral change.

When rho_l is near zero, the DiffMean direction may be picking up training confounds (topic, formatting, length differences between positive and negative examples). Steering along a confounded direction doesn't reliably produce the target behavior.

### d' Is Not Enough

d' measures how separable the training examples are along v_hat, but this separation can be driven by confounds rather than the target behavior. A high d' with low rho_l is the tell: the model has found a linear separator for the training data, but it's not the behavioral axis. **rho_l is the better diagnostic** because it validates the direction on held-out test prompts against an independent behavioral signal.

### myopic-reward Is the Clearest Failure Case

The negative d'-steerability correlation is strong evidence that the DiffMean vector for myopic-reward is capturing something other than the target concept. The more separable the training examples are at a layer, the worse steering works — consistent with overfitting to a spurious training-set feature.
