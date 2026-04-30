"""
Experiment 1: Directional Information (R^2 ratio).

Run TWO methods side-by-side and save plots / CSVs for each:

  Method 1 ("test_only"):
    v from training prompts; R^2(v), R^2(h) fit on TEST prompts only.
    Test labels are the LM-judge baseline behavior scores (factor = 0).

  Method 2 ("train_plus_test"):
    v from training prompts (same v); R^2(v), R^2(h) fit on TRAIN + TEST
    prompts combined. Training labels y_b are mapped from the contrastive
    label to the extremes of the behavior's score range
      label = pos -> y_b = score_max,
      label = neg -> y_b = score_min,
    so train activations join test activations in the same y_b space.

For each layer l (both methods):
  1. v^l = (mu_pos^l - mu_neg^l) / ||...||  from training activations only.
     Centroid midpoint m and half-range h come from the training projections.
  2. For each prompt in the chosen dataset, compute
        p_v(h_i) = (h_i . v - m) / h.
  3. Fit on that dataset:
        y_b ~ p_v(h)  (univariate OLS)              -> R^2(v)
        y_b ~ h       (RidgeCV on full activation)  -> R^2(h)
  4. Ratio = R^2(v) / R^2(h)  (clip negative R^2 to 0 before dividing).
  6. Pull steering effectiveness Delta_l from eval_results.parquet:
        Delta_l = mean(score | factor > 0, layer = l)
                  - mean(score | factor = 0, layer = l)
     and (optionally) normalize by the behavior's score range.

Outputs:
  {sweep_dir}/directional_information_analysis/
    train_activations.pt
    test_activations.pt
    r2_per_layer__test_only.csv
    r2_per_layer__train_plus_test.csv
    delta_l.csv
    analysis_summary.json
    plots/
      layer_sweep__test_only__{behavior}.png
      layer_sweep__train_plus_test__{behavior}.png
      ratio_vs_steering__test_only__{behavior}.png
      ratio_vs_steering__train_plus_test__{behavior}.png

Usage:
    uv run python axbench/scripts/projection_tactics/directional_information.py \\
        --behavior corrigible-neutral-HHH \\
        --model_name google/gemma-2-9b-it \\
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep \\
        --train_dataset_path datasets/generated/gemma-2-9b-it/corrigible-neutral-HHH/train_contrastive.json
"""
import sys
import json
import shutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats as scipy_stats
from sklearn.linear_model import RidgeCV, LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")

# Same scales as the rest of projection_tactics/.
BEHAVIOR_SCALES = {
    "survival-instinct": (-5, 5, 0),
    "myopic-reward": (-5, 5, 0),
    "corrigible-neutral-HHH": (-5, 5, 0),
    "hallucination": (0, 5, None),
    "sycophancy": (0, 10, 5),
    "refusal": (0, 10, 5),
    "coordinate-other-ais": (0, 10, 5),
}
BEHAVIORS = list(BEHAVIOR_SCALES.keys())


# ============================================================================
# Helpers (shared with projection_diffmean / projection_lda)
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def discover_layers(sweep_dir: Path) -> list[int]:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            has_eval = (d / "eval" / "eval_results.parquet").exists()
            if has_eval:
                layers.append(int(d.name.split("_")[1]))
    return sorted(layers)


def gather_multi_layer_activations(model, target_layers, inputs):
    layer_acts = {}
    handles = []
    for layer_idx in target_layers:
        def _make_hook(l):
            def hook(mod, inp, out):
                layer_acts[l] = out[0].detach()
            return hook
        h = model.model.layers[layer_idx].register_forward_hook(
            _make_hook(layer_idx), always_call=True
        )
        handles.append(h)
    _ = model.forward(**inputs)
    for h in handles:
        h.remove()
    return layer_acts


def _format_texts(tokenizer, use_chat, question, answer):
    if use_chat:
        q_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question},
             {"role": "assistant", "content": str(answer)}],
            tokenize=False, add_generation_prompt=False,
        )
    else:
        q_text = question
        full_text = question + "\n\n" + str(answer)
    return q_text, full_text


def _get_mean_response_act(model, tokenizer, target_layers, question, answer,
                           use_chat, device):
    q_text, full_text = _format_texts(tokenizer, use_chat, question, answer)

    full_inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=1024,
    ).to(device)
    q_inputs = tokenizer(
        q_text, return_tensors="pt", truncation=True, max_length=1024,
    ).to(device)

    q_len = q_inputs["input_ids"].shape[1]
    f_len = full_inputs["input_ids"].shape[1]
    if f_len <= q_len:
        return None

    layer_acts = gather_multi_layer_activations(
        model, target_layers,
        {"input_ids": full_inputs["input_ids"],
         "attention_mask": full_inputs["attention_mask"]},
    )
    result = {}
    for layer in target_layers:
        resp_acts = layer_acts[layer].float()[0, q_len:f_len]
        result[layer] = resp_acts.mean(dim=0).cpu()
    return result


# ============================================================================
# Phase 1 - training activations
# ============================================================================
@torch.no_grad()
def extract_train_activations(model, tokenizer, train_dataset_path,
                              target_layers, device):
    use_chat = supports_chat_template(tokenizer)
    model.eval()
    with open(train_dataset_path, encoding="utf-8") as f:
        train_data = json.load(f)
    logger.warning(f"Extracting training activations for {len(train_data)} pairs")

    train_acts = {l: {"pos": [], "neg": []} for l in target_layers}
    train_meta = []

    for pair_idx, item in enumerate(tqdm(train_data, desc="Train activations")):
        for label, answer_key in [(1, "answer_matching_behavior"),
                                  (0, "answer_not_matching_behavior")]:
            question = item["question"]
            answer = item[answer_key]
            mean_acts = _get_mean_response_act(
                model, tokenizer, target_layers, question, str(answer),
                use_chat, device,
            )
            if mean_acts is None:
                continue
            key = "pos" if label == 1 else "neg"
            for layer in target_layers:
                train_acts[layer][key].append(mean_acts[layer])
            train_meta.append({
                "pair_idx": pair_idx,
                "label": label,
                "label_name": "pos" if label == 1 else "neg",
            })
        if torch.cuda.is_available() and pair_idx % 5 == 0:
            torch.cuda.empty_cache()

    for layer in target_layers:
        logger.warning(
            f"  Layer {layer}: {len(train_acts[layer]['pos'])} pos, "
            f"{len(train_acts[layer]['neg'])} neg"
        )
    return train_acts, train_meta


# ============================================================================
# Phase 2 - DiffMean direction + centroid normalization
# ============================================================================
def compute_diffmean_directions(train_acts, target_layers):
    """v^l = (mu_pos - mu_neg) / ||...||  per layer.

    Also returns per-layer centroid midpoint m and half-range h (from
    train projections) for the same centroid normalization used in
    projection_diffmean / projection_lda.
    """
    directions = {}
    centroid_norm = {}
    for layer in target_layers:
        pos = torch.stack(train_acts[layer]["pos"])
        neg = torch.stack(train_acts[layer]["neg"])
        mu_pos = pos.mean(dim=0)
        mu_neg = neg.mean(dim=0)
        diff = mu_pos - mu_neg
        v = diff / max(diff.norm().item(), 1e-12)

        c_pos = mu_pos.dot(v).item()
        c_neg = mu_neg.dot(v).item()
        m = 0.5 * (c_pos + c_neg)
        h = 0.5 * (c_pos - c_neg)

        directions[layer] = v
        centroid_norm[layer] = {
            "c_pos": c_pos, "c_neg": c_neg, "midpoint": m, "half_range": h,
        }
        logger.warning(
            f"  Layer {layer:2d}: ||diff|| = {diff.norm().item():.3f}, "
            f"c+ = {c_pos:.3f}, c- = {c_neg:.3f}, m = {m:.3f}, h = {h:.3f}"
        )
    return directions, centroid_norm


# ============================================================================
# Phase 3 - test activations + projection
# ============================================================================
@torch.no_grad()
def extract_test_activations(model, tokenizer, sweep_dir, target_layers, device):
    """For each test prompt's unsteered (factor=0) generation, store the full
    mean response-token activation h_i^l + the associated baseline score."""
    use_chat = supports_chat_template(tokenizer)
    model.eval()
    sweep_dir = Path(sweep_dir)

    first_layer = target_layers[0]
    eval_df = pd.read_parquet(
        sweep_dir / f"layer_{first_layer}" / "eval" / "eval_results.parquet"
    )
    baseline_df = eval_df[eval_df["steering_factor"] == 0].copy().reset_index(drop=True)
    if baseline_df.empty:
        closest = eval_df.loc[eval_df["steering_factor"].abs().idxmin(), "steering_factor"]
        baseline_df = eval_df[eval_df["steering_factor"] == closest].copy().reset_index(drop=True)
        logger.warning(f"No factor=0 found; using factor={closest}")

    baseline_scores = {}
    for layer in target_layers:
        layer_eval = pd.read_parquet(
            sweep_dir / f"layer_{layer}" / "eval" / "eval_results.parquet"
        )
        f0 = layer_eval[layer_eval["steering_factor"] == 0]
        baseline_scores[layer] = dict(zip(f0["question_idx"], f0["behavior_score"]))

    logger.warning(f"Extracting test activations for {len(baseline_df)} prompts")

    test_mean_acts = {}   # (question_idx, layer) -> tensor
    test_metadata = []

    for idx in tqdm(range(len(baseline_df)), desc="Test activations"):
        row = baseline_df.iloc[idx]
        question = row["question"]
        generation = row["generation"]
        question_idx = int(row.get("question_idx", idx))

        if pd.isna(generation) or str(generation).strip() == "":
            continue

        mean_acts = _get_mean_response_act(
            model, tokenizer, target_layers, question, str(generation),
            use_chat, device,
        )
        if mean_acts is None:
            continue

        for layer in target_layers:
            act = mean_acts[layer]
            bs = baseline_scores.get(layer, {}).get(question_idx, np.nan)
            test_mean_acts[(question_idx, int(layer))] = act.cpu()
            test_metadata.append({
                "question_idx": question_idx,
                "layer": int(layer),
                "baseline_score": bs,
                "question": question,
                "generation": str(generation),
            })

        if torch.cuda.is_available() and idx % 5 == 0:
            torch.cuda.empty_cache()

    logger.warning(f"Extracted {len(test_metadata)} (prompt, layer) entries")
    return test_mean_acts, test_metadata


def stack_layer_test(test_mean_acts, test_metadata, layer):
    """Per-layer (H, y, qids) arrays for the regressions, TEST prompts only.
    y is the LM-judge baseline behavior score."""
    H, y, qids = [], [], []
    for entry in test_metadata:
        if int(entry["layer"]) != int(layer):
            continue
        bs = entry["baseline_score"]
        if bs is None or (isinstance(bs, float) and np.isnan(bs)):
            continue
        act = test_mean_acts.get((entry["question_idx"], int(layer)))
        if act is None:
            continue
        H.append(act.numpy())
        y.append(float(bs))
        qids.append(int(entry["question_idx"]))
    if not H:
        return np.zeros((0, 0)), np.zeros(0), []
    return np.stack(H, axis=0), np.array(y, dtype=float), qids


def train_label_to_yb(behavior, label):
    """Map a binary contrastive label to a y_b value at the behavior's
    score-range extremes (label = 1 -> max, label = 0 -> min)."""
    score_min, score_max, _ = BEHAVIOR_SCALES.get(behavior, (0.0, 1.0, None))
    return float(score_max) if int(label) == 1 else float(score_min)


def stack_layer_train(train_acts, behavior, layer):
    """Per-layer (H, y) for training prompts, with y_b mapped to the
    behavior's score-range extremes."""
    H, y = [], []
    pos_acts = train_acts.get(layer, {}).get("pos", [])
    neg_acts = train_acts.get(layer, {}).get("neg", [])
    yb_pos = train_label_to_yb(behavior, 1)
    yb_neg = train_label_to_yb(behavior, 0)
    for a in pos_acts:
        H.append(a.numpy())
        y.append(yb_pos)
    for a in neg_acts:
        H.append(a.numpy())
        y.append(yb_neg)
    if not H:
        return np.zeros((0, 0)), np.zeros(0)
    return np.stack(H, axis=0), np.array(y, dtype=float)


def stack_layer_combined(test_mean_acts, test_metadata, train_acts,
                         behavior, layer):
    """Per-layer (H, y) for TRAIN + TEST. Train rows use the mapped extreme
    y_b; test rows use the LM-judge baseline behavior score."""
    H_test, y_test, _ = stack_layer_test(test_mean_acts, test_metadata, layer)
    H_train, y_train = stack_layer_train(train_acts, behavior, layer)
    if H_test.size == 0 and H_train.size == 0:
        return np.zeros((0, 0)), np.zeros(0)
    if H_test.size == 0:
        return H_train, y_train
    if H_train.size == 0:
        return H_test, y_test
    return np.concatenate([H_train, H_test], axis=0), np.concatenate([y_train, y_test])


# ============================================================================
# Phase 4 / 5 - R^2 computations
# ============================================================================
def _r2_score(y, y_pred):
    """Standard 1 - SS_res/SS_tot."""
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = float(((y - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def compute_r2_metrics(test_mean_acts, test_metadata, directions,
                       centroid_norm, target_layers, train_acts, behavior,
                       mode="test_only",
                       ridge_alphas=(1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)):
    """Per layer, fit R^2(v) and R^2(h) on the dataset selected by `mode`:

      - "test_only":      regression dataset = test prompts only
                          (y_b = LM-judge baseline score)
      - "train_plus_test": regression dataset = train + test prompts
                           (train y_b mapped to behavior score extremes;
                            test y_b = LM-judge baseline score)

    v is always the DiffMean direction computed from the training set.
    """
    if mode not in ("test_only", "train_plus_test"):
        raise ValueError(f"Unknown mode: {mode}")

    logger.warning(f"--- compute_r2_metrics (mode = {mode}) ---")

    rows = []
    for layer in target_layers:
        # Always pull both test and train so we can log sizes consistently.
        H_test, y_test, _ = stack_layer_test(test_mean_acts, test_metadata, layer)
        H_train, y_train = stack_layer_train(train_acts, behavior, layer)
        n_test = len(y_test)
        n_train = len(y_train)
        n_train_pos = len(train_acts.get(layer, {}).get("pos", []))
        n_train_neg = len(train_acts.get(layer, {}).get("neg", []))

        if mode == "test_only":
            H_fit, y_fit = H_test, y_test
        else:
            if H_test.size == 0:
                H_fit, y_fit = H_train, y_train
            elif H_train.size == 0:
                H_fit, y_fit = H_test, y_test
            else:
                H_fit = np.concatenate([H_train, H_test], axis=0)
                y_fit = np.concatenate([y_train, y_test])
        n_fit = len(y_fit)
        d = H_fit.shape[1] if n_fit else 0

        logger.warning(
            f"  Layer {int(layer):2d} [{mode}]: "
            f"v from {n_train_pos} pos / {n_train_neg} neg train  |  "
            f"regression on n_fit = {n_fit} "
            f"({'test only' if mode == 'test_only' else f'{n_train} train + {n_test} test'})  "
            f"d(h) = {d}"
        )

        if n_fit < 4:
            rows.append({
                "layer": int(layer), "mode": mode,
                "n_train_pos": int(n_train_pos), "n_train_neg": int(n_train_neg),
                "n_test": int(n_test), "n_fit": int(n_fit), "d_h": int(d),
                "r2_v": np.nan, "r2_h": np.nan, "ratio": np.nan,
                "pearson_r_v": np.nan, "spearman_rho_v": np.nan,
                "ridge_alpha": np.nan,
            })
            continue

        v = directions[layer].numpy().astype(np.float64)
        m = centroid_norm[layer]["midpoint"]
        h = centroid_norm[layer]["half_range"]

        proj_raw = H_fit @ v
        proj = (proj_raw - m) / (h if abs(h) > 1e-12 else 1.0)

        # R^2(v) = in-sample R^2 of y_b ~ p_v(h)  (univariate OLS).
        pr, _ = scipy_stats.pearsonr(proj, y_fit)
        sr, _ = scipy_stats.spearmanr(proj, y_fit)
        r2_v = float(pr * pr)

        # R^2(h) = in-sample R^2 of y_b ~ h  with RidgeCV.
        ridge = RidgeCV(alphas=ridge_alphas)
        ridge.fit(H_fit.astype(np.float64), y_fit)
        y_pred_h = ridge.predict(H_fit.astype(np.float64))
        r2_h = _r2_score(y_fit, y_pred_h)
        ridge_alpha = float(getattr(ridge, "alpha_", np.nan))

        r2_v_clip = max(r2_v, 0.0) if not np.isnan(r2_v) else np.nan
        r2_h_clip = max(r2_h, 0.0) if not np.isnan(r2_h) else np.nan
        if (not np.isnan(r2_v_clip) and not np.isnan(r2_h_clip)
                and r2_h_clip > 1e-6):
            ratio = r2_v_clip / r2_h_clip
        else:
            ratio = np.nan

        rows.append({
            "layer": int(layer), "mode": mode,
            "n_train_pos": int(n_train_pos), "n_train_neg": int(n_train_neg),
            "n_test": int(n_test), "n_fit": int(n_fit), "d_h": int(d),
            "r2_v": float(r2_v), "r2_h": float(r2_h),
            "r2_v_clip": float(r2_v_clip), "r2_h_clip": float(r2_h_clip),
            "ratio": float(ratio) if not np.isnan(ratio) else np.nan,
            "pearson_r_v": float(pr), "spearman_rho_v": float(sr),
            "ridge_alpha": ridge_alpha,
        })
        logger.warning(
            f"  Layer {int(layer):2d} [{mode}]: "
            f"R^2(v) = {r2_v:+.3f}  R^2(h) = {r2_h:+.3f}  "
            f"ratio = {ratio if not np.isnan(ratio) else float('nan'):+.3f}  "
            f"(ridge alpha = {ridge_alpha:.2g})"
        )
    return pd.DataFrame(rows)


# ============================================================================
# Phase 6 - steering effectiveness Delta_l
# ============================================================================
def compute_delta_l(sweep_dir, target_layers, behavior, normalize=True):
    """Per layer, steering-effectiveness Delta_l.

    Matches the convention in sweep_layers_open_ended.py:summarize_eval and
    its downstream `factor_max_avg - factor_0_avg` reporting:

        Delta_l = max_f mean(score | factor = f) - mean(score | factor = 0)

    i.e. the best-achieved mean steered score across the swept factors,
    minus the baseline. Averaging across all positive factors (the older
    formulation) is not what the rest of the pipeline calls 'steering
    effectiveness': at deep layers, large factors often degrade output and
    drag that average back to baseline.

    Also returns the older mean-of-positive-factors variant under
    `delta_l_mean_pos` for reference.

    If `normalize=True`, both Delta_l columns are divided by the behavior's
    score range (max - min).
    """
    sweep_dir = Path(sweep_dir)
    score_min, score_max, _ = BEHAVIOR_SCALES.get(behavior, (None, None, None))
    score_range = (score_max - score_min) if (score_min is not None and score_max is not None) else 1.0

    rows = []
    for layer in target_layers:
        path = sweep_dir / f"layer_{layer}" / "eval" / "eval_results.parquet"
        row = {
            "layer": int(layer),
            "n_baseline": 0, "n_steered": 0,
            "baseline_mean": np.nan,
            "best_factor": np.nan, "best_factor_mean": np.nan,
            "steered_mean_pos": np.nan,
            "delta_l_raw": np.nan, "delta_l": np.nan,
            "delta_l_mean_pos_raw": np.nan, "delta_l_mean_pos": np.nan,
        }
        if not path.exists():
            rows.append(row)
            continue

        df = pd.read_parquet(path)
        base = df[df["steering_factor"] == 0]["behavior_score"].dropna()
        if len(base) == 0:
            rows.append(row)
            continue
        baseline_mean = float(base.mean())
        row.update({
            "n_baseline": int(len(base)),
            "baseline_mean": baseline_mean,
        })

        # Per-factor mean across all swept factors.
        per_factor = (
            df.dropna(subset=["behavior_score"])
              .groupby("steering_factor")["behavior_score"]
              .mean()
              .sort_index()
        )
        if per_factor.empty:
            rows.append(row)
            continue

        # Best factor in absolute terms (max avg_score).
        best_factor = float(per_factor.idxmax())
        best_factor_mean = float(per_factor.max())
        delta_raw = best_factor_mean - baseline_mean

        # Mean of positive-factor scores (legacy variant).
        pos_factors = per_factor[per_factor.index > 0]
        if not pos_factors.empty:
            steered_mean_pos = float(pos_factors.mean())
            delta_mp_raw = steered_mean_pos - baseline_mean
            n_steered = int(
                df[df["steering_factor"] > 0]["behavior_score"].notna().sum()
            )
        else:
            steered_mean_pos = np.nan
            delta_mp_raw = np.nan
            n_steered = 0

        row.update({
            "n_steered": n_steered,
            "best_factor": best_factor,
            "best_factor_mean": best_factor_mean,
            "steered_mean_pos": steered_mean_pos,
            "delta_l_raw": delta_raw,
            "delta_l": delta_raw / score_range if normalize else delta_raw,
            "delta_l_mean_pos_raw": delta_mp_raw,
            "delta_l_mean_pos": (
                delta_mp_raw / score_range if (normalize and not np.isnan(delta_mp_raw))
                else delta_mp_raw
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================
METHOD_LABELS = {
    "test_only": "Method 1: v from train; R^2 fit on TEST only",
    "train_plus_test": "Method 2: v from train; R^2 fit on TRAIN + TEST",
}


def plot_layer_sweep(metrics_df, behavior, output_path, method,
                     layer_lo=10, layer_hi=32):
    """Plot 1: 3-panel layer sweep (top R^2(h), middle R^2(v), bottom ratio)."""
    df = metrics_df.sort_values("layer").reset_index(drop=True)
    layers = df["layer"].values

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)

    # Top: R^2(h)
    ax = axes[0]
    ax.plot(layers, df["r2_h"].values, "o-", color="#1f77b4", linewidth=2,
            markersize=6, label=r"$R^2(h)$")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel(r"$R^2(h)$  (full activation)", fontsize=10)
    ax.set_title(
        f"{behavior} - Layer sweep of directional information\n"
        f"{METHOD_LABELS.get(method, method)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)

    # Middle: R^2(v)
    ax = axes[1]
    ax.plot(layers, df["r2_v"].values, "o-", color="#2ca02c", linewidth=2,
            markersize=6, label=r"$R^2(v)$")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel(r"$R^2(v)$  (DiffMean projection)", fontsize=10)
    ax.legend(loc="best", fontsize=9)

    # Bottom: ratio R^2(v) / R^2(h)
    ax = axes[2]
    ax.plot(layers, df["ratio"].values, "o-", color="#d62728", linewidth=2,
            markersize=6, label=r"$R^2(v)/R^2(h)$")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.7,
               label="ratio = 1")
    ax.set_ylabel(r"$R^2(v)/R^2(h)$", fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.legend(loc="best", fontsize=9)

    # Annotate the per-layer fit-set sizes (constant across layers in
    # practice, but captured in the metrics so we surface them).
    if not df.empty:
        n_fit = int(df["n_fit"].dropna().iloc[0]) if df["n_fit"].notna().any() else 0
        n_train_pos = int(df["n_train_pos"].dropna().iloc[0]) if df["n_train_pos"].notna().any() else 0
        n_train_neg = int(df["n_train_neg"].dropna().iloc[0]) if df["n_train_neg"].notna().any() else 0
        n_test = int(df["n_test"].dropna().iloc[0]) if df["n_test"].notna().any() else 0
        info = (f"v from train: {n_train_pos} pos / {n_train_neg} neg   "
                f"|   regression n_fit = {n_fit}   "
                f"|   test n = {n_test}")
        fig.text(0.5, 0.005, info, ha="center", fontsize=8.5,
                 color="dimgray")

    for a in axes:
        a.set_xlim(layer_lo - 0.5, layer_hi + 0.5)
        a.set_xticks(np.arange(layer_lo, layer_hi + 1))
        a.grid(True, alpha=0.4)

    plt.tight_layout(rect=(0, 0.025, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ratio_vs_steering(metrics_df, delta_df, behavior, output_path, method):
    """Plot 2: scatter of ratio vs Delta_l, one labeled point per layer."""
    # Only merge in delta_l if metrics_df doesn't already have it.
    if "delta_l" in metrics_df.columns:
        df = metrics_df.copy()
    else:
        df = metrics_df.merge(delta_df[["layer", "delta_l"]], on="layer", how="inner")
    df = df.dropna(subset=["ratio", "delta_l"]).sort_values("layer").reset_index(drop=True)
    if df.empty:
        logger.warning(f"plot_ratio_vs_steering [{method}]: empty after merge / dropna")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(df["ratio"], df["delta_l"], s=80,
               color="#2E86AB", edgecolors="black", linewidths=0.6, zorder=3)

    for _, r in df.iterrows():
        ax.annotate(
            f"{int(r['layer'])}",
            (r["ratio"], r["delta_l"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=9, color="black",
        )

    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.6, linewidth=1,
               label="ratio = 1")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.6, linewidth=1)

    if len(df) >= 4:
        rho, p_rho = scipy_stats.spearmanr(df["ratio"], df["delta_l"])
        n = len(df)
        txt = f"Spearman rho = {rho:+.3f}\np = {p_rho:.1e}\nn = {n}"
        ax.text(
            0.03, 0.97, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.85),
        )

    ax.set_xlabel(r"$R^2(v)/R^2(h)$  (directional information)", fontsize=11)
    ax.set_ylabel(r"$\Delta_l$  (normalized steering improvement)", fontsize=11)
    ax.set_title(
        f"{behavior}: directional information vs steering effectiveness\n"
        f"{METHOD_LABELS.get(method, method)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment 1: Directional information (R^2 ratio)"
    )
    parser.add_argument("--behavior", type=str, required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Path to sweep output dir (contains layer_N/ subdirs)")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="Path to train_contrastive.json")
    parser.add_argument("--layers", type=str, default="10-32",
                        help="Layer range or comma list, e.g. '10-32' or '10,11,12'")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_normalize_delta", action="store_true",
                        help="Do not divide Delta_l by behavior score range")
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip model inference; reuse cached activations + CSV")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = sweep_dir / "directional_information_analysis"
    plot_dir = out_dir / "plots"

    # Clean plots/CSVs/JSON but keep cached .pt files.
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    for csv_file in out_dir.glob("*.csv"):
        csv_file.unlink()
    for json_file in out_dir.glob("*.json"):
        json_file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Resolve layer list.
    if "-" in args.layers and "," not in args.layers:
        lo, hi = [int(x) for x in args.layers.split("-")]
        target_layers = list(range(lo, hi + 1))
    else:
        target_layers = sorted(int(x.strip()) for x in args.layers.split(","))

    # Filter to layers that actually have eval_results.parquet.
    avail = set(discover_layers(sweep_dir))
    target_layers = [l for l in target_layers if l in avail]
    if not target_layers:
        logger.error(f"No valid layers found in {sweep_dir}")
        sys.exit(1)
    layer_lo, layer_hi = min(target_layers), max(target_layers)
    logger.warning(f"Target layers: {target_layers}")

    train_acts_path = out_dir / "train_activations.pt"
    test_acts_path = out_dir / "test_activations.pt"

    # ---------- Phase 1: training activations ---------------------------
    if args.replot_only:
        if not train_acts_path.exists() or not test_acts_path.exists():
            logger.error("--replot_only requires cached train_activations.pt and "
                         "test_activations.pt. Run without --replot_only first.")
            sys.exit(1)
        cached = torch.load(train_acts_path, map_location="cpu", weights_only=False)
        train_acts = cached["train_acts"]
        cached_test = torch.load(test_acts_path, map_location="cpu", weights_only=False)
        test_mean_acts = cached_test["test_mean_acts"]
        test_metadata = cached_test["test_metadata"]
    else:
        model = None
        tokenizer = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _ensure_model_loaded():
            nonlocal model, tokenizer
            if model is not None:
                return
            logger.warning(f"Device: {device}")
            logger.warning(f"Loading model {args.model_name} ...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, model_max_length=1024, padding_side="right",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16 if args.use_bf16 else None,
                device_map=device,
            )
            model.eval()

        if train_acts_path.exists():
            logger.warning(f"Loading cached training activations from {train_acts_path}")
            cached = torch.load(train_acts_path, map_location="cpu", weights_only=False)
            train_acts = cached["train_acts"]
            train_meta = cached["train_meta"]
            # Verify cache covers the requested layers; otherwise re-extract.
            cached_layers = set(train_acts.keys())
            if not set(target_layers).issubset(cached_layers):
                logger.warning("Cached train activations missing some target layers; "
                               "re-extracting.")
                _ensure_model_loaded()
                train_acts, train_meta = extract_train_activations(
                    model, tokenizer, args.train_dataset_path, target_layers, device,
                )
                torch.save({"train_acts": train_acts, "train_meta": train_meta},
                           train_acts_path)
        else:
            _ensure_model_loaded()
            train_acts, train_meta = extract_train_activations(
                model, tokenizer, args.train_dataset_path, target_layers, device,
            )
            torch.save({"train_acts": train_acts, "train_meta": train_meta},
                       train_acts_path)

        if test_acts_path.exists():
            logger.warning(f"Loading cached test activations from {test_acts_path}")
            cached_test = torch.load(test_acts_path, map_location="cpu", weights_only=False)
            test_mean_acts = cached_test["test_mean_acts"]
            test_metadata = cached_test["test_metadata"]
            cached_layers = set(int(e["layer"]) for e in test_metadata)
            if not set(target_layers).issubset(cached_layers):
                logger.warning("Cached test activations missing some target layers; "
                               "re-extracting.")
                _ensure_model_loaded()
                test_mean_acts, test_metadata = extract_test_activations(
                    model, tokenizer, sweep_dir, target_layers, device,
                )
                torch.save({"test_mean_acts": test_mean_acts,
                            "test_metadata": test_metadata}, test_acts_path)
        else:
            _ensure_model_loaded()
            test_mean_acts, test_metadata = extract_test_activations(
                model, tokenizer, sweep_dir, target_layers, device,
            )
            torch.save({"test_mean_acts": test_mean_acts,
                        "test_metadata": test_metadata}, test_acts_path)

        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------- Phase 2: DiffMean direction + centroid normalization ----
    logger.warning("Phase 2: Computing DiffMean directions ...")
    directions, centroid_norm = compute_diffmean_directions(train_acts, target_layers)

    # ---------- Sample-size summary block --------------------------------
    n_train_per_layer = {
        l: (len(train_acts.get(l, {}).get("pos", [])),
            len(train_acts.get(l, {}).get("neg", [])))
        for l in target_layers
    }
    pos_counts = {p for (p, _) in n_train_per_layer.values()}
    neg_counts = {n for (_, n) in n_train_per_layer.values()}
    n_train_pos_str = (str(next(iter(pos_counts))) if len(pos_counts) == 1
                       else f"varies: {sorted(pos_counts)}")
    n_train_neg_str = (str(next(iter(neg_counts))) if len(neg_counts) == 1
                       else f"varies: {sorted(neg_counts)}")

    test_per_layer = {}
    for entry in test_metadata:
        l = int(entry["layer"])
        test_per_layer[l] = test_per_layer.get(l, 0) + 1
    test_counts = set(test_per_layer.values())
    n_test_str = (str(next(iter(test_counts))) if len(test_counts) == 1
                  else f"varies: {sorted(test_counts)}")

    # Pull d(h) from one stacked test layer (or first train layer if test empty).
    d_h = 0
    if target_layers:
        Hp, _, _ = stack_layer_test(test_mean_acts, test_metadata, target_layers[0])
        if Hp.size:
            d_h = int(Hp.shape[1])
        elif n_train_per_layer.get(target_layers[0], (0, 0))[0] > 0:
            d_h = int(train_acts[target_layers[0]]["pos"][0].shape[0])

    # Per-method n_fit (test_only = n_test; train_plus_test = n_train + n_test).
    if len(pos_counts) == 1 and len(neg_counts) == 1 and len(test_counts) == 1:
        n_train = next(iter(pos_counts)) + next(iter(neg_counts))
        n_test = next(iter(test_counts))
        n_fit_test_only = n_test
        n_fit_combined = n_train + n_test
    else:
        n_fit_test_only = "varies"
        n_fit_combined = "varies"

    logger.warning("=" * 60)
    logger.warning("Sample-size summary (per layer)")
    logger.warning("-" * 60)
    logger.warning(f"  Training pairs:           {n_train_pos_str} pos / {n_train_neg_str} neg")
    logger.warning(f"  Training activations:     {n_train_pos_str} pos + {n_train_neg_str} neg = "
                   f"{n_train if isinstance(n_fit_combined, int) else 'varies'} rows")
    logger.warning(f"  Test prompts:             {n_test_str}")
    logger.warning(f"  Layers:                   {len(target_layers)}  [{layer_lo}..{layer_hi}]")
    logger.warning(f"  d(h) (hidden size):       {d_h}")
    logger.warning(f"  Method 1 (test_only)        n_fit = {n_fit_test_only}")
    logger.warning(f"  Method 2 (train_plus_test)  n_fit = {n_fit_combined}")
    logger.warning("=" * 60)

    # ---------- Phase 6: Delta_l from eval parquets ---------------------
    logger.warning("Phase 6: Computing Delta_l from eval parquets ...")
    delta_df = compute_delta_l(
        sweep_dir, target_layers, args.behavior,
        normalize=not args.no_normalize_delta,
    )
    delta_df.to_csv(out_dir / "delta_l.csv", index=False)

    # ---------- Phases 4 / 5: R^2 metrics for BOTH methods -------------
    method_results = {}
    for mode in ("test_only", "train_plus_test"):
        logger.warning(f"Phases 4/5 [{mode}]: Computing R^2(v), R^2(h), ratio ...")
        metrics_df = compute_r2_metrics(
            test_mean_acts, test_metadata, directions, centroid_norm,
            target_layers, train_acts, args.behavior, mode=mode,
        )
        out_df = metrics_df.merge(delta_df, on="layer", how="left")
        out_df.to_csv(out_dir / f"r2_per_layer__{mode}.csv", index=False)

        logger.warning(f"Per-layer summary [{mode}]:")
        for _, r in out_df.iterrows():
            logger.warning(
                f"  Layer {int(r['layer']):2d}: "
                f"R^2(v) = {r['r2_v']:+.3f}, R^2(h) = {r['r2_h']:+.3f}, "
                f"ratio = {r['ratio']:.3f}, "
                f"Delta_l = {r['delta_l']:+.3f}"
            )

        # Plots for this method
        plot_layer_sweep(
            out_df, args.behavior,
            plot_dir / f"layer_sweep__{mode}__{args.behavior}.png",
            method=mode, layer_lo=layer_lo, layer_hi=layer_hi,
        )
        plot_ratio_vs_steering(
            out_df, delta_df, args.behavior,
            plot_dir / f"ratio_vs_steering__{mode}__{args.behavior}.png",
            method=mode,
        )
        method_results[mode] = out_df

    # ---------- Summary -------------------------------------------------
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "direction": "DiffMean",
        "sweep_dir": str(sweep_dir),
        "train_dataset_path": args.train_dataset_path,
        "layers": target_layers,
        "fit_strategy": "in_sample (both methods)",
        "method_descriptions": METHOD_LABELS,
        "delta_l_normalized": not args.no_normalize_delta,
        "metrics_test_only": method_results["test_only"].to_dict(orient="records"),
        "metrics_train_plus_test": method_results["train_plus_test"].to_dict(orient="records"),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
