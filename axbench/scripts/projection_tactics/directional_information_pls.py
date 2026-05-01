"""
Experiment 2: Directional Information with a PLS denominator.

Same numerator as Experiment 1 (directional_information.py):
    R^2(v) = R^2 of  y_b ~ p_v(h)   (univariate, projection along DiffMean v)

Different denominator. Instead of fitting y_b ~ h directly (high-d, overfits),
we project h onto a k-dimensional PLS subspace W in H that maximally covaries
with y_b, and fit
    R^2(h_PLS, k) = in-sample R^2 of  y_b ~ W h
We sweep k = 1..k_max, fit a single PLS at each k, and pick k* by a
kneedle-style geometric elbow on the (k, R^2) curve. The reported ratio is

    R^2(v) / R^2(h_PLS, k*)

No cross-validation: every R^2 is the in-sample fit on the regression
dataset (mirrors directional_information.py's design). Note that with no
held-out evaluation, in-sample R^2(h_PLS, k) is monotonically
non-decreasing in k and saturates near 1 once k approaches min(n_fit, d).
The elbow chooses where the R^2 curve flattens.

Run TWO methods side-by-side (same as Exp 1):
  Method 1 ("test_only"):      v from train; R^2 fit on TEST only
                                (y_b = LM-judge baseline behavior score)
  Method 2 ("train_plus_test"): v from train; R^2 fit on TRAIN + TEST
                                (train y_b mapped to behavior score extremes:
                                 pos -> score_max, neg -> score_min)

Activation caches reused if available (Exp 1's
`{sweep_dir}/directional_information_analysis/{train,test}_activations.pt`),
otherwise this script will extract + cache its own.

Outputs:
  {sweep_dir}/directional_information_pls_analysis/
    train_activations.pt              (cached or symlinked from Exp 1)
    test_activations.pt
    delta_l.csv                       (best-factor improvement, same as Exp 1)
    pls_k_sweep__test_only.csv        (per (layer, k) in-sample R^2)
    pls_k_sweep__train_plus_test.csv
    r2_per_layer__test_only.csv       (final R^2(v), R^2(h_PLS, k*), ratio)
    r2_per_layer__train_plus_test.csv
    analysis_summary.json
    plots/
      layer_sweep__test_only__{behavior}.png
      layer_sweep__train_plus_test__{behavior}.png
      ratio_vs_steering__test_only__{behavior}.png
      ratio_vs_steering__train_plus_test__{behavior}.png
      pls_k_elbow__test_only__{behavior}.png
      pls_k_elbow__train_plus_test__{behavior}.png

Usage:
    uv run python axbench/scripts/projection_tactics/directional_information_pls.py \\
        --behavior corrigible-neutral-HHH \\
        --model_name google/gemma-2-9b-it \\
        --sweep_dir /vol/filtered_sweep/gemma-2-9b-it/corrigible-neutral-HHH-sweep \\
        --train_dataset_path /vol/expanded_datasets/generated/corrigible-neutral-HHH/train_contrastive.json
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
from sklearn.cross_decomposition import PLSRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")

# Suppress sklearn PLS warnings about y residual being too small (expected
# when k approaches the rank of y).
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cross_decomposition")

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

METHOD_LABELS = {
    "test_only": "Method 1: v from train; R^2 fit on TEST only",
    "train_plus_test": "Method 2: v from train; R^2 fit on TRAIN + TEST",
}


# ============================================================================
# Helpers (identical to directional_information.py)
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
                "pair_idx": pair_idx, "label": label,
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


@torch.no_grad()
def extract_test_activations(model, tokenizer, sweep_dir, target_layers, device):
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
    test_mean_acts = {}
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
                "question_idx": question_idx, "layer": int(layer),
                "baseline_score": bs, "question": question,
                "generation": str(generation),
            })
        if torch.cuda.is_available() and idx % 5 == 0:
            torch.cuda.empty_cache()
    logger.warning(f"Extracted {len(test_metadata)} (prompt, layer) entries")
    return test_mean_acts, test_metadata


def compute_diffmean_directions(train_acts, target_layers):
    """v^l = (mu_pos - mu_neg) / ||...||  per layer, plus centroid m, h."""
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


def stack_layer_test(test_mean_acts, test_metadata, layer):
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
    score_min, score_max, _ = BEHAVIOR_SCALES.get(behavior, (0.0, 1.0, None))
    return float(score_max) if int(label) == 1 else float(score_min)


def stack_layer_train(train_acts, behavior, layer):
    H, y = [], []
    pos_acts = train_acts.get(layer, {}).get("pos", [])
    neg_acts = train_acts.get(layer, {}).get("neg", [])
    yb_pos = train_label_to_yb(behavior, 1)
    yb_neg = train_label_to_yb(behavior, 0)
    for a in pos_acts:
        H.append(a.numpy()); y.append(yb_pos)
    for a in neg_acts:
        H.append(a.numpy()); y.append(yb_neg)
    if not H:
        return np.zeros((0, 0)), np.zeros(0)
    return np.stack(H, axis=0), np.array(y, dtype=float)


# ============================================================================
# Delta_l (best-factor improvement, identical to Exp 1)
# ============================================================================
def compute_delta_l(sweep_dir, target_layers, behavior, normalize=True):
    sweep_dir = Path(sweep_dir)
    score_min, score_max, _ = BEHAVIOR_SCALES.get(behavior, (None, None, None))
    score_range = (score_max - score_min) if (score_min is not None and score_max is not None) else 1.0

    rows = []
    for layer in target_layers:
        path = sweep_dir / f"layer_{layer}" / "eval" / "eval_results.parquet"
        row = {
            "layer": int(layer), "n_baseline": 0, "n_steered": 0,
            "baseline_mean": np.nan,
            "best_factor": np.nan, "best_factor_mean": np.nan,
            "steered_mean_pos": np.nan,
            "delta_l_raw": np.nan, "delta_l": np.nan,
            "delta_l_mean_pos_raw": np.nan, "delta_l_mean_pos": np.nan,
        }
        if not path.exists():
            rows.append(row); continue
        df = pd.read_parquet(path)
        base = df[df["steering_factor"] == 0]["behavior_score"].dropna()
        if len(base) == 0:
            rows.append(row); continue
        baseline_mean = float(base.mean())
        row.update({"n_baseline": int(len(base)), "baseline_mean": baseline_mean})

        per_factor = (
            df.dropna(subset=["behavior_score"])
              .groupby("steering_factor")["behavior_score"]
              .mean().sort_index()
        )
        if per_factor.empty:
            rows.append(row); continue

        best_factor = float(per_factor.idxmax())
        best_factor_mean = float(per_factor.max())
        delta_raw = best_factor_mean - baseline_mean

        pos_factors = per_factor[per_factor.index > 0]
        if not pos_factors.empty:
            steered_mean_pos = float(pos_factors.mean())
            delta_mp_raw = steered_mean_pos - baseline_mean
            n_steered = int(df[df["steering_factor"] > 0]["behavior_score"].notna().sum())
        else:
            steered_mean_pos = np.nan
            delta_mp_raw = np.nan
            n_steered = 0

        row.update({
            "n_steered": n_steered,
            "best_factor": best_factor, "best_factor_mean": best_factor_mean,
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
# PLS denominator
# ============================================================================
def _r2_score(y, y_pred):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = float(((y - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _pls_in_sample_r2(X, y, k):
    """Single PLS fit on (X, y) with k components; return in-sample R^2."""
    if k < 1 or len(y) < 2 or X.shape[1] == 0:
        return float("nan")
    max_k = min(len(y) - 1, X.shape[1])
    k_eff = min(k, max_k)
    if k_eff < 1:
        return float("nan")
    try:
        pls = PLSRegression(n_components=k_eff, scale=True, max_iter=1000)
        pls.fit(X, y)
        y_pred = pls.predict(X).ravel()
    except Exception as e:
        logger.warning(f"PLS fit failed (k={k_eff}): {e}")
        return float("nan")
    return _r2_score(y, y_pred)


def sweep_pls_k(X, y, k_min=1, k_max=20):
    """Per-k in-sample R^2. Returns DataFrame with columns (k, r2_in_sample)."""
    n = len(y)
    d = X.shape[1] if X.size else 0
    if n < 2 or d == 0:
        return pd.DataFrame(columns=["k", "r2_in_sample"])
    hard_cap = min(n - 1, d)
    k_top = max(1, min(k_max, hard_cap))
    rows = []
    for k in range(k_min, k_top + 1):
        r2 = _pls_in_sample_r2(X, y, k)
        rows.append({"k": int(k), "r2_in_sample": float(r2)})
    return pd.DataFrame(rows)


def pick_k_elbow(k_sweep_df):
    """Pick k* via a kneedle-style geometric elbow on the (k, R^2) curve.

    R^2 in-sample is monotonically non-decreasing in k, so we treat the
    curve as a saturating one and pick the point of maximum perpendicular
    distance from the chord connecting (k_min, R^2_min) to (k_max, R^2_max).

    Returns (k_star, r2_star). Falls back to the smallest k with R^2 within
    tol of max if the geometric search fails.
    """
    col = "r2_in_sample"
    if k_sweep_df.empty or k_sweep_df[col].isna().all():
        return None, np.nan
    df = k_sweep_df.dropna(subset=[col]).sort_values("k").reset_index(drop=True)
    if len(df) == 1:
        return int(df.loc[0, "k"]), float(df.loc[0, col])

    ks = df["k"].to_numpy(dtype=float)
    r2s = df[col].to_numpy(dtype=float)

    # Normalize both axes to [0, 1] before measuring distances.
    k_span = ks.max() - ks.min()
    r2_span = r2s.max() - r2s.min()
    if k_span < 1e-12 or r2_span < 1e-12:
        # No variation in either axis -- pick smallest k with R^2 near max.
        r2_max = float(r2s.max())
        for i, r in enumerate(r2s):
            if r >= r2_max - 1e-6:
                return int(ks[i]), float(r)
        return int(ks[-1]), float(r2s[-1])

    ks_n = (ks - ks.min()) / k_span
    r2s_n = (r2s - r2s.min()) / r2_span

    # Chord from (0, 0) to (1, 1). For a saturating curve that bows upward
    # (R^2 rises fast then plateaus), R^2_n >= k_n, and the elbow is the
    # point where (R^2_n - k_n) is maximized.
    diff = r2s_n - ks_n
    elbow_idx = int(np.argmax(diff))
    k_star = int(ks[elbow_idx])
    r2_star = float(r2s[elbow_idx])

    # Sanity fallback: if the elbow distance is essentially zero, pick the
    # smallest k whose R^2 is within tol of the max.
    if diff[elbow_idx] < 1e-3:
        r2_max = float(r2s.max())
        tol = max(min(0.05 * r2_span, 0.02), 1e-6)
        for i, r in enumerate(r2s):
            if r >= r2_max - tol:
                return int(ks[i]), float(r)
    return k_star, r2_star


def compute_pls_metrics(test_mean_acts, test_metadata, directions,
                       centroid_norm, target_layers, train_acts, behavior,
                       mode="test_only", k_max=20):
    """Per layer, compute R^2(v), sweep PLS k for R^2(h_PLS), pick k*,
    return one row per layer plus the full per-(layer, k) sweep table.

    All R^2 values are in-sample (no CV).
    """
    if mode not in ("test_only", "train_plus_test"):
        raise ValueError(f"Unknown mode: {mode}")

    logger.warning(f"--- compute_pls_metrics (mode = {mode}) ---")
    rows = []
    sweep_rows = []

    for layer in target_layers:
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
            f"PLS / R^2 on n_fit = {n_fit} "
            f"({'test only' if mode == 'test_only' else f'{n_train} train + {n_test} test'})  "
            f"d(h) = {d}"
        )

        if n_fit < 4:
            rows.append({
                "layer": int(layer), "mode": mode,
                "n_train_pos": int(n_train_pos), "n_train_neg": int(n_train_neg),
                "n_test": int(n_test), "n_fit": int(n_fit), "d_h": int(d),
                "r2_v": np.nan, "r2_h_pls": np.nan, "k_star": np.nan,
                "ratio": np.nan, "pearson_r_v": np.nan, "spearman_rho_v": np.nan,
            })
            continue

        v = directions[layer].numpy().astype(np.float64)
        m = centroid_norm[layer]["midpoint"]
        h = centroid_norm[layer]["half_range"]

        proj_raw = H_fit @ v
        proj = (proj_raw - m) / (h if abs(h) > 1e-12 else 1.0)

        # R^2(v): Pearson r squared (univariate OLS in-sample = r^2).
        pr, _ = scipy_stats.pearsonr(proj, y_fit)
        sr, _ = scipy_stats.spearmanr(proj, y_fit)
        r2_v = float(pr * pr)

        # R^2(h_PLS): sweep k (in-sample), pick geometric elbow.
        k_sweep_df = sweep_pls_k(H_fit.astype(np.float64), y_fit,
                                  k_min=1, k_max=k_max)
        if k_sweep_df.empty:
            r2_h_pls = np.nan
            k_star = np.nan
        else:
            k_star, r2_h_pls = pick_k_elbow(k_sweep_df)
            if r2_h_pls is None or np.isnan(r2_h_pls):
                k_star = np.nan; r2_h_pls = np.nan

        for _, kr in k_sweep_df.iterrows():
            sweep_rows.append({
                "layer": int(layer), "mode": mode,
                "k": int(kr["k"]),
                "r2_in_sample": float(kr["r2_in_sample"]) if not pd.isna(kr["r2_in_sample"]) else np.nan,
            })

        r2_v_clip = max(r2_v, 0.0) if not np.isnan(r2_v) else np.nan
        r2_h_clip = max(r2_h_pls, 0.0) if not np.isnan(r2_h_pls) else np.nan
        if not np.isnan(r2_v_clip) and not np.isnan(r2_h_clip) and r2_h_clip > 1e-6:
            ratio = r2_v_clip / r2_h_clip
        else:
            ratio = np.nan

        rows.append({
            "layer": int(layer), "mode": mode,
            "n_train_pos": int(n_train_pos), "n_train_neg": int(n_train_neg),
            "n_test": int(n_test), "n_fit": int(n_fit), "d_h": int(d),
            "r2_v": float(r2_v),
            "r2_h_pls": float(r2_h_pls) if not np.isnan(r2_h_pls) else np.nan,
            "r2_v_clip": float(r2_v_clip) if not np.isnan(r2_v_clip) else np.nan,
            "r2_h_clip": float(r2_h_clip) if not np.isnan(r2_h_clip) else np.nan,
            "k_star": int(k_star) if k_star is not None and not (isinstance(k_star, float) and np.isnan(k_star)) else np.nan,
            "ratio": float(ratio) if not np.isnan(ratio) else np.nan,
            "pearson_r_v": float(pr), "spearman_rho_v": float(sr),
        })
        logger.warning(
            f"  Layer {int(layer):2d} [{mode}]: "
            f"R^2(v) = {r2_v:+.3f}  k* = {k_star}  R^2(h_PLS, k*) = "
            f"{r2_h_pls if not np.isnan(r2_h_pls) else float('nan'):+.3f}  "
            f"ratio = {ratio if not np.isnan(ratio) else float('nan'):+.3f}"
        )
    return pd.DataFrame(rows), pd.DataFrame(sweep_rows)


# ============================================================================
# Plots
# ============================================================================
def plot_layer_sweep(metrics_df, behavior, output_path, method,
                     layer_lo=10, layer_hi=32):
    """3-panel layer sweep: top R^2(h_PLS), middle R^2(v), bottom ratio."""
    df = metrics_df.sort_values("layer").reset_index(drop=True)
    layers = df["layer"].values

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)

    ax = axes[0]
    ax.plot(layers, df["r2_h_pls"].values, "o-", color="#1f77b4", linewidth=2,
            markersize=6, label=r"$R^2(h_{PLS}, k^\ast)$ (in-sample)")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel(r"$R^2(h_{PLS})$  (PLS subspace)", fontsize=10)
    ax.set_title(
        f"{behavior} - Layer sweep of directional information (PLS denominator)\n"
        f"{METHOD_LABELS.get(method, method)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(layers, df["r2_v"].values, "o-", color="#2ca02c", linewidth=2,
            markersize=6, label=r"$R^2(v)$")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel(r"$R^2(v)$  (DiffMean projection)", fontsize=10)
    ax.legend(loc="best", fontsize=9)

    ax = axes[2]
    ax.plot(layers, df["ratio"].values, "o-", color="#d62728", linewidth=2,
            markersize=6, label=r"$R^2(v)/R^2(h_{PLS}, k^\ast)$")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.7,
               label="ratio = 1")
    ax.set_ylabel(r"$R^2(v)/R^2(h_{PLS})$", fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.legend(loc="best", fontsize=9)

    if not df.empty:
        n_fit = int(df["n_fit"].dropna().iloc[0]) if df["n_fit"].notna().any() else 0
        n_train_pos = int(df["n_train_pos"].dropna().iloc[0]) if df["n_train_pos"].notna().any() else 0
        n_train_neg = int(df["n_train_neg"].dropna().iloc[0]) if df["n_train_neg"].notna().any() else 0
        n_test = int(df["n_test"].dropna().iloc[0]) if df["n_test"].notna().any() else 0
        info = (f"v from train: {n_train_pos} pos / {n_train_neg} neg   "
                f"|   PLS / R^2 on n_fit = {n_fit}   "
                f"|   test n = {n_test}")
        fig.text(0.5, 0.005, info, ha="center", fontsize=8.5, color="dimgray")

    for a in axes:
        a.set_xlim(layer_lo - 0.5, layer_hi + 0.5)
        a.set_xticks(np.arange(layer_lo, layer_hi + 1))
        a.grid(True, alpha=0.4)

    plt.tight_layout(rect=(0, 0.025, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ratio_vs_steering(metrics_df, delta_df, behavior, output_path, method):
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

    ax.set_xlabel(r"$R^2(v)/R^2(h_{PLS}, k^\ast)$  (PLS directional information)",
                  fontsize=11)
    ax.set_ylabel(r"$\Delta_l$  (normalized steering improvement)", fontsize=11)
    ax.set_title(
        f"{behavior}: PLS directional information vs steering effectiveness\n"
        f"{METHOD_LABELS.get(method, method)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pls_k_elbow(sweep_df, metrics_df, behavior, output_path, method):
    """Per-layer held-out R^2 vs k. One small-multiple panel per layer.
    Marks chosen k* with a vertical line."""
    if sweep_df.empty:
        logger.warning(f"plot_pls_k_elbow [{method}]: sweep_df empty")
        return

    layers = sorted(sweep_df["layer"].unique())
    n = len(layers)
    cols = min(6, n) if n > 0 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4),
                              sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    metrics_lookup = metrics_df.set_index("layer")["k_star"].to_dict() if not metrics_df.empty else {}

    for idx, layer in enumerate(layers):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        ldf = sweep_df[sweep_df["layer"] == layer].sort_values("k")
        ax.plot(ldf["k"].values, ldf["r2_in_sample"].values, "o-",
                color="#1f77b4", markersize=4, linewidth=1.5)
        k_star = metrics_lookup.get(layer, np.nan)
        if not (isinstance(k_star, float) and np.isnan(k_star)):
            ax.axvline(x=k_star, color="#d62728", linestyle="--",
                       linewidth=1.2, alpha=0.85)
            ax.text(0.97, 0.05, f"k* = {int(k_star)}", transform=ax.transAxes,
                    fontsize=8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="wheat", alpha=0.85))
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Layer {int(layer)}", fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.set_xlabel("k", fontsize=9)
        ax.set_ylabel(r"in-sample $R^2$", fontsize=9)

    # Hide unused axes.
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"{behavior}: PLS k vs in-sample $R^2$ (per layer)\n{METHOD_LABELS.get(method, method)}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment 2: Directional information with PLS denominator"
    )
    parser.add_argument("--behavior", type=str, required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--sweep_dir", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--layers", type=str, default="10-32")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--k_max", type=int, default=20,
                        help="Max PLS components to sweep")
    parser.add_argument("--no_normalize_delta", action="store_true")
    parser.add_argument("--replot_only", action="store_true")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = sweep_dir / "directional_information_pls_analysis"
    plot_dir = out_dir / "plots"

    # Clean plots / CSVs / JSON, keep cached .pt files.
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    for csv_file in out_dir.glob("*.csv"):
        csv_file.unlink()
    for json_file in out_dir.glob("*.json"):
        json_file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if "-" in args.layers and "," not in args.layers:
        lo, hi = [int(x) for x in args.layers.split("-")]
        target_layers = list(range(lo, hi + 1))
    else:
        target_layers = sorted(int(x.strip()) for x in args.layers.split(","))
    avail = set(discover_layers(sweep_dir))
    target_layers = [l for l in target_layers if l in avail]
    if not target_layers:
        logger.error(f"No valid layers in {sweep_dir}")
        sys.exit(1)
    layer_lo, layer_hi = min(target_layers), max(target_layers)
    logger.warning(f"Target layers: {target_layers}")

    # Activation cache locations: prefer this analysis dir, fall back to
    # Exp 1's dir so a previous run's caches can be reused.
    train_acts_path = out_dir / "train_activations.pt"
    test_acts_path = out_dir / "test_activations.pt"
    fallback_dir = sweep_dir / "directional_information_analysis"
    fallback_train = fallback_dir / "train_activations.pt"
    fallback_test = fallback_dir / "test_activations.pt"

    def _read_train_cache(path):
        cached = torch.load(path, map_location="cpu", weights_only=False)
        return cached["train_acts"], cached.get("train_meta", [])

    def _read_test_cache(path):
        cached = torch.load(path, map_location="cpu", weights_only=False)
        return cached["test_mean_acts"], cached["test_metadata"]

    if args.replot_only:
        candidates_train = [p for p in (train_acts_path, fallback_train) if p.exists()]
        candidates_test = [p for p in (test_acts_path, fallback_test) if p.exists()]
        if not candidates_train or not candidates_test:
            logger.error("--replot_only requires cached train + test activations.")
            sys.exit(1)
        logger.warning(f"Loading cached train activations from {candidates_train[0]}")
        train_acts, train_meta = _read_train_cache(candidates_train[0])
        logger.warning(f"Loading cached test activations from {candidates_test[0]}")
        test_mean_acts, test_metadata = _read_test_cache(candidates_test[0])
    else:
        # Try to reuse Exp 1's cache before doing any forward passes.
        used_fallback_train = False
        used_fallback_test = False
        if not train_acts_path.exists() and fallback_train.exists():
            logger.warning(f"Reusing Exp 1's train activations cache: {fallback_train}")
            train_acts, train_meta = _read_train_cache(fallback_train)
            used_fallback_train = True
        if not test_acts_path.exists() and fallback_test.exists():
            logger.warning(f"Reusing Exp 1's test activations cache: {fallback_test}")
            test_mean_acts, test_metadata = _read_test_cache(fallback_test)
            used_fallback_test = True

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

        # Train activations (load this dir's cache, fallback, or extract).
        if not used_fallback_train and not train_acts_path.exists():
            _ensure_model_loaded()
            train_acts, train_meta = extract_train_activations(
                model, tokenizer, args.train_dataset_path, target_layers, device,
            )
            torch.save({"train_acts": train_acts, "train_meta": train_meta},
                       train_acts_path)
        elif not used_fallback_train:
            train_acts, train_meta = _read_train_cache(train_acts_path)

        # Verify training cache covers target layers; re-extract if not.
        if not set(target_layers).issubset(set(train_acts.keys())):
            logger.warning("Cached train activations missing some target layers; "
                           "re-extracting.")
            _ensure_model_loaded()
            train_acts, train_meta = extract_train_activations(
                model, tokenizer, args.train_dataset_path, target_layers, device,
            )
            torch.save({"train_acts": train_acts, "train_meta": train_meta},
                       train_acts_path)

        # Test activations.
        if not used_fallback_test and not test_acts_path.exists():
            _ensure_model_loaded()
            test_mean_acts, test_metadata = extract_test_activations(
                model, tokenizer, sweep_dir, target_layers, device,
            )
            torch.save({"test_mean_acts": test_mean_acts,
                        "test_metadata": test_metadata}, test_acts_path)
        elif not used_fallback_test:
            test_mean_acts, test_metadata = _read_test_cache(test_acts_path)

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

        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    d_h = 0
    if target_layers:
        Hp, _, _ = stack_layer_test(test_mean_acts, test_metadata, target_layers[0])
        if Hp.size:
            d_h = int(Hp.shape[1])
        elif n_train_per_layer.get(target_layers[0], (0, 0))[0] > 0:
            d_h = int(train_acts[target_layers[0]]["pos"][0].shape[0])

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

    logger.warning("Phase 6: Computing Delta_l from eval parquets ...")
    delta_df = compute_delta_l(
        sweep_dir, target_layers, args.behavior,
        normalize=not args.no_normalize_delta,
    )
    delta_df.to_csv(out_dir / "delta_l.csv", index=False)

    method_results = {}
    for mode in ("test_only", "train_plus_test"):
        logger.warning(f"Phase 4/5 [{mode}]: computing R^2(v), R^2(h_PLS), ratio ...")
        metrics_df, sweep_df = compute_pls_metrics(
            test_mean_acts, test_metadata, directions, centroid_norm,
            target_layers, train_acts, args.behavior, mode=mode,
            k_max=args.k_max,
        )
        out_df = metrics_df.merge(delta_df, on="layer", how="left")
        out_df.to_csv(out_dir / f"r2_per_layer__{mode}.csv", index=False)
        sweep_df.to_csv(out_dir / f"pls_k_sweep__{mode}.csv", index=False)

        logger.warning(f"Per-layer summary [{mode}]:")
        for _, r in out_df.iterrows():
            logger.warning(
                f"  Layer {int(r['layer']):2d}: R^2(v) = {r['r2_v']:+.3f}, "
                f"k* = {r['k_star']}, R^2(h_PLS) = {r['r2_h_pls']:+.3f}, "
                f"ratio = {r['ratio']:.3f}, Delta_l = {r['delta_l']:+.3f}"
            )

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
        plot_pls_k_elbow(
            sweep_df, out_df, args.behavior,
            plot_dir / f"pls_k_elbow__{mode}__{args.behavior}.png",
            method=mode,
        )
        method_results[mode] = (out_df, sweep_df)

    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "direction": "DiffMean (numerator) + PLS (denominator)",
        "sweep_dir": str(sweep_dir),
        "train_dataset_path": args.train_dataset_path,
        "layers": target_layers,
        "k_max": args.k_max,
        "fit_strategy": "in_sample (no CV)",
        "method_descriptions": METHOD_LABELS,
        "delta_l_normalized": not args.no_normalize_delta,
        "metrics_test_only": method_results["test_only"][0].to_dict(orient="records"),
        "metrics_train_plus_test": method_results["train_plus_test"][0].to_dict(orient="records"),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
