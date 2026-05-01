"""
Experiment 1 (MCQA): Directional Information (pseudo-R^2 ratio).

MCQA-adapted port of `directional_information.py`. Because the test target
is binary (did the model greedy-decode the matching-behavior letter?), all
regressions are LOGISTIC instead of linear, and "R^2" means McFadden's
pseudo-R^2.

Run TWO methods side-by-side and save plots / CSVs for each:

  Method 1 ("test_only"):
    v from training prompts; pseudo-R^2(v), pseudo-R^2(h) fit on TEST prompts only.
    Test labels are baseline_correct in {0, 1} (whether unsteered greedy
    decode at the '(' position picks the matching-behavior letter).

  Method 2 ("train_plus_test"):
    v from training prompts (same v); pseudo-R^2(v), pseudo-R^2(h) fit on
    TRAIN + TEST prompts combined. Train labels = 1 (matching) / 0 (not
    matching). Test labels = baseline_correct.

For each layer l (both methods):
  1. v^l = (mu_pos^l - mu_neg^l) / ||...||  from training activations only.
     Centroid midpoint m and half-range h come from the training projections.
     Activations are taken at the answer-letter token position (matching
     mcqa_projection_link.py).
  2. For each prompt, compute
        p_v(h_i) = (h_i . v - m) / h.
  3. Fit on that dataset:
        y ~ p_v(h)  (univariate logistic regression)              -> R^2(v)
        y ~ h       (LogisticRegressionCV, L2 on full activation) -> R^2(h)
     R^2 = McFadden's pseudo-R^2 = 1 - LL(model) / LL(null).
  4. Ratio = R^2(v) / R^2(h)  (clip negative pseudo-R^2 to 0 before dividing).
  5. Pull steering effectiveness Delta_l from
        {mcqa_link_dir}/per_layer_summary.csv:
        Delta_l = best_steered_acc - baseline_acc.

Outputs:
  {output_dir}/
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
    uv run python axbench/scripts/projection_tactics/directional_information_mcqa.py \\
        --behavior sycophancy \\
        --model_name google/gemma-2-9b-it \\
        --mcqa_link_dir results/mcqa_projection_link/gemma-2-9b-it/sycophancy \\
        --layers 10-32
"""
import re
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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
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


BEHAVIORS = [
    "sycophancy", "survival-instinct", "corrigible-neutral-HHH",
    "hallucination", "refusal", "myopic-reward", "coordinate-other-ais",
]
TEST_PATH_MAP  = {b: f"datasets/test/{b}/test_dataset_ab.json" for b in BEHAVIORS}
TRAIN_PATH_MAP = {b: f"datasets/raw/{b}/dataset.json"          for b in BEHAVIORS}


# ============================================================================
# Helpers (shared with mcqa_projection_link.py)
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def extract_letter(answer_str: str) -> str:
    m = re.search(r"\(([A-Z])\)", answer_str.upper())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse letter from '{answer_str}'")


def build_full_ids(tokenizer, question: str, answer_letter: str):
    """Returns (prompt_ids, full_ids, letter_pos, open_paren_pos).

    letter_pos is the index of the answer-letter token; open_paren_pos is
    the '(' immediately before it. Identical to mcqa_projection_link.py.
    """
    cand = f" ({answer_letter})"
    if supports_chat_template(tokenizer):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question},
             {"role": "assistant", "content": cand}],
            tokenize=True, add_generation_prompt=False,
        )
    else:
        prompt_ids = tokenizer.encode(question)
        full_ids = tokenizer.encode(question + cand)

    letter_pos = None
    for i in range(len(full_ids) - 1, max(len(prompt_ids) - 1, 0), -1):
        if tokenizer.decode([full_ids[i]]).strip() == answer_letter:
            letter_pos = i
            break
    if letter_pos is None:
        raise RuntimeError(f"Cannot find '{answer_letter}' in suffix of full_ids")
    open_paren_pos = letter_pos - 1
    return prompt_ids, full_ids, letter_pos, open_paren_pos


def pad_batch(token_lists, pad_id, device):
    max_len = max(len(t) for t in token_lists)
    B = len(token_lists)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    lens = []
    for i, t in enumerate(token_lists):
        L = len(t)
        ids[i, :L] = torch.tensor(t, dtype=torch.long, device=device)
        mask[i, :L] = 1
        lens.append(L)
    return ids, mask, lens


def discover_layers_from_summary(per_layer_summary_path: Path) -> list[int]:
    if not per_layer_summary_path.exists():
        return []
    df = pd.read_csv(per_layer_summary_path)
    return sorted(int(l) for l in df["layer"].unique())


@torch.no_grad()
def forward_capture_letter_pos(model, layers, input_ids, attention_mask):
    """Forward pass capturing per-layer hidden states, returned on CPU as
    float32 with shape [B, L, H]. Caller indexes the desired token position."""
    storage = {}
    handles = []
    for l in layers:
        def _make_hook(layer_idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                storage[layer_idx] = h.detach()
            return hook
        h = model.model.layers[l].register_forward_hook(
            _make_hook(l), always_call=True,
        )
        handles.append(h)
    try:
        _ = model.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return {l: storage[l].float().cpu() for l in layers}


# ============================================================================
# Phase 1 - training activations
# ============================================================================
@torch.no_grad()
def extract_train_activations(model, tokenizer, train_dataset_path,
                              target_layers, device, batch_size=16,
                              max_examples=None, seed=42):
    """For each train pair, collect activations at the answer-letter position
    for both 'matching_behavior' (label=1) and 'not_matching_behavior'
    (label=0) completions."""
    model.eval()
    pad_id = tokenizer.pad_token_id

    with open(train_dataset_path, encoding="utf-8") as f:
        train_data = json.load(f)
    if max_examples is not None and max_examples < len(train_data):
        rng = np.random.default_rng(seed)
        idxs = sorted(rng.choice(len(train_data), size=max_examples, replace=False).tolist())
        train_data = [train_data[i] for i in idxs]
    logger.warning(f"Extracting training activations for {len(train_data)} pairs")

    train_flat = []
    for pair_idx, item in enumerate(train_data):
        for label, answer_key in [(1, "answer_matching_behavior"),
                                  (0, "answer_not_matching_behavior")]:
            try:
                letter = extract_letter(item[answer_key])
                _, full_ids, letter_pos, _ = build_full_ids(
                    tokenizer, item["question"], letter,
                )
            except Exception:
                continue
            train_flat.append({
                "pair_idx": pair_idx,
                "label": label,
                "full_ids": full_ids,
                "letter_pos": letter_pos,
            })

    train_acts = {l: {"pos": [], "neg": []} for l in target_layers}
    train_meta = []

    for start in tqdm(range(0, len(train_flat), batch_size), desc="Train activations"):
        batch = train_flat[start:start + batch_size]
        ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
        hiddens = forward_capture_letter_pos(model, target_layers, ids, mask)
        for i, b in enumerate(batch):
            key = "pos" if b["label"] == 1 else "neg"
            for layer in target_layers:
                act = hiddens[layer][i, b["letter_pos"], :]
                train_acts[layer][key].append(act)
            train_meta.append({
                "pair_idx": b["pair_idx"],
                "label": b["label"],
                "label_name": "pos" if b["label"] == 1 else "neg",
            })
        del hiddens, ids, mask
        if torch.cuda.is_available() and (start // batch_size) % 5 == 0:
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
    train projections).
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
def extract_test_activations(model, tokenizer, test_dataset_path,
                             baseline_correct_map, target_layers, device,
                             batch_size=16, max_test=None):
    """For each test prompt, store the activation at the matching-letter
    position together with the binary baseline_correct label (read from the
    mcqa_projection_link per_prompt_results.csv)."""
    model.eval()
    pad_id = tokenizer.pad_token_id

    with open(test_dataset_path, encoding="utf-8") as f:
        test_data = json.load(f)
    if max_test is not None:
        test_data = test_data[:max_test]

    flat_items = []
    for j, item in enumerate(test_data):
        if j not in baseline_correct_map:
            continue
        try:
            ml = extract_letter(item["answer_matching_behavior"])
            nl = extract_letter(item["answer_not_matching_behavior"])
        except Exception:
            continue
        if ml == nl:
            continue
        try:
            _, full_ids, letter_pos, _ = build_full_ids(tokenizer, item["question"], ml)
        except Exception:
            continue
        flat_items.append({
            "prompt_idx": j,
            "matching_letter": ml,
            "notmatch_letter": nl,
            "full_ids": full_ids,
            "letter_pos": letter_pos,
            "question": item["question"],
        })
    logger.warning(f"Extracting test activations for {len(flat_items)} prompts")

    test_mean_acts = {}   # (prompt_idx, layer) -> tensor
    test_metadata = []

    for start in tqdm(range(0, len(flat_items), batch_size), desc="Test activations"):
        batch = flat_items[start:start + batch_size]
        ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
        hiddens = forward_capture_letter_pos(model, target_layers, ids, mask)
        for i, b in enumerate(batch):
            j = b["prompt_idx"]
            for layer in target_layers:
                act = hiddens[layer][i, b["letter_pos"], :]
                test_mean_acts[(j, int(layer))] = act
                test_metadata.append({
                    "prompt_idx": j,
                    "layer": int(layer),
                    "baseline_correct": int(baseline_correct_map[j]),
                    "question": b["question"],
                    "matching_letter": b["matching_letter"],
                })
        del hiddens, ids, mask
        if torch.cuda.is_available() and (start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    logger.warning(f"Extracted {len(test_metadata)} (prompt, layer) entries")
    return test_mean_acts, test_metadata


def stack_layer_test(test_mean_acts, test_metadata, layer):
    """Per-layer (H, y, prompt_idxs) for TEST prompts only. y = baseline_correct."""
    H, y, qids = [], [], []
    seen = set()
    for entry in test_metadata:
        if int(entry["layer"]) != int(layer):
            continue
        j = int(entry["prompt_idx"])
        if j in seen:
            continue
        seen.add(j)
        bc = entry["baseline_correct"]
        if bc is None or (isinstance(bc, float) and np.isnan(bc)):
            continue
        act = test_mean_acts.get((j, int(layer)))
        if act is None:
            continue
        H.append(act.numpy())
        y.append(int(bc))
        qids.append(j)
    if not H:
        return np.zeros((0, 0)), np.zeros(0, dtype=int), []
    return np.stack(H, axis=0), np.array(y, dtype=int), qids


def stack_layer_train(train_acts, layer):
    """Per-layer (H, y) for TRAIN prompts. y = 1 for matching, 0 otherwise."""
    H, y = [], []
    pos_acts = train_acts.get(layer, {}).get("pos", [])
    neg_acts = train_acts.get(layer, {}).get("neg", [])
    for a in pos_acts:
        H.append(a.numpy())
        y.append(1)
    for a in neg_acts:
        H.append(a.numpy())
        y.append(0)
    if not H:
        return np.zeros((0, 0)), np.zeros(0, dtype=int)
    return np.stack(H, axis=0), np.array(y, dtype=int)


def stack_layer_combined(test_mean_acts, test_metadata, train_acts, layer):
    """Per-layer (H, y) for TRAIN + TEST. Train rows: y in {0,1}; test rows:
    y = baseline_correct."""
    H_test, y_test, _ = stack_layer_test(test_mean_acts, test_metadata, layer)
    H_train, y_train = stack_layer_train(train_acts, layer)
    if H_test.size == 0 and H_train.size == 0:
        return np.zeros((0, 0)), np.zeros(0, dtype=int)
    if H_test.size == 0:
        return H_train, y_train
    if H_train.size == 0:
        return H_test, y_test
    return (
        np.concatenate([H_train, H_test], axis=0),
        np.concatenate([y_train, y_test]).astype(int),
    )


# ============================================================================
# Phase 4 / 5 - pseudo-R^2 computations
# ============================================================================
def mcfadden_r2(y_true, y_proba):
    """McFadden's pseudo-R^2 = 1 - LL(model) / LL(null)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll_full = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p_null = float(np.mean(y))
    if p_null <= 0 or p_null >= 1:
        return float("nan")
    ll_null = len(y) * (p_null * np.log(p_null) + (1 - p_null) * np.log(1 - p_null))
    if abs(ll_null) < 1e-12:
        return float("nan")
    return 1.0 - ll_full / ll_null


def _fit_logistic_full(H, y, Cs):
    """Multivariate logistic on full activation. Uses LogisticRegressionCV
    with stratified CV when both classes have enough samples; else falls
    back to a single fit at C=1.0."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_min = min(n_pos, n_neg)
    if n_min < 2:
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(H, y)
        return clf, float("nan")
    cv_folds = max(2, min(5, n_min))
    clf = LogisticRegressionCV(
        Cs=Cs, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="neg_log_loss", solver="lbfgs", max_iter=1000,
    )
    clf.fit(H, y)
    C_used = float(clf.C_[0]) if hasattr(clf, "C_") else float("nan")
    return clf, C_used


def compute_r2_metrics(test_mean_acts, test_metadata, directions,
                       centroid_norm, target_layers, train_acts,
                       mode="test_only",
                       Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)):
    """Per layer, fit pseudo-R^2(v) and pseudo-R^2(h) on the dataset
    selected by `mode` (logistic regression).

      - "test_only":      fit set = test prompts (y = baseline_correct)
      - "train_plus_test": fit set = train + test
                           (train y in {0,1}; test y = baseline_correct)

    v is always the DiffMean direction computed from the training set.
    """
    if mode not in ("test_only", "train_plus_test"):
        raise ValueError(f"Unknown mode: {mode}")

    logger.warning(f"--- compute_r2_metrics (mode = {mode}) ---")

    rows = []
    for layer in target_layers:
        H_test, y_test, _ = stack_layer_test(test_mean_acts, test_metadata, layer)
        H_train, y_train = stack_layer_train(train_acts, layer)
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
                y_fit = np.concatenate([y_train, y_test]).astype(int)
        n_fit = len(y_fit)
        d = H_fit.shape[1] if n_fit else 0
        n_classes = int(len(np.unique(y_fit))) if n_fit else 0

        logger.warning(
            f"  Layer {int(layer):2d} [{mode}]: "
            f"v from {n_train_pos} pos / {n_train_neg} neg train  |  "
            f"regression on n_fit = {n_fit} "
            f"({'test only' if mode == 'test_only' else f'{n_train} train + {n_test} test'})  "
            f"d(h) = {d}  classes = {n_classes}"
        )

        if n_fit < 4 or n_classes < 2:
            rows.append({
                "layer": int(layer), "mode": mode,
                "n_train_pos": int(n_train_pos), "n_train_neg": int(n_train_neg),
                "n_test": int(n_test), "n_fit": int(n_fit), "d_h": int(d),
                "r2_v": np.nan, "r2_h": np.nan, "ratio": np.nan,
                "spearman_rho_v": np.nan, "logistic_C": np.nan,
            })
            continue

        v = directions[layer].numpy().astype(np.float64)
        m = centroid_norm[layer]["midpoint"]
        h = centroid_norm[layer]["half_range"]

        proj_raw = H_fit @ v
        proj = (proj_raw - m) / (h if abs(h) > 1e-12 else 1.0)

        # R^2(v) = in-sample McFadden's pseudo-R^2 of y ~ p_v(h) (univariate).
        log_v = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        log_v.fit(proj.reshape(-1, 1), y_fit)
        proba_v = log_v.predict_proba(proj.reshape(-1, 1))[:, 1]
        r2_v = mcfadden_r2(y_fit, proba_v)

        sr, _ = scipy_stats.spearmanr(proj, y_fit)

        # R^2(h) = in-sample McFadden's pseudo-R^2 of y ~ h with L2 logistic.
        log_h, C_used = _fit_logistic_full(H_fit.astype(np.float64), y_fit, list(Cs))
        proba_h = log_h.predict_proba(H_fit.astype(np.float64))[:, 1]
        r2_h = mcfadden_r2(y_fit, proba_h)

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
            "spearman_rho_v": float(sr) if not np.isnan(sr) else np.nan,
            "logistic_C": float(C_used) if not np.isnan(C_used) else np.nan,
        })
        logger.warning(
            f"  Layer {int(layer):2d} [{mode}]: "
            f"R^2(v) = {r2_v:+.3f}  R^2(h) = {r2_h:+.3f}  "
            f"ratio = {ratio if not np.isnan(ratio) else float('nan'):+.3f}  "
            f"(C = {C_used:.2g})"
        )
    return pd.DataFrame(rows)


# ============================================================================
# Phase 6 - steering effectiveness Delta_l (read from mcqa_projection_link)
# ============================================================================
def compute_delta_l(per_layer_summary_path, target_layers):
    """Per layer, steering-effectiveness Delta_l for MCQA.

        Delta_l = best_steered_acc - baseline_acc

    where both come from {mcqa_link_dir}/per_layer_summary.csv produced by
    mcqa_projection_link.py. Accuracy is already 0..1 so no extra
    normalization is applied.
    """
    path = Path(per_layer_summary_path)
    rows = []
    if not path.exists():
        logger.warning(
            f"per_layer_summary.csv not found at {path}; Delta_l will be NaN."
        )
        for layer in target_layers:
            rows.append({
                "layer": int(layer),
                "baseline_acc": np.nan,
                "best_steered_acc": np.nan,
                "best_factor": np.nan,
                "delta_l": np.nan,
                "delta_l_raw": np.nan,
            })
        return pd.DataFrame(rows)

    df = pd.read_csv(path)
    by_layer = {int(r["layer"]): r for _, r in df.iterrows()}
    for layer in target_layers:
        r = by_layer.get(int(layer))
        if r is None:
            rows.append({
                "layer": int(layer),
                "baseline_acc": np.nan,
                "best_steered_acc": np.nan,
                "best_factor": np.nan,
                "delta_l": np.nan,
                "delta_l_raw": np.nan,
            })
            continue
        baseline_acc = float(r.get("baseline_acc", np.nan))
        best_steered_acc = float(r["best_steered_acc"]) if "best_steered_acc" in r and not pd.isna(r["best_steered_acc"]) else np.nan
        best_factor = r["best_factor"] if "best_factor" in r else np.nan
        delta = best_steered_acc - baseline_acc if (not np.isnan(best_steered_acc) and not np.isnan(baseline_acc)) else np.nan
        rows.append({
            "layer": int(layer),
            "baseline_acc": baseline_acc,
            "best_steered_acc": best_steered_acc,
            "best_factor": best_factor,
            "delta_l_raw": delta,
            "delta_l": delta,
        })
    return pd.DataFrame(rows)


def load_baseline_correct(per_prompt_csv_path):
    """Returns dict prompt_idx -> baseline_correct (int 0/1).

    baseline_correct in mcqa_projection_link's per_prompt_results.csv is
    constant across layers for a given prompt, so we collapse on the first
    occurrence per prompt_idx.
    """
    path = Path(per_prompt_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"per_prompt_results.csv not found at {path}. Run "
            f"mcqa_projection_link.py for this behavior first."
        )
    df = pd.read_csv(path)
    bc = (
        df.groupby("prompt_idx")["baseline_correct"]
          .first()
          .astype(int)
          .to_dict()
    )
    return {int(k): int(v) for k, v in bc.items()}


# ============================================================================
# Plots
# ============================================================================
METHOD_LABELS = {
    "test_only": "Method 1: v from train; pseudo-R^2 fit on TEST only",
    "train_plus_test": "Method 2: v from train; pseudo-R^2 fit on TRAIN + TEST",
}


def plot_layer_sweep(metrics_df, behavior, output_path, method,
                     layer_lo=10, layer_hi=32):
    """Plot 1: 3-panel layer sweep (top R^2(h), middle R^2(v), bottom ratio)."""
    df = metrics_df.sort_values("layer").reset_index(drop=True)
    layers = df["layer"].values

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)

    ax = axes[0]
    ax.plot(layers, df["r2_h"].values, "o-", color="#1f77b4", linewidth=2,
            markersize=6, label=r"pseudo-$R^2(h)$")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel(r"pseudo-$R^2(h)$  (full activation)", fontsize=10)
    ax.set_title(
        f"{behavior} (MCQA) - Layer sweep of directional information\n"
        f"{METHOD_LABELS.get(method, method)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(layers, df["r2_v"].values, "o-", color="#2ca02c", linewidth=2,
            markersize=6, label=r"pseudo-$R^2(v)$")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel(r"pseudo-$R^2(v)$  (DiffMean projection)", fontsize=10)
    ax.legend(loc="best", fontsize=9)

    ax = axes[2]
    ax.plot(layers, df["ratio"].values, "o-", color="#d62728", linewidth=2,
            markersize=6, label=r"pseudo-$R^2(v)$/pseudo-$R^2(h)$")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.7,
               label="ratio = 1")
    ax.set_ylabel(r"pseudo-$R^2(v)/$pseudo-$R^2(h)$", fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.legend(loc="best", fontsize=9)

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

    ax.set_xlabel(r"pseudo-$R^2(v)/$pseudo-$R^2(h)$  (directional information)", fontsize=11)
    ax.set_ylabel(r"$\Delta_l$  (best steered acc - baseline acc)", fontsize=11)
    ax.set_title(
        f"{behavior} (MCQA): directional information vs steering effectiveness\n"
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
        description="Experiment 1 (MCQA): Directional information (pseudo-R^2 ratio)"
    )
    parser.add_argument("--behavior", type=str, required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--mcqa_link_dir", type=str, required=True,
                        help="Dir with per_prompt_results.csv + per_layer_summary.csv "
                             "(produced by mcqa_projection_link.py)")
    parser.add_argument("--train_dataset_path", type=str, default=None,
                        help="Path to MCQA train dataset.json "
                             "(default: datasets/raw/{behavior}/dataset.json)")
    parser.add_argument("--test_dataset_path", type=str, default=None,
                        help="Path to MCQA test_dataset_ab.json "
                             "(default: datasets/test/{behavior}/test_dataset_ab.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: {mcqa_link_dir}/directional_information_analysis)")
    parser.add_argument("--layers", type=str, default="10-32",
                        help="Layer range or comma list, e.g. '10-32' or '10,11,12'")
    parser.add_argument("--max_examples", type=int, default=300,
                        help="Max training pairs for DiffMean (random sample)")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Max test prompts to use")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip model inference; reuse cached activations + CSV")
    args = parser.parse_args()

    mcqa_link_dir = Path(args.mcqa_link_dir)
    train_dataset_path = Path(
        args.train_dataset_path or TRAIN_PATH_MAP[args.behavior]
    )
    test_dataset_path = Path(
        args.test_dataset_path or TEST_PATH_MAP[args.behavior]
    )
    out_dir = Path(args.output_dir) if args.output_dir else mcqa_link_dir / "directional_information_analysis"
    plot_dir = out_dir / "plots"

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

    per_prompt_csv = mcqa_link_dir / "per_prompt_results.csv"
    per_layer_summary_csv = mcqa_link_dir / "per_layer_summary.csv"

    # Filter to layers actually present in the mcqa_projection_link summary
    # (so Delta_l aligns); fall back to args.layers if summary missing.
    avail = set(discover_layers_from_summary(per_layer_summary_csv))
    if avail:
        target_layers = [l for l in target_layers if l in avail]
    if not target_layers:
        logger.error(
            f"No valid layers found. Check {per_layer_summary_csv} "
            f"or pass --layers explicitly."
        )
        sys.exit(1)
    layer_lo, layer_hi = min(target_layers), max(target_layers)
    logger.warning(f"Target layers: {target_layers}")

    baseline_correct_map = load_baseline_correct(per_prompt_csv)
    logger.warning(
        f"Loaded baseline_correct for {len(baseline_correct_map)} prompts "
        f"from {per_prompt_csv}"
    )

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
            cached_layers = set(train_acts.keys())
            if not set(target_layers).issubset(cached_layers):
                logger.warning("Cached train activations missing some target layers; "
                               "re-extracting.")
                _ensure_model_loaded()
                train_acts, train_meta = extract_train_activations(
                    model, tokenizer, train_dataset_path, target_layers, device,
                    batch_size=args.batch_size, max_examples=args.max_examples,
                    seed=args.seed,
                )
                torch.save({"train_acts": train_acts, "train_meta": train_meta},
                           train_acts_path)
        else:
            _ensure_model_loaded()
            train_acts, train_meta = extract_train_activations(
                model, tokenizer, train_dataset_path, target_layers, device,
                batch_size=args.batch_size, max_examples=args.max_examples,
                seed=args.seed,
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
                    model, tokenizer, test_dataset_path, baseline_correct_map,
                    target_layers, device,
                    batch_size=args.batch_size, max_test=args.max_test,
                )
                torch.save({"test_mean_acts": test_mean_acts,
                            "test_metadata": test_metadata}, test_acts_path)
        else:
            _ensure_model_loaded()
            test_mean_acts, test_metadata = extract_test_activations(
                model, tokenizer, test_dataset_path, baseline_correct_map,
                target_layers, device,
                batch_size=args.batch_size, max_test=args.max_test,
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

    # ---------- Phase 6: Delta_l from mcqa_projection_link summary ------
    logger.warning("Phase 6: Reading Delta_l from per_layer_summary.csv ...")
    delta_df = compute_delta_l(per_layer_summary_csv, target_layers)
    delta_df.to_csv(out_dir / "delta_l.csv", index=False)

    # ---------- Phases 4 / 5: pseudo-R^2 metrics for BOTH methods -------
    method_results = {}
    for mode in ("test_only", "train_plus_test"):
        logger.warning(f"Phases 4/5 [{mode}]: Computing pseudo-R^2(v), pseudo-R^2(h), ratio ...")
        metrics_df = compute_r2_metrics(
            test_mean_acts, test_metadata, directions, centroid_norm,
            target_layers, train_acts, mode=mode,
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
        "regression": "logistic (McFadden pseudo-R^2)",
        "mcqa_link_dir": str(mcqa_link_dir),
        "train_dataset_path": str(train_dataset_path),
        "test_dataset_path": str(test_dataset_path),
        "layers": target_layers,
        "fit_strategy": "in_sample (both methods)",
        "method_descriptions": METHOD_LABELS,
        "metrics_test_only": method_results["test_only"].to_dict(orient="records"),
        "metrics_train_plus_test": method_results["train_plus_test"].to_dict(orient="records"),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
