"""
Open-ended sweep → projection histograms, match-rate, and MCC-vs-d' plots.

Per behavior, this script reads the already-completed filtered_sweep results:
  {sweep_dir}/layer_{L}/DiffMean_weight.pt       — v = mu_pos - mu_neg  (train)
  {sweep_dir}/layer_{L}/separability.json        — {"dprime": ...}      (train)
  {sweep_dir}/layer_{L}/eval/eval_results.parquet — question/generation/
                                                     behavior_score per steering_factor
  {sweep_dir}/projection_analysis/centroids.csv       — per-layer midpoint/half_range
  {sweep_dir}/projection_analysis/train_projections.csv — per-layer train κ (pos/neg)

The train-side κ statistics (centroids, train pos/neg projections) are already
on disk (written by projection_vs_steering_factor.py) and are NOT recomputed
here. What's missing — and what this script computes via batched GPU forward
passes — is κ for every STEERED generation at every requested factor. That
reuses the exact mean-response-token / centroid-normalization convention
established by projection_vs_steering_factor.py, so steered κ is directly
comparable to the existing train/baseline κ:

    κ = (mean_response_act(question, generation) · v_hat - midpoint) / half_range

(pos train centroid → +1, neg train centroid → -1, matching Braun's κ_a.)

Outputs, under {sweep_dir}/open_ended_projection_plots/:
  steered_kappa.csv          — one row per (layer, factor, question_idx)
  layer_summary.csv          — one row per layer: dprime, avg score / match
                                rate / sign-MCC per factor, best-α MCC
  r_values.csv               — cross-layer Pearson r (d' vs match rate @ best α,
                                d' vs sign-MCC @ best α), computed exactly as in
                                mcqa_projection_link.py. Use --r_value to produce
                                just this table (skip plots; add --replot_only to
                                reuse cached steered_kappa.csv without GPU).
  run_summary.json           — args + which requested factors were missing
  plots/
    histogram_layer_{L}.png
    match_rate_and_dprime.png
    best_match_rate_by_layer.png (best match rate across steered factors per
                                   layer, i.e. which α is best at each layer)
    avg_score_and_dprime.png
    mcc_test_best_alpha_vs_dprime.png (best α chosen on the same test prompts
                                        the MCC is evaluated on — there's no
                                        separate val split for this sweep, so
                                        this is the only MCC-vs-α variant)

Usage:
    uv run python axbench/scripts/open_ended_projection_plots.py \\
        --behavior corrigible-neutral-HHH \\
        --sweep_dir /vol/filtered_sweep/gemma-2-9b-it/corrigible-neutral-HHH-sweep \\
        --layers 10-32 --factors 0,1,2,3,5,10
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Behavior config (mirrors open_ended_projection_link.py / projection_vs_steering_factor.py)
# ---------------------------------------------------------------------------
BEHAVIOR_SCALES = {
    "survival-instinct":      (-5, 5, 0),
    "myopic-reward":          (-5, 5, 0),
    "corrigible-neutral-HHH": (-5, 5, 0),
    "hallucination":          (0, 5, None),
}
BEHAVIORS = list(BEHAVIOR_SCALES.keys())
BEHAVIOR_THRESHOLDS = {b: (mn + mx) / 2.0 for b, (mn, mx, _) in BEHAVIOR_SCALES.items()}

DEFAULT_FACTORS = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
DEFAULT_LAYERS = list(range(10, 33))
FACTOR_TOL = 1e-6


# ---------------------------------------------------------------------------
# Small pure-numpy metrics (self-contained; duplicated from open_ended_projection_link.py
# on purpose so this script has no cross-file coupling)
# ---------------------------------------------------------------------------
def safe_mcc(pred, actual) -> float:
    pred, actual = np.asarray(pred, int), np.asarray(actual, int)
    if pred.std() < 1e-9 or actual.std() < 1e-9:
        return float("nan")
    return float(matthews_corrcoef(actual, pred))


def behavior_label(score, threshold: float):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    return int(float(score) > threshold)


def sign_mcc_with_diag(kappas: np.ndarray, labels: np.ndarray, min_examples: int = 8) -> dict:
    """Sign-MCC of predictor sign(κ)>0 against binary labels, with a diagnostic breakdown."""
    kappas = np.asarray(kappas, float)
    labels = np.asarray(labels, int)
    mask = np.isfinite(kappas) & (labels >= 0)
    n_used = int(mask.sum())
    out = {
        "mcc": float("nan"), "n_used": n_used, "n_pos": 0, "n_neg": 0,
        "reason": "too_few_examples",
    }
    if n_used < min_examples:
        return out
    k, a = kappas[mask], labels[mask]
    p = (k > 0).astype(int)
    out["n_pos"] = int((a == 1).sum())
    out["n_neg"] = int((a == 0).sum())
    if a.std() < 1e-9:
        out["reason"] = "constant_label"
        return out
    if p.std() < 1e-9:
        out["reason"] = "constant_predictor"
        return out
    out["mcc"] = safe_mcc(p, a)
    out["reason"] = "ok"
    return out


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------
def parse_layers(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(x.strip()) for x in spec.split(",") if x.strip())


def parse_factors(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# On-disk artifact loaders
# ---------------------------------------------------------------------------
def discover_layers(sweep_dir: Path, requested_layers: list[int]) -> list[int]:
    ok = []
    for l in requested_layers:
        d = sweep_dir / f"layer_{l}"
        if (d / "DiffMean_weight.pt").exists() and (d / "eval" / "eval_results.parquet").exists():
            ok.append(l)
        else:
            logger.warning(f"Layer {l}: missing DiffMean_weight.pt or eval_results.parquet — skipping")
    return ok


def load_steering_unit_vector(sweep_dir: Path, layer: int) -> torch.Tensor | None:
    p = sweep_dir / f"layer_{layer}" / "DiffMean_weight.pt"
    v = torch.load(p, map_location="cpu", weights_only=True).float()
    if v.dim() == 2:
        v = v.squeeze(0)
    norm = v.norm().item()
    if norm < 1e-12:
        logger.warning(f"Layer {layer}: near-zero steering vector norm — skipping")
        return None
    return v / norm


def load_dprime(sweep_dir: Path, layer: int) -> float:
    p = sweep_dir / f"layer_{layer}" / "separability.json"
    if not p.exists():
        return float("nan")
    with open(p) as f:
        return float(json.load(f).get("dprime", float("nan")))


def load_centroids(sweep_dir: Path) -> dict[int, tuple[float, float]]:
    p = sweep_dir / "projection_analysis" / "centroids.csv"
    if not p.exists():
        logger.warning(f"No {p} — κ-based plots (histograms, MCC) will be skipped entirely")
        return {}
    df = pd.read_csv(p)
    return {
        int(r["layer"]): (float(r["midpoint"]), float(r["half_range"]))
        for _, r in df.iterrows()
    }


def load_train_kappa(sweep_dir: Path) -> dict[int, dict[str, list[float]]]:
    p = sweep_dir / "projection_analysis" / "train_projections.csv"
    if not p.exists():
        logger.warning(f"No {p} — Train panel of projection histograms will be empty")
        return {}
    df = pd.read_csv(p)
    out: dict[int, dict[str, list[float]]] = {}
    for layer, ldf in df.groupby("layer"):
        out[int(layer)] = {
            "pos": ldf.loc[ldf["label"] == 1, "normalized_projection"].dropna().tolist(),
            "neg": ldf.loc[ldf["label"] == 0, "normalized_projection"].dropna().tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Batched GPU extraction: mean response-token activation at one layer
# ---------------------------------------------------------------------------
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def _format_texts(tokenizer, use_chat: bool, question: str, answer: str) -> tuple[str, str]:
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


def _prep_example(tokenizer, use_chat: bool, question: str, answer: str, max_length: int):
    q_text, full_text = _format_texts(tokenizer, use_chat, question, answer)
    full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]
    q_ids = tokenizer(q_text, truncation=True, max_length=max_length)["input_ids"]
    q_len, f_len = len(q_ids), len(full_ids)
    if f_len <= q_len:
        return None
    return full_ids, q_len, f_len


def _pad_batch(token_lists: list[list[int]], pad_id: int, device):
    max_len = max(len(t) for t in token_lists)
    B = len(token_lists)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, t in enumerate(token_lists):
        ids[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        mask[i, :len(t)] = 1
    return ids, mask


@torch.no_grad()
def compute_layer_mean_acts(
    model, tokenizer, layer: int, rows: list[dict], batch_size: int, device,
    max_length: int = 1024,
) -> dict[int, np.ndarray]:
    """
    rows: [{"question_idx": int, "question": str, "generation": str}, ...]
    Returns {question_idx: mean response-token activation (np.ndarray[hidden])}
    for every row whose generation was non-empty and produced a non-degenerate
    response span. Rows that don't qualify are simply absent from the result.
    """
    use_chat = supports_chat_template(tokenizer)
    pad_id = tokenizer.pad_token_id

    prepped = []
    for r in rows:
        gen = r["generation"]
        if gen is None or (isinstance(gen, float) and np.isnan(gen)) or str(gen).strip() == "":
            continue
        out = _prep_example(tokenizer, use_chat, r["question"], str(gen), max_length)
        if out is None:
            continue
        full_ids, q_len, f_len = out
        prepped.append((r["question_idx"], full_ids, q_len, f_len))

    results: dict[int, np.ndarray] = {}
    if not prepped:
        return results

    captured = {}

    def _hook(mod, inp, out):
        captured["h"] = out[0].detach()

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        for start in range(0, len(prepped), batch_size):
            chunk = prepped[start:start + batch_size]
            ids, mask = _pad_batch([c[1] for c in chunk], pad_id, device)
            model(input_ids=ids, attention_mask=mask)
            hidden = captured["h"].float()  # [B, seq, hidden]
            for i, (qidx, _, q_len, f_len) in enumerate(chunk):
                resp = hidden[i, q_len:f_len]  # [resp_len, hidden]
                if resp.shape[0] == 0:
                    continue
                results[qidx] = resp.mean(dim=0).cpu().numpy()
            del hidden
            torch.cuda.empty_cache()
    finally:
        handle.remove()

    return results


# ---------------------------------------------------------------------------
# Per-behavior pipeline
# ---------------------------------------------------------------------------
def extract_steered_kappa(
    sweep_dir: Path, layers: list[int], requested_factors: list[float],
    model, tokenizer, device, batch_size: int, max_length: int,
) -> tuple[pd.DataFrame, dict]:
    centroids = load_centroids(sweep_dir)
    missing_by_layer: dict[int, list[float]] = {}
    rows_out = []

    for layer in layers:
        v_hat = load_steering_unit_vector(sweep_dir, layer)
        if v_hat is None:
            continue
        v_hat = v_hat.to(device)

        eval_path = sweep_dir / f"layer_{layer}" / "eval" / "eval_results.parquet"
        edf = pd.read_parquet(eval_path)
        present = edf["steering_factor"].unique()

        used_factors, missing = [], []
        for f in requested_factors:
            if np.any(np.isclose(present, f, atol=FACTOR_TOL)):
                used_factors.append(f)
            else:
                missing.append(f)
        if missing:
            missing_by_layer[layer] = missing
            logger.warning(f"Layer {layer}: requested factors not in sweep data, skipping: {missing}")

        m, h = centroids.get(layer, (None, None))
        if m is None:
            logger.warning(f"Layer {layer}: no centroid — κ will be NaN for all rows")

        for f in used_factors:
            sub = edf[np.isclose(edf["steering_factor"], f, atol=FACTOR_TOL)]
            batch_rows = [
                {"question_idx": int(r["question_idx"]), "question": r["question"],
                 "generation": r["generation"]}
                for _, r in sub.iterrows()
            ]
            mean_acts = compute_layer_mean_acts(
                model, tokenizer, layer, batch_rows, batch_size, device, max_length,
            )
            for _, r in sub.iterrows():
                qidx = int(r["question_idx"])
                act = mean_acts.get(qidx)
                if act is None:
                    proj, kappa = float("nan"), float("nan")
                else:
                    proj = float(np.dot(act, v_hat.cpu().numpy()))
                    kappa = (proj - m) / h if (m is not None and abs(h) > 1e-12) else float("nan")
                rows_out.append({
                    "layer": layer, "factor": f, "question_idx": qidx,
                    "question": r["question"], "generation": r["generation"],
                    "behavior_score": r["behavior_score"],
                    "projection": proj, "kappa": kappa,
                })
        logger.warning(f"Layer {layer}: done ({len(used_factors)} factors)")

    return pd.DataFrame(rows_out), missing_by_layer


def compute_layer_summary(
    kappa_df: pd.DataFrame, layers: list[int], factors: list[float],
    sweep_dir: Path, behavior: str, min_examples: int = 8,
) -> pd.DataFrame:
    threshold = BEHAVIOR_THRESHOLDS[behavior]
    rows = []

    for layer in layers:
        ldf = kappa_df[kappa_df["layer"] == layer]
        if ldf.empty:
            continue
        row: dict = {"layer": layer, "dprime": load_dprime(sweep_dir, layer)}

        factors_here = sorted(ldf["factor"].unique())
        non_zero = [f for f in factors_here if abs(f) > FACTOR_TOL]

        for f in factors_here:
            fdf = ldf[ldf["factor"] == f]
            scores = fdf["behavior_score"].apply(
                lambda x: float(x) if x is not None and not pd.isna(x) else float("nan")
            ).values
            kappas = fdf["kappa"].astype(float).values
            labels = np.array([
                behavior_label(s, threshold) if np.isfinite(s) else -1 for s in scores
            ])
            valid = labels >= 0

            tag = "0" if abs(f) <= FACTOR_TOL else f"{f:g}"
            row[f"avg_behavior_score_{tag}"] = float(scores[valid].mean()) if valid.sum() else float("nan")
            row[f"match_rate_{tag}"] = float((labels[valid] == 1).mean()) if valid.sum() else float("nan")

            diag = sign_mcc_with_diag(kappas, labels, min_examples)
            row[f"sign_mcc_{tag}"] = diag["mcc"]
            row[f"mcc_n_used_{tag}"] = diag["n_used"]
            row[f"mcc_reason_{tag}"] = diag["reason"]

        # best α chosen on TEST itself (circular; matches mcqa's "test-best" variant)
        best_test_mcc, best_test_alpha = float("nan"), float("nan")
        for f in non_zero:
            tag = f"{f:g}"
            mcc = row.get(f"sign_mcc_{tag}", float("nan"))
            if np.isfinite(mcc) and (np.isnan(best_test_mcc) or mcc > best_test_mcc):
                best_test_mcc, best_test_alpha = mcc, f
        row["sign_mcc_test_best_alpha"] = best_test_mcc
        row["best_factor_test"] = best_test_alpha

        # best match rate across steered factors (which α gets the most prompts
        # judged as exhibiting the behavior, at this layer)
        best_match_rate, best_match_rate_alpha = float("nan"), float("nan")
        for f in non_zero:
            tag = f"{f:g}"
            mr = row.get(f"match_rate_{tag}", float("nan"))
            if np.isfinite(mr) and (np.isnan(best_match_rate) or mr > best_match_rate):
                best_match_rate, best_match_rate_alpha = mr, f
        row["best_match_rate"] = best_match_rate
        row["best_match_rate_factor"] = best_match_rate_alpha

        rows.append(row)

    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-layer Pearson r values (mirrors mcqa_projection_link.py exactly:
# mask both columns notna, require >= 3 layers, require nonzero std, then
# scipy.stats.pearsonr).
# ---------------------------------------------------------------------------
def pearson_across_layers(layer_df: pd.DataFrame, x_col: str, y_col: str) -> dict | None:
    if x_col not in layer_df.columns or y_col not in layer_df.columns:
        return None
    mask = layer_df[x_col].notna() & layer_df[y_col].notna()
    if mask.sum() < 3:
        return None
    xv = layer_df.loc[mask, x_col].values.astype(float)
    yv = layer_df.loc[mask, y_col].values.astype(float)
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return None
    r, p = scipy_stats.pearsonr(xv, yv)
    return {"pearson_r": float(r), "pearson_p": float(p), "n_layers": int(mask.sum())}


def compute_r_values(layer_df: pd.DataFrame, behavior: str) -> tuple[pd.DataFrame, dict]:
    """
    Two cross-layer Pearson r values, both at the best α chosen per layer on
    test (matching the existing 'test-best' MCC/match-rate columns):
      - d' vs best_match_rate         ("accuracy" at best α per layer)
      - d' vs sign_mcc_test_best_alpha (sign-MCC at best α per layer)
    """
    specs = [
        ("dprime_vs_match_rate_best_alpha", "dprime", "best_match_rate",
         "d' vs match rate @ best α per layer"),
        ("dprime_vs_sign_mcc_best_alpha", "dprime", "sign_mcc_test_best_alpha",
         "d' vs sign-MCC @ best α per layer"),
    ]
    rows = []
    blob: dict = {}
    for key, x_col, y_col, label in specs:
        res = pearson_across_layers(layer_df, y_col, x_col)  # r is symmetric
        blob[key] = res
        rows.append({
            "comparison": key,
            "label": label,
            "pearson_r": res["pearson_r"] if res else float("nan"),
            "pearson_p": res["pearson_p"] if res else float("nan"),
            "n_layers": res["n_layers"] if res else 0,
        })
        if res:
            sig = "*" if res["pearson_p"] < 0.05 else " "
            logger.warning(
                f"{behavior}: Pearson across layers — {label}: "
                f"r={res['pearson_r']:+.4f} (p={res['pearson_p']:.4g}){sig}  "
                f"(n={res['n_layers']} layers)"
            )
        else:
            logger.warning(f"{behavior}: Pearson across layers — {label}: insufficient data")
    return pd.DataFrame(rows), blob


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_projection_histograms(
    kappa_df: pd.DataFrame, layer: int, factors: list[float],
    train_kappa: dict, behavior: str, threshold: float, out_path: Path,
):
    ldf = kappa_df[kappa_df["layer"] == layer]
    if ldf.empty:
        return
    all_factors = sorted(ldf["factor"].unique(), key=lambda f: (abs(f), f))
    n_panels = 1 + len(all_factors)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    tp = train_kappa.get(layer, {})
    pos_proj = np.array(tp.get("pos", []), dtype=float)
    neg_proj = np.array(tp.get("neg", []), dtype=float)
    train_ax = axes_flat[0]
    if len(pos_proj) or len(neg_proj):
        all_t = np.concatenate([x for x in [pos_proj, neg_proj] if len(x)])
        pad = max(0.3, (all_t.max() - all_t.min()) * 0.05 + 0.01)
        bins = np.linspace(all_t.min() - pad, all_t.max() + pad, 25)
        if len(neg_proj):
            train_ax.hist(neg_proj, bins=bins, color="#d62728", alpha=0.55,
                          edgecolor="#a01010", linewidth=0.5, label="Non-matching")
        if len(pos_proj):
            train_ax.hist(pos_proj, bins=bins, color="#1f77b4", alpha=0.55,
                          edgecolor="#104e8b", linewidth=0.5, label="Matching")
        lo, hi = bins[0], bins[-1]
        if lo <= 0 <= hi:
            train_ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    train_ax.set_title("Train", fontsize=10, fontweight="bold")
    train_ax.set_xlabel("κ", fontsize=9)
    train_ax.set_ylabel("# examples", fontsize=9)
    train_ax.legend(fontsize=7)

    for panel_i, alpha in enumerate(all_factors):
        ax = axes_flat[1 + panel_i]
        fdf = ldf[ldf["factor"] == alpha]
        kap = fdf["kappa"].astype(float).values
        scr = fdf["behavior_score"].apply(
            lambda x: float(x) if x is not None and not pd.isna(x) else float("nan")
        ).values

        above = np.array([(s > threshold) if np.isfinite(s) else None for s in scr])
        finite_k = np.isfinite(kap)
        k_above = kap[finite_k & (above == True)]   # noqa: E712
        k_below = kap[finite_k & (above == False)]   # noqa: E712
        k_unk   = kap[finite_k & (above == None)]    # noqa: E711

        all_k = kap[finite_k]
        if len(all_k) == 0:
            ax.set_visible(False)
            continue
        pad_ = max(0.3, (all_k.max() - all_k.min()) * 0.05 + 0.01)
        bins = np.linspace(all_k.min() - pad_, all_k.max() + pad_, 20)

        if len(k_unk):
            ax.hist(k_unk, bins=bins, color="#aaaaaa", alpha=0.5,
                    edgecolor="#888888", linewidth=0.5, label="Unknown")
        if len(k_below):
            ax.hist(k_below, bins=bins, color="#d62728", alpha=0.55,
                    edgecolor="#a01010", linewidth=0.5, label="Below threshold")
        if len(k_above):
            ax.hist(k_above, bins=bins, color="#1f77b4", alpha=0.55,
                    edgecolor="#104e8b", linewidth=0.5, label="Above threshold")

        lo, hi = bins[0], bins[-1]
        if lo <= 0 <= hi:
            ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)

        ax.set_title(f"α={alpha:g}", fontsize=10, fontweight="bold")
        ax.set_xlabel("κ", fontsize=9)
        if panel_i == 0:
            ax.set_ylabel("# prompts", fontsize=9)
            ax.legend(fontsize=7)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"{behavior} — Layer {layer}: κ distribution (open-ended)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_match_rate_and_dprime(layer_df: pd.DataFrame, factors: list[float],
                               behavior: str, out_path: Path):
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(layers, layer_df.get("match_rate_0", pd.Series(dtype=float)).values,
             "D--", color="gray", linewidth=1.5, markersize=6,
             label="Baseline (α=0)", alpha=0.85, zorder=3)
    cmap = plt.get_cmap("plasma")
    non_zero = [f for f in factors if abs(f) > FACTOR_TOL]
    for i, f in enumerate(non_zero):
        col = f"match_rate_{f:g}"
        if col not in layer_df.columns:
            continue
        color = cmap(i / max(1, len(non_zero) - 1))
        ax1.plot(layers, layer_df[col].values, "o-", color=color,
                 linewidth=2, markersize=5, label=f"α={f:g}", zorder=3)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Behavior match rate (fraction of prompts)", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(layers)

    ax2 = ax1.twinx()
    if layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d' (train)", zorder=2)
        ax2.set_ylabel("d' (training discriminability)", fontsize=11, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, title="Steering factor / metric", fontsize=9,
               loc="best", framealpha=0.85)
    ax1.set_title(f"{behavior}: Behavior match rate & training d' by layer", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_best_match_rate_by_layer(layer_df: pd.DataFrame, behavior: str, out_path: Path):
    """Best match rate across steered factors per layer (which α is best), vs training d'."""
    if not np.any(np.isfinite(layer_df.get("best_match_rate", pd.Series(dtype=float)).values)):
        return
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(layers, layer_df.get("match_rate_0", pd.Series(dtype=float)).values,
             "D--", color="gray", linewidth=1.5, markersize=6,
             label="Baseline (α=0)", alpha=0.8, zorder=3)
    ax1.plot(layers, layer_df["best_match_rate"].values, "o-", color="#2E86AB",
             linewidth=2.5, markersize=7, label="Best match rate (best α per layer)", zorder=3)

    for x, y, f in zip(layers, layer_df["best_match_rate"].values,
                       layer_df["best_match_rate_factor"].values):
        if np.isfinite(y) and np.isfinite(f):
            ax1.annotate(f"α={f:g}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color="#2E86AB")

    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Behavior match rate (fraction of prompts)", fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(layers)

    ax2 = ax1.twinx()
    if layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d' (train)", zorder=2)
        ax2.set_ylabel("d' (training discriminability)", fontsize=11, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="best", framealpha=0.85)
    ax1.set_title(f"{behavior}: Best behavior match rate (best α per layer) & training d'",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_avg_score_and_dprime(layer_df: pd.DataFrame, factors: list[float],
                              behavior: str, out_path: Path):
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(13, 5))
    cmap = plt.get_cmap("plasma")
    all_factors = [0.0] + [f for f in factors if abs(f) > FACTOR_TOL]
    for i, f in enumerate(all_factors):
        tag = "0" if abs(f) <= FACTOR_TOL else f"{f:g}"
        col = f"avg_behavior_score_{tag}"
        if col not in layer_df.columns:
            continue
        color = "gray" if abs(f) <= FACTOR_TOL else cmap(i / max(1, len(all_factors) - 1))
        ls = "--" if abs(f) <= FACTOR_TOL else "-"
        ax1.plot(layers, layer_df[col].values, "o" + ls, color=color, linewidth=2,
                 markersize=5, label="Baseline (α=0)" if abs(f) <= FACTOR_TOL else f"α={f:g}")

    scale_min, scale_max, ref = BEHAVIOR_SCALES.get(behavior, (0, 10, 5))
    if ref is not None:
        ax1.axhline(ref, color="gray", linestyle=":", alpha=0.4, linewidth=0.9)
    ax1.set_ylim(scale_min - 0.5, scale_max + 0.5)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Avg behavior score", fontsize=11)
    ax1.set_xticks(layers)

    ax2 = ax1.twinx()
    if layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d'", zorder=2)
        ax2.set_ylabel("d' (training discriminability)", fontsize=11, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="best", framealpha=0.85)
    ax1.set_title(f"{behavior}: Avg behavior score & d' by layer (open-ended)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mcc_vs_dprime(layer_df: pd.DataFrame, mcc_col: str, mcc_label: str,
                       title: str, out_path: Path):
    if mcc_col not in layer_df.columns or not np.any(np.isfinite(layer_df[mcc_col].values.astype(float))):
        return
    layers = layer_df["layer"].values
    mcc = layer_df[mcc_col].values.astype(float)

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(layers, mcc, "o-", color="#C73E1D", linewidth=2, markersize=6,
             label=mcc_label, zorder=3)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Matthews correlation coefficient", fontsize=11)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xticks(layers)
    ax1.tick_params(axis="y", labelcolor="#C73E1D")

    ax2 = ax1.twinx()
    if layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d' (train)", zorder=2)
        ax2.set_ylabel("d' (training discriminability)", fontsize=11, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="best", framealpha=0.9)
    ax1.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", required=True, choices=BEHAVIORS)
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Default: /vol/filtered_sweep/gemma-2-9b-it/<behavior>-sweep")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--layers", type=str, default="10-32")
    parser.add_argument("--factors", type=str, default="0,1,2,3,5,10")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--min_examples", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip GPU extraction; reuse cached steered_kappa.csv")
    parser.add_argument("--r_value", action="store_true",
                        help="Only compute the cross-layer Pearson r table "
                             "(d' vs match rate / sign-MCC at best α per layer) "
                             "and write r_values.csv; skip all plotting. "
                             "Combine with --replot_only to reuse cached "
                             "steered_kappa.csv and avoid GPU extraction.")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir or f"/vol/filtered_sweep/gemma-2-9b-it/{args.behavior}-sweep")
    out_dir = Path(args.out_dir or sweep_dir / "open_ended_projection_plots")
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    requested_layers = parse_layers(args.layers)
    requested_factors = parse_factors(args.factors)

    kappa_csv = out_dir / "steered_kappa.csv"
    missing_by_layer: dict[int, list[float]] = {}

    if args.replot_only and kappa_csv.exists():
        logger.warning(f"--replot_only: loading cached {kappa_csv}")
        kappa_df = pd.read_csv(kappa_csv)
        layers = sorted(kappa_df["layer"].unique().tolist())
    else:
        layers = discover_layers(sweep_dir, requested_layers)
        if not layers:
            logger.error(f"No usable layers found under {sweep_dir}")
            sys.exit(1)
        logger.warning(f"{args.behavior}: layers={layers}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning(f"Device: {device}. Loading {args.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.max_length)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device,
        )
        model.eval()

        kappa_df, missing_by_layer = extract_steered_kappa(
            sweep_dir, layers, requested_factors, model, tokenizer, device,
            args.batch_size, args.max_length,
        )
        kappa_df.to_csv(kappa_csv, index=False)
        logger.warning(f"Saved {len(kappa_df)} rows → {kappa_csv}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Layer summary metrics ----
    layer_df = compute_layer_summary(
        kappa_df, layers, requested_factors, sweep_dir, args.behavior, args.min_examples,
    )
    layer_df.to_csv(out_dir / "layer_summary.csv", index=False)

    # ---- Cross-layer Pearson r table (d' vs match rate / sign-MCC @ best α) ----
    r_df, r_blob = compute_r_values(layer_df, args.behavior)
    r_df.to_csv(out_dir / "r_values.csv", index=False)
    logger.warning(f"Saved r-value table → {out_dir / 'r_values.csv'}")

    if args.r_value:
        with open(out_dir / "run_summary.json", "w") as f:
            json.dump({
                "behavior": args.behavior,
                "sweep_dir": str(sweep_dir),
                "layers": layers,
                "requested_factors": requested_factors,
                "missing_factors_by_layer": {str(k): v for k, v in missing_by_layer.items()},
                "min_examples": args.min_examples,
                "r_values": r_blob,
            }, f, indent=2)
        logger.warning(f"--r_value: r table only, skipping plots. Done → {out_dir}")
        return

    # ---- Plots ----
    train_kappa = load_train_kappa(sweep_dir)
    threshold = BEHAVIOR_THRESHOLDS[args.behavior]

    for layer in layers:
        plot_projection_histograms(
            kappa_df, layer, requested_factors, train_kappa, args.behavior,
            threshold, plot_dir / f"histogram_layer_{layer}.png",
        )

    plot_match_rate_and_dprime(layer_df, requested_factors, args.behavior,
                               plot_dir / "match_rate_and_dprime.png")
    plot_best_match_rate_by_layer(layer_df, args.behavior,
                                  plot_dir / "best_match_rate_by_layer.png")
    plot_avg_score_and_dprime(layer_df, requested_factors, args.behavior,
                              plot_dir / "avg_score_and_dprime.png")
    plot_mcc_vs_dprime(
        layer_df, "sign_mcc_test_best_alpha",
        mcc_label="MCC(sign κ vs steer match @ test-best α)",
        title=f"{args.behavior}: MCC @ best α per layer (chosen on test — circular)",
        out_path=plot_dir / "mcc_test_best_alpha_vs_dprime.png",
    )

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump({
            "behavior": args.behavior,
            "sweep_dir": str(sweep_dir),
            "layers": layers,
            "requested_factors": requested_factors,
            "missing_factors_by_layer": {str(k): v for k, v in missing_by_layer.items()},
            "min_examples": args.min_examples,
            "r_values": r_blob,
        }, f, indent=2)

    logger.warning(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
