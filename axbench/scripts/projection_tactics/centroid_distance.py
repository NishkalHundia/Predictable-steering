"""
Test-Time Centroid-Distance Analysis.

For each layer l of a contrastive training set:
  1. Per training prompt, compute a single mean activation by averaging over
     response-token positions (one vector per prompt, per class, per layer).
  2. The class centroid is the mean of those per-prompt means:
        mu_pos^l = mean_i a_bar(x_i, y_i^+)
        mu_neg^l = mean_i a_bar(x_i, y_i^-)
  3. Build two "epsilon balls" around each centroid, where
        epsilon_pos^l = radius that encompasses (almost) every training
                        positive activation that was averaged in,
     and similarly for neg. We record both the true max radius and a robust
     percentile radius (default 95).
  4. For each test prompt, take the mean response-token activation of its
     unsteered (factor=0) generation, then measure Euclidean distance from
     mu_pos^l and mu_neg^l.
  5. Correlate these distances against the baseline (unsteered) behavior
     score:
        - dist(test, mu_neg) vs score -> expected positive correlation
        - dist(test, mu_pos) vs score -> expected negative correlation
  6. Repeat for every layer in the sweep.

Requires:
  - Completed sweep (sweep_layers_open_ended.py) with eval results
  - Training contrastive dataset (for centroid computation)

Outputs:
  {sweep_dir}/centroid_distance_analysis/
    train_distances.csv           (training activations, distance to both centroids)
    test_distances.csv            (test activations, distance to both centroids + score)
    centroids.csv                 (per-layer centroid norms + epsilon radii)
    correlations.csv              (per-layer corr of both distances vs baseline score)
    analysis_summary.json
    plots/
      scatter_neg_layer_{N}.png   (dist-to-neg centroid vs baseline)
      scatter_pos_layer_{N}.png   (dist-to-pos centroid vs baseline)
      distribution_layer_{N}.png  (train vs test distance distribution)
      correlation_across_layers.png

Usage:
    uv run python axbench/scripts/projection_tactics/centroid_distance.py \\
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
# Helpers
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


def gather_multi_layer_activations(model, target_layers: list[int], inputs: dict):
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
    """Forward pass -> mean response-token activation at each target layer.

    One vector per prompt per layer: mean over response token positions.
    """
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
# Extraction: raw training activations
# ============================================================================
@torch.no_grad()
def extract_train_activations(model, tokenizer, train_dataset_path,
                              target_layers, device):
    """
    For each training pair, compute the mean response-token activation for
    the positive and negative answers at every target layer.

    Returns:
        train_acts: {layer: {"pos": [tensor, ...], "neg": [tensor, ...]}}
        train_meta: list of dicts with pair_idx, label, label_name
    """
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

        if pair_idx % 5 == 0:
            torch.cuda.empty_cache()

    for layer in target_layers:
        logger.warning(
            f"  Layer {layer}: {len(train_acts[layer]['pos'])} pos, "
            f"{len(train_acts[layer]['neg'])} neg activations"
        )

    return train_acts, train_meta


# ============================================================================
# Centroid + epsilon-ball computation
# ============================================================================
def compute_centroids_and_radii(train_acts, target_layers, percentile=95.0):
    """
    Per layer, compute:
        mu_pos, mu_neg                     — centroids (mean of per-prompt means)
        eps_pos_max, eps_pos_pct           — L2 radius covering all / `percentile`%
        eps_neg_max, eps_neg_pct           — of the training activations.
        centroid_separation                — || mu_pos - mu_neg ||_2

    Returns:
        centroids: {layer: {"mu_pos", "mu_neg"}}   (cpu tensors)
        radii_df: DataFrame with per-layer radii / separation summary
    """
    centroids = {}
    rows = []
    for layer in target_layers:
        pos = torch.stack(train_acts[layer]["pos"])  # [n_pos, hidden]
        neg = torch.stack(train_acts[layer]["neg"])  # [n_neg, hidden]

        mu_pos = pos.mean(dim=0)
        mu_neg = neg.mean(dim=0)

        d_pos = (pos - mu_pos).norm(dim=1)   # distance of each pos activation to mu_pos
        d_neg = (neg - mu_neg).norm(dim=1)

        eps_pos_max = d_pos.max().item()
        eps_neg_max = d_neg.max().item()
        eps_pos_pct = float(np.percentile(d_pos.numpy(), percentile))
        eps_neg_pct = float(np.percentile(d_neg.numpy(), percentile))

        sep = (mu_pos - mu_neg).norm().item()

        centroids[layer] = {"mu_pos": mu_pos, "mu_neg": mu_neg}
        rows.append({
            "layer": layer,
            "mu_pos_norm": mu_pos.norm().item(),
            "mu_neg_norm": mu_neg.norm().item(),
            "centroid_separation": sep,
            "eps_pos_max": eps_pos_max,
            "eps_neg_max": eps_neg_max,
            f"eps_pos_p{int(percentile)}": eps_pos_pct,
            f"eps_neg_p{int(percentile)}": eps_neg_pct,
            "train_dist_pos_mean": d_pos.mean().item(),
            "train_dist_neg_mean": d_neg.mean().item(),
            "n_pos": int(pos.shape[0]),
            "n_neg": int(neg.shape[0]),
        })
        logger.warning(
            f"  Layer {layer:2d}: centroid_sep = {sep:.3f}, "
            f"eps_pos(max/p{int(percentile)}) = {eps_pos_max:.3f}/{eps_pos_pct:.3f}, "
            f"eps_neg(max/p{int(percentile)}) = {eps_neg_max:.3f}/{eps_neg_pct:.3f}"
        )

    return centroids, pd.DataFrame(rows)


# ============================================================================
# Distance helpers
# ============================================================================
def compute_train_distances(train_acts, centroids, target_layers):
    """Per-layer, per-activation distance to both centroids + in-ball flags."""
    rows = []
    for layer in target_layers:
        mu_pos = centroids[layer]["mu_pos"]
        mu_neg = centroids[layer]["mu_neg"]
        for key, label in [("pos", 1), ("neg", 0)]:
            for i, act in enumerate(train_acts[layer][key]):
                d_pos = (act - mu_pos).norm().item()
                d_neg = (act - mu_neg).norm().item()
                rows.append({
                    "pair_idx": i,
                    "label": label,
                    "label_name": key,
                    "layer": int(layer),
                    "dist_to_pos_centroid": d_pos,
                    "dist_to_neg_centroid": d_neg,
                })
    return pd.DataFrame(rows)


@torch.no_grad()
def extract_test_distances(model, tokenizer, sweep_dir, centroids,
                           target_layers, device, return_raw=False):
    """
    For each test prompt, run the unsteered (factor=0) generation through the
    model, take the mean response-token activation at each target layer, and
    compute Euclidean distance to mu_pos and mu_neg.

    If return_raw=True, also returns the raw cached activations + metadata so
    they can be saved to disk and reused.
    """
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

    logger.warning(f"Extracting centroid distances for {len(baseline_df)} test prompts")

    rows = []
    raw_acts = {}
    raw_meta = []

    for idx in tqdm(range(len(baseline_df)), desc="Test distances"):
        row = baseline_df.iloc[idx]
        question = row["question"]
        generation = row["generation"]
        question_idx = row.get("question_idx", idx)

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
            mu_pos = centroids[layer]["mu_pos"].to(act.device)
            mu_neg = centroids[layer]["mu_neg"].to(act.device)
            d_pos = (act - mu_pos).norm().item()
            d_neg = (act - mu_neg).norm().item()
            bs = baseline_scores.get(layer, {}).get(question_idx, np.nan)

            rows.append({
                "question_idx": int(question_idx),
                "layer": int(layer),
                "dist_to_pos_centroid": d_pos,
                "dist_to_neg_centroid": d_neg,
                "baseline_score": bs,
                "question": question,
                "generation": str(generation),
            })

            if return_raw:
                raw_acts[(int(question_idx), int(layer))] = act.cpu()
                raw_meta.append({
                    "question_idx": int(question_idx),
                    "layer": int(layer),
                    "idx": idx,
                    "baseline_score": bs,
                    "question": question,
                    "generation": str(generation),
                })

        if idx % 5 == 0:
            torch.cuda.empty_cache()

    logger.warning(f"Extracted {len(rows)} test distance rows")
    df = pd.DataFrame(rows)
    if return_raw:
        return df, raw_acts, raw_meta
    return df


def annotate_in_ball(df, radii_df, percentile):
    """Add boolean columns 'inside_pos_ball' / 'inside_neg_ball' using
    the max-radius epsilon and the percentile epsilon."""
    pct_col_pos = f"eps_pos_p{int(percentile)}"
    pct_col_neg = f"eps_neg_p{int(percentile)}"
    radii_map = {
        int(r["layer"]): (
            r["eps_pos_max"], r["eps_neg_max"],
            r[pct_col_pos], r[pct_col_neg],
        )
        for _, r in radii_df.iterrows()
    }
    in_pos_max, in_neg_max, in_pos_pct, in_neg_pct = [], [], [], []
    for _, row in df.iterrows():
        eps_p_max, eps_n_max, eps_p_pct, eps_n_pct = radii_map.get(
            int(row["layer"]), (np.inf, np.inf, np.inf, np.inf)
        )
        in_pos_max.append(row["dist_to_pos_centroid"] <= eps_p_max)
        in_neg_max.append(row["dist_to_neg_centroid"] <= eps_n_max)
        in_pos_pct.append(row["dist_to_pos_centroid"] <= eps_p_pct)
        in_neg_pct.append(row["dist_to_neg_centroid"] <= eps_n_pct)
    df = df.copy()
    df["inside_pos_ball_max"] = in_pos_max
    df["inside_neg_ball_max"] = in_neg_max
    df[f"inside_pos_ball_p{int(percentile)}"] = in_pos_pct
    df[f"inside_neg_ball_p{int(percentile)}"] = in_neg_pct
    return df


# ============================================================================
# Correlations
# ============================================================================
def compute_correlations(test_df, target_layers):
    """Per-layer Pearson/Spearman correlations of both centroid distances
    vs the baseline behavior score."""
    rows = []
    for layer in target_layers:
        ldf = test_df[test_df["layer"] == layer]
        out = {"layer": layer, "n": 0}
        for dist_col, tag in [("dist_to_neg_centroid", "neg"),
                              ("dist_to_pos_centroid", "pos")]:
            valid = ldf[[dist_col, "baseline_score"]].dropna()
            if len(valid) < 4:
                out.update({
                    f"pearson_r_{tag}": np.nan,
                    f"pearson_p_{tag}": np.nan,
                    f"spearman_rho_{tag}": np.nan,
                    f"spearman_p_{tag}": np.nan,
                })
                continue
            r, p_r = scipy_stats.pearsonr(valid[dist_col], valid["baseline_score"])
            rho, p_rho = scipy_stats.spearmanr(valid[dist_col], valid["baseline_score"])
            out.update({
                f"pearson_r_{tag}": r,
                f"pearson_p_{tag}": p_r,
                f"spearman_rho_{tag}": rho,
                f"spearman_p_{tag}": p_rho,
            })
            out["n"] = len(valid)
        rows.append(out)
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================
def _scatter_dist_vs_score(test_df, layer, behavior, dist_col, centroid_label,
                           output_path, radii_df=None, percentile=None):
    ldf = test_df[test_df["layer"] == layer]
    valid = ldf[[dist_col, "baseline_score"]].dropna()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid[dist_col], valid["baseline_score"],
               alpha=0.8, s=60, edgecolors="k", linewidths=0.4,
               color="#d62728" if centroid_label == "neg" else "#2ca02c",
               zorder=3)

    # Draw epsilon radii as vertical lines.
    if radii_df is not None:
        rrow = radii_df[radii_df["layer"] == layer]
        if not rrow.empty:
            r = rrow.iloc[0]
            eps_max = r[f"eps_{centroid_label}_max"]
            ax.axvline(x=eps_max, color="gray", linestyle="--", alpha=0.7,
                       label=f"eps_max = {eps_max:.2f}")
            if percentile is not None:
                eps_pct = r[f"eps_{centroid_label}_p{int(percentile)}"]
                ax.axvline(x=eps_pct, color="black", linestyle=":", alpha=0.7,
                           label=f"eps_p{int(percentile)} = {eps_pct:.2f}")

    if len(valid) >= 4:
        r, p_r = scipy_stats.pearsonr(valid[dist_col], valid["baseline_score"])
        rho, p_rho = scipy_stats.spearmanr(valid[dist_col], valid["baseline_score"])
        z = np.polyfit(valid[dist_col], valid["baseline_score"], 1)
        xr = np.linspace(valid[dist_col].min(), valid[dist_col].max(), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="red", alpha=0.7, linewidth=1.5,
                zorder=2)
        txt = f"r = {r:.3f} (p = {p_r:.1e})\nrho = {rho:.3f} (p = {p_rho:.1e})\nn = {len(valid)}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel(f"L2 distance from {centroid_label} centroid", fontsize=10)
    ax.set_ylabel("Baseline behavior score (unsteered)", fontsize=10)
    ax.set_title(f"{behavior} - Layer {layer}: dist to {centroid_label} centroid",
                 fontsize=12, fontweight="bold")
    if radii_df is not None:
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distribution(train_df, test_df, layer, behavior, radii_df,
                      percentile, output_path):
    """Histogram of dist-to-pos and dist-to-neg for train (by class) and test."""
    train_l = train_df[train_df["layer"] == layer]
    test_l = test_df[test_df["layer"] == layer]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, centroid_label, colors in [
        (axes[0], "neg", {"train_pos": "#2ca02c", "train_neg": "#d62728", "test": "#1f77b4"}),
        (axes[1], "pos", {"train_pos": "#2ca02c", "train_neg": "#d62728", "test": "#1f77b4"}),
    ]:
        dist_col = f"dist_to_{centroid_label}_centroid"
        groups, labels, cols = [], [], []
        if not train_l.empty:
            pos_vals = train_l[train_l["label"] == 1][dist_col].dropna()
            neg_vals = train_l[train_l["label"] == 0][dist_col].dropna()
            if len(pos_vals):
                groups.append(pos_vals); labels.append(f"Train pos (n={len(pos_vals)})"); cols.append(colors["train_pos"])
            if len(neg_vals):
                groups.append(neg_vals); labels.append(f"Train neg (n={len(neg_vals)})"); cols.append(colors["train_neg"])
        if not test_l.empty:
            tv = test_l[dist_col].dropna()
            if len(tv):
                groups.append(tv); labels.append(f"Test (n={len(tv)})"); cols.append(colors["test"])

        if not groups:
            continue

        all_vals = pd.concat(groups)
        bins = np.linspace(max(0, all_vals.min() - 0.5), all_vals.max() + 0.5, 35)
        for vals, lbl, clr in zip(groups, labels, cols):
            ax.hist(vals, bins=bins, alpha=0.45, label=lbl, color=clr,
                    edgecolor="white", linewidth=0.5)

        rrow = radii_df[radii_df["layer"] == layer]
        if not rrow.empty:
            r = rrow.iloc[0]
            eps_max = r[f"eps_{centroid_label}_max"]
            eps_pct = r[f"eps_{centroid_label}_p{int(percentile)}"]
            ax.axvline(x=eps_max, color="black", linestyle="--", linewidth=2, alpha=0.8,
                       label=f"eps_max = {eps_max:.2f}")
            ax.axvline(x=eps_pct, color="black", linestyle=":", linewidth=2, alpha=0.6,
                       label=f"eps_p{int(percentile)} = {eps_pct:.2f}")

        ax.set_xlabel(f"L2 distance from {centroid_label} centroid", fontsize=10)
        ax.set_title(f"dist to {centroid_label} centroid", fontsize=11)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count", fontsize=10)
    fig.suptitle(f"{behavior} - Layer {layer}: Train vs Test Centroid Distances",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_across_layers(corr_df, behavior, output_path):
    if corr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    layers = corr_df["layer"].values

    for tag, color, expected in [
        ("neg", "#d62728", "expected > 0"),
        ("pos", "#2ca02c", "expected < 0"),
    ]:
        rho_col = f"spearman_rho_{tag}"
        p_col = f"spearman_p_{tag}"
        if rho_col not in corr_df.columns:
            continue
        rhos = corr_df[rho_col].values
        pvals = corr_df[p_col].values
        sig_mask = pvals < 0.05

        ax.plot(layers, rhos, "o-", color=color, linewidth=2, markersize=7,
                label=f"dist to {tag} centroid vs score ({expected})", zorder=2)
        ax.scatter(layers[sig_mask], rhos[sig_mask], s=140, facecolors="none",
                   edgecolors=color, linewidths=2.2, zorder=3)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Spearman rho  (centroid distance vs baseline score)", fontsize=10)
    ax.set_title(
        f"{behavior}: Centroid-Distance to Baseline-Score Correlation Across Layers",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Test-time centroid-distance analysis per layer"
    )
    parser.add_argument("--behavior", type=str, required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Path to sweep output dir (contains layer_N/ subdirs)")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="Path to train_contrastive.json")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers (default: discover from sweep_dir)")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--eps_percentile", type=float, default=95.0,
                        help="Percentile of training distances to report as the "
                             "robust epsilon radius (default 95).")
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip model inference; reuse saved CSVs")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    out_dir = sweep_dir / "centroid_distance_analysis"
    plot_dir = out_dir / "plots"
    # Clean plots/CSVs but preserve cached activation .pt files
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    for csv_file in out_dir.glob("*.csv"):
        csv_file.unlink()
    for json_file in out_dir.glob("*.json"):
        json_file.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.layers:
        target_layers = sorted(int(x.strip()) for x in args.layers.split(","))
    else:
        target_layers = discover_layers(sweep_dir)
    if not target_layers:
        logger.error(f"No valid layers found in {sweep_dir}")
        sys.exit(1)
    logger.warning(f"Target layers: {target_layers}")

    test_dist_path = out_dir / "test_distances.csv"
    train_dist_path = out_dir / "train_distances.csv"

    train_acts_path = out_dir / "train_activations.pt"
    test_acts_path = out_dir / "test_activations.pt"

    if args.replot_only:
        for p in [test_dist_path, train_dist_path]:
            if not p.exists():
                logger.error(f"Missing {p}; run without --replot_only first")
                sys.exit(1)
        test_df = pd.read_csv(test_dist_path)
        train_df = pd.read_csv(train_dist_path)
        centroids_df = pd.read_csv(out_dir / "centroids.csv")
        logger.warning(f"Loaded {len(test_df)} test, {len(train_df)} train rows")
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

        # Phase 1: training activations (from cache or model)
        if train_acts_path.exists():
            logger.warning(f"Loading cached training activations from {train_acts_path}")
            cached = torch.load(train_acts_path, map_location="cpu", weights_only=False)
            train_acts = cached["train_acts"]
            train_meta = cached["train_meta"]
        else:
            _ensure_model_loaded()
            logger.warning("Phase 1: Collecting raw training activations ...")
            train_acts, train_meta = extract_train_activations(
                model, tokenizer, args.train_dataset_path, target_layers, device,
            )
            logger.warning(f"Saving training activations to {train_acts_path}")
            torch.save({"train_acts": train_acts, "train_meta": train_meta}, train_acts_path)

        # Phase 2: centroids + epsilon radii
        logger.warning("Phase 2: Computing centroids and epsilon radii ...")
        centroids, centroids_df = compute_centroids_and_radii(
            train_acts, target_layers, percentile=args.eps_percentile,
        )

        # Phase 3: training distances (as a sanity/diagnostic reference)
        logger.warning("Phase 3: Computing training distances ...")
        train_df = compute_train_distances(train_acts, centroids, target_layers)

        # Phase 4: test activations + distances (from cache or model)
        if test_acts_path.exists():
            logger.warning(f"Loading cached test activations from {test_acts_path}")
            cached_test = torch.load(test_acts_path, map_location="cpu", weights_only=False)
            test_mean_acts = cached_test["test_mean_acts"]
            test_metadata = cached_test["test_metadata"]
            test_rows = []
            for entry in test_metadata:
                layer = entry["layer"]
                act = test_mean_acts[(entry["question_idx"], layer)]
                mu_pos = centroids[layer]["mu_pos"]
                mu_neg = centroids[layer]["mu_neg"]
                test_rows.append({
                    "question_idx": entry["question_idx"],
                    "layer": layer,
                    "dist_to_pos_centroid": (act - mu_pos).norm().item(),
                    "dist_to_neg_centroid": (act - mu_neg).norm().item(),
                    "baseline_score": entry["baseline_score"],
                    "question": entry["question"],
                    "generation": entry["generation"],
                })
            test_df = pd.DataFrame(test_rows)
        else:
            _ensure_model_loaded()
            logger.warning("Phase 4: Extracting test activations ...")
            test_df, test_mean_acts, test_metadata = extract_test_distances(
                model, tokenizer, sweep_dir, centroids, target_layers, device,
                return_raw=True,
            )
            logger.warning(f"Saving test activations to {test_acts_path}")
            torch.save({"test_mean_acts": test_mean_acts, "test_metadata": test_metadata},
                       test_acts_path)

        if model is not None:
            del model
            torch.cuda.empty_cache()

    # Annotate in-ball flags + write CSVs
    test_df = annotate_in_ball(test_df, centroids_df, args.eps_percentile)
    train_df = annotate_in_ball(train_df, centroids_df, args.eps_percentile)

    centroids_df.to_csv(out_dir / "centroids.csv", index=False)
    test_df.to_csv(test_dist_path, index=False)
    train_df.to_csv(train_dist_path, index=False)

    # Correlations
    corr_df = compute_correlations(test_df, target_layers)
    corr_df.to_csv(out_dir / "correlations.csv", index=False)

    logger.warning("Per-layer correlations (test distance vs baseline_score):")
    for _, row in corr_df.iterrows():
        sig_neg = " *" if row.get("spearman_p_neg", 1.0) < 0.05 else ""
        sig_pos = " *" if row.get("spearman_p_pos", 1.0) < 0.05 else ""
        logger.warning(
            f"  Layer {int(row['layer']):2d}: "
            f"dist_neg rho = {row['spearman_rho_neg']:+.3f} (p = {row['spearman_p_neg']:.1e}){sig_neg}, "
            f"dist_pos rho = {row['spearman_rho_pos']:+.3f} (p = {row['spearman_p_pos']:.1e}){sig_pos}, "
            f"n = {int(row['n'])}"
        )

    # Plots
    logger.warning("Generating plots ...")
    for layer in target_layers:
        _scatter_dist_vs_score(
            test_df, layer, args.behavior, "dist_to_neg_centroid", "neg",
            plot_dir / f"scatter_neg_layer_{layer}.png",
            radii_df=centroids_df, percentile=args.eps_percentile,
        )
        _scatter_dist_vs_score(
            test_df, layer, args.behavior, "dist_to_pos_centroid", "pos",
            plot_dir / f"scatter_pos_layer_{layer}.png",
            radii_df=centroids_df, percentile=args.eps_percentile,
        )
        plot_distribution(train_df, test_df, layer, args.behavior,
                          centroids_df, args.eps_percentile,
                          plot_dir / f"distribution_layer_{layer}.png")

    plot_correlation_across_layers(corr_df, args.behavior,
                                   plot_dir / "correlation_across_layers.png")

    # Summary
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "direction": "CentroidDistance",
        "eps_percentile": args.eps_percentile,
        "sweep_dir": str(sweep_dir),
        "train_dataset_path": args.train_dataset_path,
        "layers": target_layers,
        "n_test_prompts": int(test_df.groupby("layer")["question_idx"].nunique().median()),
        "n_train_pos": int(train_df[train_df["label"] == 1].groupby("layer").size().median()),
        "n_train_neg": int(train_df[train_df["label"] == 0].groupby("layer").size().median()),
        "centroids": centroids_df.to_dict(orient="records"),
        "correlations": corr_df.to_dict(orient="records"),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
