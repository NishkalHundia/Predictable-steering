"""
Projection vs Steering Factor Analysis.

For each test prompt at each layer:
  1. Retrieve unsteered (factor=0) generation from existing sweep eval results
  2. Forward pass through model to extract mean response-token activations
  3. Project onto the steering direction (DiffMean vector)
  4. Compute per-prompt steerability metrics from existing behavior scores
  5. Correlate projection with steerability metrics
  6. Generate diagnostic plots

Hypothesis: prompts whose unsteered activations project more strongly toward the
positive (behavior-matching) centroid need smaller steering factors for effective
steering, while prompts projecting toward the negative centroid need larger factors.

Requires:
  - Completed sweep (sweep_layers_open_ended.py) with eval results
  - Optionally rejudged (rejudge_sweep.py)

Directory structure (reads from):
  {sweep_dir}/
    layer_{N}/
      DiffMean_weight.pt
      eval/eval_results.parquet

Outputs:
  {sweep_dir}/projection_analysis/
    projections.csv
    steerability_metrics.csv
    merged_data.csv
    correlations.csv
    plots/
      scatter_panel_layer_{N}.png
      curves_by_projection_layer_{N}.png
      correlation_summary.png

Usage:
    uv run python axbench/scripts/projection_vs_steering_factor.py \\
        --behavior sycophancy \\
        --model_name google/gemma-2-9b-it \\
        --sweep_dir results/gemma-2-9b-it/sycophancy-sweep

    # Specific layers:
    ... --layers 15,20,25

    # Replot only (skip model inference, reuse saved projections.csv):
    ... --replot_only
"""
import os
import sys
import json
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================================
# Behavior scales
# ============================================================================
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


def get_behavior_scale(behavior: str):
    return BEHAVIOR_SCALES.get(behavior, (0, 10, 5))


# ============================================================================
# Helpers
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def discover_layers(sweep_dir: Path) -> list[int]:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            has_sv = (d / "DiffMean_weight.pt").exists()
            has_eval = (d / "eval" / "eval_results.parquet").exists()
            if has_sv and has_eval:
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


# ============================================================================
# Phase 1: Extract projections
# ============================================================================
@torch.no_grad()
def extract_projections(model, tokenizer, sweep_dir, target_layers, device):
    """
    For each test prompt, get the unsteered generation (factor=0), run a forward
    pass to collect mean response-token activations at each target layer, and
    project onto the steering direction.

    Uses the same activation-extraction logic as sweep_layers_open_ended.py:
    mean of response-token hidden states, projected as a.dot(sv) / ||sv||.
    """
    use_chat = supports_chat_template(tokenizer)
    model.eval()
    sweep_dir = Path(sweep_dir)

    steering_vectors = {}
    for layer in target_layers:
        sv = torch.load(
            sweep_dir / f"layer_{layer}" / "DiffMean_weight.pt",
            map_location="cpu", weights_only=True,
        )
        if sv.dim() == 2:
            sv = sv.squeeze(0)
        steering_vectors[layer] = sv

    first_layer = target_layers[0]
    eval_df = pd.read_parquet(
        sweep_dir / f"layer_{first_layer}" / "eval" / "eval_results.parquet"
    )

    baseline_df = eval_df[eval_df["steering_factor"] == 0].copy().reset_index(drop=True)
    if baseline_df.empty:
        closest = eval_df.loc[eval_df["steering_factor"].abs().idxmin(), "steering_factor"]
        baseline_df = eval_df[eval_df["steering_factor"] == closest].copy().reset_index(drop=True)
        logger.warning(f"No factor=0 found; using factor={closest} as baseline")
    logger.warning(f"Extracting projections for {len(baseline_df)} test prompts")

    rows = []
    for idx in tqdm(range(len(baseline_df)), desc="Extracting projections"):
        row = baseline_df.iloc[idx]
        question = row["question"]
        generation = row["generation"]
        question_idx = row.get("question_idx", idx)

        if pd.isna(generation) or str(generation).strip() == "":
            logger.warning(f"  Prompt {question_idx}: empty baseline generation, skipping")
            continue

        if use_chat:
            q_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": str(generation)}],
                tokenize=False, add_generation_prompt=False,
            )
        else:
            q_text = question
            full_text = question + "\n\n" + str(generation)

        full_inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=1024,
        ).to(device)
        q_inputs = tokenizer(
            q_text, return_tensors="pt", truncation=True, max_length=1024,
        ).to(device)

        q_len = q_inputs["input_ids"].shape[1]
        f_len = full_inputs["input_ids"].shape[1]
        if f_len <= q_len:
            logger.warning(f"  Prompt {question_idx}: no response tokens, skipping")
            continue

        layer_acts = gather_multi_layer_activations(
            model, target_layers,
            {"input_ids": full_inputs["input_ids"],
             "attention_mask": full_inputs["attention_mask"]},
        )

        for layer in target_layers:
            resp_acts = layer_acts[layer].float()[0, q_len:f_len]
            mean_act = resp_acts.mean(dim=0)

            sv = steering_vectors[layer].to(mean_act.device)
            sv_norm = sv.norm().item()
            proj = mean_act.dot(sv).item() / sv_norm if sv_norm > 1e-12 else 0.0

            rows.append({
                "question_idx": int(question_idx),
                "layer": int(layer),
                "projection": proj,
                "sv_norm": sv_norm,
                "question": question,
                "generation": str(generation),
                "n_response_tokens": int(f_len - q_len),
            })

        if idx % 5 == 0:
            torch.cuda.empty_cache()

    logger.warning(f"Extracted {len(rows)} projection values")
    return pd.DataFrame(rows)


# ============================================================================
# Phase 2: Per-prompt steerability metrics
# ============================================================================
def _load_layer_eval(sweep_dir, layer, fluency_filter=None):
    """
    Load behavior eval data for a layer, optionally joined with axbench fluency
    data and filtered to fluency_score >= fluency_filter.

    Returns (filtered_df, n_per_factor) where n_per_factor is a dict mapping
    steering_factor -> count of surviving examples (for min_valid checks).
    """
    eval_path = sweep_dir / f"layer_{layer}" / "eval" / "eval_results.parquet"
    if not eval_path.exists():
        return pd.DataFrame(), {}

    eval_df = pd.read_parquet(eval_path)

    if fluency_filter is not None:
        ax_path = sweep_dir / f"layer_{layer}" / "eval_axbench" / "eval_results.parquet"
        if ax_path.exists():
            ax_df = pd.read_parquet(ax_path)
            if "fluency_score" in ax_df.columns:
                ax_cols = ["question_idx", "steering_factor", "fluency_score"]
                merged = eval_df.merge(ax_df[ax_cols],
                                       on=["question_idx", "steering_factor"],
                                       how="inner")
                mask = merged["fluency_score"] >= fluency_filter
                n_before, n_after = len(mask), int(mask.sum())
                logger.warning(
                    f"  Layer {layer}: fluency >= {fluency_filter}: "
                    f"{n_after}/{n_before} rows survive"
                )
                eval_df = merged[mask].reset_index(drop=True)
            else:
                logger.warning(f"  Layer {layer}: axbench data has no fluency_score column")
        else:
            logger.warning(f"  Layer {layer}: no axbench data at {ax_path}, skipping fluency filter")

    n_per_factor = (
        eval_df.dropna(subset=["behavior_score"])
        .groupby("steering_factor")["behavior_score"]
        .count()
        .to_dict()
    )
    return eval_df, n_per_factor


def compute_steerability_metrics(sweep_dir, target_layers, behavior,
                                 fluency_filter=None, min_valid=None):
    """
    From existing eval data, compute per-prompt steerability metrics at each layer.

    If fluency_filter is set, only (prompt, factor) rows where the axbench
    fluency_score >= fluency_filter are used.  If min_valid is set, aggregate
    "best factor" metrics only consider factors with >= min_valid surviving
    examples (matching rejudge_sweep.py logic).
    """
    min_val, max_val, _ = get_behavior_scale(behavior)
    scale_range = max_val - min_val
    sweep_dir = Path(sweep_dir)
    all_rows = []

    for layer in target_layers:
        eval_df, n_per_factor = _load_layer_eval(sweep_dir, layer, fluency_filter)
        if eval_df.empty:
            continue

        valid_factors = set(n_per_factor.keys())
        if min_valid:
            valid_factors = {f for f, n in n_per_factor.items() if n >= min_valid}

        for qidx in sorted(eval_df["question_idx"].unique()):
            pdf = eval_df[eval_df["question_idx"] == qidx].dropna(subset=["behavior_score"])
            if pdf.empty:
                continue

            scores = dict(zip(pdf["steering_factor"].values, pdf["behavior_score"].values))
            baseline = scores.get(0.0, scores.get(0, np.nan))

            scores_valid = {f: s for f, s in scores.items() if f in valid_factors}

            pos = sorted([(f, s) for f, s in scores.items() if f > 0], key=lambda x: x[0])
            pos_valid = sorted([(f, s) for f, s in scores_valid.items() if f > 0],
                               key=lambda x: x[0])
            all_sorted = sorted(scores.items(), key=lambda x: x[0])

            m = {"question_idx": int(qidx), "layer": int(layer), "baseline_score": baseline}

            # --- max positive score & its factor (respects min_valid) ---
            if pos_valid:
                best_f, best_s = max(pos_valid, key=lambda x: x[1])
                m["max_pos_score"] = best_s
                m["factor_for_max_pos"] = best_f
                m["delta_from_baseline"] = (best_s - baseline) if not np.isnan(baseline) else np.nan
            else:
                m["max_pos_score"] = m["factor_for_max_pos"] = m["delta_from_baseline"] = np.nan

            # --- min factor to exceed baseline + delta (10/20/30 % of scale) ---
            # uses all positive-factor scores (not min_valid gated) since this
            # is a per-prompt metric, not an aggregate
            for pct_label, pct in [("10", 0.10), ("20", 0.20), ("30", 0.30)]:
                key = f"min_factor_delta_{pct_label}"
                if np.isnan(baseline) or not pos:
                    m[key] = np.nan
                else:
                    thresh = baseline + pct * scale_range
                    m[key] = next((f for f, s in pos if s >= thresh), np.nan)

            # --- sensitivity slope (all factors) ---
            if len(all_sorted) >= 3:
                fa = np.array([x[0] for x in all_sorted])
                sa = np.array([x[1] for x in all_sorted])
                m["sensitivity_slope"] = scipy_stats.linregress(fa, sa).slope
            else:
                m["sensitivity_slope"] = np.nan

            # --- positive-factors-only sensitivity slope ---
            if len(pos) >= 3:
                fa = np.array([x[0] for x in pos])
                sa = np.array([x[1] for x in pos])
                m["pos_sensitivity_slope"] = scipy_stats.linregress(fa, sa).slope
            else:
                m["pos_sensitivity_slope"] = np.nan

            # --- scores at specific factors ---
            for f_val in [1.0, 2.0, 5.0, 10.0]:
                col = f"score_at_f{int(f_val)}"
                m[col] = scores.get(f_val, scores.get(int(f_val), np.nan))

            all_rows.append(m)

    return pd.DataFrame(all_rows)


# ============================================================================
# Phase 3: Correlation analysis
# ============================================================================
SCATTER_METRICS = [
    ("baseline_score", "Baseline Score (f=0)", "+"),
    ("factor_for_max_pos", "Factor for Max Score (pos)", "-"),
    ("min_factor_delta_20", "Min Factor for Delta 20%", "-"),
    ("sensitivity_slope", "Sensitivity (slope, all factors)", "+"),
    ("score_at_f1", "Score at f=1", "+"),
    ("delta_from_baseline", "Delta from Baseline (max-base)", "-"),
]

ALL_METRIC_COLS = [
    "baseline_score", "max_pos_score", "factor_for_max_pos",
    "delta_from_baseline",
    "min_factor_delta_10", "min_factor_delta_20", "min_factor_delta_30",
    "sensitivity_slope", "pos_sensitivity_slope",
    "score_at_f1", "score_at_f2", "score_at_f5", "score_at_f10",
]


def compute_correlations(merged_df, target_layers):
    rows = []
    for layer in target_layers:
        ldf = merged_df[merged_df["layer"] == layer]
        for col in ALL_METRIC_COLS:
            if col not in ldf.columns:
                continue
            valid = ldf[["projection", col]].dropna()
            if len(valid) < 4:
                continue
            r, p_r = scipy_stats.pearsonr(valid["projection"], valid[col])
            rho, p_rho = scipy_stats.spearmanr(valid["projection"], valid[col])
            rows.append({
                "layer": layer, "metric": col,
                "pearson_r": r, "pearson_p": p_r,
                "spearman_rho": rho, "spearman_p": p_rho,
                "n": len(valid),
            })
    return pd.DataFrame(rows)


# ============================================================================
# Phase 4: Plots
# ============================================================================
def _annotate_corr(ax, x, y, label=""):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 4:
        return
    r, p_r = scipy_stats.pearsonr(valid["x"], valid["y"])
    rho, p_rho = scipy_stats.spearmanr(valid["x"], valid["y"])
    txt = f"r={r:.3f} (p={p_r:.3f})\nrho={rho:.3f} (p={p_rho:.3f})\nn={len(valid)}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    # regression line
    z = np.polyfit(valid["x"], valid["y"], 1)
    xr = np.linspace(valid["x"].min(), valid["x"].max(), 50)
    ax.plot(xr, np.polyval(z, xr), "--", color="red", alpha=0.6, linewidth=1.2)


def plot_scatter_panel(merged_df, layer, behavior, output_path):
    """6-panel scatter: projection vs each key metric."""
    ldf = merged_df[merged_df["layer"] == layer]
    if ldf.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, (col, label, expected) in enumerate(SCATTER_METRICS):
        ax = axes[i]
        if col not in ldf.columns:
            ax.set_visible(False)
            continue
        valid = ldf[["projection", col]].dropna()
        if valid.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=9)
            continue
        ax.scatter(valid["projection"], valid[col], alpha=0.7, s=40, edgecolors="k",
                   linewidths=0.3, color="#2E86AB")
        _annotate_corr(ax, valid["projection"], valid[col])
        ax.set_xlabel("Projection onto steering dir.", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(f"{label}  (expect {expected})", fontsize=9)

    fig.suptitle(f"{behavior} - Layer {layer}: Projection vs Steerability",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_curves_by_projection(merged_df, eval_dfs, layer, behavior, output_path):
    """
    Behavior-score vs steering-factor curves for every prompt, colored by
    projection value. If the hypothesis holds, high-projection prompts (blue)
    should show behaviour rising at lower factors than low-projection prompts (red).
    """
    ldf = merged_df[merged_df["layer"] == layer]
    if ldf.empty or layer not in eval_dfs:
        return
    edf = eval_dfs[layer]
    min_val, max_val, ref_line = get_behavior_scale(behavior)

    proj_map = dict(zip(ldf["question_idx"], ldf["projection"]))
    qidxs = sorted(proj_map.keys())
    projs = np.array([proj_map[q] for q in qidxs])
    norm = Normalize(vmin=projs.min(), vmax=projs.max())
    cmap = plt.cm.coolwarm

    fig, ax = plt.subplots(figsize=(10, 6))
    for qidx in qidxs:
        pdf = edf[edf["question_idx"] == qidx].sort_values("steering_factor")
        if pdf.empty:
            continue
        color = cmap(norm(proj_map[qidx]))
        ax.plot(pdf["steering_factor"], pdf["behavior_score"],
                "-o", color=color, alpha=0.55, linewidth=1.2, markersize=3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Projection (low -> high)", fontsize=9)

    if ref_line is not None:
        ax.axhline(y=ref_line, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Steering Factor", fontsize=11)
    ax.set_ylabel(f"Behavior Score ({min_val:.0f}-{max_val:.0f})", fontsize=11)
    ax.set_ylim(min_val - 0.3, max_val + 0.3)
    ax.set_title(f"{behavior} - Layer {layer}: Per-Prompt Curves Colored by Projection",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_curves_by_projection_quantile(merged_df, eval_dfs, layer, behavior, output_path,
                                       n_quantiles=3):
    """
    Mean behavior-score curve per projection quantile (tercile by default).
    Cleaner aggregated view than the per-prompt version.
    """
    ldf = merged_df[merged_df["layer"] == layer]
    if ldf.empty or layer not in eval_dfs:
        return
    edf = eval_dfs[layer]
    min_val, max_val, ref_line = get_behavior_scale(behavior)

    proj_map = dict(zip(ldf["question_idx"], ldf["projection"]))

    enriched = edf.copy()
    enriched["projection"] = enriched["question_idx"].map(proj_map)
    enriched = enriched.dropna(subset=["projection", "behavior_score"])
    if enriched.empty:
        return

    enriched["proj_quantile"] = pd.qcut(enriched["projection"], n_quantiles,
                                         labels=False, duplicates="drop")

    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, enriched["proj_quantile"].nunique()))
    quantile_bounds = enriched.groupby("proj_quantile")["projection"].agg(["min", "max"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for q_idx, color in zip(sorted(enriched["proj_quantile"].unique()), colors):
        qdf = enriched[enriched["proj_quantile"] == q_idx]
        agg = qdf.groupby("steering_factor")["behavior_score"].agg(["mean", "std"]).reset_index()
        lo, hi = quantile_bounds.loc[q_idx, "min"], quantile_bounds.loc[q_idx, "max"]
        label = f"Q{int(q_idx)}: proj [{lo:.1f}, {hi:.1f}]"
        ax.errorbar(agg["steering_factor"], agg["mean"], yerr=agg["std"],
                    marker="o", linewidth=2, capsize=3, color=color, label=label, markersize=4)

    if ref_line is not None:
        ax.axhline(y=ref_line, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Steering Factor", fontsize=11)
    ax.set_ylabel(f"Mean Behavior Score ({min_val:.0f}-{max_val:.0f})", fontsize=11)
    ax.set_ylim(min_val - 0.3, max_val + 0.3)
    ax.set_title(f"{behavior} - Layer {layer}: Mean Curves by Projection Quantile",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_summary(corr_df, behavior, target_layers, output_path):
    """Pearson r and Spearman rho across layers for each key metric."""
    key_metrics = [col for col, _, _ in SCATTER_METRICS]
    avail = corr_df[corr_df["metric"].isin(key_metrics)]
    if avail.empty:
        return

    metrics_present = [m for m in key_metrics if m in avail["metric"].values]
    n = len(metrics_present)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, stat_col, stat_name in [
        (axes[0], "pearson_r", "Pearson r"),
        (axes[1], "spearman_rho", "Spearman rho"),
    ]:
        cmap = plt.cm.tab10(np.linspace(0, 1, n))
        for i, metric in enumerate(metrics_present):
            mdf = avail[avail["metric"] == metric].sort_values("layer")
            if mdf.empty:
                continue
            ax.plot(mdf["layer"], mdf[stat_col], "o-", label=metric.replace("_", " "),
                    color=cmap[i], linewidth=1.8, markersize=5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel(stat_name)
        ax.set_title(f"{stat_name} across Layers")
        if target_layers:
            ax.set_xticks(target_layers)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"{behavior}: Projection-Steerability Correlation by Layer",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_histogram(merged_df, layer, behavior, output_path):
    """Histogram of projection values with baseline_score color."""
    ldf = merged_df[merged_df["layer"] == layer]
    if ldf.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ldf["projection"].dropna(), bins=20, color="#2E86AB", edgecolor="white",
            alpha=0.8)
    ax.set_xlabel("Projection onto steering direction", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{behavior} - Layer {layer}: Test-Prompt Projection Distribution",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze projection of unsteered activations vs required steering factor"
    )
    parser.add_argument("--behavior", type=str, required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Path to sweep output dir (contains layer_N/ subdirs)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers (default: discover from sweep_dir)")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--fluency_filter", type=float, default=None,
                        help="Only use (prompt, factor) rows with fluency_score >= this "
                             "(requires eval_axbench/ data from rejudge_sweep.py)")
    parser.add_argument("--min_valid", type=int, default=None,
                        help="Only consider a factor 'valid' for best/worst metrics if "
                             ">= this many examples survive fluency filtering")
    parser.add_argument("--replot_only", action="store_true",
                        help="Skip model inference; reuse saved projections.csv")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = sweep_dir / "projection_analysis"
    plot_dir = out_dir / "plots"
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

    # -- Extract or load projections ------------------------------------
    proj_path = out_dir / "projections.csv"
    if args.replot_only:
        if not proj_path.exists():
            logger.error(f"No projections.csv at {proj_path}; run without --replot_only first")
            sys.exit(1)
        proj_df = pd.read_csv(proj_path)
        logger.warning(f"Loaded {len(proj_df)} saved projections")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning(f"Using device: {device}")
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

        proj_df = extract_projections(model, tokenizer, sweep_dir, target_layers, device)
        proj_df.to_csv(proj_path, index=False)
        logger.warning(f"Saved projections to {proj_path}")

        del model
        torch.cuda.empty_cache()

    # -- Compute steerability metrics ------------------------------------
    if args.fluency_filter is not None:
        logger.warning(f"Fluency filter: >= {args.fluency_filter}")
    if args.min_valid is not None:
        logger.warning(f"Min valid: >= {args.min_valid} examples per factor")
    logger.warning("Computing per-prompt steerability metrics ...")
    metrics_df = compute_steerability_metrics(
        sweep_dir, target_layers, args.behavior,
        fluency_filter=args.fluency_filter, min_valid=args.min_valid,
    )
    metrics_df.to_csv(out_dir / "steerability_metrics.csv", index=False)

    # -- Merge projections + metrics -------------------------------------
    merged = proj_df.merge(metrics_df, on=["question_idx", "layer"], how="inner")
    merged.to_csv(out_dir / "merged_data.csv", index=False)
    logger.warning(f"Merged data: {len(merged)} rows")

    # -- Correlations -----------------------------------------------------
    corr_df = compute_correlations(merged, target_layers)
    corr_df.to_csv(out_dir / "correlations.csv", index=False)
    logger.warning(f"Computed {len(corr_df)} correlation entries")

    for _, row in corr_df.iterrows():
        sig = "*" if row["spearman_p"] < 0.05 else ""
        logger.warning(
            f"  L{int(row['layer']):2d} {row['metric']:>25s}: "
            f"r={row['pearson_r']:+.3f} (p={row['pearson_p']:.3f})  "
            f"rho={row['spearman_rho']:+.3f} (p={row['spearman_p']:.3f}) {sig}"
        )

    # -- Load eval data for curve plots (with same fluency filter) --------
    eval_dfs = {}
    for layer in target_layers:
        filtered_df, _ = _load_layer_eval(sweep_dir, layer, args.fluency_filter)
        if not filtered_df.empty:
            eval_dfs[layer] = filtered_df

    # -- Generate plots ---------------------------------------------------
    logger.warning("Generating plots ...")
    for layer in target_layers:
        plot_scatter_panel(merged, layer, args.behavior,
                           plot_dir / f"scatter_panel_layer_{layer}.png")
        plot_curves_by_projection(merged, eval_dfs, layer, args.behavior,
                                  plot_dir / f"curves_by_projection_layer_{layer}.png")
        plot_curves_by_projection_quantile(merged, eval_dfs, layer, args.behavior,
                                           plot_dir / f"curves_quantile_layer_{layer}.png")
        plot_projection_histogram(merged, layer, args.behavior,
                                  plot_dir / f"projection_hist_layer_{layer}.png")

    plot_correlation_summary(corr_df, args.behavior, target_layers,
                             plot_dir / "correlation_summary.png")

    # -- Summary ----------------------------------------------------------
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "sweep_dir": str(sweep_dir),
        "layers": target_layers,
        "fluency_filter": args.fluency_filter,
        "min_valid": args.min_valid,
        "n_prompts_per_layer": int(merged.groupby("layer")["question_idx"].nunique().median()),
        "correlations": corr_df.to_dict(orient="records"),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nAnalysis complete! Results in {out_dir}")
    logger.warning(f"Plots in {plot_dir}")


if __name__ == "__main__":
    main()
