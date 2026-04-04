"""
Per-Prompt Steerability Analysis.

Tests whether a test prompt's projection onto the steering direction predicts
its per-prompt steerability outcomes.  Avoids the statistical weakness of
meta-correlating a noisy per-layer rho by working at the prompt level.

Three analyses run for every layer:
  A. Within-layer correlation
       rho(p_tilde, best_steered_score)  and  rho(p_tilde, delta)
       -- N = number of test prompts (typically 32), not 1

  B. Group comparison (median split)
       High-projection prompts vs low-projection prompts:
       Mann-Whitney U on steerability, Cohen d effect size

  C. Aggregate layer-level (supplementary)
       Cross-layer correlation of per-layer rho / d' vs aggregate steerability
       (kept for reference, but NOT the primary evidence)

Fluency filtering: per-prompt x factor, fluency_score >= threshold.
No min_valid needed for per-prompt analysis (only for aggregate metrics).

Outputs:
  {sweep_dir}/steerability_link/
    per_prompt_steerability.csv        per-prompt steerability + projection
    within_layer_results.csv           per-layer correlation + group comparison
    layer_metrics.csv                  aggregate per-layer rho, d', steerability
    cross_layer_correlations.csv       supplementary meta-correlation
    summary.json
    plots/
      within_layer_rho.png             per-prompt rho across layers
      group_comparison.png             effect sizes across layers
      scatter_layer_{N}.png            top K illustrative scatters
      metrics_by_layer.png             overview panel

Usage:
    uv run python axbench/scripts/projection_steerability_link.py \\
        --behavior corrigible-neutral-HHH \\
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep

    # Custom thresholds:
    ... --fluency_threshold 1.0 --min_valid 25
"""
import json
import shutil
import sys
import numpy as np
import pandas as pd
from pathlib import Path
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

STEER_METRIC_COLS = ["best_steered_score", "delta", "score_f1", "score_f2", "score_f5"]


# ============================================================================
# Discovery & data loading
# ============================================================================
def discover_layers(sweep_dir: Path) -> list[int]:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            if (d / "eval" / "eval_results.parquet").exists():
                layers.append(int(d.name.split("_")[1]))
    return sorted(layers)


def load_projection_data(sweep_dir: Path) -> pd.DataFrame:
    """Load normalized projections from projection_analysis/test_projections.csv."""
    path = sweep_dir / "projection_analysis" / "test_projections.csv"
    if not path.exists():
        logger.error(f"Missing {path}. Run projection_vs_steering_factor.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"question_idx", "layer", "normalized_projection", "baseline_score"}
    if not required.issubset(df.columns):
        logger.error(f"test_projections.csv missing columns: {required - set(df.columns)}")
        sys.exit(1)
    return df


def load_separability(sweep_dir: Path, layers: list[int]) -> pd.DataFrame:
    rows = []
    for layer in layers:
        sep_path = sweep_dir / f"layer_{layer}" / "separability.json"
        if sep_path.exists():
            sep = json.loads(sep_path.read_text())
            rows.append({"layer": layer, "dprime": sep.get("dprime"), "auroc": sep.get("auroc")})
        else:
            rows.append({"layer": layer, "dprime": None, "auroc": None})
    return pd.DataFrame(rows)


def load_projection_correlations(sweep_dir: Path) -> pd.DataFrame:
    path = sweep_dir / "projection_analysis" / "correlations.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)[["layer", "spearman_rho", "spearman_p"]].copy()


# ============================================================================
# Per-prompt steerability (primary analysis)
# ============================================================================
def compute_per_prompt_steerability(
    sweep_dir: Path,
    proj_df: pd.DataFrame,
    layers: list[int],
    fluency_threshold: float,
) -> pd.DataFrame:
    """
    For each prompt x layer, compute per-prompt steerability from eval data
    with per-prompt fluency filtering.

    Returns a DataFrame with one row per (prompt, layer).
    """
    all_rows = []

    for layer in layers:
        layer_dir = sweep_dir / f"layer_{layer}"
        eval_path = layer_dir / "eval" / "eval_results.parquet"
        axbench_path = layer_dir / "eval_axbench" / "eval_results.parquet"

        if not eval_path.exists():
            continue

        eval_df = pd.read_parquet(eval_path)
        if "behavior_score" not in eval_df.columns:
            continue

        working = eval_df[["question_idx", "steering_factor", "behavior_score"]].copy()
        working = working.dropna(subset=["behavior_score"])

        if axbench_path.exists():
            ax_df = pd.read_parquet(axbench_path)
            if "fluency_score" in ax_df.columns:
                working = working.merge(
                    ax_df[["question_idx", "steering_factor", "fluency_score"]],
                    on=["question_idx", "steering_factor"], how="inner",
                )
                n_before = len(working)
                working = working[working["fluency_score"] >= fluency_threshold]
                logger.warning(
                    f"  Layer {layer}: fluency filter kept "
                    f"{len(working)}/{n_before} prompt-factor pairs"
                )

        prompts = working["question_idx"].unique()

        for qidx in prompts:
            prompt_data = working[working["question_idx"] == qidx]

            f0 = prompt_data[prompt_data["steering_factor"] == 0]
            baseline = float(f0["behavior_score"].values[0]) if not f0.empty else np.nan

            pos_data = prompt_data[prompt_data["steering_factor"] > 0]

            if pos_data.empty:
                best_steered = np.nan
                best_factor = np.nan
                delta = np.nan
            else:
                best_row = pos_data.loc[pos_data["behavior_score"].idxmax()]
                best_steered = float(best_row["behavior_score"])
                best_factor = float(best_row["steering_factor"])
                delta = best_steered - baseline if not np.isnan(baseline) else np.nan

            def _score_at(f):
                r = prompt_data[prompt_data["steering_factor"] == f]
                return float(r["behavior_score"].values[0]) if not r.empty else np.nan

            all_rows.append({
                "question_idx": int(qidx),
                "layer": int(layer),
                "baseline_score": baseline,
                "best_steered_score": best_steered,
                "best_factor": best_factor,
                "delta": delta,
                "score_f1": _score_at(1),
                "score_f2": _score_at(2),
                "score_f5": _score_at(5),
            })

    steer_df = pd.DataFrame(all_rows)

    merged = steer_df.merge(
        proj_df[["question_idx", "layer", "normalized_projection"]],
        on=["question_idx", "layer"], how="inner",
    )

    for layer in layers:
        ldf = merged[merged["layer"] == layer]
        if ldf.empty:
            continue
        median_proj = ldf["normalized_projection"].median()
        merged.loc[merged["layer"] == layer, "projection_group"] = np.where(
            merged.loc[merged["layer"] == layer, "normalized_projection"] >= median_proj,
            "high", "low"
        )

    return merged


# ============================================================================
# Within-layer analysis
# ============================================================================
def _cohens_d(group_a, group_b):
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((na - 1) * group_a.std()**2 + (nb - 1) * group_b.std()**2) / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return (group_a.mean() - group_b.mean()) / pooled_std


def within_layer_analysis(per_prompt_df: pd.DataFrame, layers: list[int]) -> pd.DataFrame:
    """
    For each layer, compute:
      - Spearman rho(p_tilde, best_steered_score), rho(p_tilde, delta)
      - Group comparison: Mann-Whitney U, Cohen d
    """
    rows = []
    for layer in layers:
        ldf = per_prompt_df[per_prompt_df["layer"] == layer]
        if len(ldf) < 4:
            continue

        entry = {"layer": layer, "n_prompts": len(ldf)}

        for metric in ["best_steered_score", "delta"]:
            valid = ldf[["normalized_projection", metric]].dropna()
            if len(valid) >= 4:
                rho, p = scipy_stats.spearmanr(valid["normalized_projection"], valid[metric])
                entry[f"rho_vs_{metric}"] = rho
                entry[f"p_vs_{metric}"] = p
            else:
                entry[f"rho_vs_{metric}"] = np.nan
                entry[f"p_vs_{metric}"] = np.nan

        if "projection_group" in ldf.columns:
            high = ldf[ldf["projection_group"] == "high"]
            low = ldf[ldf["projection_group"] == "low"]

            for metric in ["best_steered_score", "delta"]:
                hv = high[metric].dropna()
                lv = low[metric].dropna()
                entry[f"mean_{metric}_high"] = hv.mean() if len(hv) else np.nan
                entry[f"mean_{metric}_low"] = lv.mean() if len(lv) else np.nan
                if len(hv) >= 2 and len(lv) >= 2:
                    u_stat, u_p = scipy_stats.mannwhitneyu(hv, lv, alternative="greater")
                    entry[f"U_{metric}"] = u_stat
                    entry[f"U_p_{metric}"] = u_p
                    entry[f"cohend_{metric}"] = _cohens_d(hv, lv)
                else:
                    entry[f"U_{metric}"] = np.nan
                    entry[f"U_p_{metric}"] = np.nan
                    entry[f"cohend_{metric}"] = np.nan

        rows.append(entry)
    return pd.DataFrame(rows)


# ============================================================================
# Aggregate steerability (supplementary, for layer_metrics)
# ============================================================================
def compute_aggregate_steerability(
    sweep_dir: Path, layers: list[int],
    fluency_threshold: float, min_valid: int,
) -> pd.DataFrame:
    rows = []
    for layer in layers:
        layer_dir = sweep_dir / f"layer_{layer}"
        eval_path = layer_dir / "eval" / "eval_results.parquet"
        axbench_path = layer_dir / "eval_axbench" / "eval_results.parquet"

        if not eval_path.exists():
            continue

        eval_df = pd.read_parquet(eval_path)
        if "behavior_score" not in eval_df.columns:
            continue

        working = eval_df[["question_idx", "steering_factor", "behavior_score"]].copy()
        working = working.dropna(subset=["behavior_score"])

        if axbench_path.exists():
            ax_df = pd.read_parquet(axbench_path)
            if "fluency_score" in ax_df.columns:
                working = working.merge(
                    ax_df[["question_idx", "steering_factor", "fluency_score"]],
                    on=["question_idx", "steering_factor"], how="inner",
                )
                working = working[working["fluency_score"] >= fluency_threshold]

        per_factor = (
            working.groupby("steering_factor")["behavior_score"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_score", "count": "n_valid"})
        )
        valid_factors = per_factor[per_factor["n_valid"] >= min_valid]

        f0 = valid_factors[valid_factors["steering_factor"] == 0]
        baseline_mean = float(f0["mean_score"].values[0]) if not f0.empty else np.nan

        pos_valid = valid_factors[valid_factors["steering_factor"] > 0]
        if pos_valid.empty:
            best_agg_score = np.nan
            best_agg_factor = np.nan
            agg_delta = np.nan
        else:
            best_row = pos_valid.loc[pos_valid["mean_score"].idxmax()]
            best_agg_score = float(best_row["mean_score"])
            best_agg_factor = float(best_row["steering_factor"])
            agg_delta = best_agg_score - baseline_mean if not np.isnan(baseline_mean) else np.nan

        rows.append({
            "layer": layer,
            "agg_baseline": baseline_mean,
            "agg_best_score": best_agg_score,
            "agg_best_factor": best_agg_factor,
            "agg_delta": agg_delta,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Cross-layer correlation (supplementary)
# ============================================================================
def cross_layer_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    predictors = [c for c in ["proj_baseline_rho", "dprime"] if c in merged.columns]
    targets = [c for c in ["agg_delta", "agg_best_score",
                           "mean_rho_vs_best_steered", "mean_rho_vs_delta"]
               if c in merged.columns]
    rows = []
    for pred in predictors:
        for tgt in targets:
            valid = merged[[pred, tgt]].dropna()
            if len(valid) < 4:
                continue
            rho, p = scipy_stats.spearmanr(valid[pred], valid[tgt])
            r, pr = scipy_stats.pearsonr(valid[pred], valid[tgt])
            rows.append({
                "predictor": pred, "target": tgt,
                "spearman_rho": rho, "spearman_p": p,
                "pearson_r": r, "pearson_p": pr,
                "n": len(valid),
            })
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================
def plot_within_layer_rho(wl_df: pd.DataFrame, behavior: str, output_path: Path):
    """Per-prompt rho(p_tilde, best_steered) and rho(p_tilde, delta) across layers."""
    if wl_df.empty:
        return
    layers = wl_df["layer"].values
    fig, ax = plt.subplots(figsize=(12, 5))

    for col, label, color, marker in [
        ("rho_vs_best_steered_score", "rho(proj, best_steered)", "#2E86AB", "o"),
        ("rho_vs_delta", "rho(proj, delta)", "#E94F37", "s"),
    ]:
        if col not in wl_df.columns:
            continue
        vals = wl_df[col].values
        p_col = col.replace("rho_", "p_")
        pvals = wl_df[p_col].values if p_col in wl_df.columns else np.ones_like(vals)

        ax.plot(layers, vals, f"{marker}-", color=color, linewidth=2, markersize=6,
                label=label, zorder=2)
        sig = pvals < 0.05
        ax.scatter(layers[sig], vals[sig], s=120, facecolors="none",
                   edgecolors=color, linewidths=2.5, zorder=3)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Spearman rho (within-layer, per-prompt)", fontsize=10)
    ax.set_title(f"{behavior}: Projection vs Per-Prompt Steerability",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=9)

    n_sig_best = (wl_df.get("p_vs_best_steered_score", pd.Series(dtype=float)) < 0.05).sum()
    n_sig_delta = (wl_df.get("p_vs_delta", pd.Series(dtype=float)) < 0.05).sum()
    ax.text(0.02, 0.02,
            f"Circled = p < 0.05\n"
            f"Sig layers: best_steered={n_sig_best}/{len(layers)}, delta={n_sig_delta}/{len(layers)}",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_group_comparison(wl_df: pd.DataFrame, behavior: str, output_path: Path):
    """Cohen d effect sizes for high-proj vs low-proj across layers."""
    if wl_df.empty:
        return
    layers = wl_df["layer"].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, (metric, title) in enumerate([
        ("best_steered_score", "Best Steered Score"),
        ("delta", "Delta (Best - Baseline)"),
    ]):
        ax = axes[i]
        d_col = f"cohend_{metric}"
        p_col = f"U_p_{metric}"
        mean_h = f"mean_{metric}_high"
        mean_l = f"mean_{metric}_low"

        if d_col not in wl_df.columns:
            continue

        ds = wl_df[d_col].values
        ps = wl_df[p_col].values if p_col in wl_df.columns else np.ones_like(ds)

        colors = ["#2ca02c" if d > 0 else "#d62728" for d in ds]
        ax.bar(layers, ds, color=colors, alpha=0.7, edgecolor="k", linewidth=0.5)
        for j, (l, d, p) in enumerate(zip(layers, ds, ps)):
            if p < 0.05:
                ax.annotate("*", (l, d), ha="center",
                            va="bottom" if d > 0 else "top",
                            fontsize=14, fontweight="bold", color="black")

        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.axhline(0.5, color="blue", linestyle=":", alpha=0.3, label="Cohen d = 0.5 (medium)")
        ax.axhline(-0.5, color="blue", linestyle=":", alpha=0.3)
        ax.set_ylabel(f"Cohen d\n({title})", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")

        if mean_h in wl_df.columns:
            for _, row in wl_df.iterrows():
                ax.annotate(
                    f"H={row.get(mean_h, 0):.1f}\nL={row.get(mean_l, 0):.1f}",
                    (row["layer"], 0), fontsize=5, alpha=0.6, ha="center",
                    textcoords="offset points", xytext=(0, -20),
                )

    axes[0].set_title(
        f"{behavior}: High-Projection vs Low-Projection Group Steerability\n"
        f"(green = high-proj group steers better, * = p < 0.05)",
        fontsize=11, fontweight="bold",
    )
    axes[-1].set_xlabel("Layer", fontsize=11)
    axes[-1].set_xticks(layers)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter_for_layer(
    per_prompt_df: pd.DataFrame, layer: int, behavior: str, output_path: Path,
):
    """Scatter of p_tilde vs best_steered_score and delta for one layer."""
    ldf = per_prompt_df[per_prompt_df["layer"] == layer]
    if len(ldf) < 4:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (y_col, y_label) in enumerate([
        ("best_steered_score", "Best Steered Score (fluency-filtered)"),
        ("delta", "Delta (best steered - baseline)"),
    ]):
        ax = axes[i]
        valid = ldf[["normalized_projection", y_col]].dropna()
        if len(valid) < 4:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", transform=ax.transAxes)
            continue

        if "projection_group" in ldf.columns:
            high = ldf[ldf["projection_group"] == "high"]
            low = ldf[ldf["projection_group"] == "low"]
            ax.scatter(high["normalized_projection"], high[y_col], s=50, color="#2ca02c",
                       edgecolors="k", linewidths=0.4, label="High proj", zorder=3, alpha=0.8)
            ax.scatter(low["normalized_projection"], low[y_col], s=50, color="#d62728",
                       edgecolors="k", linewidths=0.4, label="Low proj", zorder=3, alpha=0.8)
        else:
            ax.scatter(valid["normalized_projection"], valid[y_col], s=50, color="#2E86AB",
                       edgecolors="k", linewidths=0.4, zorder=3, alpha=0.8)

        rho, p = scipy_stats.spearmanr(valid["normalized_projection"], valid[y_col])
        z = np.polyfit(valid["normalized_projection"], valid[y_col], 1)
        xr = np.linspace(valid["normalized_projection"].min(),
                         valid["normalized_projection"].max(), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="black", alpha=0.5, linewidth=1.5)

        sig = "*" if p < 0.05 else ""
        ax.set_title(f"rho = {rho:.3f} (p = {p:.2e}){sig}", fontsize=10)
        ax.set_xlabel("Normalized projection (p~)", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.axvline(x=-1, color="#d62728", linestyle=":", alpha=0.4)
        ax.axvline(x=+1, color="#2ca02c", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"{behavior} - Layer {layer}: Projection vs Per-Prompt Steerability",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_by_layer(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    """Overview: rho, d', and aggregate delta across layers."""
    layers = layer_df["layer"].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    for col, label, color in [
        ("proj_baseline_rho", "rho(proj, baseline)", "#2E86AB"),
        ("mean_rho_vs_best_steered", "rho(proj, best_steered) [per-prompt]", "#44AF69"),
    ]:
        if col in layer_df.columns and layer_df[col].notna().any():
            ax.plot(layers, layer_df[col], "o-", color=color, linewidth=2,
                    markersize=5, label=label)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Spearman rho", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title(f"{behavior}: Metrics by Layer", fontsize=12, fontweight="bold")

    ax = axes[1]
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax.plot(layers, layer_df["dprime"], "s-", color="#E94F37", linewidth=2, markersize=5)
    ax.set_ylabel("d'", fontsize=10)

    ax = axes[2]
    if "agg_delta" in layer_df.columns:
        ax.plot(layers, layer_df["agg_delta"], "D-", color="#6B2D5C", linewidth=2, markersize=5)
    ax.set_ylabel("Aggregate delta\n(best - baseline)", fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)

    for a in axes:
        a.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Per-prompt steerability analysis: does projection predict steering outcomes?"
    )
    parser.add_argument("--behavior", type=str, required=True)
    parser.add_argument("--sweep_dir", type=str, required=True)
    parser.add_argument("--fluency_threshold", type=float, default=1.0)
    parser.add_argument("--min_valid", type=int, default=25,
                        help="Min examples per factor for aggregate metrics (default: 25)")
    parser.add_argument("--top_k_scatter", type=int, default=5,
                        help="Number of layers to produce scatter plots for (default: 5)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = sweep_dir / "steerability_link"
    if out_dir.exists():
        logger.warning(f"Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    layers = discover_layers(sweep_dir)
    if not layers:
        logger.error(f"No layers found in {sweep_dir}")
        sys.exit(1)
    logger.warning(f"Layers: {layers}")

    # --- Load projection data ---
    logger.warning("Loading projection data ...")
    proj_df = load_projection_data(sweep_dir)
    logger.warning(f"  {len(proj_df)} projection rows loaded")

    # --- Compute per-prompt steerability ---
    logger.warning(f"Computing per-prompt steerability (fluency >= {args.fluency_threshold}) ...")
    pp_df = compute_per_prompt_steerability(
        sweep_dir, proj_df, layers, args.fluency_threshold,
    )
    pp_df.to_csv(out_dir / "per_prompt_steerability.csv", index=False)
    logger.warning(f"  {len(pp_df)} per-prompt steerability rows")

    # --- Within-layer analysis (PRIMARY) ---
    logger.warning("\n=== PRIMARY: Within-Layer Per-Prompt Analysis ===")
    wl_df = within_layer_analysis(pp_df, layers)
    wl_df.to_csv(out_dir / "within_layer_results.csv", index=False)

    logger.warning("\nWithin-layer results:")
    for _, row in wl_df.iterrows():
        rho_best = row.get("rho_vs_best_steered_score", np.nan)
        p_best = row.get("p_vs_best_steered_score", np.nan)
        rho_delta = row.get("rho_vs_delta", np.nan)
        p_delta = row.get("p_vs_delta", np.nan)
        d_best = row.get("cohend_best_steered_score", np.nan)
        sig_b = "*" if pd.notna(p_best) and p_best < 0.05 else " "
        sig_d = "*" if pd.notna(p_delta) and p_delta < 0.05 else " "
        logger.warning(
            f"  Layer {int(row['layer']):2d} (n={int(row.get('n_prompts', 0)):2d}): "
            f"rho_best={rho_best:+.3f}{sig_b} (p={p_best:.2e})  "
            f"rho_delta={rho_delta:+.3f}{sig_d} (p={p_delta:.2e})  "
            f"cohen_d={d_best:+.2f}"
        )

    n_sig = (wl_df.get("p_vs_best_steered_score", pd.Series(dtype=float)) < 0.05).sum()
    logger.warning(
        f"\n  Summary: {n_sig}/{len(wl_df)} layers show significant "
        f"rho(projection, best_steered_score) at p<0.05"
    )

    # --- Aggregate steerability (supplementary) ---
    logger.warning("\n=== SUPPLEMENTARY: Aggregate Layer Metrics ===")
    sep_df = load_separability(sweep_dir, layers)
    agg_df = compute_aggregate_steerability(
        sweep_dir, layers, args.fluency_threshold, args.min_valid,
    )
    rho_df = load_projection_correlations(sweep_dir)

    layer_df = sep_df.merge(agg_df, on="layer", how="outer")
    if not rho_df.empty:
        layer_df = layer_df.merge(
            rho_df.rename(columns={"spearman_rho": "proj_baseline_rho",
                                   "spearman_p": "proj_baseline_p"}),
            on="layer", how="outer",
        )

    wl_summary = wl_df[["layer", "rho_vs_best_steered_score", "rho_vs_delta"]].rename(
        columns={"rho_vs_best_steered_score": "mean_rho_vs_best_steered",
                 "rho_vs_delta": "mean_rho_vs_delta"}
    )
    layer_df = layer_df.merge(wl_summary, on="layer", how="outer")
    layer_df = layer_df.sort_values("layer").reset_index(drop=True)
    layer_df.to_csv(out_dir / "layer_metrics.csv", index=False)

    logger.warning("\nPer-layer summary:")
    for _, row in layer_df.iterrows():
        dp = f"d'={row.get('dprime', np.nan):.2f}" if pd.notna(row.get("dprime")) else "d'=N/A"
        delta = f"agg_delta={row.get('agg_delta', np.nan):+.2f}" if pd.notna(row.get("agg_delta")) else "agg_delta=N/A"
        logger.warning(f"  Layer {int(row['layer']):2d}: {dp}  {delta}")

    # --- Cross-layer correlations (supplementary) ---
    xl_df = cross_layer_correlations(layer_df)
    if not xl_df.empty:
        xl_df.to_csv(out_dir / "cross_layer_correlations.csv", index=False)
        logger.warning("\nCross-layer correlations (supplementary):")
        for _, row in xl_df.iterrows():
            sig = "*" if row["spearman_p"] < 0.05 else ""
            logger.warning(
                f"  {row['predictor']:20s} vs {row['target']:25s}: "
                f"rho={row['spearman_rho']:+.3f} (p={row['spearman_p']:.2e}){sig}"
            )

    # --- Plots ---
    logger.warning("\nGenerating plots ...")
    plot_within_layer_rho(wl_df, args.behavior, plot_dir / "within_layer_rho.png")
    plot_group_comparison(wl_df, args.behavior, plot_dir / "group_comparison.png")
    plot_metrics_by_layer(layer_df, args.behavior, plot_dir / "metrics_by_layer.png")

    if not wl_df.empty and "rho_vs_best_steered_score" in wl_df.columns:
        ranked = wl_df.dropna(subset=["rho_vs_best_steered_score"])
        ranked = ranked.nlargest(min(args.top_k_scatter, len(ranked)),
                                 "rho_vs_best_steered_score")
        for _, row in ranked.iterrows():
            l = int(row["layer"])
            plot_scatter_for_layer(pp_df, l, args.behavior,
                                  plot_dir / f"scatter_layer_{l}.png")

    # --- Summary JSON ---
    summary = {
        "behavior": args.behavior,
        "sweep_dir": str(sweep_dir),
        "fluency_threshold": args.fluency_threshold,
        "min_valid_aggregate": args.min_valid,
        "n_layers": len(layers),
        "layers": layers,
        "n_prompts_per_layer": int(pp_df.groupby("layer")["question_idx"].nunique().median())
            if not pp_df.empty else 0,
        "primary_analysis": {
            "method": "within-layer per-prompt correlation + group comparison",
            "n_layers_sig_best_steered": int(n_sig),
            "within_layer_results": wl_df.to_dict(orient="records") if not wl_df.empty else [],
        },
        "supplementary_cross_layer": xl_df.to_dict(orient="records") if not xl_df.empty else [],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
