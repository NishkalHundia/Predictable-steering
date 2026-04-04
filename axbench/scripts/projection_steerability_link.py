"""
Link projection-baseline correlation (rho) to steerability across layers.

For each layer, collects three families of metrics:
  1. rho  -- Spearman correlation of normalized_projection vs baseline_score
             (already computed by projection_vs_steering_factor.py)
  2. d'   -- training separability from separability.json
  3. steerability -- computed here from per-prompt eval data with fluency + min_valid filtering

Then correlates (rho, d') against steerability across layers to test
whether layers where projections are more predictive of behavior also
steer more effectively.

Steerability metrics (all use ONLY fluency-valid, min-valid factors):
  - best_mean_score: highest mean behavior_score across valid positive factors
  - best_factor:     the factor achieving best_mean_score
  - delta:           best_mean_score - baseline_mean
  - score_f1/f2/f5:  mean behavior_score at factor=1/2/5 (NaN if not valid)
  - efficiency:      delta / best_factor
  - auc_positive:    average of mean scores across all valid positive factors

Outputs:
  {sweep_dir}/steerability_link/
    layer_metrics.csv
    cross_layer_correlations.csv
    summary.json
    plots/
      rho_vs_steerability.png
      dprime_vs_steerability.png
      metrics_by_layer.png
      predictor_comparison.png

Usage:
    uv run python axbench/scripts/projection_steerability_link.py \\
        --behavior corrigible-neutral-HHH \\
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep

    # Custom thresholds:
    uv run python axbench/scripts/projection_steerability_link.py \\
        --behavior corrigible-neutral-HHH \\
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep \\
        --fluency_threshold 1.0 --min_valid 25
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


# ============================================================================
# Discovery
# ============================================================================
def discover_layers(sweep_dir: Path) -> list[int]:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            if (d / "eval" / "eval_results.parquet").exists():
                layers.append(int(d.name.split("_")[1]))
    return sorted(layers)


# ============================================================================
# Load existing analysis data (rho, d')
# ============================================================================
def load_projection_correlations(sweep_dir: Path) -> pd.DataFrame:
    """Load rho values from projection_analysis/correlations.csv."""
    corr_path = sweep_dir / "projection_analysis" / "correlations.csv"
    if not corr_path.exists():
        logger.error(f"Missing {corr_path}. Run projection_vs_steering_factor.py first.")
        sys.exit(1)
    df = pd.read_csv(corr_path)
    return df[["layer", "spearman_rho", "spearman_p", "pearson_r", "pearson_p", "n"]].copy()


def load_separability(sweep_dir: Path, layers: list[int]) -> pd.DataFrame:
    """Load d' and auroc from layer_N/separability.json."""
    rows = []
    for layer in layers:
        sep_path = sweep_dir / f"layer_{layer}" / "separability.json"
        if sep_path.exists():
            sep = json.loads(sep_path.read_text())
            rows.append({
                "layer": layer,
                "dprime": sep.get("dprime"),
                "auroc": sep.get("auroc"),
                "sv_norm": sep.get("norm"),
            })
        else:
            rows.append({"layer": layer, "dprime": None, "auroc": None, "sv_norm": None})
    return pd.DataFrame(rows)


# ============================================================================
# Compute steerability with fluency + min_valid filtering
# ============================================================================
def compute_steerability_per_layer(
    sweep_dir: Path,
    layers: list[int],
    fluency_threshold: float = 1.0,
    min_valid: int = 25,
) -> pd.DataFrame:
    """
    For each layer, load per-prompt eval data, apply fluency filter, then:
      - group by steering_factor
      - keep only factors with >= min_valid fluency-valid examples
      - compute steerability metrics from valid factors
    """
    all_rows = []
    for layer in layers:
        layer_dir = sweep_dir / f"layer_{layer}"
        eval_path = layer_dir / "eval" / "eval_results.parquet"
        axbench_path = layer_dir / "eval_axbench" / "eval_results.parquet"

        if not eval_path.exists():
            logger.warning(f"  Layer {layer}: no eval_results.parquet, skipping")
            continue

        eval_df = pd.read_parquet(eval_path)
        if "behavior_score" not in eval_df.columns:
            logger.warning(f"  Layer {layer}: no behavior_score column, skipping")
            continue

        working_df = eval_df[["question_idx", "steering_factor", "behavior_score"]].copy()
        working_df = working_df.dropna(subset=["behavior_score"])

        if axbench_path.exists():
            ax_df = pd.read_parquet(axbench_path)
            if "fluency_score" in ax_df.columns:
                fluency_cols = ax_df[["question_idx", "steering_factor", "fluency_score"]]
                working_df = working_df.merge(
                    fluency_cols, on=["question_idx", "steering_factor"], how="inner"
                )
                n_before = len(working_df)
                working_df = working_df[working_df["fluency_score"] >= fluency_threshold]
                n_after = len(working_df)
                logger.warning(
                    f"  Layer {layer}: fluency >= {fluency_threshold}: "
                    f"{n_after}/{n_before} rows kept"
                )
            else:
                logger.warning(f"  Layer {layer}: no fluency_score in axbench, skipping filter")
        else:
            logger.warning(f"  Layer {layer}: no eval_axbench, skipping fluency filter")

        per_factor = (
            working_df.groupby("steering_factor")["behavior_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_score", "count": "n_valid"})
        )
        valid_factors = per_factor[per_factor["n_valid"] >= min_valid]

        f0 = valid_factors[valid_factors["steering_factor"] == 0]
        baseline_mean = float(f0["mean_score"].values[0]) if not f0.empty else np.nan

        positive_valid = valid_factors[valid_factors["steering_factor"] > 0]

        if positive_valid.empty:
            best_mean_score = np.nan
            best_factor = np.nan
            delta = np.nan
            efficiency = np.nan
            auc_positive = np.nan
        else:
            best_row = positive_valid.loc[positive_valid["mean_score"].idxmax()]
            best_mean_score = float(best_row["mean_score"])
            best_factor = float(best_row["steering_factor"])
            delta = best_mean_score - baseline_mean if not np.isnan(baseline_mean) else np.nan
            efficiency = delta / best_factor if best_factor > 0 and not np.isnan(delta) else np.nan
            auc_positive = float(positive_valid["mean_score"].mean())

        def _score_at_factor(f):
            row = valid_factors[valid_factors["steering_factor"] == f]
            return float(row["mean_score"].values[0]) if not row.empty else np.nan

        all_rows.append({
            "layer": layer,
            "baseline_mean": baseline_mean,
            "best_mean_score": best_mean_score,
            "best_factor": best_factor,
            "delta": delta,
            "score_f1": _score_at_factor(1),
            "score_f2": _score_at_factor(2),
            "score_f5": _score_at_factor(5),
            "efficiency": efficiency,
            "auc_positive": auc_positive,
            "n_valid_factors": len(positive_valid),
        })

    return pd.DataFrame(all_rows)


# ============================================================================
# Cross-layer correlation
# ============================================================================
STEERABILITY_COLS = [
    "best_mean_score", "delta", "score_f1", "score_f2", "score_f5",
    "efficiency", "auc_positive",
]


def correlate_predictor_vs_steerability(
    merged: pd.DataFrame,
    predictor_col: str,
    steerability_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Spearman + Pearson of predictor vs each steerability metric across layers."""
    if steerability_cols is None:
        steerability_cols = STEERABILITY_COLS
    rows = []
    for col in steerability_cols:
        valid = merged[[predictor_col, col]].dropna()
        if len(valid) < 4:
            continue
        rho, p_rho = scipy_stats.spearmanr(valid[predictor_col], valid[col])
        r, p_r = scipy_stats.pearsonr(valid[predictor_col], valid[col])
        rows.append({
            "predictor": predictor_col,
            "steerability_metric": col,
            "spearman_rho": rho, "spearman_p": p_rho,
            "pearson_r": r, "pearson_p": p_r,
            "n_layers": len(valid),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================
def plot_predictor_vs_steerability(
    merged: pd.DataFrame,
    predictor_col: str,
    predictor_label: str,
    behavior: str,
    output_path: Path,
):
    """Multi-panel scatter: predictor vs each steerability metric."""
    cols = [c for c in STEERABILITY_COLS if c in merged.columns and merged[c].notna().sum() >= 4]
    if not cols:
        return
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        valid = merged[[predictor_col, col, "layer"]].dropna()
        if len(valid) < 4:
            ax.set_visible(False)
            continue
        ax.scatter(valid[predictor_col], valid[col], s=50, edgecolors="k",
                   linewidths=0.4, color="#2E86AB", zorder=3)
        for _, row in valid.iterrows():
            ax.annotate(f"{int(row['layer'])}", (row[predictor_col], row[col]),
                        fontsize=6, alpha=0.7, textcoords="offset points",
                        xytext=(4, 4))
        rho, p = scipy_stats.spearmanr(valid[predictor_col], valid[col])
        z = np.polyfit(valid[predictor_col], valid[col], 1)
        xr = np.linspace(valid[predictor_col].min(), valid[predictor_col].max(), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="red", alpha=0.6, linewidth=1.5)
        sig = "*" if p < 0.05 else ""
        ax.set_title(f"vs {col}\nrho={rho:.3f} (p={p:.2e}){sig}", fontsize=9)
        ax.set_xlabel(predictor_label, fontsize=9)
        ax.set_ylabel(col, fontsize=9)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{behavior}: {predictor_label} vs Steerability Metrics",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_by_layer(merged: pd.DataFrame, behavior: str, output_path: Path):
    """Line plot of rho, d', and key steerability metric across layers."""
    layers = merged["layer"].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: rho
    ax = axes[0]
    if "spearman_rho" in merged.columns:
        ax.plot(layers, merged["spearman_rho"], "o-", color="#2E86AB", linewidth=2,
                markersize=6, label="Spearman rho")
        sig = merged["spearman_p"] < 0.05
        ax.scatter(layers[sig], merged["spearman_rho"].values[sig], s=100,
                   facecolors="none", edgecolors="red", linewidths=2, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Spearman rho\n(proj vs baseline)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title(f"{behavior}: Metrics by Layer", fontsize=12, fontweight="bold")

    # Panel 2: d'
    ax = axes[1]
    if "dprime" in merged.columns and merged["dprime"].notna().any():
        ax.plot(layers, merged["dprime"], "s-", color="#E94F37", linewidth=2,
                markersize=6, label="d'")
        ax.legend(fontsize=8)
    ax.set_ylabel("d' (training\nseparability)", fontsize=10)

    # Panel 3: steerability delta
    ax = axes[2]
    if "delta" in merged.columns:
        ax.plot(layers, merged["delta"], "D-", color="#44AF69", linewidth=2,
                markersize=6, label="Delta (best - baseline)")
    if "auc_positive" in merged.columns:
        ax2 = ax.twinx()
        ax2.plot(layers, merged["auc_positive"], "^--", color="#6B2D5C", linewidth=1.5,
                 markersize=5, alpha=0.7, label="AUC positive")
        ax2.set_ylabel("AUC positive", fontsize=9, color="#6B2D5C")
        ax2.tick_params(axis="y", labelcolor="#6B2D5C")
    ax.set_ylabel("Delta\n(best - baseline)", fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    for a in axes:
        a.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictor_comparison(merged: pd.DataFrame, behavior: str, output_path: Path):
    """Side-by-side: rho vs delta and d' vs delta, to compare which predicts better."""
    has_rho = "spearman_rho" in merged.columns and merged["spearman_rho"].notna().sum() >= 4
    has_dprime = "dprime" in merged.columns and merged["dprime"].notna().sum() >= 4
    has_delta = "delta" in merged.columns and merged["delta"].notna().sum() >= 4

    if not has_delta or (not has_rho and not has_dprime):
        return

    ncols = int(has_rho) + int(has_dprime)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    panel_idx = 0
    for pred_col, pred_label, color in [
        ("spearman_rho", "Spearman rho (proj vs baseline)", "#2E86AB"),
        ("dprime", "d' (training separability)", "#E94F37"),
    ]:
        if pred_col not in merged.columns or merged[pred_col].notna().sum() < 4:
            continue
        ax = axes[panel_idx]
        valid = merged[[pred_col, "delta", "layer"]].dropna()
        ax.scatter(valid[pred_col], valid["delta"], s=60, color=color,
                   edgecolors="k", linewidths=0.4, zorder=3)
        for _, row in valid.iterrows():
            ax.annotate(f"L{int(row['layer'])}", (row[pred_col], row["delta"]),
                        fontsize=7, alpha=0.7, textcoords="offset points", xytext=(4, 4))

        rho, p = scipy_stats.spearmanr(valid[pred_col], valid["delta"])
        z = np.polyfit(valid[pred_col], valid["delta"], 1)
        xr = np.linspace(valid[pred_col].min(), valid[pred_col].max(), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="red", alpha=0.6, linewidth=1.5)
        sig = " *" if p < 0.05 else ""
        ax.set_title(f"rho = {rho:.3f} (p = {p:.2e}){sig}", fontsize=10)
        ax.set_xlabel(pred_label, fontsize=10)
        ax.set_ylabel("Delta (best steered - baseline)", fontsize=10)
        panel_idx += 1

    fig.suptitle(f"{behavior}: Which predictor explains steerability?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Link projection-baseline rho to steerability across layers"
    )
    parser.add_argument("--behavior", type=str, required=True)
    parser.add_argument("--sweep_dir", type=str, required=True)
    parser.add_argument("--fluency_threshold", type=float, default=1.0,
                        help="Min fluency_score to keep (default: 1.0)")
    parser.add_argument("--min_valid", type=int, default=25,
                        help="Min examples per factor to be valid (default: 25)")
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

    # --- 1. Load rho from projection analysis ---
    logger.warning("Loading projection correlations ...")
    rho_df = load_projection_correlations(sweep_dir)

    # --- 2. Load d' from separability.json ---
    logger.warning("Loading separability (d') ...")
    sep_df = load_separability(sweep_dir, layers)

    # --- 3. Compute steerability with fluency + min_valid ---
    logger.warning(
        f"Computing steerability (fluency >= {args.fluency_threshold}, "
        f"min_valid >= {args.min_valid}) ..."
    )
    steer_df = compute_steerability_per_layer(
        sweep_dir, layers, args.fluency_threshold, args.min_valid
    )

    # --- 4. Merge everything ---
    merged = rho_df.merge(sep_df, on="layer", how="outer")
    merged = merged.merge(steer_df, on="layer", how="outer")
    merged = merged.sort_values("layer").reset_index(drop=True)

    merged.to_csv(out_dir / "layer_metrics.csv", index=False)
    logger.warning(f"Saved layer_metrics.csv ({len(merged)} layers)")

    logger.warning("\nPer-layer summary:")
    for _, row in merged.iterrows():
        rho_str = f"rho={row.get('spearman_rho', np.nan):+.3f}" if pd.notna(row.get("spearman_rho")) else "rho=N/A"
        dp_str = f"d'={row.get('dprime', np.nan):.2f}" if pd.notna(row.get("dprime")) else "d'=N/A"
        delta_str = f"delta={row.get('delta', np.nan):+.2f}" if pd.notna(row.get("delta")) else "delta=N/A"
        best_str = f"best={row.get('best_mean_score', np.nan):.2f}@f={row.get('best_factor', np.nan):.0f}" if pd.notna(row.get("best_mean_score")) else "best=N/A"
        logger.warning(f"  Layer {int(row['layer']):2d}: {rho_str}  {dp_str}  {delta_str}  {best_str}")

    # --- 5. Cross-layer correlations ---
    logger.warning("\nCross-layer correlations:")
    corr_rows = []

    for pred_col, pred_label in [("spearman_rho", "rho"), ("dprime", "d'")]:
        if pred_col not in merged.columns or merged[pred_col].notna().sum() < 4:
            logger.warning(f"  {pred_label}: not enough data, skipping")
            continue
        corr = correlate_predictor_vs_steerability(merged, pred_col)
        corr_rows.append(corr)
        logger.warning(f"\n  {pred_label} vs steerability:")
        for _, row in corr.iterrows():
            sig = " *" if row["spearman_p"] < 0.05 else ""
            logger.warning(
                f"    vs {row['steerability_metric']:20s}: "
                f"rho={row['spearman_rho']:+.3f} (p={row['spearman_p']:.2e}), "
                f"r={row['pearson_r']:+.3f} (p={row['pearson_p']:.2e}), "
                f"n={int(row['n_layers'])}{sig}"
            )

    if corr_rows:
        all_corr = pd.concat(corr_rows, ignore_index=True)
        all_corr.to_csv(out_dir / "cross_layer_correlations.csv", index=False)
    else:
        all_corr = pd.DataFrame()

    # --- 6. Plots ---
    logger.warning("\nGenerating plots ...")
    plot_predictor_vs_steerability(
        merged, "spearman_rho", "Spearman rho (proj vs baseline)", args.behavior,
        plot_dir / "rho_vs_steerability.png",
    )
    plot_predictor_vs_steerability(
        merged, "dprime", "d' (training separability)", args.behavior,
        plot_dir / "dprime_vs_steerability.png",
    )
    plot_metrics_by_layer(merged, args.behavior, plot_dir / "metrics_by_layer.png")
    plot_predictor_comparison(merged, args.behavior, plot_dir / "predictor_comparison.png")

    # --- 7. Summary JSON ---
    summary = {
        "behavior": args.behavior,
        "sweep_dir": str(sweep_dir),
        "fluency_threshold": args.fluency_threshold,
        "min_valid": args.min_valid,
        "n_layers": len(merged),
        "layers": layers,
        "cross_layer_correlations": all_corr.to_dict(orient="records") if not all_corr.empty else [],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
