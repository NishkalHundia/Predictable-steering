"""
Plot KL divergence results from eval_kl_divergence.py.

Three modes:
1. Single-concept: Plot KL vs. steering factor for one behavior
2. Multi-concept:  Overlay multiple behaviors on the same plot using effective_strength
3. AxBench auto:   --axbench --axbench_dir <dir> auto-discovers all concept subdirs

Usage:
    # Single concept:
    uv run python axbench/scripts/plot_kl_divergence.py \
        --results_dirs results/gemma-2-9b-it/sycophancy-open-ended/kl_eval \
        --behaviors sycophancy

    # Multi-concept comparison:
    uv run python axbench/scripts/plot_kl_divergence.py \
        --results_dirs results/gemma-2-9b-it/sycophancy-open-ended/kl_eval \
                       results/gemma-2-9b-it/hallucination-open-ended/kl_eval \
                       results/gemma-2-9b-it/myopic-reward-open-ended/kl_eval \
        --behaviors sycophancy hallucination myopic-reward

    # AxBench auto-discover:
    uv run python axbench/scripts/plot_kl_divergence.py \
        --axbench --axbench_dir results/gemma-2-9b-it/axbench_concepts
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')

BEHAVIOR_COLORS = {
    "sycophancy": "#E94F37",
    "hallucination": "#2E86AB",
    "survival-instinct": "#44AF69",
    "corrigible-neutral-HHH": "#6B2D5C",
    "refusal": "#F4A261",
    "coordinate-other-ais": "#264653",
    "myopic-reward": "#E76F51",
}

BEHAVIOR_LABELS = {
    "sycophancy": "Sycophancy",
    "hallucination": "Hallucination",
    "survival-instinct": "Survival Instinct",
    "corrigible-neutral-HHH": "Corrigibility",
    "refusal": "Refusal",
    "coordinate-other-ais": "AI Coordination",
    "myopic-reward": "Myopic Reward",
}


def load_results(results_dir, behavior):
    """Load kl_summary.csv and kl_results.parquet from a results directory."""
    results_dir = Path(results_dir)

    summary_path = results_dir / "kl_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    summary_df = pd.read_csv(summary_path)

    detail_path = results_dir / "kl_results.parquet"
    detail_df = None
    if detail_path.exists():
        detail_df = pd.read_parquet(detail_path)

    return summary_df, detail_df


# =========================================================================
# Single-concept plots
# =========================================================================
def plot_single_kl_vs_factor(summary_df, behavior, output_dir):
    """Plot KL divergence vs steering factor for a single concept."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    label = BEHAVIOR_LABELS.get(behavior, behavior)
    color = BEHAVIOR_COLORS.get(behavior, "#2E86AB")

    factors = summary_df["steering_factor"].values
    kl_mean = summary_df["kl_mean"].values
    kl_std = summary_df["kl_std"].values

    # --- Plot 1: KL vs factor ---
    ax1 = axes[0]
    ax1.errorbar(
        factors, kl_mean, yerr=kl_std,
        marker='o', markersize=8, linewidth=2,
        capsize=5, capthick=1.5, color=color, ecolor=color, alpha=0.85,
    )
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Steering Factor', fontsize=12)
    ax1.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax1.set_title(f'{label}: KL(Steered ‖ Base) vs Factor', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # value labels
    for f, k in zip(factors, kl_mean):
        ax1.annotate(f'{k:.2f}', (f, k), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # --- Plot 2: KL vs effective strength ---
    ax2 = axes[1]
    if "effective_strength" in summary_df.columns:
        eff = summary_df["effective_strength"].values
    else:
        # fallback: we don't have vector_norm, just use factor
        eff = factors
        logger.warning("effective_strength column not found, using raw factor")

    ax2.errorbar(
        eff, kl_mean, yerr=kl_std,
        marker='s', markersize=8, linewidth=2,
        capsize=5, capthick=1.5, color=color, ecolor=color, alpha=0.85,
    )
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Effective Strength (factor × ‖v‖)', fontsize=12)
    ax2.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax2.set_title(f'{label}: KL vs Effective Strength', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    if "vector_norm" in summary_df.columns:
        norm_val = summary_df["vector_norm"].iloc[0]
        ax2.annotate(f'‖v‖ = {norm_val:.1f}', xy=(0.02, 0.95),
                     xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = output_dir / "kl_vs_factor.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved: {plot_path}")


def plot_single_kl_distribution(detail_df, behavior, output_dir):
    """Box plot of per-prompt KL at each factor."""
    if detail_df is None:
        return

    label = BEHAVIOR_LABELS.get(behavior, behavior)
    color = BEHAVIOR_COLORS.get(behavior, "#2E86AB")

    factors = sorted(detail_df["steering_factor"].unique())
    data_by_factor = [
        detail_df[detail_df["steering_factor"] == f]["kl_divergence"].values
        for f in factors
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(factors) * 0.8), 6))
    bp = ax.boxplot(
        data_by_factor, positions=range(len(factors)), widths=0.5,
        patch_artist=True, showfliers=True, flierprops=dict(markersize=3, alpha=0.4),
    )
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([f'{f}' for f in factors], fontsize=9)
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax.set_title(f'{label}: Per-Prompt KL Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / "kl_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved: {plot_path}")


# =========================================================================
# Multi-concept comparison plots
# =========================================================================
def plot_multi_kl_vs_factor(all_summaries, output_dir):
    """Overlay KL vs factor for multiple concepts."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for behavior, summary_df in all_summaries.items():
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        color = BEHAVIOR_COLORS.get(behavior, None)

        factors = summary_df["steering_factor"].values
        kl_mean = summary_df["kl_mean"].values

        ax.plot(factors, kl_mean, 'o-', label=label, color=color,
                linewidth=2, markersize=7, alpha=0.85)

    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax.set_title('KL(Steered ‖ Base) by Concept — Raw Factor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "multi_kl_vs_factor.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved: {plot_path}")


def plot_multi_kl_vs_effective_strength(all_summaries, output_dir):
    """Overlay KL vs effective_strength for multiple concepts (the key comparison plot)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for behavior, summary_df in all_summaries.items():
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        color = BEHAVIOR_COLORS.get(behavior, None)

        if "effective_strength" in summary_df.columns:
            x = summary_df["effective_strength"].values
        else:
            x = summary_df["steering_factor"].values

        kl_mean = summary_df["kl_mean"].values

        # Annotate with vector norm
        norm_str = ""
        if "vector_norm" in summary_df.columns:
            norm_str = f" (‖v‖={summary_df['vector_norm'].iloc[0]:.1f})"

        ax.plot(x, kl_mean, 'o-', label=f'{label}{norm_str}', color=color,
                linewidth=2, markersize=7, alpha=0.85)

    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Effective Strength (factor × ‖v‖)', fontsize=12)
    ax.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax.set_title('KL(Steered ‖ Base) by Concept — Normalized Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "multi_kl_vs_effective_strength.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved: {plot_path}")


def plot_multi_kl_bar_at_factor(all_summaries, output_dir, target_factors=None):
    """
    Bar chart comparing mean KL across concepts at specific factors.
    Useful for a quick "which concept shifts the distribution most?" view.
    """
    if target_factors is None:
        # Pick a representative subset
        all_factors = set()
        for df in all_summaries.values():
            all_factors.update(df["steering_factor"].tolist())
        # Try to pick some reasonable ones
        candidates = [-100, -50, -10, -2, -1, 1, 2, 10, 50, 100]
        target_factors = [f for f in candidates if f in all_factors]
        if not target_factors:
            target_factors = sorted(all_factors)

    behaviors = list(all_summaries.keys())
    n_behaviors = len(behaviors)
    n_factors = len(target_factors)

    fig, ax = plt.subplots(figsize=(max(10, n_factors * 2), 6))
    x = np.arange(n_factors)
    width = 0.8 / n_behaviors

    for i, behavior in enumerate(behaviors):
        summary_df = all_summaries[behavior]
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        color = BEHAVIOR_COLORS.get(behavior, None)

        kl_vals = []
        for f in target_factors:
            row = summary_df[summary_df["steering_factor"] == f]
            kl_vals.append(row["kl_mean"].values[0] if len(row) > 0 else 0)

        offset = (i - n_behaviors / 2 + 0.5) * width
        ax.bar(x + offset, kl_vals, width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in target_factors])
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Mean KL Divergence (nats)', fontsize=12)
    ax.set_title('KL Divergence Comparison Across Concepts', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / "multi_kl_bar_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved: {plot_path}")


def print_summary_table(all_summaries):
    """Print a combined summary table to stdout."""
    print("\n" + "=" * 80)
    print("KL DIVERGENCE SUMMARY ACROSS CONCEPTS")
    print("=" * 80)

    for behavior, summary_df in all_summaries.items():
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        norm = summary_df["vector_norm"].iloc[0] if "vector_norm" in summary_df.columns else "?"
        print(f"\n--- {label} (‖v‖ = {norm}) ---")
        cols = ["steering_factor", "effective_strength", "kl_mean", "kl_std", "kl_median"]
        cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[cols].to_string(index=False))


def discover_axbench_kl_results(axbench_dir):
    """
    Auto-discover concept subdirs that have kl_eval/kl_summary.csv.
    Returns list of (label, kl_eval_dir) tuples.
    """
    axbench_dir = Path(axbench_dir)
    concepts = []

    for subdir in sorted(axbench_dir.iterdir()):
        if not subdir.is_dir():
            continue
        kl_eval_dir = subdir / "kl_eval"
        summary_path = kl_eval_dir / "kl_summary.csv"
        if not summary_path.exists():
            continue

        # Try to load config for a nice label
        config_path = subdir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            concept_name = config.get("concept_name", subdir.name)
            concept_id = config.get("concept_id", "")
            label = f"{concept_id}_{concept_name}" if concept_id else concept_name
        else:
            label = subdir.name

        concepts.append((label, kl_eval_dir))

    return concepts


def main():
    parser = argparse.ArgumentParser(
        description="Plot KL divergence results (single, multi-concept, or axbench auto)."
    )
    parser.add_argument("--results_dirs", type=str, nargs='+', default=None,
                        help="One or more directories containing kl_summary.csv")
    parser.add_argument("--behaviors", type=str, nargs='+', default=None,
                        help="Behavior name(s), one per results_dir")
    parser.add_argument("--axbench", action="store_true",
                        help="Auto-discover all AxBench concept results in --axbench_dir")
    parser.add_argument("--axbench_dir", type=str, default=None,
                        help="Directory containing concept subdirs with kl_eval/ results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save plots. Defaults to first results_dir for single, "
                             "axbench_dir/kl_comparison for axbench, "
                             "or a shared parent for multi.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve which mode we're in and build results_dirs / behaviors
    # ------------------------------------------------------------------
    if args.axbench:
        if not args.axbench_dir:
            print("ERROR: --axbench_dir is required when --axbench is set")
            return
        discovered = discover_axbench_kl_results(args.axbench_dir)
        if not discovered:
            print(f"ERROR: No concept subdirs with kl_eval/kl_summary.csv found in {args.axbench_dir}")
            return
        logger.warning(f"Auto-discovered {len(discovered)} AxBench concepts in {args.axbench_dir}")
        behaviors = [label for label, _ in discovered]
        results_dirs = [str(kl_dir) for _, kl_dir in discovered]

        if args.output_dir is None:
            args.output_dir = str(Path(args.axbench_dir) / "kl_comparison")
    else:
        if not args.results_dirs or not args.behaviors:
            print("ERROR: --results_dirs and --behaviors are required "
                  "(or use --axbench --axbench_dir)")
            return
        if len(args.results_dirs) != len(args.behaviors):
            print("ERROR: Must provide the same number of --results_dirs and --behaviors")
            return
        results_dirs = args.results_dirs
        behaviors = args.behaviors

    # ------------------------------------------------------------------
    # Load all results
    # ------------------------------------------------------------------
    all_summaries = {}
    all_details = {}
    for rdir, behavior in zip(results_dirs, behaviors):
        summary_df, detail_df = load_results(rdir, behavior)
        all_summaries[behavior] = summary_df
        all_details[behavior] = detail_df

    # Determine output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif len(results_dirs) == 1:
        output_dir = Path(results_dirs[0])
    else:
        output_dir = Path(results_dirs[0]).parent / "kl_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(behaviors) == 1:
        # ── Single-concept plots ──
        behavior = behaviors[0]
        summary_df = all_summaries[behavior]
        detail_df = all_details[behavior]

        plot_single_kl_vs_factor(summary_df, behavior, output_dir)
        plot_single_kl_distribution(detail_df, behavior, output_dir)
    else:
        # ── Multi-concept comparison plots ──
        plot_multi_kl_vs_factor(all_summaries, output_dir)
        plot_multi_kl_vs_effective_strength(all_summaries, output_dir)
        plot_multi_kl_bar_at_factor(all_summaries, output_dir)

    # Always print table
    print_summary_table(all_summaries)

    logger.warning(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
