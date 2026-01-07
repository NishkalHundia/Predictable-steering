"""
Generate plots from open-ended evaluation results.

Usage:
    uv run python -m axbench.scripts.open_ended_plot \
        --results_dir results/sycophancy-open-ended \
        --behavior sycophancy
"""
import argparse
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


# Behavior descriptions for plot titles (with score interpretation)
# Higher score means MORE of the behavior for most, except survival-instinct & corrigibility
BEHAVIOR_LABELS = {
    "hallucination": ("Hallucination Score", "0=accurate, 10=made up"),
    "sycophancy": ("Sycophancy Score", "0=honest, 10=sycophantic"),
    "survival-instinct": ("Shutdown Acceptance Score", "0=self-preservation, 10=accepts shutdown"),
    "corrigible-neutral-HHH": ("Corrigibility Score", "0=rigid, 10=corrigible"),
    "refusal": ("Refusal Score", "0=compliant, 10=refuses"),
    "myopic-reward": ("Myopic Reward Score", "0=long-term, 10=myopic"),
    "coordinate-other-ais": ("AI Coordination Score", "0=no coordination, 10=strong coordination"),
}


def plot_behavior_vs_factor(summary_df, behavior, output_dir):
    """Plot behavior score distribution vs steering factor (0-10 scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    factors = summary_df["steering_factor"].values
    avg_score = summary_df["avg_score"].values
    
    label_info = BEHAVIOR_LABELS.get(behavior, (behavior, ""))
    title = label_info[0] if isinstance(label_info, tuple) else label_info
    subtitle = label_info[1] if isinstance(label_info, tuple) else ""
    
    # Plot 1: Average score trend
    ax1 = axes[0]
    ax1.plot(factors, avg_score, 'b-o', linewidth=2, markersize=10)
    ax1.fill_between(factors, 0, avg_score, alpha=0.2)
    ax1.set_xlabel('Steering Factor', fontsize=12)
    ax1.set_ylabel('Average Score (0-10)', fontsize=12)
    ax1.set_title(f'{title} vs Steering Factor\n({subtitle})', fontsize=14)
    ax1.set_ylim(-0.5, 10.5)
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Neutral (5)')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='No steering')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for f, s in zip(factors, avg_score):
        ax1.annotate(f'{s:.1f}', (f, s), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: Score distribution (stacked bar)
    ax2 = axes[1]
    
    low_pct = summary_df["low_pct"].values
    medium_pct = summary_df["medium_pct"].values
    high_pct = summary_df["high_pct"].values
    
    x = np.arange(len(factors))
    width = 0.6
    
    ax2.bar(x, low_pct, width, label='Low (0-3)', color='green', alpha=0.7)
    ax2.bar(x, medium_pct, width, bottom=low_pct, label='Medium (4-6)', color='orange', alpha=0.7)
    ax2.bar(x, high_pct, width, bottom=low_pct+medium_pct, label='High (7-10)', color='red', alpha=0.7)
    
    ax2.set_xlabel('Steering Factor', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Score Distribution by Steering Factor', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{f:.1f}' for f in factors])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "behavior_vs_factor.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved plot to {plot_path}")


def plot_score_trend(summary_df, behavior, output_dir):
    """Plot average behavior score trend with error bars (0-10 scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = summary_df["steering_factor"].values
    avg_score = summary_df["avg_score"].values
    std_score = summary_df["std_score"].values if "std_score" in summary_df.columns else np.zeros_like(avg_score)
    
    label_info = BEHAVIOR_LABELS.get(behavior, (behavior, ""))
    title = label_info[0] if isinstance(label_info, tuple) else label_info
    subtitle = label_info[1] if isinstance(label_info, tuple) else ""
    
    # Plot with error bars
    ax.errorbar(factors, avg_score, yerr=std_score, fmt='b-o', linewidth=2, markersize=10,
                capsize=5, capthick=2, ecolor='lightblue', alpha=0.8)
    
    # Fill area under curve
    ax.fill_between(factors, 0, avg_score, alpha=0.15)
    
    # Add annotations
    for f, s in zip(factors, avg_score):
        ax.annotate(f'{s:.2f}', (f, s), textcoords="offset points", 
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Average Score (0-10)', fontsize=12)
    ax.set_title(f'{title} vs Steering Factor\n({subtitle})', fontsize=14)
    ax.set_ylim(-0.5, 10.5)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Neutral (5)')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='No steering')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "score_trend.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved plot to {plot_path}")


def plot_effectiveness(summary_df, behavior, output_dir):
    """Plot steering effectiveness (change from baseline, 0-10 scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = summary_df["steering_factor"].values
    avg_score = summary_df["avg_score"].values
    
    label_info = BEHAVIOR_LABELS.get(behavior, (behavior, ""))
    title = label_info[0] if isinstance(label_info, tuple) else label_info
    
    # Find baseline (factor=0)
    baseline_idx = np.where(factors == 0)[0]
    if len(baseline_idx) > 0:
        baseline_score = avg_score[baseline_idx[0]]
    else:
        baseline_score = avg_score[len(avg_score)//2]  # Use middle as baseline
    
    # Calculate change from baseline
    change_from_baseline = avg_score - baseline_score
    
    # Bar plot with positive/negative coloring
    colors = ['green' if c < 0 else 'red' for c in change_from_baseline]
    bars = ax.bar(factors, change_from_baseline, width=0.4, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Change in Score from Baseline', fontsize=12)
    ax.set_title(f'Steering Effectiveness: Change in {title}\nfrom Baseline (factor=0, baseline={baseline_score:.2f})', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, change_from_baseline):
        height = bar.get_height()
        ax.annotate(f'{val:+.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / "effectiveness.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing open_ended_results.parquet and summary.csv")
    parser.add_argument("--behavior", type=str, required=True,
                        help="Behavior name for plot labels")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load summary
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return
    
    summary_df = pd.read_csv(summary_path)
    logger.warning(f"Loaded summary with {len(summary_df)} steering factors")
    
    # Generate plots
    plot_behavior_vs_factor(summary_df, args.behavior, results_dir)
    plot_score_trend(summary_df, args.behavior, results_dir)
    plot_effectiveness(summary_df, args.behavior, results_dir)
    
    logger.warning("All plots generated!")


if __name__ == "__main__":
    main()

