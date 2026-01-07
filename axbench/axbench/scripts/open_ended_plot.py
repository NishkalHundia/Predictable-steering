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


# Behavior descriptions for plot titles
BEHAVIOR_LABELS = {
    "hallucination": "Hallucination Rate",
    "sycophancy": "Sycophancy Rate", 
    "survival-instinct": "Self-Preservation Rate",
    "corrigible-neutral-HHH": "Resistance to Modification Rate",
    "refusal": "Compliance Rate (lower = more refusal)",
    "myopic-reward": "Myopic Reward Preference Rate",
}


def plot_behavior_vs_factor(summary_df, behavior, output_dir):
    """Plot behavior rate vs steering factor."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    factors = summary_df["steering_factor"].values
    behavior_rate = summary_df["behavior_rate_pct"].values
    avg_score = summary_df["avg_behavior_score"].values
    
    # Plot 1: Behavior rate
    ax1 = axes[0]
    ax1.plot(factors, behavior_rate, 'b-o', linewidth=2, markersize=8, label='Behavior Rate (score=2)')
    ax1.fill_between(factors, 0, behavior_rate, alpha=0.2)
    ax1.set_xlabel('Steering Factor', fontsize=12)
    ax1.set_ylabel('Behavior Rate (%)', fontsize=12)
    ax1.set_title(f'{BEHAVIOR_LABELS.get(behavior, behavior)} vs Steering Factor', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='No steering')
    ax1.legend()
    
    # Plot 2: Average score
    ax2 = axes[1]
    colors = ['green', 'orange', 'red']
    
    # Stacked bar chart
    non_behavior = summary_df["non_behavior_rate_pct"].values
    partial = summary_df["partial_rate_pct"].values
    behavior_full = summary_df["behavior_rate_pct"].values
    
    x = np.arange(len(factors))
    width = 0.6
    
    ax2.bar(x, non_behavior, width, label='Non-behavior (score=0)', color='green', alpha=0.7)
    ax2.bar(x, partial, width, bottom=non_behavior, label='Partial (score=1)', color='orange', alpha=0.7)
    ax2.bar(x, behavior_full, width, bottom=non_behavior+partial, label='Full behavior (score=2)', color='red', alpha=0.7)
    
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
    """Plot average behavior score trend."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = summary_df["steering_factor"].values
    avg_score = summary_df["avg_behavior_score"].values
    
    # Create gradient color based on score
    ax.plot(factors, avg_score, 'b-o', linewidth=2, markersize=10)
    
    # Add annotations
    for f, s in zip(factors, avg_score):
        ax.annotate(f'{s:.2f}', (f, s), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Average Behavior Score (0-2)', fontsize=12)
    ax.set_title(f'Average {BEHAVIOR_LABELS.get(behavior, behavior).replace(" Rate", " Score")} vs Steering Factor', fontsize=14)
    ax.set_ylim(-0.1, 2.1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "score_trend.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.warning(f"Saved plot to {plot_path}")


def plot_effectiveness(summary_df, behavior, output_dir):
    """Plot steering effectiveness (change from baseline)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = summary_df["steering_factor"].values
    behavior_rate = summary_df["behavior_rate_pct"].values
    
    # Find baseline (factor=0)
    baseline_idx = np.where(factors == 0)[0]
    if len(baseline_idx) > 0:
        baseline_rate = behavior_rate[baseline_idx[0]]
    else:
        baseline_rate = behavior_rate[len(behavior_rate)//2]  # Use middle as baseline
    
    # Calculate change from baseline
    change_from_baseline = behavior_rate - baseline_rate
    
    # Bar plot with positive/negative coloring
    colors = ['green' if c < 0 else 'red' for c in change_from_baseline]
    bars = ax.bar(factors, change_from_baseline, width=0.4, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Change in Behavior Rate (pp)', fontsize=12)
    ax.set_title(f'Steering Effectiveness: Change in {BEHAVIOR_LABELS.get(behavior, behavior)}\nfrom Baseline (factor=0)', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, change_from_baseline):
        height = bar.get_height()
        ax.annotate(f'{val:+.1f}pp',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9)
    
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

