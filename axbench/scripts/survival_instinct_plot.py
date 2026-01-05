"""
Create accuracy plots for survival-instinct CAA evaluation results.
"""
import os
import sys
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/survival-instinct")
    parser.add_argument("--output_dir", type=str, default="results/survival-instinct")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load accuracy summary
    summary_path = results_dir / "accuracy_summary.csv"
    if not summary_path.exists():
        logger.error(f"Could not find {summary_path}. Run evaluation first!")
        sys.exit(1)
    
    summary_df = pd.read_csv(summary_path)
    logger.warning(f"Loaded accuracy summary from {summary_path}")
    
    # Also load all generations for additional analysis
    parquet_path = results_dir / "all_generations.parquet"
    if parquet_path.exists():
        all_results = pd.read_parquet(parquet_path)
        logger.warning(f"Loaded {len(all_results)} generation results")
    else:
        all_results = None
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ========================================
    # Plot 1: Accuracy vs Steering Factor
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = summary_df["steering_factor"].values
    acc_with_parens = summary_df["accuracy_with_parens"].values
    acc_without_parens = summary_df["accuracy_without_parens"].values
    
    ax.plot(factors, acc_with_parens, 'o-', linewidth=2, markersize=8, 
            label='Accuracy (with parentheses)', color='#2563eb')
    ax.plot(factors, acc_without_parens, 's--', linewidth=2, markersize=8,
            label='Accuracy (without parentheses)', color='#dc2626')
    
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.7, label='Random baseline (50%)')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Survival Instinct CAA: Accuracy vs Steering Factor\n(Gemma-2-2B-IT, Layer 10)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(factors.min() - 0.2, factors.max() + 0.2)
    ax.set_ylim(0, 100)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i, (f, acc) in enumerate(zip(factors, acc_with_parens)):
        if f in [-2, 0, 2]:
            ax.annotate(f'{acc:.1f}%', (f, acc), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = output_dir / "accuracy_vs_steering_factor.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.warning(f"Saved plot to {plot_path}")
    plt.close()
    
    # ========================================
    # Plot 2: Accuracy difference from baseline
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get baseline (factor=0) accuracy
    baseline_idx = np.where(factors == 0)[0]
    if len(baseline_idx) > 0:
        baseline_acc = acc_with_parens[baseline_idx[0]]
    else:
        baseline_acc = acc_with_parens[len(factors)//2]
    
    acc_diff = acc_with_parens - baseline_acc
    
    colors = ['#22c55e' if d >= 0 else '#ef4444' for d in acc_diff]
    bars = ax.bar(factors, acc_diff, color=colors, width=0.3, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Accuracy Change from Baseline (%)', fontsize=12)
    ax.set_title(f'Accuracy Change vs Steering Factor\n(Baseline at factor=0: {baseline_acc:.1f}%)', fontsize=14)
    ax.set_xlim(factors.min() - 0.5, factors.max() + 0.5)
    
    # Add value labels on bars
    for bar, diff in zip(bars, acc_diff):
        height = bar.get_height()
        ax.annotate(f'{diff:+.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9)
    
    plt.tight_layout()
    
    plot_path = output_dir / "accuracy_change_from_baseline.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.warning(f"Saved plot to {plot_path}")
    plt.close()
    
    # ========================================
    # Plot 3: Both accuracy types side by side
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(factors))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, acc_with_parens, width, label='With Parentheses', color='#3b82f6')
    bars2 = ax.bar(x + width/2, acc_without_parens, width, label='Without Parentheses', color='#f97316')
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random (50%)')
    
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Survival Instinct CAA: Accuracy Comparison\n(Gemma-2-2B-IT, Layer 10)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.1f}' for f in factors])
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 100)
    
    # Add value labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    plot_path = output_dir / "accuracy_comparison_bar.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.warning(f"Saved plot to {plot_path}")
    plt.close()
    
    # ========================================
    # Plot 4: Heatmap style visualization
    # ========================================
    if all_results is not None:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create pivot table for heatmap
        steering_factors = sorted(all_results["steering_factor"].unique())
        pivot = all_results.pivot_table(
            values='correct_with_parens',
            index='question_idx',
            columns='steering_factor',
            aggfunc='first'
        )
        
        # Plot heatmap
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(steering_factors)))
        ax.set_xticklabels([f'{f:.1f}' for f in steering_factors])
        ax.set_xlabel('Steering Factor', fontsize=12)
        ax.set_ylabel('Question Index', fontsize=12)
        ax.set_title('Per-Question Correctness Across Steering Factors\n(Green=Correct, Red=Incorrect)', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Correct')
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Incorrect', '', 'Correct'])
        
        plt.tight_layout()
        
        plot_path = output_dir / "per_question_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.warning(f"Saved plot to {plot_path}")
        plt.close()
    
    # ========================================
    # Print Summary Statistics
    # ========================================
    logger.warning("\n" + "="*60)
    logger.warning("SUMMARY STATISTICS")
    logger.warning("="*60)
    
    print("\nAccuracy Summary Table:")
    print(summary_df.to_string(index=False))
    
    print(f"\n\nKey Findings:")
    print(f"  - Best accuracy (with parens): {acc_with_parens.max():.1f}% at factor={factors[np.argmax(acc_with_parens)]:.1f}")
    print(f"  - Worst accuracy (with parens): {acc_with_parens.min():.1f}% at factor={factors[np.argmin(acc_with_parens)]:.1f}")
    print(f"  - Baseline (factor=0) accuracy: {baseline_acc:.1f}%")
    print(f"  - Accuracy range: {acc_with_parens.max() - acc_with_parens.min():.1f}%")
    
    # Effect of steering direction
    pos_mask = factors > 0
    neg_mask = factors < 0
    
    if pos_mask.any():
        avg_positive = acc_with_parens[pos_mask].mean()
        print(f"  - Average accuracy (positive steering): {avg_positive:.1f}%")
    if neg_mask.any():
        avg_negative = acc_with_parens[neg_mask].mean()
        print(f"  - Average accuracy (negative steering): {avg_negative:.1f}%")
    
    logger.warning("\nPlotting complete!")


if __name__ == "__main__":
    main()

