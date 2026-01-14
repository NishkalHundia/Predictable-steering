"""
Plot CAA steering evaluation results.

Usage:
    uv run python axbench/scripts/plot_caa_results.py \
        --results_dir results/gemma-2-9b-it/sycophancy-open-ended/eval \
        --behavior sycophancy \
        --output_path results/gemma-2-9b-it/sycophancy-open-ended/eval/steering_plot.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use a nicer style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_steering_results(results_dir: str, behavior: str, output_path: str = None):
    """Plot steering factor vs behavior score."""
    
    results_dir = Path(results_dir)
    
    # Load summary
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        print(f"Summary not found at {summary_path}")
        return
    
    df = pd.read_csv(summary_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average score vs steering factor
    ax1 = axes[0]
    ax1.errorbar(
        df['steering_factor'], 
        df['avg_score'], 
        yerr=df['std_score'],
        marker='o', 
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color='#2E86AB'
    )
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Neutral (5)')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Steering Factor', fontsize=12)
    ax1.set_ylabel('Behavior Score (0-10)', fontsize=12)
    ax1.set_title(f'{behavior.replace("-", " ").title()}: Score vs Steering', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 10)
    ax1.legend()
    
    # Add annotations for min/max
    min_idx = df['avg_score'].idxmin()
    max_idx = df['avg_score'].idxmax()
    ax1.annotate(f"{df.loc[min_idx, 'avg_score']:.1f}", 
                 (df.loc[min_idx, 'steering_factor'], df.loc[min_idx, 'avg_score']),
                 textcoords="offset points", xytext=(0, -15), ha='center', fontsize=10)
    ax1.annotate(f"{df.loc[max_idx, 'avg_score']:.1f}", 
                 (df.loc[max_idx, 'steering_factor'], df.loc[max_idx, 'avg_score']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
    
    # Plot 2: High/Low percentage bar chart
    ax2 = axes[1]
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df['high_pct'], width, label='High (≥7)', color='#E94F37')
    bars2 = ax2.bar(x + width/2, df['low_pct'], width, label='Low (≤3)', color='#44AF69')
    
    ax2.set_xlabel('Steering Factor', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title(f'{behavior.replace("-", " ").title()}: Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{f:.1f}" for f in df['steering_factor']])
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 5:
            ax2.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 5:
            ax2.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    # Print summary table
    print("\n" + "="*60)
    print(f"SUMMARY: {behavior}")
    print("="*60)
    print(df.to_string(index=False))
    
    # Compute steering effect
    if 0 in df['steering_factor'].values:
        baseline = df[df['steering_factor'] == 0]['avg_score'].values[0]
        max_positive = df[df['steering_factor'] > 0]['avg_score'].max() if len(df[df['steering_factor'] > 0]) > 0 else baseline
        max_negative = df[df['steering_factor'] < 0]['avg_score'].min() if len(df[df['steering_factor'] < 0]) > 0 else baseline
        
        print(f"\nBaseline (factor=0): {baseline:.2f}")
        print(f"Max positive steering: {max_positive:.2f} (Δ = +{max_positive - baseline:.2f})")
        print(f"Max negative steering: {max_negative:.2f} (Δ = {max_negative - baseline:.2f})")
        print(f"Total range: {max_positive - max_negative:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing summary.csv")
    parser.add_argument("--behavior", type=str, required=True,
                        help="Behavior name for plot title")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save plot (if not specified, shows interactively)")
    args = parser.parse_args()
    
    plot_steering_results(args.results_dir, args.behavior, args.output_path)


if __name__ == "__main__":
    main()
