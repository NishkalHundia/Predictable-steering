"""
Plot CAA steering evaluation results.

Supports three modes:
1. Simple mode (summary.csv): Single score per factor
2. Axbench mode (axbench_summary.csv): 3 metrics + harmonic mean
3. AxBench auto-discover: --axbench --axbench_dir auto-discovers all concept subdirs

Usage:
    uv run python axbench/scripts/plot_caa_results.py \
        --results_dir results/gemma-2-9b-it/sycophancy-open-ended/eval \
        --behavior sycophancy \
        --output_path results/gemma-2-9b-it/sycophancy-open-ended/eval/steering_plot.png

    # For axbench-style 3-metric results:
    uv run python axbench/scripts/plot_caa_results.py \
        --results_dir results/gemma-2-9b-it/sycophancy-open-ended/eval-axbench \
        --behavior sycophancy \
        --mode axbench

    # Auto-discover all AxBench concept results:
    uv run python axbench/scripts/plot_caa_results.py \
        --axbench --axbench_dir results/gemma-2-9b-it/axbench_concepts
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use a nicer style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_axbench_results(results_dir: str, behavior: str, output_path: str = None):
    """Plot axbench-style 3-metric results."""
    
    results_dir = Path(results_dir)
    summary_path = results_dir / "axbench_summary.csv"
    
    if not summary_path.exists():
        print(f"Axbench summary not found at {summary_path}")
        return
    
    df = pd.read_csv(summary_path)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All 3 metrics + harmonic mean
    ax1 = axes[0]
    colors = ['#E94F37', '#2E86AB', '#44AF69', '#6B2D5C']
    
    ax1.plot(df['steering_factor'], df['behavior_score'], 'o-', 
             label='Behavior', color=colors[0], linewidth=2, markersize=8)
    ax1.plot(df['steering_factor'], df['instruction_score'], 's-', 
             label='Instruction', color=colors[1], linewidth=2, markersize=8)
    ax1.plot(df['steering_factor'], df['fluency_score'], '^-', 
             label='Fluency', color=colors[2], linewidth=2, markersize=8)
    ax1.plot(df['steering_factor'], df['harmonic_mean'], 'D-', 
             label='Harmonic Mean', color=colors[3], linewidth=2.5, markersize=9)
    
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Threshold (1)')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Steering Factor', fontsize=12)
    ax1.set_ylabel('Score (0-2)', fontsize=12)
    ax1.set_title(f'{behavior.replace("-", " ").title()}: Axbench Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 2.1)
    ax1.legend(loc='best')
    
    # Plot 2: Bar chart of harmonic mean with error bars
    ax2 = axes[1]
    x = np.arange(len(df))
    bars = ax2.bar(x, df['harmonic_mean'], yerr=df['harmonic_std'], 
                   capsize=5, color='#6B2D5C', alpha=0.8)
    
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Steering Factor', fontsize=12)
    ax2.set_ylabel('Harmonic Mean (0-2)', fontsize=12)
    ax2.set_title(f'{behavior.replace("-", " ").title()}: Final Score', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{f:.1f}" for f in df['steering_factor']])
    ax2.set_ylim(0, 2.1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print(f"AXBENCH SUMMARY: {behavior}")
    print("="*70)
    print(df.to_string(index=False))
    
    # Compute steering effect
    if 0 in df['steering_factor'].values:
        baseline = df[df['steering_factor'] == 0]['harmonic_mean'].values[0]
        max_score = df['harmonic_mean'].max()
        min_score = df['harmonic_mean'].min()
        
        print(f"\nBaseline (factor=0): {baseline:.3f}")
        print(f"Max score: {max_score:.3f}")
        print(f"Min score: {min_score:.3f}")
        print(f"Total range: {max_score - min_score:.3f}")


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


def discover_axbench_judge_results(axbench_dir):
    """
    Auto-discover concept subdirs that have axbench_judge_eval/axbench_summary.csv.
    Returns list of (label, concept_name, judge_eval_dir) tuples.
    """
    axbench_dir = Path(axbench_dir)
    concepts = []

    for subdir in sorted(axbench_dir.iterdir()):
        if not subdir.is_dir():
            continue
        judge_dir = subdir / "axbench_judge_eval"
        summary_path = judge_dir / "axbench_summary.csv"
        if not summary_path.exists():
            continue

        config_path = subdir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            concept_name = config.get("concept_name", subdir.name)
            concept_id = config.get("concept_id", "")
            label = f"axbench_{concept_id}" if concept_id else subdir.name
        else:
            concept_name = subdir.name
            label = subdir.name

        concepts.append((label, concept_name, judge_dir))

    return concepts


def plot_multi_axbench_harmonic_mean(all_summaries, output_dir):
    """Overlay harmonic mean vs steering factor for all concepts."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, df) in enumerate(all_summaries.items()):
        ax.plot(df['steering_factor'], df['harmonic_mean'], 'o-',
                label=label, linewidth=2, markersize=7, alpha=0.85)

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Threshold (1)')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Harmonic Mean Score (0-2)', fontsize=12)
    ax.set_title('AxBench Judge: Harmonic Mean by Concept', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 2.1)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "multi_harmonic_mean_vs_factor.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_multi_axbench_metrics(all_summaries, output_dir):
    """Grid plot: one subplot per metric, all concepts overlaid."""
    metrics = ['behavior_score', 'instruction_score', 'fluency_score', 'harmonic_mean']
    titles = ['Behavior Relevance', 'Instruction Relevance', 'Fluency', 'Harmonic Mean']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        for i, (label, df) in enumerate(all_summaries.items()):
            ax.plot(df['steering_factor'], df[metric], 'o-',
                    label=label, linewidth=1.5, markersize=5, alpha=0.85)

        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Steering Factor', fontsize=10)
        ax.set_ylabel('Score (0-2)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 2.1)
        ax.grid(True, alpha=0.3)

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(all_summaries), 4),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plot_path = output_dir / "multi_metrics_grid.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_multi_axbench_bar(all_summaries, output_dir, target_factors=None):
    """Bar chart comparing harmonic mean across concepts at select factors."""
    if target_factors is None:
        all_factors = set()
        for df in all_summaries.values():
            all_factors.update(df["steering_factor"].tolist())
        candidates = [-100, -50, -10, -2, -1, 0, 1, 2, 10, 50, 100]
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
        df = all_summaries[behavior]
        vals = []
        for f in target_factors:
            row = df[df["steering_factor"] == f]
            vals.append(row["harmonic_mean"].values[0] if len(row) > 0 else 0)

        offset = (i - n_behaviors / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=behavior, alpha=0.8)

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in target_factors])
    ax.set_xlabel('Steering Factor', fontsize=12)
    ax.set_ylabel('Harmonic Mean (0-2)', fontsize=12)
    ax.set_title('AxBench Judge: Harmonic Mean Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 2.1)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / "multi_harmonic_bar_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing summary.csv or axbench_summary.csv")
    parser.add_argument("--behavior", type=str, default=None,
                        help="Behavior name for plot title")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save plot (if not specified, shows interactively)")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "simple", "axbench"],
                        help="Evaluation mode: simple (0-10 score), axbench (3 metrics, 0-2)")
    parser.add_argument("--axbench", action="store_true",
                        help="Auto-discover all AxBench concept results in --axbench_dir")
    parser.add_argument("--axbench_dir", type=str, default=None,
                        help="Directory containing concept subdirs with axbench_judge_eval/ results")
    args = parser.parse_args()

    if args.axbench:
        # ==========================================================
        # AXBENCH AUTO-DISCOVER MODE
        # ==========================================================
        if not args.axbench_dir:
            print("ERROR: --axbench_dir is required when --axbench is set")
            return

        discovered = discover_axbench_judge_results(args.axbench_dir)
        if not discovered:
            print(f"ERROR: No concept subdirs with axbench_judge_eval/axbench_summary.csv "
                  f"found in {args.axbench_dir}")
            return

        print(f"Auto-discovered {len(discovered)} AxBench concepts in {args.axbench_dir}")

        # Load all summaries
        all_summaries = {}
        for label, concept_name, judge_dir in discovered:
            df = pd.read_csv(judge_dir / "axbench_summary.csv")
            display_label = f"{label} ({concept_name[:30]})"
            all_summaries[display_label] = df

            # Also generate per-concept single plot
            plot_axbench_results(
                str(judge_dir), display_label,
                str(judge_dir / "steering_plot.png")
            )

        # Multi-concept comparison plots
        output_dir = Path(args.output_path) if args.output_path else Path(args.axbench_dir) / "judge_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_multi_axbench_harmonic_mean(all_summaries, output_dir)
        plot_multi_axbench_metrics(all_summaries, output_dir)
        plot_multi_axbench_bar(all_summaries, output_dir)

        # Print combined table
        print("\n" + "=" * 80)
        print("AXBENCH JUDGE SUMMARY ACROSS CONCEPTS")
        print("=" * 80)
        for label, df in all_summaries.items():
            print(f"\n--- {label} ---")
            print(df[['steering_factor', 'behavior_score', 'instruction_score',
                       'fluency_score', 'harmonic_mean']].to_string(index=False))

        print(f"\nAll multi-concept plots saved to {output_dir}")

    else:
        # ==========================================================
        # SINGLE CONCEPT MODE (original behavior)
        # ==========================================================
        if not args.results_dir or not args.behavior:
            print("ERROR: --results_dir and --behavior are required "
                  "(or use --axbench --axbench_dir)")
            return

        results_dir = Path(args.results_dir)

        # Auto-detect mode
        if args.mode == "auto":
            if (results_dir / "axbench_summary.csv").exists():
                args.mode = "axbench"
            else:
                args.mode = "simple"

        if args.mode == "axbench":
            plot_axbench_results(args.results_dir, args.behavior, args.output_path)
        else:
            plot_steering_results(args.results_dir, args.behavior, args.output_path)


if __name__ == "__main__":
    main()
