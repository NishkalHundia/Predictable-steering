"""
Plot comparison of MCQA-trained vs open-ended-trained steering vectors,
both evaluated on open-ended generation with LLM judge (0-10 scale).

For each behavior, generates TWO plots:
  1. Score vs Steering Factor (raw multiplier)
  2. Score vs Effective Strength (norm * factor) — fair comparison across methods

Expected directory structure:
    MCQA results:
        results/mcqa/gemma-2-9b-it/<behavior>/config.json
        results/mcqa/gemma-2-9b-it/<behavior>/open_ended_eval/summary.csv

    Open-ended results (note the `-open-ended` suffix on behavior dirs):
        results/open_ended/gemma-2-9b-it/<behavior>-open-ended/config.json
        results/open_ended/gemma-2-9b-it/<behavior>-open-ended/eval/summary.csv

Usage:
    uv run python axbench/scripts/plot_mcqa_vs_open_ended.py \
        --mcqa_dir results/mcqa/gemma-2-9b-it \
        --open_ended_dir results/open_ended/gemma-2-9b-it \
        --output_dir results/plots

    # Specific behaviors only:
    uv run python axbench/scripts/plot_mcqa_vs_open_ended.py \
        --mcqa_dir results/mcqa/gemma-2-9b-it \
        --open_ended_dir results/open_ended/gemma-2-9b-it \
        --output_dir results/plots \
        --behaviors "sycophancy,refusal"
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

BEHAVIORS = [
    "sycophancy",
    "survival-instinct",
    "corrigible-neutral-HHH",
    "hallucination",
    "refusal",
    "myopic-reward",
    "coordinate-other-ais",
]

BEHAVIOR_LABELS = {
    "sycophancy": "Sycophancy",
    "survival-instinct": "Survival Instinct",
    "corrigible-neutral-HHH": "Corrigibility",
    "hallucination": "Hallucination",
    "refusal": "Refusal",
    "myopic-reward": "Myopic Reward",
    "coordinate-other-ais": "Coordinate Other AIs",
}


def _behavior_dir_candidates(base_dir: Path, behavior: str) -> list[Path]:
    """Return possible behavior directory paths (handles `-open-ended` suffix convention)."""
    return [
        base_dir / behavior,
        base_dir / f"{behavior}-open-ended",
    ]


def find_summary(base_dir: Path, behavior: str) -> Path | None:
    """Find summary.csv for a behavior, searching common directory and eval subdirectory names."""
    for bdir in _behavior_dir_candidates(base_dir, behavior):
        candidates = [
            bdir / "open_ended_eval" / "summary.csv",
            bdir / "eval" / "summary.csv",
            bdir / "summary.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
    return None


def find_norm(base_dir: Path, behavior: str) -> float | None:
    """Load steering vector norm from config.json."""
    for bdir in _behavior_dir_candidates(base_dir, behavior):
        config_path = bdir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            norm = config.get("steering_vector_norm")
            if norm is not None:
                return norm
    return None


def load_summaries(base_dir: Path, factor_range=(-10, 10)):
    """Load all behavior summaries and norms from a results directory."""
    summaries = {}
    norms = {}
    for behavior in BEHAVIORS:
        summary_path = find_summary(base_dir, behavior)
        if summary_path is None:
            continue
        df = pd.read_csv(summary_path)
        df = df[(df["steering_factor"] >= factor_range[0]) &
                (df["steering_factor"] <= factor_range[1])]
        df = df.sort_values("steering_factor")
        if len(df) > 0:
            summaries[behavior] = df
            norm = find_norm(base_dir, behavior)
            if norm is not None:
                norms[behavior] = norm
    return summaries, norms


def plot_single_behavior(behavior, mcqa_df, oe_df, mcqa_norm, oe_norm, output_dir, factor_range):
    """Save two plots for one behavior: vs factor and vs effective strength."""
    label = BEHAVIOR_LABELS.get(behavior, behavior)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- Left: Score vs Steering Factor ----
    ax1 = axes[0]

    if mcqa_df is not None:
        norm_str = f" (‖v‖={mcqa_norm:.1f})" if mcqa_norm else ""
        ax1.errorbar(
            mcqa_df["steering_factor"], mcqa_df["avg_score"],
            yerr=mcqa_df.get("std_score", None),
            marker="o", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#E94F37", label=f"MCQA-trained{norm_str}"
        )

    if oe_df is not None:
        norm_str = f" (‖v‖={oe_norm:.1f})" if oe_norm else ""
        ax1.errorbar(
            oe_df["steering_factor"], oe_df["avg_score"],
            yerr=oe_df.get("std_score", None),
            marker="s", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#2E86AB", label=f"Open-ended-trained{norm_str}"
        )

    ax1.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="Neutral (5)")
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Steering Factor", fontsize=12)
    ax1.set_ylabel("Behavior Score (0-10)", fontsize=12)
    ax1.set_title(f"{label}: Score vs Steering Factor", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.5, 10.5)
    ax1.set_xlim(factor_range[0] - 0.5, factor_range[1] + 0.5)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)

    # ---- Right: Score vs Effective Strength (norm * factor) ----
    ax2 = axes[1]
    has_effective = False

    if mcqa_df is not None and mcqa_norm:
        eff = mcqa_df["steering_factor"] * mcqa_norm
        ax2.errorbar(
            eff, mcqa_df["avg_score"],
            yerr=mcqa_df.get("std_score", None),
            marker="o", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#E94F37", label=f"MCQA-trained (‖v‖={mcqa_norm:.1f})"
        )
        has_effective = True

    if oe_df is not None and oe_norm:
        eff = oe_df["steering_factor"] * oe_norm
        ax2.errorbar(
            eff, oe_df["avg_score"],
            yerr=oe_df.get("std_score", None),
            marker="s", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#2E86AB", label=f"Open-ended-trained (‖v‖={oe_norm:.1f})"
        )
        has_effective = True

    ax2.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="Neutral (5)")
    ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Effective Strength (‖v‖ × factor)", fontsize=12)
    ax2.set_ylabel("Behavior Score (0-10)", fontsize=12)
    ax2.set_title(f"{label}: Score vs Effective Strength", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.5, 10.5)
    if has_effective:
        ax2.legend(fontsize=9, loc="best")
    else:
        ax2.text(0.5, 0.5, "No norm data in config.json", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = output_dir / f"{behavior}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_all_behaviors_individually(mcqa_summaries, oe_summaries, mcqa_norms, oe_norms,
                                    output_dir, factor_range):
    """Save one plot per behavior."""
    all_behaviors = sorted(set(list(mcqa_summaries.keys()) + list(oe_summaries.keys())))
    if not all_behaviors:
        print("No results to plot.")
        return

    for behavior in all_behaviors:
        plot_single_behavior(
            behavior,
            mcqa_summaries.get(behavior),
            oe_summaries.get(behavior),
            mcqa_norms.get(behavior),
            oe_norms.get(behavior),
            output_dir,
            factor_range,
        )


def plot_overlay_all_behaviors(mcqa_summaries, oe_summaries, output_dir, factor_range):
    """Two subplots: all MCQA behaviors overlaid, all open-ended behaviors overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    datasets = [
        ("MCQA-trained", mcqa_summaries, axes[0]),
        ("Open-ended-trained", oe_summaries, axes[1]),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(BEHAVIORS)))

    for title, summaries, ax in datasets:
        for i, behavior in enumerate(BEHAVIORS):
            if behavior not in summaries:
                continue
            df = summaries[behavior]
            blabel = BEHAVIOR_LABELS.get(behavior, behavior)
            ax.plot(
                df["steering_factor"], df["avg_score"],
                "o-", color=colors[i], label=blabel, linewidth=2, markersize=6, alpha=0.85
            )

        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Factor", fontsize=12)
        ax.set_ylabel("Behavior Score (0-10)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(-0.5, 10.5)
        ax.set_xlim(factor_range[0] - 0.5, factor_range[1] + 0.5)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("All Behaviors: Steering Factor vs Behavior Score", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / "all_behaviors_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_delta_from_baseline(mcqa_summaries, oe_summaries, output_dir):
    """Bar chart: Δ score (max factor - baseline) per behavior, MCQA vs open-ended."""
    behaviors = sorted(set(list(mcqa_summaries.keys()) + list(oe_summaries.keys())))
    if not behaviors:
        return

    mcqa_deltas = []
    oe_deltas = []
    labels = []

    for behavior in behaviors:
        labels.append(BEHAVIOR_LABELS.get(behavior, behavior))

        for summaries, deltas in [(mcqa_summaries, mcqa_deltas), (oe_summaries, oe_deltas)]:
            if behavior in summaries:
                df = summaries[behavior]
                baseline_rows = df[df["steering_factor"] == 0]
                baseline = baseline_rows["avg_score"].values[0] if len(baseline_rows) > 0 else df["avg_score"].median()
                max_score = df["avg_score"].max()
                deltas.append(max_score - baseline)
            else:
                deltas.append(0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))

    if any(d != 0 for d in mcqa_deltas):
        ax.bar(x - width / 2, mcqa_deltas, width, label="MCQA-trained", color="#E94F37", alpha=0.85)
    if any(d != 0 for d in oe_deltas):
        ax.bar(x + width / 2, oe_deltas, width, label="Open-ended-trained", color="#2E86AB", alpha=0.85)

    ax.set_xlabel("Behavior", fontsize=12)
    ax.set_ylabel("Δ Score (max - baseline)", fontsize=12)
    ax.set_title("Steering Effect: Δ from Baseline (factor=0)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / "delta_from_baseline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary_table(mcqa_summaries, oe_summaries, mcqa_norms, oe_norms):
    """Print a combined summary table to stdout."""
    behaviors = sorted(set(list(mcqa_summaries.keys()) + list(oe_summaries.keys())))

    print("\n" + "=" * 100)
    print(f"{'Behavior':<25} {'Method':<20} {'‖v‖':<10} {'Base(0)':<10} {'Max':<8} {'Min':<8} {'Δ':<8}")
    print("=" * 100)

    for behavior in behaviors:
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        for method, summaries, norms in [("MCQA", mcqa_summaries, mcqa_norms),
                                          ("Open-ended", oe_summaries, oe_norms)]:
            if behavior not in summaries:
                continue
            df = summaries[behavior]
            norm = norms.get(behavior)
            baseline_rows = df[df["steering_factor"] == 0]
            baseline = baseline_rows["avg_score"].values[0] if len(baseline_rows) > 0 else float("nan")
            max_s = df["avg_score"].max()
            min_s = df["avg_score"].min()
            delta = max_s - baseline if not np.isnan(baseline) else float("nan")
            norm_str = f"{norm:.2f}" if norm else "N/A"
            print(f"{label:<25} {method:<20} {norm_str:<10} {baseline:<10.2f} {max_s:<8.2f} {min_s:<8.2f} {delta:<8.2f}")
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcqa_dir", type=str, default=None,
                        help="Base dir for MCQA results (e.g., results/mcqa/gemma-2-9b-it)")
    parser.add_argument("--open_ended_dir", type=str, default=None,
                        help="Base dir for open-ended results (e.g., results/open_ended/gemma-2-9b-it)")
    parser.add_argument("--output_dir", type=str, default="results/plots",
                        help="Directory to save plots")
    parser.add_argument("--behaviors", type=str, default=None,
                        help="Comma-separated list of behaviors to plot (default: all found)")
    parser.add_argument("--factor_min", type=float, default=-10,
                        help="Min steering factor to include")
    parser.add_argument("--factor_max", type=float, default=10,
                        help="Max steering factor to include")
    args = parser.parse_args()

    if args.mcqa_dir is None and args.open_ended_dir is None:
        print("ERROR: Provide at least one of --mcqa_dir or --open_ended_dir")
        return

    factor_range = (args.factor_min, args.factor_max)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    mcqa_summaries, mcqa_norms = {}, {}
    oe_summaries, oe_norms = {}, {}

    if args.mcqa_dir:
        mcqa_summaries, mcqa_norms = load_summaries(Path(args.mcqa_dir), factor_range)
        print(f"Loaded MCQA results for {len(mcqa_summaries)} behaviors: {list(mcqa_summaries.keys())}")
        for b, n in mcqa_norms.items():
            print(f"  {b}: ‖v‖ = {n:.4f}")

    if args.open_ended_dir:
        oe_summaries, oe_norms = load_summaries(Path(args.open_ended_dir), factor_range)
        print(f"Loaded open-ended results for {len(oe_summaries)} behaviors: {list(oe_summaries.keys())}")
        for b, n in oe_norms.items():
            print(f"  {b}: ‖v‖ = {n:.4f}")

    if not mcqa_summaries and not oe_summaries:
        print("ERROR: No summary.csv files found in either directory.")
        return

    # Filter to requested behaviors
    if args.behaviors:
        requested = [b.strip() for b in args.behaviors.split(",")]
        mcqa_summaries = {k: v for k, v in mcqa_summaries.items() if k in requested}
        mcqa_norms = {k: v for k, v in mcqa_norms.items() if k in requested}
        oe_summaries = {k: v for k, v in oe_summaries.items() if k in requested}
        oe_norms = {k: v for k, v in oe_norms.items() if k in requested}
        if not mcqa_summaries and not oe_summaries:
            print(f"ERROR: No results found for requested behaviors: {requested}")
            return

    # Generate per-behavior plots (one PNG each, 2 panels: vs factor + vs effective strength)
    plot_all_behaviors_individually(mcqa_summaries, oe_summaries, mcqa_norms, oe_norms,
                                    output_dir, factor_range)

    # Generate summary plots
    plot_overlay_all_behaviors(mcqa_summaries, oe_summaries, output_dir, factor_range)
    plot_delta_from_baseline(mcqa_summaries, oe_summaries, output_dir)

    # Print table
    print_summary_table(mcqa_summaries, oe_summaries, mcqa_norms, oe_norms)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
