"""
Plot comparison of MCQA-trained vs open-ended-trained steering vectors,
evaluated with the axbench-style LLM judge (3 metrics + harmonic mean, 0-2 scale).

For each behavior, generates a 2×2 panel:
  - Top-left:     Behavior Score vs Steering Factor
  - Top-right:    Instruction Score vs Steering Factor
  - Bottom-left:  Fluency Score vs Steering Factor
  - Bottom-right: Harmonic Mean vs Steering Factor

Also generates per-behavior 1×2 panel for harmonic mean:
  - Left:  Harmonic Mean vs Steering Factor
  - Right: Harmonic Mean vs Effective Strength (norm × factor)

Expected directory structure (axbench_summary.csv produced by eval_caa_axbench_judge.py):
    MCQA results:
        results/mcqa/gemma-2-9b-it/<behavior>/eval-axbench/axbench_summary.csv
        results/mcqa/gemma-2-9b-it/<behavior>/config.json  (has steering_vector_norm)

    Open-ended results (note the `-open-ended` suffix on behavior dirs):
        results/open_ended/gemma-2-9b-it/<behavior>-open-ended/eval-axbench/axbench_summary.csv
        results/open_ended/gemma-2-9b-it/<behavior>-open-ended/config.json

Usage:
    uv run python axbench/scripts/plot_axbench_comparison.py \
        --mcqa_dir results/mcqa/gemma-2-9b-it \
        --open_ended_dir results/open_ended/gemma-2-9b-it \
        --output_dir results/plots/axbench

    # Specific behaviors only:
    uv run python axbench/scripts/plot_axbench_comparison.py \
        --mcqa_dir results/mcqa/gemma-2-9b-it \
        --open_ended_dir results/open_ended/gemma-2-9b-it \
        --output_dir results/plots/axbench \
        --behaviors "sycophancy,survival-instinct"
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

# Axbench judge metrics (columns in axbench_summary.csv)
METRICS = [
    ("behavior_score", "Behavior Relevance"),
    ("instruction_score", "Instruction Relevance"),
    ("fluency_score", "Fluency"),
    ("harmonic_mean", "Harmonic Mean"),
]


def _behavior_dir_candidates(base_dir: Path, behavior: str) -> list[Path]:
    """Return possible behavior directory paths (handles `-open-ended` suffix convention)."""
    return [
        base_dir / behavior,
        base_dir / f"{behavior}-open-ended",
    ]


def find_axbench_summary(base_dir: Path, behavior: str) -> Path | None:
    """Find axbench_summary.csv for a behavior."""
    for bdir in _behavior_dir_candidates(base_dir, behavior):
        candidates = [
            bdir / "eval-axbench" / "axbench_summary.csv",
            bdir / "axbench_eval" / "axbench_summary.csv",
            bdir / "open_ended_eval" / "eval-axbench" / "axbench_summary.csv",
            bdir / "axbench_summary.csv",
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
    """Load all behavior axbench summaries and norms from a results directory."""
    summaries = {}
    norms = {}
    for behavior in BEHAVIORS:
        summary_path = find_axbench_summary(base_dir, behavior)
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


def plot_4panel(behavior, mcqa_df, oe_df, output_dir, factor_range):
    """2×2 panel: all 4 metrics vs steering factor, MCQA vs open-ended."""
    label = BEHAVIOR_LABELS.get(behavior, behavior)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, metric_name) in zip(axes, METRICS):
        if mcqa_df is not None and col in mcqa_df.columns:
            ax.plot(
                mcqa_df["steering_factor"], mcqa_df[col],
                "o-", color="#E94F37", linewidth=2.5, markersize=7,
                label="MCQA-trained"
            )
        if oe_df is not None and col in oe_df.columns:
            ax.plot(
                oe_df["steering_factor"], oe_df[col],
                "s-", color="#2E86AB", linewidth=2.5, markersize=7,
                label="Open-ended-trained"
            )

        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Factor", fontsize=11)
        ax.set_ylabel(f"{metric_name} (0-2)", fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.1, 2.1)
        ax.set_xlim(factor_range[0] - 0.5, factor_range[1] + 0.5)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{label} — AxBench Judge (3 metrics + harmonic)",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()

    out_path = output_dir / f"{behavior}_4panel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_harmonic_with_effective(behavior, mcqa_df, oe_df, mcqa_norm, oe_norm,
                                 output_dir, factor_range):
    """1×2 panel: harmonic mean vs factor (left) and vs effective strength (right)."""
    label = BEHAVIOR_LABELS.get(behavior, behavior)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- Left: harmonic mean vs steering factor ----
    ax1 = axes[0]
    if mcqa_df is not None:
        norm_str = f" (‖v‖={mcqa_norm:.1f})" if mcqa_norm else ""
        std_col = "harmonic_std" if "harmonic_std" in mcqa_df.columns else None
        ax1.errorbar(
            mcqa_df["steering_factor"], mcqa_df["harmonic_mean"],
            yerr=mcqa_df[std_col] if std_col else None,
            marker="o", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#E94F37", label=f"MCQA-trained{norm_str}"
        )
    if oe_df is not None:
        norm_str = f" (‖v‖={oe_norm:.1f})" if oe_norm else ""
        std_col = "harmonic_std" if "harmonic_std" in oe_df.columns else None
        ax1.errorbar(
            oe_df["steering_factor"], oe_df["harmonic_mean"],
            yerr=oe_df[std_col] if std_col else None,
            marker="s", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#2E86AB", label=f"Open-ended-trained{norm_str}"
        )

    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Neutral (1)")
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Steering Factor", fontsize=12)
    ax1.set_ylabel("Harmonic Mean (0-2)", fontsize=12)
    ax1.set_title(f"{label}: Harmonic Mean vs Factor", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.1, 2.1)
    ax1.set_xlim(factor_range[0] - 0.5, factor_range[1] + 0.5)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)

    # ---- Right: harmonic mean vs effective strength ----
    ax2 = axes[1]
    has_effective = False

    if mcqa_df is not None and mcqa_norm:
        eff = mcqa_df["steering_factor"] * mcqa_norm
        std_col = "harmonic_std" if "harmonic_std" in mcqa_df.columns else None
        ax2.errorbar(
            eff, mcqa_df["harmonic_mean"],
            yerr=mcqa_df[std_col] if std_col else None,
            marker="o", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#E94F37", label=f"MCQA-trained (‖v‖={mcqa_norm:.1f})"
        )
        has_effective = True

    if oe_df is not None and oe_norm:
        eff = oe_df["steering_factor"] * oe_norm
        std_col = "harmonic_std" if "harmonic_std" in oe_df.columns else None
        ax2.errorbar(
            eff, oe_df["harmonic_mean"],
            yerr=oe_df[std_col] if std_col else None,
            marker="s", markersize=8, linewidth=2.5, capsize=4, capthick=1.5,
            color="#2E86AB", label=f"Open-ended-trained (‖v‖={oe_norm:.1f})"
        )
        has_effective = True

    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Neutral (1)")
    ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Effective Strength (‖v‖ × factor)", fontsize=12)
    ax2.set_ylabel("Harmonic Mean (0-2)", fontsize=12)
    ax2.set_title(f"{label}: Harmonic Mean vs Effective Strength", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.1, 2.1)
    if has_effective:
        ax2.legend(fontsize=9, loc="best")
    else:
        ax2.text(0.5, 0.5, "No norm data in config.json", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{behavior}_harmonic.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_overlay_harmonic(mcqa_summaries, oe_summaries, output_dir, factor_range):
    """All behaviors overlaid on one plot, harmonic mean only."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(BEHAVIORS)))

    for title, summaries, ax in [
        ("MCQA-trained", mcqa_summaries, axes[0]),
        ("Open-ended-trained", oe_summaries, axes[1]),
    ]:
        for i, behavior in enumerate(BEHAVIORS):
            if behavior not in summaries:
                continue
            df = summaries[behavior]
            blabel = BEHAVIOR_LABELS.get(behavior, behavior)
            ax.plot(
                df["steering_factor"], df["harmonic_mean"],
                "o-", color=colors[i], label=blabel, linewidth=2, markersize=6, alpha=0.85
            )

        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Factor", fontsize=12)
        ax.set_ylabel("Harmonic Mean (0-2)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(-0.1, 2.1)
        ax.set_xlim(factor_range[0] - 0.5, factor_range[1] + 0.5)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("AxBench Judge: Harmonic Mean (all behaviors)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / "all_behaviors_harmonic_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary_table(mcqa_summaries, oe_summaries, mcqa_norms, oe_norms):
    """Print combined summary table to stdout."""
    behaviors = sorted(set(list(mcqa_summaries.keys()) + list(oe_summaries.keys())))

    print("\n" + "=" * 120)
    print(f"{'Behavior':<25} {'Method':<18} {'‖v‖':<10} "
          f"{'Behav(0)':<10} {'Instr(0)':<10} {'Flu(0)':<10} {'H(0)':<10} "
          f"{'H-max':<8} {'H-min':<8}")
    print("=" * 120)

    for behavior in behaviors:
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        for method, summaries, norms in [("MCQA", mcqa_summaries, mcqa_norms),
                                          ("Open-ended", oe_summaries, oe_norms)]:
            if behavior not in summaries:
                continue
            df = summaries[behavior]
            norm = norms.get(behavior)
            baseline = df[df["steering_factor"] == 0]
            if len(baseline) > 0:
                b0 = baseline["behavior_score"].values[0]
                i0 = baseline["instruction_score"].values[0]
                f0 = baseline["fluency_score"].values[0]
                h0 = baseline["harmonic_mean"].values[0]
            else:
                b0 = i0 = f0 = h0 = float("nan")

            h_max = df["harmonic_mean"].max()
            h_min = df["harmonic_mean"].min()
            norm_str = f"{norm:.2f}" if norm else "N/A"
            print(f"{label:<25} {method:<18} {norm_str:<10} "
                  f"{b0:<10.3f} {i0:<10.3f} {f0:<10.3f} {h0:<10.3f} "
                  f"{h_max:<8.3f} {h_min:<8.3f}")
        print("-" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Plot axbench judge comparison: MCQA-trained vs open-ended-trained"
    )
    parser.add_argument("--mcqa_dir", type=str, default=None,
                        help="Base dir for MCQA results (e.g., results/mcqa/gemma-2-9b-it)")
    parser.add_argument("--open_ended_dir", type=str, default=None,
                        help="Base dir for open-ended results (e.g., results/open_ended/gemma-2-9b-it)")
    parser.add_argument("--output_dir", type=str, default="results/plots/axbench",
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
        print(f"Loaded MCQA axbench results for {len(mcqa_summaries)} behaviors: {list(mcqa_summaries.keys())}")
        for b, n in mcqa_norms.items():
            print(f"  {b}: ‖v‖ = {n:.4f}")

    if args.open_ended_dir:
        oe_summaries, oe_norms = load_summaries(Path(args.open_ended_dir), factor_range)
        print(f"Loaded open-ended axbench results for {len(oe_summaries)} behaviors: {list(oe_summaries.keys())}")
        for b, n in oe_norms.items():
            print(f"  {b}: ‖v‖ = {n:.4f}")

    if not mcqa_summaries and not oe_summaries:
        print("ERROR: No axbench_summary.csv files found in either directory.")
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

    # Generate per-behavior plots
    all_behaviors = sorted(set(list(mcqa_summaries.keys()) + list(oe_summaries.keys())))
    for behavior in all_behaviors:
        # 2×2 panel with all 4 metrics
        plot_4panel(
            behavior,
            mcqa_summaries.get(behavior),
            oe_summaries.get(behavior),
            output_dir, factor_range,
        )
        # 1×2 panel: harmonic vs factor + harmonic vs effective strength
        plot_harmonic_with_effective(
            behavior,
            mcqa_summaries.get(behavior),
            oe_summaries.get(behavior),
            mcqa_norms.get(behavior),
            oe_norms.get(behavior),
            output_dir, factor_range,
        )

    # Summary overlay
    plot_overlay_harmonic(mcqa_summaries, oe_summaries, output_dir, factor_range)

    # Print table
    print_summary_table(mcqa_summaries, oe_summaries, mcqa_norms, oe_norms)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
