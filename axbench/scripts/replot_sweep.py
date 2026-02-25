"""
Regenerate sweep plots from an existing sweep_summary.json.

Usage:
    uv run python axbench/scripts/replot_sweep.py \
        --sweep_dir results/gemma-2-9b-it/corrigible-neutral-HHH-sweep \
        --behavior corrigible-neutral-HHH
"""
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def recompute_sweep_summary(sweep_dir: Path, target_layers: list[int]):
    """Rebuild sweep_summary from per-layer artifacts, fixing max/min score logic."""
    entries = []
    for l in target_layers:
        layer_dir = sweep_dir / f"layer_{l}"
        if not layer_dir.exists():
            continue

        sep_path = layer_dir / "separability.json"
        sep = json.loads(sep_path.read_text()) if sep_path.exists() else {}

        entry = {
            "layer": l,
            "dprime": sep.get("dprime"),
            "auroc": sep.get("auroc"),
            "norm": sep.get("norm"),
        }

        summary_path = layer_dir / "eval" / "summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            entry["eval_scores"] = df.to_dict(orient="records")

            f0 = df[df["steering_factor"] == 0]
            if not f0.empty:
                entry["factor_0_avg"] = float(f0["avg_score"].values[0])

            entry["factor_max_avg"] = float(df["avg_score"].max())
            entry["factor_min_avg"] = float(df["avg_score"].min())
            entry["best_positive_factor"] = float(
                df.loc[df["avg_score"].idxmax(), "steering_factor"]
            )
            entry["best_negative_factor"] = float(
                df.loc[df["avg_score"].idxmin(), "steering_factor"]
            )

        entries.append(entry)
    return entries


def plot_sweep_summary(sweep_summary, behavior, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [s["layer"] for s in sweep_summary]
    dprimes = [s["dprime"] for s in sweep_summary]
    aurocs = [s["auroc"] for s in sweep_summary]
    norms = [s["norm"] for s in sweep_summary]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, dprimes, "o-", linewidth=2, markersize=7, color="#E94F37")
    ax.set_xlabel("Layer")
    ax.set_ylabel("d' (discriminability)")
    ax.set_title(f"{behavior}: d' by Layer")
    ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(output_dir / "dprime_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, aurocs, "s-", linewidth=2, markersize=7, color="#2E86AB")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{behavior}: AUROC by Layer")
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(layers)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "auroc_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, norms, "^-", linewidth=2, markersize=7, color="#44AF69")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering Vector Norm")
    ax.set_title(f"{behavior}: Steering Vector Norm by Layer")
    ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(output_dir / "norm_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()

    has_eval = any("eval_scores" in s for s in sweep_summary)
    if has_eval:
        fig, ax = plt.subplots(figsize=(8, 4))
        for factor_label in ["factor_0_avg", "factor_max_avg", "factor_min_avg"]:
            vals = [s.get(factor_label) for s in sweep_summary]
            if all(v is not None for v in vals):
                style = {
                    "factor_0_avg": ("Baseline (factor=0)", "gray", "--"),
                    "factor_max_avg": ("Best score (max across factors)", "#E94F37", "-"),
                    "factor_min_avg": ("Worst score (min across factors)", "#2E86AB", "-"),
                }
                label, color, ls = style[factor_label]
                ax.plot(layers, vals, "o-", label=label, color=color, linestyle=ls,
                        linewidth=2, markersize=6)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Avg Behavior Score (0-10)")
        ax.set_title(f"{behavior}: Steering Eval by Layer")
        ax.set_xticks(layers)
        ax.set_ylim(0, 10)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "score_by_layer.png", dpi=150, bbox_inches="tight")
        plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(layers, dprimes, "o-", color="#E94F37", linewidth=2, markersize=6)
    axes[0].set_title("d'")
    axes[0].set_xlabel("Layer")
    axes[0].set_xticks(layers)
    axes[1].plot(layers, aurocs, "s-", color="#2E86AB", linewidth=2, markersize=6)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_title("AUROC")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylim(0.4, 1.05)
    axes[1].set_xticks(layers)
    axes[2].plot(layers, norms, "^-", color="#44AF69", linewidth=2, markersize=6)
    axes[2].set_title("Steering Norm")
    axes[2].set_xlabel("Layer")
    axes[2].set_xticks(layers)
    fig.suptitle(f"{behavior}: Layer Sweep Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "combined_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate sweep plots from existing results")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Path to sweep output dir (contains layer_N/ subdirs)")
    parser.add_argument("--behavior", type=str, required=True,
                        help="Behavior name (for plot titles)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layer range override, e.g. '10-32'. Auto-detected if omitted.")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if args.layers:
        if "-" in args.layers and "," not in args.layers:
            lo, hi = args.layers.split("-")
            target_layers = list(range(int(lo), int(hi) + 1))
        else:
            target_layers = sorted(int(x) for x in args.layers.split(","))
    else:
        target_layers = sorted(
            int(d.name.split("_")[1])
            for d in sweep_dir.iterdir()
            if d.is_dir() and d.name.startswith("layer_")
        )

    print(f"Recomputing summary for layers: {target_layers}")
    sweep_summary = recompute_sweep_summary(sweep_dir, target_layers)

    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"Updated {sweep_dir / 'sweep_summary.json'}")

    plot_sweep_summary(sweep_summary, args.behavior, sweep_dir / "sweep_plots")


if __name__ == "__main__":
    main()
