"""
Render paper-ready figures from CSVs already produced by mcqa_projection_link.py.

For each behavior, this rebuilds three figures with the exact styling used in
the paper (large axis labels, consistent layout, no titles/legends):

    paper_plots/<behavior>/<short>_steer_new.png        (Figs 1, 4)
    paper_plots/<behavior>/<short>_mcc_new.png          (Figs 3, 7)
    paper_plots/<behavior>/projection_hist_postgen_layer_<L>.png (Figs 2, 5, 6)

Inputs (read from results dir, default `results/mcqa_projection_link/<model>/<behavior>/`):
    per_layer_summary.csv
    per_prompt_results.csv
    train_projections.json
    summary.json   (used to recover --factors)

Usage:
    python scripts/make_paper_plots.py --all
    python scripts/make_paper_plots.py --behavior myopic-reward --target-layer 21
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Reuse the histogram plotter from the main script unmodified, then patch its
# typography for the paper version.
import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import mcqa_projection_link as m  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")

DEFAULT_MODEL_SHORT = "gemma-2-9b-it"
BEHAVIORS = [
    "sycophancy",
    "survival-instinct",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
]
BEHAVIOR_SHORT = {
    "sycophancy": "sycophancy",
    "survival-instinct": "survival",
    "corrigible-neutral-HHH": "corrigible",
    "hallucination": "hallucination",
    "myopic-reward": "myopic",
}
DEFAULT_LAYER = 21
BEHAVIOR_DEFAULT_LAYER = {
    "hallucination": 19,
}


# ---------------------------------------------------------------------------
# Paper-style plotters
# ---------------------------------------------------------------------------
def plot_steering_and_dprime_paper(layer_df, factors, out_path):
    """Per-layer accuracy (one line per α) + d' on twin axis. Big tick labels,
    no title, no legend (legend is added in the paper layout)."""
    layers = layer_df["layer"].values
    even_layers = [int(l) for l in layers if int(l) % 2 == 0]

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(layers, layer_df["baseline_acc"].values, "D--", color="gray",
             linewidth=1.5, markersize=6, label="Baseline (α=0)", alpha=0.85, zorder=3)
    cmap = plt.get_cmap("plasma")
    factor_cols = [f for f in factors if f"steered_acc_{f:g}" in layer_df.columns]
    for i, f in enumerate(factor_cols):
        color = cmap(i / max(1, len(factor_cols) - 1))
        ax1.plot(layers, layer_df[f"steered_acc_{f:g}"].values, "o-", color=color,
                 linewidth=2, markersize=5, label=f"α={f:g}", zorder=3)
    ax1.set_xlabel("Layer", fontsize=16)
    ax1.set_ylabel("Greedy accuracy", fontsize=16)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(even_layers)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)

    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d'", zorder=2)
        ax2.set_ylabel("d' (train)", fontsize=16, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=14)
        ax2.set_ylim(bottom=0)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mcc_vs_dprime_paper(layer_df, out_path, mcc_col="sign_kappa_mcc_val_best_on_test"):
    """Per-layer MCC vs d'. Defaults to val-chosen-α MCC evaluated on test —
    this is the column used in Figures 3 & 7 of the paper."""
    if mcc_col not in layer_df.columns:
        return
    mcc = layer_df[mcc_col].values.astype(float)
    if not np.any(np.isfinite(mcc)):
        return

    layers = layer_df["layer"].values
    even_layers = [int(l) for l in layers if int(l) % 2 == 0]

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(layers, mcc, "o-", color="#C73E1D", linewidth=2, markersize=6,
             label="MCC(sign κ vs steer match | val-chosen α, test prompts)", zorder=3)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.set_xlabel("Layer", fontsize=16)
    ax1.set_ylabel("MCC", fontsize=16)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xticks(even_layers)
    ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelcolor="#C73E1D", labelsize=14)

    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d' (train)", zorder=2)
        ax2.set_ylabel("d' (train)", fontsize=16, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=14)
        ax2.set_ylim(bottom=0)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_histograms_paper(prompt_df, factors, behavior, layer,
                                     train_projections, out_path):
    """Reuse the main script's plot_projection_histograms (postgen=True), but
    monkey-patch typography for the paper. The patches are scoped to this call."""
    TICK_FS = 14
    LABEL_FS = TICK_FS + 1
    TITLE_FS = TICK_FS + 1

    original_suptitle   = Figure.suptitle
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel
    original_set_title  = Axes.set_title
    original_legend     = Axes.legend
    original_savefig    = Figure.savefig

    def _xlabel_override(self, *a, **kw):
        kw["fontsize"] = LABEL_FS
        return original_set_xlabel(self, *a, **kw)

    def _title_override(self, *a, **kw):
        kw["fontsize"] = TITLE_FS
        kw["fontweight"] = "bold"
        return original_set_title(self, *a, **kw)

    def _savefig_override(self, *a, **kw):
        self.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.10,
                             wspace=0.30, hspace=0.40)
        for i, ax in enumerate(self.axes):
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            if i == 0:
                original_set_ylabel(ax, "# prompts", fontsize=LABEL_FS)
            else:
                original_set_ylabel(ax, "")
        kw.pop("bbox_inches", None)
        kw.pop("pad_inches", None)
        return original_savefig(self, *a, **kw)

    Figure.suptitle  = lambda self, *a, **kw: None
    Axes.set_xlabel  = _xlabel_override
    Axes.set_title   = _title_override
    Axes.legend      = lambda self, *a, **kw: None
    Figure.savefig   = _savefig_override
    try:
        m.plot_projection_histograms(
            prompt_df, factors, behavior, layer,
            train_projections, out_path, postgen=True,
        )
    finally:
        Figure.suptitle  = original_suptitle
        Axes.set_xlabel  = original_set_xlabel
        Axes.set_title   = original_set_title
        Axes.legend      = original_legend
        Figure.savefig   = original_savefig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _load_factors(out_dir: Path, layer_df: pd.DataFrame) -> list:
    sp = out_dir / "summary.json"
    if sp.exists():
        try:
            blob = json.loads(sp.read_text())
            f_list = blob.get("factors")
            if f_list:
                return sorted(float(x) for x in f_list)
        except Exception:
            pass
    factors = set()
    for c in layer_df.columns:
        if c.startswith("steered_acc_"):
            tail = c.replace("steered_acc_", "")
            try:
                factors.add(float(tail))
            except ValueError:
                pass
    return sorted(factors)


def replot_one(results_root: Path, model_short: str, behavior: str,
               target_layer: int, paper_root: Path) -> None:
    src = results_root / "mcqa_projection_link" / model_short / behavior
    per_layer_csv = src / "per_layer_summary.csv"
    per_prompt_csv = src / "per_prompt_results.csv"
    if not per_layer_csv.exists() or not per_prompt_csv.exists():
        print(f"[skip] {behavior}: missing CSVs in {src}")
        return

    per_layer = pd.read_csv(per_layer_csv)
    per_prompt = pd.read_csv(per_prompt_csv)

    train_projections = {}
    tp_path = src / "train_projections.json"
    if tp_path.exists():
        raw = json.loads(tp_path.read_text())
        train_projections = {
            int(k): {"pos": v["pos"], "neg": v["neg"]} for k, v in raw.items()
        }

    factors = _load_factors(src, per_layer)

    available_layers = {int(x) for x in per_layer["layer"].values}
    if target_layer not in available_layers:
        print(f"[warn] {behavior}: layer {target_layer} not in CSV "
              f"(have {sorted(available_layers)}); skipping histogram.")
        target_layer = None

    out_dir = paper_root / behavior
    out_dir.mkdir(parents=True, exist_ok=True)
    short = BEHAVIOR_SHORT.get(behavior, behavior.split("-")[0])

    plot_steering_and_dprime_paper(
        per_layer, factors,
        out_dir / f"{short}_steer_new.png",
    )
    plot_mcc_vs_dprime_paper(
        per_layer,
        out_dir / f"{short}_mcc_new.png",
        mcc_col="sign_kappa_mcc_val_best_on_test",
    )
    if target_layer is not None:
        try:
            plot_projection_histograms_paper(
                per_prompt, factors, behavior, target_layer,
                train_projections,
                out_dir / f"projection_hist_postgen_layer_{target_layer}.png",
            )
        except (KeyError, ValueError) as e:
            print(f"[warn] {behavior}: skipping histogram ({e})")
    print(f"[ok] {behavior}: wrote → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results",
                        help="Root containing mcqa_projection_link/<model_short>/<behavior>/...")
    parser.add_argument("--model_short", default=DEFAULT_MODEL_SHORT)
    parser.add_argument("--behavior", default=None,
                        help="Single behavior; overrides --all.")
    parser.add_argument("--all", action="store_true",
                        help="Render all five paper behaviors.")
    parser.add_argument("--target_layer", type=int, default=-1,
                        help="Layer for the post-gen histogram. "
                             "-1 = use paper default per behavior (21, hallucination=19).")
    parser.add_argument("--out_dir", default="paper_plots")
    args = parser.parse_args()

    if not args.behavior and not args.all:
        parser.error("Pass --behavior <name> or --all.")

    targets = BEHAVIORS if args.all else [args.behavior]
    results_root = Path(args.results_dir)
    paper_root = Path(args.out_dir)
    paper_root.mkdir(parents=True, exist_ok=True)

    for b in targets:
        layer = (
            args.target_layer if args.target_layer >= 0
            else BEHAVIOR_DEFAULT_LAYER.get(b, DEFAULT_LAYER)
        )
        replot_one(results_root, args.model_short, b, layer, paper_root)


if __name__ == "__main__":
    main()
