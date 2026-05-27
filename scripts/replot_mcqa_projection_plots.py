#!/usr/bin/env python3
"""
Replot selected MCQA projection-link figures from downloaded CSV/JSON only.

Steering + MCC figures match mcqa_projection_link.py layout (colors, line widths, markers,
dpi) with scaled axis tick/label fonts (--font-scale). Titles and legends are omitted from
the main PNGs; companion files `<stem>_legend.png` carry the legends for compositing.

Optional: hist_postgen (needs per_prompt_results.csv + train_projections.json); exports
`projection_hist_postgen_layer_<L>_legend.png` with a horizontal category legend.

Inputs under --data_dir:
  per_layer_summary.csv — required for steering_dprime / mcc_val_best
  per_prompt_results.csv + train_projections.json — required for hist_postgen

Histogram subsetting:
  --histogram-only only runs hist_postgen.
  --hist-factors 2 or --hist-factors 0,2 selects α panels (default inferred factors prepend α=0).
  --hist-skip-train omits the Train panel (train JSON not loaded).
  --layers 21 limits which layers are rendered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-whitegrid")

# Same canvas as mcqa_projection_link.py steering/MCC exports.
_PIPELINE_FIGSIZE = (13.0, 5.0)
_DEFAULT_FONT_SCALE = round(1.2 * 1.75, 3)  # 2.1 vs pipeline (prior default was 1.2)


def _scaled_fs(base: float, font_scale: float) -> float:
    """Scale a reference pt size (pipeline uses 9–12 pt for these plots)."""
    return round(base * font_scale, 2)


def _axis_labelpad(font_scale: float) -> float:
    """Extra space between axis spines and axis titles."""
    return round(10 + 2 * font_scale, 1)


def _hist_postgen_legend_proxy_handles() -> tuple[list[Patch], list[str]]:
    """Matching / Non-matching / Gibberish — same styling as _hist_overlapping & train panel."""
    return (
        [
            Patch(facecolor="#1f77b4", edgecolor="#104e8b", linewidth=0.5, alpha=0.55),
            Patch(facecolor="#d62728", edgecolor="#a01010", linewidth=0.5, alpha=0.55),
            Patch(facecolor="#aaaaaa", edgecolor="#888888", linewidth=0.5, alpha=0.55),
        ],
        ["Matching", "Non-matching", "Gibberish"],
    )


def _save_standalone_legend(
    handles,
    labels,
    out_path: Path,
    *,
    font_scale: float,
    title: str | None = None,
    ncol: int = 3,
    legend_fontsize: float | None = None,
) -> None:
    """Minimal PNG containing only a legend (for slides/compositing)."""
    fs = legend_fontsize if legend_fontsize is not None else _scaled_fs(9, font_scale)
    n = len(handles)
    if n == 0:
        return
    ncol = max(1, min(ncol, n))
    nrow = (n + ncol - 1) // ncol
    # Wide enough for a single horizontal row when ncol == n (e.g. steering).
    fig_w = max(4.0, min(34.0, ncol * 2.05 + (3.5 if title else 0)))
    fig_h = max(0.6, 0.4 + nrow * 0.45 + (0.45 if title else 0))
    fig = plt.figure(figsize=(fig_w, fig_h))
    legend_kwargs = dict(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=ncol,
        fontsize=fs,
        frameon=True,
        columnspacing=1.35 if ncol >= 4 else 1.15,
        handlelength=2,
        borderpad=0.65,
    )
    if title:
        leg = fig.legend(title=title, **legend_kwargs)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(fs)
    else:
        fig.legend(**legend_kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_steering_and_dprime(
    layer_df,
    factors,
    out_path,
    *,
    font_scale: float = _DEFAULT_FONT_SCALE,
):
    """Steering accuracy for all factors + baseline + d' on one plot with dual y-axes."""
    fs_axis = _scaled_fs(11, font_scale)
    fs_tick = _scaled_fs(10, font_scale)
    lp = _axis_labelpad(font_scale)

    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=_PIPELINE_FIGSIZE)

    # --- left axis: accuracy ---
    ax1.plot(layers, layer_df["baseline_acc"].values, "D--", color="gray",
             linewidth=1.5, markersize=6, label="Baseline (α=0)", alpha=0.85, zorder=3)
    cmap = plt.get_cmap("plasma")
    factor_cols = [f for f in factors if f"steered_acc_{f:g}" in layer_df.columns]
    for i, f in enumerate(factor_cols):
        color = cmap(i / max(1, len(factor_cols) - 1))
        ax1.plot(layers, layer_df[f"steered_acc_{f:g}"].values, "o-", color=color,
                 linewidth=2, markersize=5, label=f"α={f:g}", zorder=3)
    ax1.set_xlabel("Layer", fontsize=fs_axis, labelpad=lp)
    ax1.set_ylabel("Greedy accuracy", fontsize=fs_axis, labelpad=lp)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(layers)
    ax1.tick_params(axis="both", labelsize=fs_tick)

    # --- right axis: d' ---
    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d'", zorder=2)
        ax2.set_ylabel("d' (train)", fontsize=fs_axis, color="steelblue", labelpad=lp)
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=fs_tick)
        ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    legend_path = out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}")
    nleg = len(h1) + len(h2)
    _save_standalone_legend(
        h1 + h2,
        l1 + l2,
        legend_path,
        font_scale=font_scale,
        title="Steering factor / metric",
        ncol=nleg,
    )


def plot_mcc_vs_dprime_for_column(
    layer_df,
    out_path,
    mcc_col: str,
    mcc_label: str = "MCC(sign κ vs actual match)",
    *,
    font_scale: float = _DEFAULT_FONT_SCALE,
):
    """Dual-axis: arbitrary per-layer MCC column vs training d'."""
    if mcc_col not in layer_df.columns:
        return
    mcc = layer_df[mcc_col].values.astype(float)
    if not np.any(np.isfinite(mcc)):
        return

    fs_axis = _scaled_fs(11, font_scale)
    fs_tick = _scaled_fs(10, font_scale)
    lp = _axis_labelpad(font_scale)

    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=_PIPELINE_FIGSIZE)
    ax1.plot(layers, mcc, "o-", color="#C73E1D", linewidth=2, markersize=6,
             label=mcc_label, zorder=3)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.set_xlabel("Layer", fontsize=fs_axis, labelpad=lp)
    ax1.set_ylabel("MCC", fontsize=fs_axis, labelpad=lp)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xticks(layers)
    ax1.tick_params(axis="x", labelsize=fs_tick)
    ax1.tick_params(axis="y", labelcolor="#C73E1D", labelsize=fs_tick)

    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d' (train)", zorder=2)
        ax2.set_ylabel("d' (train)", fontsize=fs_axis, color="steelblue", labelpad=lp)
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=fs_tick)
        ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    legend_path = out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}")
    _save_standalone_legend(
        h1 + h2,
        l1 + l2,
        legend_path,
        font_scale=font_scale,
        title=None,
        ncol=2,
    )


def plot_mcc_val_best_alpha_on_test_vs_dprime(
    layer_df,
    out_path,
    *,
    font_scale: float = _DEFAULT_FONT_SCALE,
):
    """Best α from validation; MCC evaluated on held-out test prompts."""
    plot_mcc_vs_dprime_for_column(
        layer_df,
        out_path,
        mcc_col="sign_kappa_mcc_val_best_on_test",
        mcc_label="MCC(sign κ vs steer match | val-chosen α, test prompts)",
        font_scale=font_scale,
    )


# --- Optional histograms ----------------------------------------------------


def _hist_overlapping(ax, k_match, k_nonmatch, k_gibber, bins):
    """Overlapping semi-transparent histograms — all three distributions visible at once."""
    if len(k_gibber):
        ax.hist(
            k_gibber,
            bins=bins,
            color="#aaaaaa",
            alpha=0.55,
            edgecolor="#888888",
            linewidth=0.5,
            label="Gibberish",
        )
    if len(k_nonmatch):
        ax.hist(
            k_nonmatch,
            bins=bins,
            color="#d62728",
            alpha=0.55,
            edgecolor="#a01010",
            linewidth=0.5,
            label="Non-matching",
        )
    if len(k_match):
        ax.hist(
            k_match,
            bins=bins,
            color="#1f77b4",
            alpha=0.55,
            edgecolor="#104e8b",
            linewidth=0.5,
            label="Matching",
        )


def plot_projection_histograms_postgen(
    prompt_df: pd.DataFrame,
    hist_factors: list[float],
    behavior: str,
    layer: int,
    train_projections: dict,
    out_path: Path,
    *,
    include_train: bool = True,
    font_scale: float = _DEFAULT_FONT_SCALE,
) -> None:
    """Post-gen κ_a histograms — fonts use _scaled_fs like steering/MCC (bases: title 10, axes 9, ticks 10, legend 9)."""
    fs_title = _scaled_fs(10, font_scale)
    fs_axis = _scaled_fs(9, font_scale)
    fs_tick = _scaled_fs(10, font_scale)

    # Pipeline figure cell geometry: figsize (16, 7) with 2×4 subplots → 4×3.5 in per panel.
    base_col = "kappa_a_postgen"
    ldf = prompt_df[prompt_df["layer"] == layer].dropna(subset=[base_col]).copy()
    if len(ldf) < 4:
        print(f"Skip hist_postgen L{layer}: too few rows", file=sys.stderr)
        return

    if not include_train and len(hist_factors) == 0:
        print(f"Skip hist_postgen L{layer}: no panels (--hist-skip-train and empty factors)", file=sys.stderr)
        return

    for f in hist_factors:
        if f == 0:
            continue
        col_k = f"kappa_a_postgen_{f:g}"
        if col_k not in ldf.columns:
            sys.exit(
                f"Missing column {col_k} in per_prompt_results for layer {layer}. "
                "Use --hist-factors/--factors that exist in the CSV.",
            )

    n_panels = (1 if include_train else 0) + len(hist_factors)
    _cell_w, _cell_h = 4.0, 3.5
    fig_w = _cell_w * n_panels
    fig_h = _cell_h
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat: list = list(axes.flatten())

    ax_cursor = 0

    if include_train:
        train_ax = axes_flat[ax_cursor]
        ax_cursor += 1
        tp = train_projections.get(str(layer), train_projections.get(layer, {}))
        if isinstance(tp, dict):
            pos_proj = np.array(tp.get("pos", []), dtype=float)
            neg_proj = np.array(tp.get("neg", []), dtype=float)
        else:
            pos_proj = np.array([], dtype=float)
            neg_proj = np.array([], dtype=float)

        if len(pos_proj) > 0 or len(neg_proj) > 0:
            all_train = (
                np.concatenate([pos_proj, neg_proj])
                if len(pos_proj) and len(neg_proj)
                else (pos_proj if len(pos_proj) else neg_proj)
            )
            pad = max(0.3, (all_train.max() - all_train.min()) * 0.05 + 0.01)
            bins_t = np.linspace(all_train.min() - pad, all_train.max() + pad, 25)
            if len(neg_proj):
                train_ax.hist(
                    neg_proj,
                    bins=bins_t,
                    color="#d62728",
                    alpha=0.55,
                    edgecolor="#a01010",
                    linewidth=0.5,
                    label="Non-matching",
                )
            if len(pos_proj):
                train_ax.hist(
                    pos_proj,
                    bins=bins_t,
                    color="#1f77b4",
                    alpha=0.55,
                    edgecolor="#104e8b",
                    linewidth=0.5,
                    label="Matching",
                )
            x_lo, x_hi = bins_t[0], bins_t[-1]
            if x_lo <= 0 <= x_hi:
                train_ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        train_ax.set_title("Train", fontsize=fs_title, fontweight="bold")
        train_ax.set_xlabel("κ_a", fontsize=fs_axis)
        train_ax.set_ylabel("# examples", fontsize=fs_axis)
        train_ax.tick_params(axis="both", labelsize=fs_tick)

    postgen = True
    for j, f in enumerate(hist_factors):
        ax = axes_flat[ax_cursor]
        ax_cursor += 1

        if f == 0:
            kappa_vals = ldf[base_col].values
        elif postgen:
            col_k = f"kappa_a_postgen_{f:g}"
            kappa_vals = ldf[col_k].values
        else:
            kappa_vals = ldf["kappa_a"].values + 2.0 * f

        if f == 0:
            matching = ldf["baseline_correct"].astype(bool).values
            on_format = ldf["baseline_on_format"].astype(bool).values
        else:
            col_c = f"steered_correct_{f:g}"
            col_f = f"steered_on_format_{f:g}"
            if col_c not in ldf.columns:
                sys.exit(f"Missing column {col_c} for layer {layer}")
            matching = ldf[col_c].astype(bool).values
            on_format = ldf[col_f].astype(bool).values

        k_match = kappa_vals[matching]
        k_nonmatch = kappa_vals[~matching & on_format]
        k_gibber = kappa_vals[~matching & ~on_format]

        pad = max(0.3, (kappa_vals.max() - kappa_vals.min()) * 0.05 + 0.01)
        x_lo = kappa_vals.min() - pad
        x_hi = kappa_vals.max() + pad
        bins = np.linspace(x_lo, x_hi, 20)

        _hist_overlapping(ax, k_match, k_nonmatch, k_gibber, bins)

        ax.set_xlim(x_lo, x_hi)
        if x_lo <= 0 <= x_hi:
            ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        xlab = "κ_a (post-gen token)"
        ax.set_xlabel(xlab, fontsize=fs_axis)
        ax.set_title(f"α={f:g}", fontsize=fs_title, fontweight="bold")
        ax.tick_params(axis="both", labelsize=fs_tick)
        if j == 0:
            ax.set_ylabel("# prompts", fontsize=fs_axis)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    leg_h, leg_l = _hist_postgen_legend_proxy_handles()
    legend_path = out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}")
    _save_standalone_legend(
        leg_h,
        leg_l,
        legend_path,
        font_scale=font_scale,
        title=None,
        ncol=3,
    )


def load_train_projections(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def infer_factors_from_summary(layer_df: pd.DataFrame) -> list[float]:
    cols = [c for c in layer_df.columns if c.startswith("steered_acc_")]
    return sorted({float(c.replace("steered_acc_", "")) for c in cols})


def _infer_factors_from_prompt_df(df: pd.DataFrame) -> list[float]:
    cols = [
        c.replace("kappa_a_postgen_", "")
        for c in df.columns
        if c.startswith("kappa_a_postgen_") and c != "kappa_a_postgen"
    ]
    out = sorted({float(x) for x in cols})
    if not out:
        sys.exit(
            "Could not infer steering factors from per_prompt_results.csv "
            "(need kappa_a_postgen_<alpha> columns).",
        )
    return out


def parse_layers(spec: str | None, fallback_from_df: pd.Series) -> list[int]:
    if spec:
        if "-" in spec and "," not in spec:
            lo, hi = spec.split("-")
            return list(range(int(lo), int(hi) + 1))
        return sorted(int(x.strip()) for x in spec.split(","))
    return sorted(int(x) for x in fallback_from_df.unique())


def main():
    ap = argparse.ArgumentParser(description="Replot MCQA projection-link figures from CSV/JSON.")
    ap.add_argument("--data_dir", required=True, type=Path, help="Folder for one behavior (plot_data/<behavior>)")
    ap.add_argument("--behavior", required=True, help="Behavior name (plot titles only).")
    ap.add_argument(
        "--plots",
        default="steering_dprime,mcc_val_best",
        help="Comma-separated when not using --histogram-only. Default: steering_dprime,mcc_val_best. Optional: hist_postgen",
    )
    ap.add_argument(
        "--histogram-only",
        action="store_true",
        help="Only replot hist_postgen (ignores --plots). Requires per_prompt_results.csv.",
    )
    ap.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: ./replots/<behavior>)")
    ap.add_argument(
        "--factors",
        default=None,
        help="Steering factors for steering_dprime, or histogram panels when --hist-factors is omitted (e.g. 1,2,3,5,10)",
    )
    ap.add_argument(
        "--hist-factors",
        default=None,
        help="Histogram α panels only, e.g. 2 or 0,2. Overrides --factors for hist_postgen. "
        "Default: infer from CSV (prepends α=0 baseline panel unless --hist-skip-train)",
    )
    ap.add_argument(
        "--hist-skip-train",
        action="store_true",
        help="Histogram layout: omit Train panel and do not read train_projections.json.",
    )
    ap.add_argument("--layers", default=None, help="Layers for hist_postgen only, e.g. 21 or 10-32")
    ap.add_argument(
        "--font-scale",
        type=float,
        default=_DEFAULT_FONT_SCALE,
        help=(
            f"Multiply pipeline reference pt sizes for steering/MCC/histogram replots (default {_DEFAULT_FONT_SCALE}). "
            "Histogram panels use bases title=10, axes=9, ticks=10 (same tick scaling as steering/MCC); standalone hist legend uses base 9."
        ),
    )
    args = ap.parse_args()

    if args.font_scale <= 0:
        sys.exit("--font-scale must be positive")

    data_dir = args.data_dir.expanduser().resolve()
    summary_csv = data_dir / "per_layer_summary.csv"
    prompt_csv = data_dir / "per_prompt_results.csv"
    train_proj_json = data_dir / "train_projections.json"

    want = {"hist_postgen"} if args.histogram_only else {x.strip() for x in args.plots.split(",") if x.strip()}
    valid = {"steering_dprime", "mcc_val_best", "hist_postgen"}
    bad = want - valid
    if bad:
        sys.exit(f"Unknown --plots entries: {bad}. Use: {sorted(valid)}")

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else Path("replots") / args.behavior
    )

    fs = args.font_scale

    if {"steering_dprime", "mcc_val_best"} & want:
        if not summary_csv.is_file():
            sys.exit(f"Missing {summary_csv}")
        layer_df = pd.read_csv(summary_csv)
        factors = (
            sorted({float(x.strip()) for x in args.factors.split(",") if x.strip()})
            if args.factors
            else infer_factors_from_summary(layer_df)
        )
        if not factors:
            sys.exit("No steered_acc_* columns found in per_layer_summary.csv")

        if "steering_dprime" in want:
            plot_steering_and_dprime(
                layer_df,
                factors,
                out_dir / "steering_acc_and_dprime.png",
                font_scale=fs,
            )
        if "mcc_val_best" in want:
            plot_mcc_val_best_alpha_on_test_vs_dprime(
                layer_df,
                out_dir / "mcc_best_val_alpha_on_test_vs_dprime.png",
                font_scale=fs,
            )

    if "hist_postgen" in want:
        if not prompt_csv.is_file():
            sys.exit(f"Missing {prompt_csv}")
        prompt_df = pd.read_csv(prompt_csv)
        include_train = not args.hist_skip_train
        train_proj: dict = {}
        if include_train:
            if not train_proj_json.is_file():
                sys.exit(f"Missing {train_proj_json} (use --hist-skip-train if you have no train projections)")
            train_proj = load_train_projections(train_proj_json)

        if args.hist_factors:
            hist_factors_list = sorted(
                {float(x.strip()) for x in args.hist_factors.split(",") if x.strip()},
            )
        elif args.factors:
            hist_factors_list = sorted(
                {float(x.strip()) for x in args.factors.split(",") if x.strip()},
            )
        else:
            hist_factors_list = [0.0] + _infer_factors_from_prompt_df(prompt_df)

        if not include_train and len(hist_factors_list) == 0:
            sys.exit("With --hist-skip-train, provide at least one α via --hist-factors or --factors.")

        hist_layers = parse_layers(args.layers, prompt_df["layer"])
        for layer in hist_layers:
            plot_projection_histograms_postgen(
                prompt_df,
                hist_factors_list,
                args.behavior,
                layer,
                train_proj,
                out_dir / f"projection_hist_postgen_layer_{layer}.png",
                include_train=include_train,
                font_scale=fs,
            )

    print(f"Done. Outputs under {out_dir}")


if __name__ == "__main__":
    main()
