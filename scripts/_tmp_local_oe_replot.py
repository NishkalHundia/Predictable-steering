#!/usr/bin/env python3
"""Temp local OE replot (no Modal).

1) steering_score_and_dprime.png — no title
2) projection_hist_postgen_layer_*.png — panel titles + axis labels at
   paper hist fonts + 1pt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "local_oe_replot"
HIST_LAYERS = list(range(10, 33))
FLUENCY_THRESHOLD = 1.0

BEHAVIORS = [
    "myopic-reward",
    "sycophancy",
    "hallucination",
    "survival-instinct",
    "corrigible-neutral-HHH",
]
BEHAVIOR_SCALES = {
    "survival-instinct": (-5, 5, 0),
    "myopic-reward": (-5, 5, 0),
    "corrigible-neutral-HHH": (-5, 5, 0),
    "hallucination": (0, 5, None),
    "sycophancy": (0, 10, 5),
}
BEHAVIOR_THRESHOLDS = {b: (mn + mx) / 2.0 for b, (mn, mx, _) in BEHAVIOR_SCALES.items()}

_FACTOR_CMAP = plt.get_cmap("plasma")
_BASELINE_COLOR = "gray"
_DPRIME_COLOR = "steelblue"
_PAPER_FONT_SCALE = 2.1
_FS_AXIS = round(11 * _PAPER_FONT_SCALE, 2)
_FS_TICK = round(10 * _PAPER_FONT_SCALE, 2)
_AXIS_LABELPAD = round(10 + 2 * _PAPER_FONT_SCALE, 1)

# Paper hist bases are title=10, axes=9; bump those bases by +2 *before* scale
# so the change is visible on the wide (18") hist grids.
_FS_HIST_TITLE = round((10 + 2) * _PAPER_FONT_SCALE, 2)  # 25.2
_FS_HIST_AXIS = round((9 + 2) * _PAPER_FONT_SCALE, 2)    # 23.1
_FS_HIST_SUP = round(12 * _PAPER_FONT_SCALE, 2)           # 25.2 (unchanged)

plt.style.use("seaborn-v0_8-whitegrid")


def _factor_color(f: float, non_zero_factors: list[float]) -> str:
    if abs(f) < 1e-9:
        return _BASELINE_COLOR
    nz = [x for x in non_zero_factors if abs(x) > 1e-9]
    if not nz:
        return _BASELINE_COLOR
    try:
        i = sorted(nz).index(float(f))
    except ValueError:
        i = 0
    return matplotlib.colors.to_hex(_FACTOR_CMAP(i / max(1, len(nz) - 1)))


def _infer_nonzero_factors_from_summary(layer_df: pd.DataFrame) -> list[float]:
    factors = []
    for c in layer_df.columns:
        m = re.fullmatch(r"avg_steered_behavior_(.+)", c)
        if m:
            factors.append(float(m.group(1)))
    return sorted(factors)


def _infer_nonzero_factors_from_prompt(prompt_df: pd.DataFrame, prefix: str) -> list[float]:
    factors = []
    for c in prompt_df.columns:
        if c.startswith(prefix):
            try:
                factors.append(float(c[len(prefix):]))
            except ValueError:
                continue
    return sorted(set(factors))


def plot_steering_score_and_dprime(
    layer_df: pd.DataFrame,
    non_zero_factors: list[float],
    behavior: str,
    out_path: Path,
) -> None:
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(13, 5))

    rescale_syc = behavior == "sycophancy"
    score_scale = 0.5 if rescale_syc else 1.0

    nz = [f for f in non_zero_factors if f != 0]
    for f in [0.0] + nz:
        col = "avg_behavior_score_0" if f == 0 else f"avg_steered_behavior_{f:g}"
        if col not in layer_df.columns:
            continue
        y = layer_df[col].to_numpy(dtype=float) * score_scale
        color = _factor_color(f, nz)
        if f == 0:
            ax1.plot(
                layers, y, "D--", color=color,
                linewidth=1.5, markersize=6, alpha=0.85, zorder=3,
            )
        else:
            ax1.plot(
                layers, y, "o-", color=color,
                linewidth=2, markersize=5, zorder=3,
            )

    if rescale_syc:
        scale_min, scale_max, ref = 0.0, 5.0, 2.5
    else:
        scale_min, scale_max, ref = BEHAVIOR_SCALES.get(behavior, (0, 10, 5))
    if ref is not None:
        ax1.axhline(ref, color="gray", linestyle=":", alpha=0.4, linewidth=0.9)
    ax1.set_ylim(scale_min - 0.5, scale_max + 0.5)
    ax1.set_xlabel("Layer", fontsize=_FS_AXIS, labelpad=_AXIS_LABELPAD)
    ax1.set_ylabel("Avg behavior score", fontsize=_FS_AXIS, labelpad=_AXIS_LABELPAD)
    ax1.set_xticks(layers)
    ax1.tick_params(axis="both", labelsize=_FS_TICK)

    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color=_DPRIME_COLOR)
        ax2.plot(
            layers, layer_df["dprime"].values, "s:", color=_DPRIME_COLOR,
            linewidth=1.5, markersize=5, zorder=2,
        )
        ax2.set_ylabel(
            "d' (train)", fontsize=_FS_AXIS, color=_DPRIME_COLOR, labelpad=_AXIS_LABELPAD,
        )
        ax2.tick_params(axis="y", labelcolor=_DPRIME_COLOR, labelsize=_FS_TICK)
        ax2.set_ylim(bottom=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_histograms_oe(
    prompt_df: pd.DataFrame,
    layer: int,
    non_zero_factors: list[float],
    behavior: str,
    train_projections: dict,
    threshold: float,
    out_path: Path,
    *,
    fluency_threshold: float = FLUENCY_THRESHOLD,
    kappa_postgen_col: str = "kappa_postgen",
    steered_kappa_prefix: str = "steered_kappa_",
    train_title: str = "Train",
    train_xlabel: str = "κ",
    kappa_site_label: str | None = None,
    suptitle: str | None = None,
) -> None:
    ldf = prompt_df[prompt_df["layer"] == layer].copy()
    if len(ldf) < 4:
        return

    all_factors = [0.0] + [f for f in non_zero_factors if f != 0.0]
    n_factor_panels = len(all_factors)
    n_panels = 1 + n_factor_panels

    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    title_fs, label_fs, tick_fs, sup_fs = (
        _FS_HIST_TITLE, _FS_HIST_AXIS, _FS_TICK, _FS_HIST_SUP,
    )

    tp = train_projections.get(layer, {}) or train_projections.get(str(layer), {})
    pos_proj = np.array(tp.get("pos", []), dtype=float)
    neg_proj = np.array(tp.get("neg", []), dtype=float)
    train_ax = axes_flat[0]
    if len(pos_proj) > 0 or len(neg_proj) > 0:
        all_t = np.concatenate([x for x in [pos_proj, neg_proj] if len(x) > 0])
        pad = max(0.3, (all_t.max() - all_t.min()) * 0.05 + 0.01)
        bins = np.linspace(all_t.min() - pad, all_t.max() + pad, 25)
        if len(neg_proj):
            train_ax.hist(
                neg_proj, bins=bins, color="#d62728", alpha=0.55,
                edgecolor="#a01010", linewidth=0.5,
            )
        if len(pos_proj):
            train_ax.hist(
                pos_proj, bins=bins, color="#1f77b4", alpha=0.55,
                edgecolor="#104e8b", linewidth=0.5,
            )
        lo, hi = bins[0], bins[-1]
        if lo <= 0 <= hi:
            train_ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    train_ax.set_title(train_title, fontsize=title_fs, fontweight="bold")
    train_ax.set_xlabel(train_xlabel, fontsize=label_fs)
    train_ax.set_ylabel("# prompts", fontsize=label_fs)
    train_ax.tick_params(axis="both", labelsize=tick_fs)

    for panel_i, alpha in enumerate(all_factors):
        ax = axes_flat[1 + panel_i]
        if alpha == 0.0:
            kap_col = kappa_postgen_col
            scr_col = "behavior_score_0"
            flu_col = "fluency_score_0"
        else:
            kap_col = f"{steered_kappa_prefix}{alpha:g}"
            scr_col = f"steered_behavior_score_{alpha:g}"
            flu_col = f"steered_fluency_score_{alpha:g}"

        if kap_col not in ldf.columns:
            ax.set_visible(False)
            continue

        flu_mask = ldf[flu_col].apply(
            lambda x: x is not None
            and not (isinstance(x, float) and np.isnan(x))
            and float(x) >= fluency_threshold
        )
        sub = ldf[flu_mask]
        if len(sub) < 2:
            ax.set_visible(False)
            continue

        kap = sub[kap_col].astype(float).values
        scr = sub[scr_col].apply(
            lambda x: float(x) if x is not None else float("nan")
        ).values
        above = np.array([(s > threshold) if np.isfinite(s) else None for s in scr])
        k_above = kap[above == True]   # noqa: E712
        k_below = kap[above == False]  # noqa: E712
        k_unk = kap[above == None]

        all_k = kap[np.isfinite(kap)]
        if len(all_k) == 0:
            ax.set_visible(False)
            continue
        pad_ = max(0.3, (all_k.max() - all_k.min()) * 0.05 + 0.01)
        bins = np.linspace(all_k.min() - pad_, all_k.max() + pad_, 20)

        if len(k_unk):
            ax.hist(k_unk, bins=bins, color="#aaaaaa", alpha=0.5,
                    edgecolor="#888888", linewidth=0.5)
        if len(k_below):
            ax.hist(k_below, bins=bins, color="#d62728", alpha=0.55,
                    edgecolor="#a01010", linewidth=0.5)
        if len(k_above):
            ax.hist(k_above, bins=bins, color="#1f77b4", alpha=0.55,
                    edgecolor="#104e8b", linewidth=0.5)

        lo, hi = bins[0], bins[-1]
        if lo <= 0 <= hi:
            ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)

        ax.set_title(f"α={alpha:g}", fontsize=title_fs, fontweight="bold")
        if kappa_site_label is None:
            ax.set_xlabel("κ_postgen", fontsize=label_fs)
        else:
            ax.set_xlabel(f"κ_postgen ({kappa_site_label})", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        if panel_i == 0 and kappa_site_label is None:
            ax.set_ylabel("# prompts", fontsize=label_fs)

    for ax in axes_flat[1 + n_factor_panels:]:
        ax.set_visible(False)

    if suptitle is None:
        suptitle = (
            f"{behavior} — Layer {layer}: Postgen κ distribution "
            f"(open-ended, fluency≥{fluency_threshold})"
        )
    fig.suptitle(suptitle, fontsize=sup_fs, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def score_jobs() -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for regime in (
        "open_ended_projection_average",
        "open_ended_projection_last_token",
    ):
        for b in BEHAVIORS:
            d = DATA / regime / b
            out.append((b, d / "per_layer_summary.csv", d / "plots" / "steering_score_and_dprime.png"))
    for b in BEHAVIORS:
        d = DATA / "open_ended_projection_link_prompted" / b
        for site in ("avg_token", "last_token"):
            out.append((
                b,
                d / f"per_layer_summary_{site}.csv",
                d / "plots" / site / "steering_score_and_dprime.png",
            ))
    return out


def hist_jobs() -> list[dict]:
    jobs: list[dict] = []
    for regime in (
        "open_ended_projection_average",
        "open_ended_projection_last_token",
    ):
        for b in BEHAVIORS:
            d = DATA / regime / b
            jobs.append({
                "behavior": b,
                "dir": d,
                "plots_dir": d / "plots",
                "kappa_postgen_col": "kappa_postgen",
                "steered_kappa_prefix": "steered_kappa_",
                "train_title": "Train",
                "train_xlabel": "κ",
                "kappa_site_label": None,
                "suptitle_fn": lambda behavior, layer: (
                    f"{behavior} — Layer {layer}: Postgen κ distribution "
                    f"(open-ended, fluency≥{FLUENCY_THRESHOLD})"
                ),
            })
    for b in BEHAVIORS:
        d = DATA / "open_ended_projection_link_prompted" / b
        for site, kap_col, prefix, site_label in (
            ("last_token", "kappa_postgen", "steered_kappa_", "response last tok"),
            ("avg_token", "kappa_postgen_avg", "steered_kappa_avg_", "response avg tok"),
        ):
            jobs.append({
                "behavior": b,
                "dir": d,
                "plots_dir": d / "plots" / site,
                "kappa_postgen_col": kap_col,
                "steered_kappa_prefix": prefix,
                "train_title": "Train (cue + question)",
                "train_xlabel": "κ (cue + question last tok)",
                "kappa_site_label": site_label,
                "suptitle_fn": (
                    lambda behavior, layer, sl=site_label: (
                        f"{behavior} — Layer {layer}: Train=cue + question last tok; α=postgen "
                        f"{sl} (fluency≥{FLUENCY_THRESHOLD})"
                    )
                ),
            })
    return jobs


def main() -> int:
    do_scores = "--hist-only" not in sys.argv
    print(
        f"Hist fonts: title={_FS_HIST_TITLE} axis={_FS_HIST_AXIS} "
        f"(bases +2 then ×{_PAPER_FONT_SCALE}) tick={_FS_TICK} sup={_FS_HIST_SUP}"
        + ("" if do_scores else " [hist-only]")
    )
    fails = 0

    # --- score plots ---
    if do_scores:
        for behavior, csv_path, png_path in score_jobs():
            tag = f"{png_path.relative_to(DATA)}"
            if not csv_path.exists():
                print(f"[SKIP score] {tag}")
                fails += 1
                continue
            df = pd.read_csv(csv_path)
            plot_steering_score_and_dprime(
                df, _infer_nonzero_factors_from_summary(df), behavior, png_path,
            )
            print(f"[OK score] {tag}")

    # --- hist plots ---
    for job in hist_jobs():
        d = job["dir"]
        prompt_csv = d / "per_prompt_results.csv"
        train_json = d / "train_projections.json"
        tag_base = f"{job['plots_dir'].relative_to(DATA)}"
        if not prompt_csv.exists() or not train_json.exists():
            print(f"[SKIP hist] {tag_base}")
            fails += 1
            continue
        prompt_df = pd.read_csv(prompt_csv)
        with open(train_json) as f:
            raw = json.load(f)
        train_projs = {int(k): v for k, v in raw.items()}
        factors = _infer_nonzero_factors_from_prompt(
            prompt_df, job["steered_kappa_prefix"],
        )
        thr = BEHAVIOR_THRESHOLDS[job["behavior"]]
        n_ok = 0
        for layer in HIST_LAYERS:
            out = job["plots_dir"] / f"projection_hist_postgen_layer_{layer}.png"
            plot_projection_histograms_oe(
                prompt_df, layer, factors, job["behavior"], train_projs, thr, out,
                kappa_postgen_col=job["kappa_postgen_col"],
                steered_kappa_prefix=job["steered_kappa_prefix"],
                train_title=job["train_title"],
                train_xlabel=job["train_xlabel"],
                kappa_site_label=job["kappa_site_label"],
                suptitle=job["suptitle_fn"](job["behavior"], layer),
            )
            if out.exists():
                n_ok += 1
        print(f"[OK hist]  {tag_base} ({n_ok} layers)")

    print(f"Done. fails={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
