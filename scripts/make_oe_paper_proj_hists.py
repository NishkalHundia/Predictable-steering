#!/usr/bin/env python3
"""Paper 1×2 OE projection hists: Train | best-α (avg / last only).

Reads caches under results/local_oe_replot/{open_ended_projection_average,
open_ended_projection_last_token}/<dataset_behavior>/ and writes:

  paper_plots/<short>_proj_<mean|last>.png

Fonts match the local OE hist replot (title/axis bases +2 × 2.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "local_oe_replot"
OUT_DIR = ROOT / "paper_plots"
FLUENCY_THRESHOLD = 1.0

BEHAVIOR_SCALES = {
    "survival-instinct": (-5, 5, 0),
    "myopic-reward": (-5, 5, 0),
    "corrigible-neutral-HHH": (-5, 5, 0),
    "hallucination": (0, 5, None),
    "sycophancy": (0, 10, 5),
}
BEHAVIOR_THRESHOLDS = {b: (mn + mx) / 2.0 for b, (mn, mx, _) in BEHAVIOR_SCALES.items()}

# Short paper name → dataset folder name
DATASET = {
    "corrigibility": "corrigible-neutral-HHH",
    "survival": "survival-instinct",
    "syco": "sycophancy",
    "myopic": "myopic-reward",
    "hallucination": "hallucination",
}

# Panel letter order matches the lists below.
SPECS = {
    "mean": {  # open_ended_projection_average
        "regime": "open_ended_projection_average",
        "tag": "mean",
        "panels": [
            ("a", "corrigibility", 20, 3),
            ("b", "survival", 16, 10),
            ("c", "syco", 16, 5),
            ("d", "myopic", 24, 3),
            ("e", "hallucination", 12, 10),
        ],
    },
    "last": {  # open_ended_projection_last_token
        "regime": "open_ended_projection_last_token",
        "tag": "last",
        "panels": [
            ("a", "corrigibility", 21, 2),
            ("b", "survival", 18, 5),
            ("c", "syco", 26, 1),
            ("d", "myopic", 23, 2),
            ("e", "hallucination", 17, 2),
        ],
    },
}

_PAPER_FONT_SCALE = 2.1
_FS_HIST_TITLE = round((10 + 2) * _PAPER_FONT_SCALE, 2)  # 25.2
_FS_HIST_AXIS = round((9 + 2) * _PAPER_FONT_SCALE, 2)    # 23.1
_FS_TICK = round(10 * _PAPER_FONT_SCALE, 2)              # 21.0

plt.style.use("seaborn-v0_8-whitegrid")


def _hist_pair(
    ax,
    values_neg: np.ndarray,
    values_pos: np.ndarray,
    values_unk: np.ndarray | None,
    n_bins: int,
) -> None:
    unk = values_unk if values_unk is not None else np.array([], dtype=float)
    parts = [v for v in (values_neg, values_pos, unk) if len(v)]
    if not parts:
        ax.set_visible(False)
        return
    all_v = np.concatenate(parts)
    all_v = all_v[np.isfinite(all_v)]
    if len(all_v) == 0:
        ax.set_visible(False)
        return
    pad = max(0.3, (all_v.max() - all_v.min()) * 0.05 + 0.01)
    bins = np.linspace(all_v.min() - pad, all_v.max() + pad, n_bins)
    if len(unk):
        ax.hist(
            unk, bins=bins, color="#aaaaaa", alpha=0.5,
            edgecolor="#888888", linewidth=0.5,
        )
    if len(values_neg):
        ax.hist(
            values_neg, bins=bins, color="#d62728", alpha=0.55,
            edgecolor="#a01010", linewidth=0.5,
        )
    if len(values_pos):
        ax.hist(
            values_pos, bins=bins, color="#1f77b4", alpha=0.55,
            edgecolor="#104e8b", linewidth=0.5,
        )
    lo, hi = bins[0], bins[-1]
    if lo <= 0 <= hi:
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)


def _split_by_label(kap: np.ndarray, scr: np.ndarray, threshold: float):
    above = np.array([(s > threshold) if np.isfinite(s) else None for s in scr])
    return (
        kap[above == False],  # noqa: E712
        kap[above == True],   # noqa: E712
        kap[above == None],
    )


def plot_train_best_alpha(
    prompt_df: pd.DataFrame,
    train_projections: dict,
    *,
    dataset_behavior: str,
    layer: int,
    best_alpha: float,
    out_path: Path,
) -> None:
    threshold = BEHAVIOR_THRESHOLDS[dataset_behavior]
    title_fs, label_fs, tick_fs = _FS_HIST_TITLE, _FS_HIST_AXIS, _FS_TICK

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # ---- Train ----
    tp = train_projections.get(layer, {}) or train_projections.get(str(layer), {})
    pos = np.array(tp.get("pos", []), dtype=float)
    neg = np.array(tp.get("neg", []), dtype=float)
    _hist_pair(axes[0], neg, pos, None, n_bins=25)
    axes[0].set_title("Train", fontsize=title_fs, fontweight="bold")
    axes[0].set_xlabel(r"$\kappa_a$", fontsize=label_fs)
    axes[0].set_ylabel("# prompts", fontsize=label_fs)
    axes[0].tick_params(axis="both", labelsize=tick_fs)

    # ---- Best α ----
    ldf = prompt_df[prompt_df["layer"] == layer].copy()
    if abs(best_alpha) < 1e-12:
        kap_col, scr_col, flu_col = "kappa_postgen", "behavior_score_0", "fluency_score_0"
    else:
        kap_col = f"steered_kappa_{best_alpha:g}"
        scr_col = f"steered_behavior_score_{best_alpha:g}"
        flu_col = f"steered_fluency_score_{best_alpha:g}"

    if kap_col not in ldf.columns:
        raise KeyError(f"missing {kap_col} for {dataset_behavior} L{layer}")

    flu_mask = ldf[flu_col].apply(
        lambda x: x is not None
        and not (isinstance(x, float) and np.isnan(x))
        and float(x) >= FLUENCY_THRESHOLD
    )
    sub = ldf[flu_mask]
    kap = sub[kap_col].astype(float).values
    scr = sub[scr_col].apply(
        lambda x: float(x) if x is not None else float("nan")
    ).values
    k_neg, k_pos, k_unk = _split_by_label(kap, scr, threshold)
    _hist_pair(axes[1], k_neg, k_pos, k_unk, n_bins=20)
    axes[1].set_title(rf"$\alpha={best_alpha:g}$", fontsize=title_fs, fontweight="bold")
    axes[1].set_xlabel(r"$\kappa_a$ (post-gen token)", fontsize=label_fs)
    axes[1].tick_params(axis="both", labelsize=tick_fs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_open_proj_legend(out_path: Path) -> None:
    """Standalone horizontal legend: blue=above threshold, red=below threshold."""
    fs = round(9 * _PAPER_FONT_SCALE, 2)
    handles = [
        Patch(facecolor="#1f77b4", edgecolor="#104e8b", linewidth=0.5, alpha=0.55),
        Patch(facecolor="#d62728", edgecolor="#a01010", linewidth=0.5, alpha=0.55),
    ]
    labels = ["above threshold", "below threshold"]
    ncol = 2
    fig_w = max(4.0, min(34.0, ncol * 2.05))
    fig_h = max(0.6, 0.4 + 0.45)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=ncol,
        fontsize=fs,
        frameon=True,
        columnspacing=1.15,
        handlelength=2,
        borderpad=0.65,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for method, cfg in SPECS.items():
        regime = cfg["regime"]
        tag = cfg["tag"]
        for _letter, short, layer, alpha in cfg["panels"]:
            dataset = DATASET[short]
            d = DATA / regime / dataset
            prompt_csv = d / "per_prompt_results.csv"
            train_json = d / "train_projections.json"
            if not prompt_csv.exists() or not train_json.exists():
                raise FileNotFoundError(f"missing caches under {d}")
            prompt_df = pd.read_csv(prompt_csv)
            with open(train_json) as f:
                train_projs = {int(k): v for k, v in json.load(f).items()}
            out = OUT_DIR / f"{short}_proj_{tag}.png"
            plot_train_best_alpha(
                prompt_df, train_projs,
                dataset_behavior=dataset,
                layer=layer,
                best_alpha=float(alpha),
                out_path=out,
            )
            print(f"[OK] {out.relative_to(ROOT)}  (L{layer}, α={alpha})")
            n += 1
    legend_path = OUT_DIR / "open_proj_legend.png"
    save_open_proj_legend(legend_path)
    print(f"[OK] {legend_path.relative_to(ROOT)}")
    print(f"Wrote {n} figures + legend → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
