#!/usr/bin/env python3
"""Replot OE sign_mcc_by_layer → paper_plots/open_mcc/<short>_mcc_<tag>.png

Paper style: same fonts as steering_score_and_dprime, no title, no legend.
Tags: mean | last | prompted  (prompted = cue DiffMean, last_token postgen κ).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "local_oe_replot"
OUT_DIR = ROOT / "paper_plots" / "open_mcc"

SHORT = {
    "corrigible-neutral-HHH": "corrigibility",
    "survival-instinct": "survival",
    "sycophancy": "syco",
    "myopic-reward": "myopic",
    "hallucination": "hallucination",
}

JOBS = [
    # (tag, regime, relative plots subdir or ".")
    ("mean", "open_ended_projection_average", "."),
    ("last", "open_ended_projection_last_token", "."),
    ("prompted", "open_ended_projection_link_prompted", "last_token"),
]

_PAPER_FONT_SCALE = 2.1
_FS_AXIS = round(11 * _PAPER_FONT_SCALE, 2)  # 23.1
_FS_TICK = round(10 * _PAPER_FONT_SCALE, 2)  # 21.0
_AXIS_LABELPAD = round(10 + 2 * _PAPER_FONT_SCALE, 1)  # 14.2

plt.style.use("seaborn-v0_8-whitegrid")


def _infer_nonzero_factors(layer_df: pd.DataFrame) -> list[float]:
    factors = []
    for c in layer_df.columns:
        m = re.fullmatch(r"steered_sign_mcc_(.+)", c)
        if m:
            factors.append(float(m.group(1)))
    return sorted(factors)


def plot_mcc_by_layer(layer_df: pd.DataFrame, non_zero_factors: list[float], out_path: Path) -> None:
    layers = layer_df["layer"].values
    fig, ax = plt.subplots(figsize=(13, 5))

    if "sign_kappa_mcc" in layer_df.columns:
        ax.plot(
            layers, layer_df["sign_kappa_mcc"].values, "D--", color="gray",
            linewidth=1.5, markersize=6, alpha=0.8, zorder=3,
        )

    cmap = plt.get_cmap("plasma")
    nz = [f for f in non_zero_factors if abs(f) > 1e-9]
    nf = len(nz)
    for i, f in enumerate(nz):
        col = f"steered_sign_mcc_{f:g}"
        if col not in layer_df.columns:
            continue
        ax.plot(
            layers, layer_df[col].values, "o-",
            color=cmap(i / max(1, nf - 1)), linewidth=2, markersize=5, zorder=3,
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=_FS_AXIS, labelpad=_AXIS_LABELPAD)
    ax.set_ylabel("MCC", fontsize=_FS_AXIS, labelpad=_AXIS_LABELPAD)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(layers)
    ax.tick_params(axis="both", labelsize=_FS_TICK)
    # no title, no legend

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for tag, regime, sub in JOBS:
        for dataset, short in SHORT.items():
            d = DATA / regime / dataset
            if sub == ".":
                csv_path = d / "per_layer_summary.csv"
            else:
                # prompted writes site-tagged summaries
                csv_path = d / f"per_layer_summary_{sub}.csv"
                if not csv_path.exists():
                    csv_path = d / "per_layer_summary.csv"
            if not csv_path.exists():
                print(f"[SKIP] {short}_mcc_{tag} (missing {csv_path.name})")
                continue
            df = pd.read_csv(csv_path)
            out = OUT_DIR / f"{short}_mcc_{tag}.png"
            plot_mcc_by_layer(df, _infer_nonzero_factors(df), out)
            print(f"[OK] {out.relative_to(ROOT)}")
            n += 1
    print(f"Wrote {n} figures → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
