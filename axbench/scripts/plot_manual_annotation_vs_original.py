"""
Plot LM-judge (rejudge) scores vs human annotation scores from manual_annotation.xlsx.

Compares rejudged behavior scores with sarah_anno, swastik_anno, navita_anno.
Handles negative scores and 0-5 / -5 to 5 scales.

Usage:
    uv run python axbench/scripts/plot_manual_annotation_vs_original.py

    uv run python axbench/scripts/plot_manual_annotation_vs_original.py \
        --xlsx_path results/gemma-2-9b-it/manual_annotation.xlsx \
        --output_dir results/gemma-2-9b-it/manual_annotation_plots
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_HUMAN_SHEETS = [
    "sarah_anno",
    "swastik_anno",
    "navita_anno",
]

METRICS = ["Behavior score", "Fluence score"]
KEY_COLUMNS = ["Prompt", "Response", "Behavior"]
PAIRING_COLUMN_CANDIDATES = ["category", "pairing", "pair", "bucket"]


def _find_column(df: pd.DataFrame, desired_name: str) -> str | None:
    normalized = {c.strip().lower(): c for c in df.columns}
    desired_key = desired_name.strip().lower()
    if desired_key in normalized:
        return normalized[desired_key]

    # Handle common typo variant: Fluence vs Fluency
    if desired_key == "fluence score" and "fluency score" in normalized:
        return normalized["fluency score"]
    if desired_key == "fluency score" and "fluence score" in normalized:
        return normalized["fluence score"]
    return None


def _find_matching_keys(orig: pd.DataFrame, human: pd.DataFrame) -> list[tuple[str, str]]:
    """Return [(orig_col, human_col), ...] for merge, using case-insensitive match."""
    orig_norm = {c.strip().lower(): c for c in orig.columns}
    human_norm = {c.strip().lower(): c for c in human.columns}
    out = []
    for k in KEY_COLUMNS:
        kk = k.strip().lower()
        if kk in orig_norm and kk in human_norm:
            out.append((orig_norm[kk], human_norm[kk]))
    return out


def _prepare_metric_pairs(
    original_df: pd.DataFrame,
    human_df: pd.DataFrame,
    human_sheet: str,
) -> list[pd.DataFrame]:
    key_pairs = _find_matching_keys(original_df, human_df)
    if not key_pairs:
        raise ValueError(
            f"Cannot align sheet '{human_sheet}' with LM sheet: no shared key columns among {KEY_COLUMNS}."
        )
    orig_keys = [p[0] for p in key_pairs]
    human_keys = [p[1] for p in key_pairs]

    out = []
    orig_behavior_col = _find_column(original_df, "Behavior score")
    orig_fluence_col = _find_column(original_df, "Fluence score")

    pairing_col = None
    for cand in PAIRING_COLUMN_CANDIDATES:
        found = _find_column(original_df, cand) or _find_column(human_df, cand)
        if found is not None:
            pairing_col = found
            break

    for metric in METRICS:
        orig_metric = _find_column(original_df, metric)
        human_metric = _find_column(human_df, metric)
        if orig_metric is None or human_metric is None:
            continue

        original_keep = orig_keys + [orig_metric]
        human_keep = human_keys + [human_metric]
        if orig_behavior_col is not None:
            original_keep.append(orig_behavior_col)
        if orig_fluence_col is not None:
            original_keep.append(orig_fluence_col)
        if pairing_col is not None:
            if pairing_col in original_df.columns:
                original_keep.append(pairing_col)
            if pairing_col in human_df.columns:
                human_keep.append(pairing_col)

        original_keep = list(dict.fromkeys(original_keep))
        human_keep = list(dict.fromkeys(human_keep))

        original_slice = original_df[original_keep].copy()
        human_slice = human_df[human_keep].copy()

        human_renamed = human_slice.rename(columns=dict(zip(human_keys, orig_keys)))
        merged = original_slice.merge(
            human_renamed,
            on=orig_keys,
            how="inner",
            suffixes=("_original", "_human"),
        )
        orig_metric_merged = _resolve_merged_col(merged, orig_metric, prefer="original")
        human_metric_merged = _resolve_merged_col(merged, human_metric, prefer="human")
        merged["original_score"] = pd.to_numeric(merged[orig_metric_merged], errors="coerce")
        merged["human_score"] = pd.to_numeric(merged[human_metric_merged], errors="coerce")
        merged = merged.dropna(subset=["original_score", "human_score"]).copy()

        if merged.empty:
            continue

        merged["metric"] = metric
        merged["annotator"] = human_sheet
        merged["pairing"] = _derive_pairing_labels(
            merged=merged,
            explicit_pairing_col=pairing_col,
            original_behavior_col=orig_behavior_col,
            original_fluence_col=orig_fluence_col,
        )

        out.append(merged[["annotator", "pairing", "metric", "original_score", "human_score"]])

    return out


def _resolve_merged_col(df: pd.DataFrame, base_col: str, prefer: str) -> str:
    if base_col in df.columns:
        return base_col
    preferred = f"{base_col}_{prefer}"
    if preferred in df.columns:
        return preferred
    if prefer == "original":
        fallback = f"{base_col}_human"
    else:
        fallback = f"{base_col}_original"
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Could not resolve merged column for '{base_col}' (prefer={prefer}).")


def _derive_pairing_labels(
    merged: pd.DataFrame,
    explicit_pairing_col: str | None,
    original_behavior_col: str | None,
    original_fluence_col: str | None,
) -> pd.Series:
    labels = pd.Series(["unknown"] * len(merged), index=merged.index, dtype=object)

    if explicit_pairing_col:
        explicit_candidates = [explicit_pairing_col, f"{explicit_pairing_col}_original", f"{explicit_pairing_col}_human"]
        for col in explicit_candidates:
            if col in merged.columns:
                values = merged[col].astype(str).str.strip()
                mask = values.notna() & (values != "") & (values.str.lower() != "nan")
                labels.loc[mask] = values.loc[mask]
                break

    if original_behavior_col is None or original_fluence_col is None:
        return labels

    behavior_candidates = [original_behavior_col, f"{original_behavior_col}_original"]
    fluence_candidates = [original_fluence_col, f"{original_fluence_col}_original"]
    behavior_col = next((c for c in behavior_candidates if c in merged.columns), None)
    fluence_col = next((c for c in fluence_candidates if c in merged.columns), None)
    if behavior_col is None or fluence_col is None:
        return labels

    b = pd.to_numeric(merged[behavior_col], errors="coerce")
    f = pd.to_numeric(merged[fluence_col], errors="coerce")
    b_valid = b.dropna()
    f_round = f.round().astype("Int64")

    # Support 0-10, 0-5, -5 to 5 scales
    b_min, b_max = float(b_valid.min()), float(b_valid.max())
    if b_min < 0:
        high_b, low_b = 5, -5
    elif b_max <= 5:
        high_b, low_b = 5, 0
    else:
        high_b, low_b = 10, 0

    derived = labels.copy()
    derived[(b >= high_b - 0.5) & (f_round == 0)] = "high_behavior_low_fluency"
    derived[(b <= low_b + 0.5) & (f_round == 2)] = "low_behavior_high_fluency"
    derived[(b >= high_b - 0.5) & (f_round == 2)] = "high_behavior_high_fluency"
    derived[(b <= low_b + 0.5) & (f_round == 0)] = "low_behavior_low_fluency"

    return derived


def _plot_pairs(all_pairs: pd.DataFrame, output_path: Path) -> None:
    metrics_present = [m for m in METRICS if m in set(all_pairs["metric"])]
    if not metrics_present:
        raise ValueError("No usable metric rows found after dropping missing scores.")

    fig, axes = plt.subplots(1, len(metrics_present), figsize=(7 * len(metrics_present), 6), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics_present):
        metric_df = all_pairs[all_pairs["metric"] == metric].copy()
        corr = metric_df["original_score"].corr(metric_df["human_score"])

        score_min = float(
            min(
                metric_df["original_score"].min(),
                metric_df["human_score"].min(),
            )
        )
        score_max = float(
            max(
                metric_df["original_score"].max(),
                metric_df["human_score"].max(),
            )
        )
        max_unique = max(metric_df["original_score"].nunique(), metric_df["human_score"].nunique())
        if max_unique > 40:
            bin_size = 0.5
        elif max_unique > 20:
            bin_size = 0.25
        else:
            bin_size = 0.0

        if bin_size > 0:
            metric_df["original_plot"] = np.round(metric_df["original_score"] / bin_size) * bin_size
            metric_df["human_plot"] = np.round(metric_df["human_score"] / bin_size) * bin_size
            bin_text = f"bin={bin_size:g}"
        else:
            metric_df["original_plot"] = metric_df["original_score"]
            metric_df["human_plot"] = metric_df["human_score"]
            bin_text = "bin=exact"

        bubbles = (
            metric_df.groupby(["original_plot", "human_plot"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        marker_size = 35 + 22 * np.sqrt(bubbles["count"])
        sc = ax.scatter(
            bubbles["original_plot"],
            bubbles["human_plot"],
            s=marker_size,
            c=bubbles["count"],
            cmap="viridis",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )

        pad = max(0.5, (score_max - score_min) * 0.05)
        axis_min = score_min - pad if score_min < 0 else max(0, score_min - pad)
        axis_max = score_max + pad
        ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1.2, color="black", alpha=0.7)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_xlabel("Rejudge LM score")
        ax.set_ylabel("Human score")
        ax.set_title(
            f"{metric} (bubble=count)\nN={len(metric_df)}, r={corr:.3f}, {bin_text}"
        )
        ax.grid(alpha=0.25)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Overlapping points")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _compute_agreement_stats(metric_df: pd.DataFrame) -> dict:
    """Compute agreement/disagreement stats for LM vs human (handles negatives and 0)."""
    lm = metric_df["original_score"].values
    hu = metric_df["human_score"].values
    n = len(lm)

    exact = np.sum(lm == hu)
    within_1 = np.sum(np.abs(lm - hu) <= 1)
    within_2 = np.sum(np.abs(lm - hu) <= 2)

    # Sign agreement (only meaningful when scale includes negative)
    both_pos = np.sum((lm > 0) & (hu > 0))
    both_neg = np.sum((lm < 0) & (hu < 0))
    both_zero = np.sum((lm == 0) & (hu == 0))
    same_sign = both_pos + both_neg + both_zero

    opp_pos_neg = np.sum((lm > 0) & (hu < 0))
    opp_neg_pos = np.sum((lm < 0) & (hu > 0))
    opposite_sign = opp_pos_neg + opp_neg_pos

    # One zero, other non-zero (mild disagreement)
    lm_zero_hu_not = np.sum((lm == 0) & (hu != 0))
    hu_zero_lm_not = np.sum((lm != 0) & (hu == 0))
    zero_mismatch = lm_zero_hu_not + hu_zero_lm_not

    return {
        "n": n,
        "exact_pct": 100 * exact / n if n else 0,
        "within_1pt_pct": 100 * within_1 / n if n else 0,
        "within_2pt_pct": 100 * within_2 / n if n else 0,
        "same_sign_pct": 100 * same_sign / n if n else 0,
        "opposite_sign_pct": 100 * opposite_sign / n if n else 0,
        "zero_mismatch_pct": 100 * zero_mismatch / n if n else 0,
        "mae": float(np.mean(np.abs(lm - hu))) if n else 0,
    }


def _plot_agreement_summary(all_pairs: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Plot bar chart of agreement rates: how often LM agrees vs disagrees with humans."""
    behavior_df = all_pairs[all_pairs["metric"] == "Behavior score"]
    if behavior_df.empty:
        return pd.DataFrame()

    rows = []
    for annotator in behavior_df["annotator"].unique():
        sub = behavior_df[behavior_df["annotator"] == annotator]
        stats = _compute_agreement_stats(sub)
        rows.append({"annotator": annotator, **stats})

    agg_df = behavior_df.copy()
    stats_all = _compute_agreement_stats(agg_df)
    rows.append({"annotator": "all", **stats_all})

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: agreement metrics (positive = LM agrees)
    ax1 = axes[0]
    annotators = summary["annotator"].tolist()
    x = np.arange(len(annotators))
    w = 0.2

    ax1.bar(x - 1.5 * w, summary["exact_pct"], w, label="Exact match", color="#2ecc71")
    ax1.bar(x - 0.5 * w, summary["within_1pt_pct"], w, label="Within 1 pt", color="#3498db")
    ax1.bar(x + 0.5 * w, summary["within_2pt_pct"], w, label="Within 2 pt", color="#9b59b6")
    ax1.bar(x + 1.5 * w, summary["same_sign_pct"], w, label="Same sign", color="#1abc9c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(annotators)
    ax1.set_ylabel("% of pairs")
    ax1.set_title("Behavior score: LM judge agreement with human")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: disagreement metrics (LM disagrees)
    ax2 = axes[1]
    ax2.bar(x - 0.5 * w, summary["opposite_sign_pct"], w, label="Opposite sign (strong disagree)", color="#e74c3c")
    ax2.bar(x + 0.5 * w, summary["zero_mismatch_pct"], w, label="Zero vs non-zero", color="#e67e22")
    ax2.set_xticks(x)
    ax2.set_xticklabels(annotators)
    ax2.set_ylabel("% of pairs")
    ax2.set_title("Behavior score: LM judge disagreement with human")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    return summary


def _plot_pairing_disagreement(all_pairs: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    paired = all_pairs[all_pairs["pairing"].fillna("unknown") != "unknown"].copy()
    if paired.empty:
        raise ValueError("No pairing/category information found to compute disagreement.")

    summary = (
        paired.groupby(["metric", "pairing"], as_index=False)
        .agg(
            n=("human_score", "count"),
            original_mean=("original_score", "mean"),
            human_mean=("human_score", "mean"),
            mean_delta=("human_score", lambda x: float(np.mean(x - paired.loc[x.index, "original_score"]))),
            mae=("human_score", lambda x: float(np.mean(np.abs(x - paired.loc[x.index, "original_score"])))),
        )
    )

    metrics_present = [m for m in METRICS if m in set(summary["metric"])]
    fig, axes = plt.subplots(1, len(metrics_present), figsize=(8 * len(metrics_present), 6), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics_present):
        sdf = summary[summary["metric"] == metric].sort_values("mae", ascending=False).reset_index(drop=True)
        y = np.arange(len(sdf))

        for i, row in sdf.iterrows():
            ax.plot(
                [row["original_mean"], row["human_mean"]],
                [i, i],
                color="gray",
                linewidth=2.0,
                alpha=0.8,
            )

        ax.scatter(sdf["original_mean"], y, s=60, label="Rejudge LM mean", color="#1f77b4", zorder=3)
        ax.scatter(sdf["human_mean"], y, s=60, label="Human mean", color="#d62728", zorder=3)

        for i, row in sdf.iterrows():
            ax.text(
                max(row["original_mean"], row["human_mean"]) + 0.08,
                i,
                f"|delta|={abs(row['mean_delta']):.2f}, n={int(row['n'])}",
                va="center",
                fontsize=9,
            )

        x_min = min(float(sdf["original_mean"].min()), float(sdf["human_mean"].min()))
        x_max = max(float(sdf["original_mean"].max()), float(sdf["human_mean"].max()))
        pad = max(0.5, (x_max - x_min) * 0.05)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_yticks(y)
        ax.set_yticklabels(sdf["pairing"])
        ax.invert_yaxis()
        ax.set_xlabel("Score")
        ax.set_title(f"{metric}: Rejudge LM vs Human by pairing")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    return summary.sort_values(["metric", "mae"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx_path",
        type=str,
        default="results/gemma-2-9b-it/manual_annotation.xlsx",
        help="Path to manual annotation workbook.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gemma-2-9b-it/manual_annotation_plots",
        help="Directory for output plot and CSV.",
    )
    parser.add_argument(
        "--original_sheet",
        type=str,
        default="rejudge",
        help="Sheet name containing LM judge scores (rejudged).",
    )
    parser.add_argument(
        "--human_sheets",
        type=str,
        default=",".join(DEFAULT_HUMAN_SHEETS),
        help="Comma-separated list of human annotation sheet names.",
    )
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    try:
        workbook = pd.read_excel(xlsx_path, sheet_name=None)
    except ImportError as e:
        raise ImportError("Reading .xlsx requires openpyxl. Install it in your environment.") from e

    if args.original_sheet not in workbook:
        raise ValueError(f"Missing required sheet '{args.original_sheet}' in {xlsx_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_df = workbook[args.original_sheet]
    target_human_sheets = [s.strip() for s in args.human_sheets.split(",") if s.strip()]

    pair_frames = []
    for sheet in target_human_sheets:
        if sheet not in workbook:
            print(f"[skip] Sheet not found: {sheet}")
            continue
        pair_frames.extend(_prepare_metric_pairs(original_df, workbook[sheet], sheet))

    if not pair_frames:
        raise ValueError("No aligned, non-empty score pairs found across requested sheets.")

    all_pairs = pd.concat(pair_frames, ignore_index=True)

    out_csv = output_dir / "rejudge_vs_human_pairs.csv"
    all_pairs.to_csv(out_csv, index=False)

    summary = (
        all_pairs.groupby(["annotator", "metric"], as_index=False)
        .agg(
            n=("human_score", "count"),
            original_mean=("original_score", "mean"),
            human_mean=("human_score", "mean"),
            pearson_r=("human_score", lambda x: x.corr(all_pairs.loc[x.index, "original_score"])),
            mae=("human_score", lambda x: np.mean(np.abs(x - all_pairs.loc[x.index, "original_score"]))),
        )
        .sort_values(["metric", "annotator"])
    )
    out_summary = output_dir / "rejudge_vs_human_summary.csv"
    summary.to_csv(out_summary, index=False)

    out_plot = output_dir / "rejudge_vs_human_bubble.png"
    _plot_pairs(all_pairs, out_plot)

    # Agreement/disagreement summary for behavior scores (handles negatives, 0)
    agreement_summary = None
    try:
        agreement_plot = output_dir / "rejudge_agreement_summary.png"
        agreement_summary = _plot_agreement_summary(all_pairs, agreement_plot)
        if not agreement_summary.empty:
            agreement_summary.to_csv(output_dir / "rejudge_agreement_summary.csv", index=False)
            print(f"Wrote agreement plot: {agreement_plot}")
            print(f"Wrote agreement summary CSV: {output_dir / 'rejudge_agreement_summary.csv'}")
    except Exception as e:
        print(f"[skip] Agreement summary plot: {e}")

    pairing_summary = None
    try:
        pairing_plot = output_dir / "rejudge_pairing_disagreement_dumbbell.png"
        pairing_summary = _plot_pairing_disagreement(all_pairs, pairing_plot)
        pairing_summary_path = output_dir / "pairing_disagreement_summary.csv"
        pairing_summary.to_csv(pairing_summary_path, index=False)
    except ValueError as e:
        pairing_plot = None
        pairing_summary_path = None
        print(f"[skip] Pairing disagreement plot not generated: {e}")

    print(f"Loaded workbook: {xlsx_path}")
    print(f"Wrote pairs CSV: {out_csv}")
    print(f"Wrote summary CSV: {out_summary}")
    print(f"Wrote plot: {out_plot}")
    if pairing_plot is not None:
        print(f"Wrote plot: {pairing_plot}")
    if pairing_summary_path is not None:
        print(f"Wrote pairing summary CSV: {pairing_summary_path}")
    print("\nPer-sheet available points:")
    for _, row in summary.iterrows():
        print(f"  {row['annotator']} | {row['metric']}: n={int(row['n'])}, r={row['pearson_r']:.3f}, MAE={row['mae']:.3f}")
    if agreement_summary is not None and not agreement_summary.empty:
        print("\nBehavior score agreement (rejudge vs human):")
        for _, row in agreement_summary.iterrows():
            print(
                f"  {row['annotator']}: exact={row['exact_pct']:.1f}%, within_1pt={row['within_1pt_pct']:.1f}%, "
                f"same_sign={row['same_sign_pct']:.1f}%, opposite_sign={row['opposite_sign_pct']:.1f}%, MAE={row['mae']:.3f}"
            )
    if pairing_summary is not None and not pairing_summary.empty:
        print("\nHighest disagreement by pairing (per metric):")
        top = pairing_summary.groupby("metric", as_index=False).first()
        for _, row in top.iterrows():
            print(
                f"  {row['metric']}: {row['pairing']} "
                f"(MAE={row['mae']:.3f}, mean_delta={row['mean_delta']:.3f}, n={int(row['n'])})"
            )


if __name__ == "__main__":
    main()
