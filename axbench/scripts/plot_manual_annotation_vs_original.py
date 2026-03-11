"""
Plot original LM-judge scores vs human annotation scores from manual_annotation.xlsx.

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
    "swastik_anno",
    "navita_anno",
    "nishkal_anno",
    "sarah_anno",
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


def _prepare_metric_pairs(
    original_df: pd.DataFrame,
    human_df: pd.DataFrame,
    human_sheet: str,
) -> list[pd.DataFrame]:
    usable_keys = [k for k in KEY_COLUMNS if k in original_df.columns and k in human_df.columns]
    if not usable_keys:
        raise ValueError(
            f"Cannot align sheet '{human_sheet}' with 'original': no shared key columns among {KEY_COLUMNS}."
        )

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

        original_keep = usable_keys + [orig_metric]
        human_keep = usable_keys + [human_metric]
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

        merged = original_slice.merge(human_slice, on=usable_keys, how="inner", suffixes=("_original", "_human"))
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
    b_round = b.round().astype("Int64")
    f_round = f.round().astype("Int64")

    derived = labels.copy()
    derived[(b_round == 10) & (f_round == 0)] = "high_behavior_low_fluency"
    derived[(b_round == 0) & (f_round == 2)] = "low_behavior_high_fluency"
    derived[(b_round == 10) & (f_round == 2)] = "high_behavior_high_fluency"
    derived[(b_round == 0) & (f_round == 0)] = "low_behavior_low_fluency"

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

        max_score = float(
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

        axis_upper = 10.5 if max_score > 2.5 else 2.2
        ax.plot([0, axis_upper], [0, axis_upper], linestyle="--", linewidth=1.2, color="black", alpha=0.7)
        ax.set_xlim(0, axis_upper)
        ax.set_ylim(0, axis_upper)
        ax.set_xlabel("Original LM judge score")
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

        ax.scatter(sdf["original_mean"], y, s=60, label="Original mean", color="#1f77b4", zorder=3)
        ax.scatter(sdf["human_mean"], y, s=60, label="Human mean", color="#d62728", zorder=3)

        for i, row in sdf.iterrows():
            ax.text(
                max(row["original_mean"], row["human_mean"]) + 0.08,
                i,
                f"|delta|={abs(row['mean_delta']):.2f}, n={int(row['n'])}",
                va="center",
                fontsize=9,
            )

        max_score = max(float(sdf["original_mean"].max()), float(sdf["human_mean"].max()))
        x_upper = 10.5 if max_score > 2.5 else 2.2
        ax.set_xlim(-0.2, x_upper)
        ax.set_yticks(y)
        ax.set_yticklabels(sdf["pairing"])
        ax.invert_yaxis()
        ax.set_xlabel("Score")
        ax.set_title(f"{metric}: Original vs Human by pairing")
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
        default="original",
        help="Sheet name containing original LM judge scores.",
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

    out_csv = output_dir / "original_vs_human_pairs.csv"
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
    out_summary = output_dir / "original_vs_human_summary.csv"
    summary.to_csv(out_summary, index=False)

    out_plot = output_dir / "original_vs_human_bubble.png"
    _plot_pairs(all_pairs, out_plot)

    pairing_summary = None
    try:
        pairing_plot = output_dir / "pairing_disagreement_dumbbell.png"
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
