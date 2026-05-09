"""
Reproduce Tables 1 and 2 from the paper from already-generated CSVs.

Table 1: Pearson correlation between per-layer training d' and per-layer test
         steering accuracy at α* (chosen on validation per layer).
Table 2: Pearson correlation between per-layer training d' and per-layer MCC
         of sign(κ_a) predicting steered correctness at the val-chosen α.

Inputs (per behavior):
    results/mcqa_projection_link/<model_short>/<behavior>/per_layer_summary.csv

Outputs:
    paper_plots/tables.md           (markdown rendering of both tables)
    paper_plots/tables.csv          (long-format machine-readable)

Usage:
    python scripts/make_paper_tables.py
    python scripts/make_paper_tables.py --results_dir results --out_dir paper_plots
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DEFAULT_MODEL_SHORT = "gemma-2-9b-it"
BEHAVIORS = [
    "sycophancy",
    "survival-instinct",
    "myopic-reward",
    "hallucination",
    "corrigible-neutral-HHH",
]
PRETTY = {
    "sycophancy":              "Sycophancy",
    "survival-instinct":       "Survival-instinct",
    "myopic-reward":           "Myopic-reward",
    "hallucination":           "Hallucination",
    "corrigible-neutral-HHH":  "Corrigibility",
}


def pearson(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan"), float("nan"), int(mask.sum())
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def fmt(r, p):
    if not np.isfinite(r):
        return "n/a"
    star = "*" if p < 0.05 else " "
    return f"{r:.3f}{star}".rstrip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_short", default=DEFAULT_MODEL_SHORT)
    parser.add_argument("--out_dir", default="paper_plots")
    args = parser.parse_args()

    base = Path(args.results_dir) / "mcqa_projection_link" / args.model_short
    rows = []
    table1, table2 = [], []
    for b in BEHAVIORS:
        per_layer_csv = base / b / "per_layer_summary.csv"
        if not per_layer_csv.exists():
            print(f"[skip] {b}: {per_layer_csv} missing")
            continue
        df = pd.read_csv(per_layer_csv).sort_values("layer").reset_index(drop=True)

        x_dprime = df["dprime"].values.astype(float)

        # Table 1: per-layer d' vs per-layer test acc @ val-chosen α.
        if "test_steered_acc_at_val_best_alpha" in df.columns:
            y_acc = df["test_steered_acc_at_val_best_alpha"].values.astype(float)
        else:
            y_acc = np.full(len(df), np.nan)
        r1, p1, n1 = pearson(x_dprime, y_acc)

        # Table 2: per-layer d' vs per-layer MCC @ val-chosen α (test prompts).
        if "sign_kappa_mcc_val_best_on_test" in df.columns:
            y_mcc = df["sign_kappa_mcc_val_best_on_test"].values.astype(float)
        else:
            y_mcc = np.full(len(df), np.nan)
        r2, p2, n2 = pearson(x_dprime, y_mcc)

        table1.append((PRETTY[b], r1, p1, n1))
        table2.append((PRETTY[b], r2, p2, n2))
        rows.append({
            "behavior": b,
            "pretty": PRETTY[b],
            "table1_pearson_r": r1, "table1_pearson_p": p1, "table1_n_layers": n1,
            "table2_pearson_r": r2, "table2_pearson_p": p2, "table2_n_layers": n2,
        })

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Long-format CSV.
    if rows:
        with open(out_root / "tables.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Markdown rendering, ordered as in the paper.
    md_lines = ["# Paper Tables\n"]

    md_lines.append("## Table 1 — Per-layer d' vs per-layer test steering accuracy")
    md_lines.append("")
    md_lines.append("| Dataset | Pearson r | n |")
    md_lines.append("|---|---|---|")
    for pretty, r, p, n in sorted(table1, key=lambda x: (-x[1] if np.isfinite(x[1]) else 1, x[0])):
        md_lines.append(f"| {pretty} | {fmt(r, p)} | {n} |")
    md_lines.append("")

    md_lines.append("## Table 2 — Per-layer d' vs per-layer MCC(sign κ_a, steer success)")
    md_lines.append("")
    md_lines.append("| Dataset | Pearson r | n |")
    md_lines.append("|---|---|---|")
    for pretty, r, p, n in sorted(table2, key=lambda x: (-x[1] if np.isfinite(x[1]) else 1, x[0])):
        md_lines.append(f"| {pretty} | {fmt(r, p)} | {n} |")
    md_lines.append("")
    md_lines.append("(*) denotes p < 0.05.")
    md_lines.append("")

    md_path = out_root / "tables.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}")
    print(f"Wrote {out_root / 'tables.csv'}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
