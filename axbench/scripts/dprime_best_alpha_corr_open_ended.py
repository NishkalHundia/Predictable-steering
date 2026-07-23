"""
d' vs best-α metrics correlation for the OPEN-ENDED projection-link sweep.

Replicates mcqa_projection_link.py's methodology EXACTLY, with the one
substitution the open-ended regime requires:

  MCQA is inherently BINARY — the greedily generated letter is either the correct
  option or not — so its per-layer target is *greedy accuracy* (steered_acc_{α}).
  Open-ended generation is NOT binary, so we swap accuracy → **match_rate** =
  fraction of fluent, labelled prompts whose LM-judge score exceeds the behavior
  threshold (i.e. the response "displays the behavior").

MCQA method  (mcqa_projection_link.py:1408-1429):
  1. best α per layer = argmax over factors of steered_acc_{α}   (ONE α per layer)
  2. accuracy@bestα   = steered_acc at that α          (== best_steered_acc)
  3. MCC@bestα        = MCC(sign(κ), steered_correct) at that SAME α
  4. correlate d' vs accuracy@bestα  and  d' vs MCC@bestα  across layers

Open-ended analog (this script):
  1. best α per layer = argmax over factors of match_rate_{α}    (ONE α per layer)
  2. accuracy@bestα   = match_rate_{α*}                (== best_match_rate)
  3. MCC@bestα        = steered_sign_mcc_{α*}          (sign(κ) vs behavior label, SAME α*)
  4. correlate d' vs match_rate@bestα  and  d' vs MCC@bestα  across layers

This is the "single shared best α" definition: ONE α is chosen per layer (by
match rate), and BOTH the accuracy and the MCC are read off at that same α —
exactly as MCQA reads MCC at its accuracy-chosen best α. (Contrast: taking each
metric at its own argmax α, which is a different, looser definition.)

Input: per_layer_summary.csv produced by open_ended_projection_link.py — it
already carries `dprime`, `match_rate_{α}`, and `steered_sign_mcc_{α}` columns.
No GPU, no model, no re-generation: this is pure post-hoc arithmetic on the CSV.

Usage:
    python axbench/scripts/dprime_best_alpha_corr_open_ended.py \
        --root /path/with/<behavior>/per_layer_summary.csv \
        --behaviors myopic-reward survival-instinct hallucination corrigible-neutral-HHH
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# Non-zero steering factors present in the sweep (α=0 is the unsteered baseline).
DEFAULT_FACTORS = [1.0, 2.0, 3.0, 5.0, 10.0]
DEFAULT_BEHAVIORS = [
    "myopic-reward", "survival-instinct", "hallucination", "corrigible-neutral-HHH",
]


def best_alpha_metrics(df: pd.DataFrame, factors):
    """Per layer, pick ONE best α = argmax match_rate over factors, then read the
    accuracy (match_rate) AND the sign-MCC at that same α. Mirrors MCQA's
    idxmax-on-accuracy + MCC-at-that-α (mcqa_projection_link.py:1411-1429).

    Returns (best_alpha[str], acc_at_best[float], mcc_at_best[float]) as arrays
    aligned with df's rows. Layers whose match_rate is all-NaN yield NaN.
    """
    mr_cols = [f"match_rate_{f:g}" for f in factors if f"match_rate_{f:g}" in df.columns]
    mr = df[mr_cols].to_numpy(dtype=float)                       # [n_layers, n_factors]
    fac_of_col = [c.replace("match_rate_", "") for c in mr_cols]

    best_alpha = np.full(len(df), None, dtype=object)
    acc_at_best = np.full(len(df), np.nan, dtype=float)
    mcc_at_best = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        row = mr[i]
        if np.all(np.isnan(row)):
            continue
        j = int(np.nanargmax(row))                              # accuracy-chosen α
        a = fac_of_col[j]
        best_alpha[i] = a
        acc_at_best[i] = row[j]
        mcc_col = f"steered_sign_mcc_{a}"                        # MCC at that SAME α
        if mcc_col in df.columns:
            mcc_at_best[i] = float(df.iloc[i][mcc_col])
    return best_alpha, acc_at_best, mcc_at_best


def corr(x, y):
    """Pearson + Spearman across layers, dropping NaN and zero-variance (mirrors
    the safe_* guards used in both projection-link scripts)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
        return dict(r=np.nan, p=np.nan, rho=np.nan, rho_p=np.nan, n=int(m.sum()))
    r, p = stats.pearsonr(x[m], y[m])
    rho, rho_p = stats.spearmanr(x[m], y[m])
    return dict(r=float(r), p=float(p), rho=float(rho), rho_p=float(rho_p), n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Dir containing <behavior>/per_layer_summary.csv")
    ap.add_argument("--behaviors", nargs="+", default=DEFAULT_BEHAVIORS)
    ap.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    ap.add_argument("--out_csv", default=None,
                    help="Optional path to write the per-behavior r-values table")
    args = ap.parse_args()

    rows = []
    for b in args.behaviors:
        csv = os.path.join(args.root, b, "per_layer_summary.csv")
        if not os.path.exists(csv):
            print(f"!! {b}: missing {csv}")
            continue
        df = pd.read_csv(csv).sort_values("layer").reset_index(drop=True)
        _, acc_at_best, mcc_at_best = best_alpha_metrics(df, args.factors)
        dprime = df["dprime"].to_numpy(float)

        c_acc = corr(dprime, acc_at_best)   # d' vs accuracy(match_rate) @ best α
        c_mcc = corr(dprime, mcc_at_best)   # d' vs sign-MCC             @ best α
        rows.append({
            "behavior": b,
            "n_layers": len(df),
            "r_dprime_vs_acc":  c_acc["r"],  "p_acc":  c_acc["p"],
            "rho_dprime_vs_acc": c_acc["rho"], "n_acc": c_acc["n"],
            "r_dprime_vs_mcc":  c_mcc["r"],  "p_mcc":  c_mcc["p"],
            "rho_dprime_vs_mcc": c_mcc["rho"], "n_mcc": c_mcc["n"],
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\nSingle shared best α (α chosen per layer by match_rate; accuracy = match_rate).")
    print("d' vs accuracy@bestα  and  d' vs sign-MCC@bestα  — Pearson r (across layers):\n")
    print(f"{'behavior':24s} {'r(d,acc)':>10s} {'p':>8s} | {'r(d,MCC)':>10s} {'p':>8s}   n")
    print("-" * 72)
    for r in rows:
        print(f"{r['behavior']:24s} "
              f"{r['r_dprime_vs_acc']:+.3f} {r['p_acc']:8.2g} | "
              f"{r['r_dprime_vs_mcc']:+.3f} {r['p_mcc']:8.2g}   "
              f"{r['n_acc']}/{r['n_mcc']}")

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
