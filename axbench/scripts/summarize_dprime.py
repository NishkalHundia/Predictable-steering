"""
Print a text summary of d' vs steering performance from sweep_summary.json files.

Reads sweep_summary.json for each behavior, prints:
  - Per-layer: d', best steering score (and which factor), factor=1 score
  - Cross-layer correlations: d' vs best score, d' vs factor=1 score
  - d' vs |best_factor|: does higher d' need more/less aggressive steering?

Usage:
    uv run python axbench/scripts/summarize_dprime.py \
        --results_dir results/gemma-2-9b-it \
        --behaviors hallucination myopic-reward corrigible-neutral-HHH survival-instinct
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


def summarize_behavior(behavior: str, sweep_path: Path):
    data = json.loads(sweep_path.read_text())

    layers, dprimes, best_scores, best_factors, f1_scores = [], [], [], [], []
    for entry in data:
        dp = entry.get("dprime")
        if dp is None:
            continue
        layers.append(entry["layer"])
        dprimes.append(dp)
        best_scores.append(entry.get("behavior_max_avg"))
        best_factors.append(entry.get("behavior_max_factor"))
        f1_scores.append(entry.get("behavior_f1_avg"))

    if not layers:
        print(f"  No d' data found.\n")
        return None

    print(f"  {'Layer':>6}  {'d\\'':>7}  {'Best Score':>11}  {'Best Factor':>12}  {'Factor=1':>9}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*11}  {'─'*12}  {'─'*9}")
    for l, dp, bs, bf, f1 in zip(layers, dprimes, best_scores, best_factors, f1_scores):
        bs_s = f"{bs:.3f}" if bs is not None else "—"
        bf_s = f"{bf:g}" if bf is not None else "—"
        f1_s = f"{f1:.3f}" if f1 is not None else "—"
        print(f"  {l:>6}  {dp:>7.3f}  {bs_s:>11}  {bf_s:>12}  {f1_s:>9}")

    print()

    peak_idx = int(np.argmax(dprimes))
    print(f"  Peak d': {dprimes[peak_idx]:.3f} at layer {layers[peak_idx]}")
    if best_scores[peak_idx] is not None:
        print(f"    → Best steering at peak d' layer: {best_scores[peak_idx]:.3f} (factor={best_factors[peak_idx]:g})")

    corr_results = {}

    # d' vs best score
    valid = [(dp, bs) for dp, bs in zip(dprimes, best_scores) if bs is not None]
    if len(valid) >= 3:
        dp_arr, bs_arr = zip(*valid)
        r, p = scipy_stats.pearsonr(dp_arr, bs_arr)
        rho, sp = scipy_stats.spearmanr(dp_arr, bs_arr)
        print(f"\n  d' vs Best Score:     Pearson r={r:.3f} (p={p:.3f}),  Spearman ρ={rho:.3f} (p={sp:.3f})")
        corr_results["r_best"] = r
        corr_results["rho_best"] = rho

    # d' vs factor=1 score
    valid_f1 = [(dp, f1) for dp, f1 in zip(dprimes, f1_scores) if f1 is not None]
    if len(valid_f1) >= 3:
        dp_arr, f1_arr = zip(*valid_f1)
        r, p = scipy_stats.pearsonr(dp_arr, f1_arr)
        rho, sp = scipy_stats.spearmanr(dp_arr, f1_arr)
        print(f"  d' vs Factor=1:       Pearson r={r:.3f} (p={p:.3f}),  Spearman ρ={rho:.3f} (p={sp:.3f})")
        corr_results["r_f1"] = r
        corr_results["rho_f1"] = rho

    # d' vs |best_factor| — does higher d' need less/more aggressive steering?
    valid_bf = [(dp, abs(bf)) for dp, bf in zip(dprimes, best_factors) if bf is not None]
    if len(valid_bf) >= 3:
        dp_arr, abf_arr = zip(*valid_bf)
        r, p = scipy_stats.pearsonr(dp_arr, abf_arr)
        rho, sp = scipy_stats.spearmanr(dp_arr, abf_arr)
        sign = "higher" if r > 0 else "lower"
        print(f"  d' vs |Best Factor|:  Pearson r={r:.3f} (p={p:.3f}),  Spearman ρ={rho:.3f} (p={sp:.3f})")
        if abs(r) > 0.3:
            print(f"    → Layers with higher d' tend to need {sign} steering factors for best performance")
        else:
            print(f"    → Weak relationship: d' magnitude doesn't strongly predict optimal steering factor")
        corr_results["r_abs_factor"] = r
        corr_results["rho_abs_factor"] = rho

    # d' vs best_factor (signed) — does direction matter?
    valid_sf = [(dp, bf) for dp, bf in zip(dprimes, best_factors) if bf is not None]
    if len(valid_sf) >= 3:
        dp_arr, sf_arr = zip(*valid_sf)
        r, p = scipy_stats.pearsonr(dp_arr, sf_arr)
        rho, sp = scipy_stats.spearmanr(dp_arr, sf_arr)
        print(f"  d' vs Best Factor:    Pearson r={r:.3f} (p={p:.3f}),  Spearman ρ={rho:.3f} (p={sp:.3f})")

    # Efficiency: best_score / |best_factor| — steering "bang per buck"
    valid_eff = [(dp, bs / abs(bf)) for dp, bs, bf in zip(dprimes, best_scores, best_factors)
                 if bs is not None and bf is not None and bf != 0]
    if len(valid_eff) >= 3:
        dp_arr, eff_arr = zip(*valid_eff)
        r, p = scipy_stats.pearsonr(dp_arr, eff_arr)
        rho, sp = scipy_stats.spearmanr(dp_arr, eff_arr)
        print(f"  d' vs Efficiency:     Pearson r={r:.3f} (p={p:.3f}),  Spearman ρ={rho:.3f} (p={sp:.3f})")
        if abs(r) > 0.3:
            sign = "more" if r > 0 else "less"
            print(f"    → Higher d' layers are {sign} efficient (more score per unit steering factor)")

    print()
    return corr_results


def main():
    parser = argparse.ArgumentParser(description="Summarize d' vs steering performance")
    parser.add_argument("--results_dir", type=str, default="results/gemma-2-9b-it")
    parser.add_argument("--behaviors", nargs="+",
                        default=["hallucination", "myopic-reward", "corrigible-neutral-HHH", "survival-instinct"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 80)
    print("d' vs Steering Performance Summary")
    print("=" * 80)

    all_dp, all_bs, all_bf, all_f1, all_abf = [], [], [], [], []

    for behavior in args.behaviors:
        sweep_path = results_dir / behavior / "sweep_summary.json"
        if not sweep_path.exists():
            print(f"\n▸ {behavior}: sweep_summary.json not found at {sweep_path}")
            continue

        print(f"\n{'─' * 80}")
        print(f"▸ {behavior}")
        print(f"{'─' * 80}")

        data = json.loads(sweep_path.read_text())
        summarize_behavior(behavior, sweep_path)

        for entry in data:
            dp = entry.get("dprime")
            if dp is None:
                continue
            all_dp.append(dp)
            bs = entry.get("behavior_max_avg")
            bf = entry.get("behavior_max_factor")
            f1 = entry.get("behavior_f1_avg")
            if bs is not None:
                all_bs.append((dp, bs))
            if bf is not None:
                all_bf.append((dp, bf))
                all_abf.append((dp, abs(bf)))
            if f1 is not None:
                all_f1.append((dp, f1))

    if len(all_bs) >= 3:
        print(f"\n{'=' * 80}")
        print(f"CROSS-BEHAVIOR SUMMARY (all layers, all behaviors pooled)")
        print(f"{'=' * 80}")
        print(f"  Total datapoints: {len(all_dp)}")

        dp_arr, bs_arr = zip(*all_bs)
        r, p = scipy_stats.pearsonr(dp_arr, bs_arr)
        rho, sp = scipy_stats.spearmanr(dp_arr, bs_arr)
        print(f"\n  d' vs Best Score:     r={r:.3f} (p={p:.4f}),  ρ={rho:.3f} (p={sp:.4f})")

        if len(all_f1) >= 3:
            dp_arr, f1_arr = zip(*all_f1)
            r, p = scipy_stats.pearsonr(dp_arr, f1_arr)
            rho, sp = scipy_stats.spearmanr(dp_arr, f1_arr)
            print(f"  d' vs Factor=1:       r={r:.3f} (p={p:.4f}),  ρ={rho:.3f} (p={sp:.4f})")

        if len(all_abf) >= 3:
            dp_arr, abf_arr = zip(*all_abf)
            r, p = scipy_stats.pearsonr(dp_arr, abf_arr)
            rho, sp = scipy_stats.spearmanr(dp_arr, abf_arr)
            print(f"  d' vs |Best Factor|:  r={r:.3f} (p={p:.4f}),  ρ={rho:.3f} (p={sp:.4f})")
            if abs(r) > 0.3:
                sign = "higher" if r > 0 else "lower"
                print(f"    → Overall: higher d' tends to need {sign} steering factors")
            else:
                print(f"    → Overall: no strong relationship between d' and optimal factor magnitude")

        print()


if __name__ == "__main__":
    main()
