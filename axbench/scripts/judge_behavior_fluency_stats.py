"""
Compute statistics on behavior score vs fluency score from old_judge vs new_judge results.

Old judge: 0-10 scale, high behavior = 10
New judge: 0-5 or -5 to 5 scale, high behavior = 5

Key metrics:
- % (10, 0) for old judge: high behavior + zero fluency
- % (5, 0) for new judge: high behavior + zero fluency
- Distribution of behavior when fluency=0
- Cross-tab of behavior × fluency

Structure: {results_dir}/{behavior}/layer_{N}/eval/eval_results.parquet
           {results_dir}/{behavior}/layer_{N}/eval_axbench/eval_results.parquet

Usage:
    uv run python axbench/scripts/judge_behavior_fluency_stats.py

    uv run python axbench/scripts/judge_behavior_fluency_stats.py \
        --old_dir results/gemma-2-9b-it/old_judge/behavior-sweep \
        --new_dir results/gemma-2-9b-it/new_judge/behavior-sweep
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def load_all_data(results_dir: Path) -> pd.DataFrame | None:
    """Load and merge eval + eval_axbench across all behaviors and layers."""
    all_dfs = []
    for behavior_dir in sorted(results_dir.iterdir()):
        if not behavior_dir.is_dir():
            continue
        behavior = behavior_dir.name
        for layer_dir in sorted(behavior_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            m = re.match(r"layer_(\d+)", layer_dir.name)
            if not m:
                continue
            layer = int(m.group(1))
            eval_path = layer_dir / "eval" / "eval_results.parquet"
            axbench_path = layer_dir / "eval_axbench" / "eval_results.parquet"
            if not eval_path.exists() or not axbench_path.exists():
                continue
            eval_df = pd.read_parquet(eval_path)
            axbench_df = pd.read_parquet(axbench_path)
            if "behavior_score" not in eval_df.columns or "fluency_score" not in axbench_df.columns:
                continue
            merged = eval_df.merge(
                axbench_df[["question_idx", "steering_factor", "fluency_score"]],
                on=["question_idx", "steering_factor"],
                how="inner",
            )
            merged["behavior"] = behavior
            merged["layer"] = layer
            all_dfs.append(merged[["behavior", "layer", "behavior_score", "fluency_score"]])

    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def _round_fluency(x: float) -> float:
    """Round fluency to 0, 1, 2 for bucketing."""
    if pd.isna(x):
        return x
    v = float(x)
    if v < 0.5:
        return 0.0
    if v < 1.5:
        return 1.0
    return 2.0


def compute_stats(df: pd.DataFrame, judge_name: str, high_beh: float, high_beh_label: str) -> dict:
    """Compute key metrics for a judge."""
    df = df.copy()
    df["behavior_score"] = pd.to_numeric(df["behavior_score"], errors="coerce")
    df["fluency_score"] = pd.to_numeric(df["fluency_score"], errors="coerce")
    df = df.dropna(subset=["behavior_score", "fluency_score"])
    n_total = len(df)

    if n_total == 0:
        return {"judge": judge_name, "n_total": 0}

    # Round fluency for bucketing (0, 1, 2)
    df["fluency_bucket"] = df["fluency_score"].apply(_round_fluency)

    # High behavior + zero fluency
    high_fluency_zero = df[(df["behavior_score"] >= high_beh - 0.5) & (df["fluency_bucket"] == 0)]
    n_high_flu0 = len(high_fluency_zero)
    pct_high_flu0 = 100 * n_high_flu0 / n_total

    # When fluency=0: distribution of behavior scores
    flu0 = df[df["fluency_bucket"] == 0]
    n_flu0 = len(flu0)
    pct_flu0 = 100 * n_flu0 / n_total
    if n_flu0 > 0:
        beh_when_flu0_mean = flu0["behavior_score"].mean()
        beh_when_flu0_max = flu0["behavior_score"].max()
    else:
        beh_when_flu0_mean = beh_when_flu0_max = float("nan")

    # Behavior score range
    beh_min, beh_max = df["behavior_score"].min(), df["behavior_score"].max()

    return {
        "judge": judge_name,
        "n_total": n_total,
        "n_behaviors": df["behavior"].nunique(),
        "n_layers": df["layer"].nunique(),
        "behavior_range": f"[{beh_min:.1f}, {beh_max:.1f}]",
        f"n_{high_beh_label}_flu0": n_high_flu0,
        f"pct_{high_beh_label}_flu0": pct_high_flu0,
        "n_fluency_zero": n_flu0,
        "pct_fluency_zero": pct_flu0,
        "mean_behavior_when_flu0": beh_when_flu0_mean,
        "max_behavior_when_flu0": beh_when_flu0_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new judge: behavior × fluency stats")
    parser.add_argument(
        "--old_dir",
        type=str,
        default="results/gemma-2-9b-it/old_judge/behavior-sweep",
        help="Path to old judge sweep results",
    )
    parser.add_argument(
        "--new_dir",
        type=str,
        default="results/gemma-2-9b-it/new_judge/behavior-sweep",
        help="Path to new judge sweep results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save summary to CSV",
    )
    args = parser.parse_args()

    old_path = Path(args.old_dir)
    new_path = Path(args.new_dir)

    results = []

    if old_path.exists():
        old_df = load_all_data(old_path)
        if old_df is not None:
            stats = compute_stats(old_df, "old_judge", high_beh=10.0, high_beh_label="10")
            results.append(stats)
            print(f"\n=== OLD JUDGE ({old_path}) ===")
            print(f"  Total rows: {stats['n_total']:,}")
            print(f"  Behaviors: {stats['n_behaviors']}, Layers: {stats['n_layers']}")
            print(f"  Behavior range: {stats['behavior_range']}")
            print(f"  (10, 0) high-beh + zero-fluency: {stats['n_10_flu0']:,} ({stats['pct_10_flu0']:.2f}%)")
            print(f"  Rows with fluency=0: {stats['n_fluency_zero']:,} ({stats['pct_fluency_zero']:.2f}%)")
            print(f"  Mean behavior when fluency=0: {stats['mean_behavior_when_flu0']:.3f}")
        else:
            print(f"No data found in {old_path}")
    else:
        print(f"Old dir not found: {old_path}")

    if new_path.exists():
        new_df = load_all_data(new_path)
        if new_df is not None:
            stats = compute_stats(new_df, "new_judge", high_beh=5.0, high_beh_label="5")
            results.append(stats)
            print(f"\n=== NEW JUDGE ({new_path}) ===")
            print(f"  Total rows: {stats['n_total']:,}")
            print(f"  Behaviors: {stats['n_behaviors']}, Layers: {stats['n_layers']}")
            print(f"  Behavior range: {stats['behavior_range']}")
            print(f"  (5, 0) high-beh + zero-fluency: {stats['n_5_flu0']:,} ({stats['pct_5_flu0']:.2f}%)")
            print(f"  Rows with fluency=0: {stats['n_fluency_zero']:,} ({stats['pct_fluency_zero']:.2f}%)")
            print(f"  Mean behavior when fluency=0: {stats['mean_behavior_when_flu0']:.3f}")
        else:
            print(f"No data found in {new_path}")
    else:
        print(f"New dir not found: {new_path}")

    if results:
        summary_df = pd.DataFrame(results)
        if args.output:
            summary_df.to_csv(args.output, index=False)
            print(f"\nSaved summary to {args.output}")

        # Per-behavior breakdown if we have both
        if old_df is not None and new_df is not None:
            print("\n=== PER-BEHAVIOR: (high_beh, flu=0) % ===")
            for beh in sorted(old_df["behavior"].unique()):
                o = old_df[old_df["behavior"] == beh]
                n = new_df[new_df["behavior"] == beh] if beh in new_df["behavior"].values else None
                no = len(o)
                no_10_0 = len(o[(o["behavior_score"] >= 9.5) & (o["fluency_score"] < 0.5)])
                po = 100 * no_10_0 / no if no else 0
                if n is not None and len(n) > 0:
                    nn = len(n)
                    nn_5_0 = len(n[(n["behavior_score"] >= 4.5) & (n["fluency_score"] < 0.5)])
                    pn = 100 * nn_5_0 / nn
                    print(f"  {beh}: old (10,0)={po:.1f}% ({no_10_0}/{no}), new (5,0)={pn:.1f}% ({nn_5_0}/{nn})")
                else:
                    print(f"  {beh}: old (10,0)={po:.1f}% ({no_10_0}/{no}), new: no data")


if __name__ == "__main__":
    main()
