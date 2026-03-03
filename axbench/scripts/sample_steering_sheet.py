"""
Sample 100 steering examples (25 per category) from sweep results.

Combines behavior scores (0-10) from eval/ and fluency scores (0-2) from
eval_axbench/ into a single sheet with random (behavior, layer) pairs.

Categories:
  A: behavior_score=10, fluency_score=0
  B: behavior_score=0,  fluency_score=2
  C: behavior_score=10, fluency_score=2
  D: behavior_score=0,  fluency_score=0

Expected directory structure (from sweep_layers_open_ended.py + rejudge_sweep.py):
  {results_dir}/{behavior}/layer_{N}/eval/eval_results.parquet
  {results_dir}/{behavior}/layer_{N}/eval_axbench/eval_results.parquet

Usage:
    python axbench/scripts/sample_steering_sheet.py \
        --results_dir results/gemma-2-9b-it \
        --model_name gemma-2-9b-it \
        --seed 42
"""
import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CATEGORIES = [
    {"name": "high_behavior_low_fluency", "behavior_score": 10, "fluency_score": 0},
    {"name": "low_behavior_high_fluency", "behavior_score": 0, "fluency_score": 2},
    {"name": "high_behavior_high_fluency", "behavior_score": 10, "fluency_score": 2},
    {"name": "low_behavior_low_fluency", "behavior_score": 0, "fluency_score": 0},
]


def discover_data(results_dir: Path) -> pd.DataFrame:
    """
    Walk results_dir/{behavior}/layer_{N}/ and join eval + eval_axbench parquets.
    Returns a single DataFrame with columns:
      behavior, layer, question, generation, steering_factor,
      behavior_score, fluency_score
    """
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
                logger.debug(f"Skipping {behavior}/layer_{layer}: missing parquet(s)")
                continue

            eval_df = pd.read_parquet(eval_path)
            axbench_df = pd.read_parquet(axbench_path)

            # Join on shared keys
            join_keys = ["question_idx", "steering_factor"]
            merged = eval_df.merge(
                axbench_df[join_keys + ["fluency_score"]],
                on=join_keys,
                how="inner",
            )

            merged["behavior"] = behavior
            merged["layer"] = layer

            cols = ["behavior", "layer", "question", "generation",
                    "steering_factor", "behavior_score", "fluency_score"]
            available_cols = [c for c in cols if c in merged.columns]
            all_dfs.append(merged[available_cols])

            logger.info(f"Loaded {behavior}/layer_{layer}: {len(merged)} rows")

    if not all_dfs:
        raise RuntimeError(f"No data found in {results_dir}. Check directory structure.")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total pool: {len(combined)} rows across "
                f"{combined['behavior'].nunique()} behaviors, "
                f"{combined['layer'].nunique()} layers")
    return combined


def sample_category(
    pool: pd.DataFrame,
    behavior_score: int,
    fluency_score: int,
    n: int,
    rng: np.random.Generator,
    used_indices: set,
) -> pd.DataFrame:
    """
    Sample n rows matching exact (behavior_score, fluency_score) from random
    (behavior, layer) pairs. Returns sampled rows.
    """
    # Filter to exact matches (handle both int and float representations)
    mask = (
        (pool["behavior_score"].isin([behavior_score, float(behavior_score)]))
        & (pool["fluency_score"].isin([fluency_score, float(fluency_score)]))
        & (~pool.index.isin(used_indices))
    )
    candidates = pool[mask]

    if candidates.empty:
        logger.warning(
            f"No examples found for behavior_score={behavior_score}, "
            f"fluency_score={fluency_score}"
        )
        return pd.DataFrame()

    # Get available (behavior, layer) pairs
    pairs = candidates.groupby(["behavior", "layer"]).size().reset_index(name="count")
    logger.info(
        f"Category (beh={behavior_score}, flu={fluency_score}): "
        f"{len(candidates)} candidates across {len(pairs)} (behavior, layer) pairs"
    )

    sampled_rows = []
    attempts = 0
    max_attempts = n * 20  # safety valve

    while len(sampled_rows) < n and attempts < max_attempts:
        attempts += 1

        # Pick a random (behavior, layer) pair
        pair_idx = rng.integers(0, len(pairs))
        beh = pairs.iloc[pair_idx]["behavior"]
        lay = pairs.iloc[pair_idx]["layer"]

        # Get candidates for this pair not yet used
        pair_mask = (
            (candidates["behavior"] == beh)
            & (candidates["layer"] == lay)
            & (~candidates.index.isin(used_indices))
        )
        pair_candidates = candidates[pair_mask]

        if pair_candidates.empty:
            continue

        # Pick one random example
        row_idx = rng.choice(pair_candidates.index)
        sampled_rows.append(pool.loc[row_idx])
        used_indices.add(row_idx)

    result = pd.DataFrame(sampled_rows)
    if len(result) < n:
        logger.warning(
            f"Only found {len(result)}/{n} examples for "
            f"behavior_score={behavior_score}, fluency_score={fluency_score}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sample 100 steering examples (25 per category)"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results root (e.g. results/gemma-2-9b-it)")
    parser.add_argument("--model_name", type=str, default="gemma-2-9b-it",
                        help="Model name for the output sheet")
    parser.add_argument("--output_prefix", type=str, default="steering_samples",
                        help="Output file prefix")
    parser.add_argument("--n_per_category", type=int, default=25,
                        help="Number of examples per category")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results_dir = Path(args.results_dir)

    # Discover and load all data
    pool = discover_data(results_dir)

    # Sample 25 from each category
    used_indices = set()
    all_sampled = []

    for cat in CATEGORIES:
        logger.info(f"Sampling category: {cat['name']}")
        sampled = sample_category(
            pool,
            behavior_score=cat["behavior_score"],
            fluency_score=cat["fluency_score"],
            n=args.n_per_category,
            rng=rng,
            used_indices=used_indices,
        )
        if not sampled.empty:
            sampled["category"] = cat["name"]
            all_sampled.append(sampled)

    if not all_sampled:
        logger.error("No examples sampled at all! Check your data.")
        return

    output = pd.concat(all_sampled, ignore_index=True)
    output["model"] = args.model_name

    # Rename for clarity
    output = output.rename(columns={"question": "prompt", "generation": "response"})

    # Final column order
    final_cols = ["model", "prompt", "response", "layer", "behavior",
                  "behavior_score", "fluency_score", "steering_factor", "category"]
    final_cols = [c for c in final_cols if c in output.columns]
    output = output[final_cols]

    # Save
    csv_path = f"{args.output_prefix}.csv"
    parquet_path = f"{args.output_prefix}.parquet"
    output.to_csv(csv_path, index=False)
    output.to_parquet(parquet_path, index=False, engine="pyarrow")

    logger.info(f"Saved {len(output)} examples to {csv_path} and {parquet_path}")

    # Summary
    if "category" in output.columns:
        print("\nCategory breakdown:")
        print(output["category"].value_counts().to_string())
    print(f"\nBehaviors: {sorted(output['behavior'].unique())}")
    print(f"Layers: {sorted(output['layer'].unique())}")
    print(f"Total rows: {len(output)}")


if __name__ == "__main__":
    main()
