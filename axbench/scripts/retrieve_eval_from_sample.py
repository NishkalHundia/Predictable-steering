"""
Retrieve full eval entries for rows in a sample CSV produced by sample_steering_sheet.py.

Given a steering_samples CSV, loads the corresponding rows from the eval and eval_axbench
parquets, returning the full eval data (all columns) for those exact entries.

Usage:
    python axbench/scripts/retrieve_eval_from_sample.py \
        --sample_csv steering_samples.csv \
        --results_dir results/gemma-2-9b-it \
        --output retrieved_eval.parquet
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def discover_and_merge(results_dir: Path, behavior: str, layer: int) -> pd.DataFrame | None:
    """Load and merge eval + eval_axbench for a single (behavior, layer)."""
    layer_dir = results_dir / behavior / f"layer_{layer}"
    eval_path = layer_dir / "eval" / "eval_results.parquet"
    axbench_path = layer_dir / "eval_axbench" / "eval_results.parquet"

    if not eval_path.exists() or not axbench_path.exists():
        return None

    eval_df = pd.read_parquet(eval_path)
    axbench_df = pd.read_parquet(axbench_path)
    merged = eval_df.merge(
        axbench_df[["question_idx", "steering_factor", "fluency_score"]],
        on=["question_idx", "steering_factor"],
        how="inner",
    )
    merged["behavior"] = behavior
    merged["layer"] = layer
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve full eval entries matching a sample_steering_sheet CSV"
    )
    parser.add_argument("--sample_csv", type=str, required=True,
                        help="Path to CSV from sample_steering_sheet.py")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results root (e.g. results/gemma-2-9b-it)")
    parser.add_argument("--output", type=str, default="retrieved_eval.parquet",
                        help="Output parquet path")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional: also save as CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sample = pd.read_csv(args.sample_csv)

    # sample_steering_sheet renames question->prompt, generation->response
    if "prompt" in sample.columns:
        sample = sample.rename(columns={"prompt": "question", "response": "generation"})
    elif "question" not in sample.columns:
        raise ValueError("Sample CSV must have 'prompt'/'response' or 'question'/'generation'")

    # Match keys: behavior, layer, question, generation, steering_factor
    match_cols = ["behavior", "layer", "question", "generation", "steering_factor"]
    for c in match_cols:
        if c not in sample.columns:
            raise ValueError(f"Sample CSV missing column: {c}")

    retrieved = []
    seen = set()

    for (behavior, layer), group in sample.groupby(["behavior", "layer"]):
        merged = discover_and_merge(results_dir, str(behavior), int(layer))
        if merged is None:
            print(f"Warning: no data for {behavior}/layer_{layer}")
            continue

        for _, row in group.iterrows():
            q = str(row["question"]).strip() if pd.notna(row["question"]) else ""
            g = str(row["generation"]).strip() if pd.notna(row["generation"]) else ""
            sf = row["steering_factor"]
            mask = (
                (merged["question"].astype(str).str.strip() == q)
                & (merged["generation"].astype(str).str.strip() == g)
                & (merged["steering_factor"] == sf)
            )
            matches = merged[mask]
            if len(matches) == 0:
                print(f"Warning: no match for {behavior}/layer_{layer} steering_factor={row['steering_factor']}")
                continue
            if len(matches) > 1:
                matches = matches.head(1)
            key = (behavior, layer, row["question"], row["generation"], row["steering_factor"])
            if key in seen:
                continue
            seen.add(key)
            retrieved.append(matches.iloc[0])

    if not retrieved:
        raise RuntimeError("No matching entries found. Check paths and CSV columns.")

    out_df = pd.DataFrame(retrieved)
    out_df.to_parquet(args.output, index=False, engine="pyarrow")
    print(f"Saved {len(out_df)} rows to {args.output}")

    if args.output_csv:
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(out_df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
