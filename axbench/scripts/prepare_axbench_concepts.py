"""
Prepare AxBench concept500 concepts for KL divergence analysis.

For each selected concept, this script:
1. Extracts positive/negative examples from the concept500 parquet
2. Computes DiffMean steering vector (mean_positive - mean_negative activations)
3. Saves the steering vector as DiffMean_weight.pt
4. Saves test prompts as JSON for eval_kl_divergence.py

Usage:
    # Specific concepts:
    uv run python axbench/scripts/prepare_axbench_concepts.py \
        --model_name google/gemma-2-9b-it \
        --layer 20 \
        --concept_ids 5,42,100 \
        --output_dir results/gemma-2-9b-it/axbench_concepts

    # Random selection:
    uv run python axbench/scripts/prepare_axbench_concepts.py \
        --model_name google/gemma-2-9b-it \
        --layer 20 \
        --num_random 5 \
        --output_dir results/gemma-2-9b-it/axbench_concepts
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from axbench.utils.constants import CHAT_MODELS, EMPTY_CONCEPT
from axbench.utils.model_utils import gather_residual_activations, get_prefix_length

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path):
    """Load metadata from a JSONL file."""
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def get_concept_data(all_df, metadata, concept_id):
    """Get positive/negative examples for a concept."""
    concept = metadata[concept_id]["concept"]
    genre_map = metadata[concept_id]["concept_genres_map"]
    genre = genre_map[concept][0]

    concept_df = all_df[all_df['concept_id'] == concept_id]
    positive_df = concept_df[
        (concept_df["output_concept"] == concept) & (concept_df["category"] == "positive")
    ]

    negative_df = all_df[
        (all_df["output_concept"] == EMPTY_CONCEPT) & (all_df["category"] == "negative")
    ]
    negative_df = negative_df[negative_df["concept_genre"] == genre]

    return positive_df, negative_df, concept


@torch.no_grad()
def compute_diffmean_vector(model, tokenizer, positive_df, negative_df, layer, device, is_chat_model):
    """
    Compute DiffMean steering vector from AxBench concept data.
    Averages activations over output tokens for positive and negative examples.
    Returns the steering vector.
    """
    hidden_size = model.config.hidden_size
    positive_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    negative_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    positive_count = 0
    negative_count = 0

    all_examples = pd.concat([
        positive_df.assign(label=1),
        negative_df.assign(label=0),
    ], ignore_index=True)

    for row in tqdm(all_examples.itertuples(), total=len(all_examples), desc="Collecting activations"):
        prompt = row.input
        completion = row.output or ""

        if is_chat_model:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            prompt_only = [{"role": "user", "content": prompt}]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_only, tokenize=True, add_generation_prompt=True
            )
        else:
            full_text = prompt + "\n\n" + completion
            full_ids = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(prompt)

        output_len = len(full_ids) - len(prompt_ids)
        if output_len <= 0:
            continue

        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        inputs = {"input_ids": full_tensor, "attention_mask": torch.ones_like(full_tensor)}

        activations = gather_residual_activations(
            model, layer, inputs
        ).squeeze(0).detach().float()  # [seq_len, hidden_size]

        # Average over output tokens only
        output_activations = activations[-output_len:]
        mean_act = output_activations.mean(dim=0)  # [hidden_size]

        if row.label == 1:
            positive_sum += mean_act
            positive_count += 1
        else:
            negative_sum += mean_act
            negative_count += 1

    if positive_count == 0 or negative_count == 0:
        raise ValueError(f"Not enough examples: pos={positive_count}, neg={negative_count}")

    mean_positive = positive_sum / positive_count
    mean_negative = negative_sum / negative_count
    steering_vector = mean_positive - mean_negative

    logger.warning(f"  Pos examples: {positive_count}, Neg examples: {negative_count}")
    logger.warning(f"  Steering vector norm: {steering_vector.norm().item():.4f}")

    return steering_vector


def create_test_json(positive_df, max_test=50):
    """
    Create test JSON from positive examples' input prompts.
    Format: [{"question": "..."}, ...]
    """
    prompts = positive_df["input"].tolist()
    if len(prompts) > max_test:
        prompts = prompts[:max_test]
    return [{"question": p} for p in prompts]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Prepare AxBench concepts for KL divergence analysis."
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--train_parquet", type=str,
                        default="axbench/concept500/prod_9b_l20_v1/generate/train_data.parquet",
                        help="Path to concept500 training parquet")
    parser.add_argument("--metadata_jsonl", type=str,
                        default="axbench/concept500/prod_9b_l20_v1/generate/metadata.jsonl",
                        help="Path to concept500 metadata JSONL")
    parser.add_argument("--concept_ids", type=str, default=None,
                        help="Comma-separated concept IDs to process")
    parser.add_argument("--num_random", type=int, default=None,
                        help="Number of random concepts to select (alternative to --concept_ids)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    logger.warning(f"Loading data from {args.train_parquet}...")
    train_df = pd.read_parquet(args.train_parquet)
    metadata = load_metadata(args.metadata_jsonl)
    logger.warning(f"Loaded {len(train_df)} rows, {len(metadata)} concepts")

    # Select concept IDs
    if args.concept_ids:
        concept_ids = [int(x.strip()) for x in args.concept_ids.split(",")]
    elif args.num_random:
        all_ids = sorted(train_df[train_df['concept_id'] >= 0]['concept_id'].unique())
        np.random.seed(args.seed)
        concept_ids = sorted(np.random.choice(all_ids, size=min(args.num_random, len(all_ids)), replace=False).tolist())
    else:
        print("ERROR: Must provide --concept_ids or --num_random")
        sys.exit(1)

    logger.warning(f"Selected concept IDs: {concept_ids}")
    for cid in concept_ids:
        logger.warning(f"  [{cid}] {metadata[cid]['concept']}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Loading model {args.model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_chat_model = args.model_name in CHAT_MODELS

    # Process each concept
    output_dir = Path(args.output_dir)
    summary = []

    for concept_id in concept_ids:
        concept_name = metadata[concept_id]["concept"]
        # Sanitize for directory name
        safe_name = concept_name[:50].replace(" ", "_").replace("/", "-")
        concept_dir = output_dir / f"concept_{concept_id}_{safe_name}"
        concept_dir.mkdir(parents=True, exist_ok=True)

        logger.warning(f"\n{'='*60}")
        logger.warning(f"Processing concept {concept_id}: {concept_name}")
        logger.warning(f"{'='*60}")

        try:
            positive_df, negative_df, concept = get_concept_data(train_df, metadata, concept_id)
        except Exception as e:
            logger.error(f"  Failed to get data for concept {concept_id}: {e}")
            continue

        if positive_df.empty or negative_df.empty:
            logger.warning(f"  Skipping: pos={len(positive_df)}, neg={len(negative_df)}")
            continue

        logger.warning(f"  Positive examples: {len(positive_df)}, Negative examples: {len(negative_df)}")

        # Compute DiffMean steering vector
        try:
            steering_vector = compute_diffmean_vector(
                model, tokenizer, positive_df, negative_df, args.layer, device, is_chat_model
            )
        except Exception as e:
            logger.error(f"  Failed to compute DiffMean: {e}")
            continue

        # Save steering vector (same format as train_caa_open_ended.py)
        weight_path = concept_dir / "DiffMean_weight.pt"
        bias_path = concept_dir / "DiffMean_bias.pt"
        torch.save(steering_vector.unsqueeze(0).cpu(), weight_path)
        torch.save(torch.zeros(1), bias_path)
        logger.warning(f"  Saved steering vector to {weight_path}")

        # Create and save test JSON
        test_data = create_test_json(positive_df)
        test_path = concept_dir / "test_prompts.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        logger.warning(f"  Saved {len(test_data)} test prompts to {test_path}")

        # Save concept config
        config = {
            "concept_id": concept_id,
            "concept_name": concept_name,
            "model_name": args.model_name,
            "layer": args.layer,
            "n_positive": len(positive_df),
            "n_negative": len(negative_df),
            "n_test_prompts": len(test_data),
            "steering_vector_norm": steering_vector.norm().item(),
            "ref": metadata[concept_id].get("ref", ""),
        }
        with open(concept_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        summary.append(config)

        # Clear cache
        torch.cuda.empty_cache()

    # Save overall summary
    summary_path = output_dir / "concepts_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("PREPARED AXBENCH CONCEPTS")
    print("=" * 70)
    for s in summary:
        print(f"  [{s['concept_id']:>3d}] {s['concept_name'][:60]:<60s}  ||v||={s['steering_vector_norm']:.2f}")
    print(f"\nOutput: {output_dir}")
    print(f"Total concepts prepared: {len(summary)}")

    # Print next-step commands
    print("\n" + "=" * 70)
    print("NEXT STEPS — Run KL divergence for each concept:")
    print("=" * 70)
    for s in summary:
        cid = s['concept_id']
        safe_name = s['concept_name'][:50].replace(" ", "_").replace("/", "-")
        cdir = f"{output_dir}/concept_{cid}_{safe_name}"
        print(f'uv run python axbench/scripts/eval_kl_divergence.py '
              f'--behavior "axbench_concept_{cid}" '
              f'--model_name {args.model_name} '
              f'--layer {args.layer} '
              f'--steering_vector_path {cdir}/DiffMean_weight.pt '
              f'--test_dataset_path {cdir}/test_prompts.json '
              f'--output_dir {cdir}/kl_eval '
              f'--steering_factors "-500,-200,-100,-50,-10,-5,-2,-1,0,1,2,5,10,50,100,200,500" '
              f'--batch_size 12')
        print()


if __name__ == "__main__":
    main()
