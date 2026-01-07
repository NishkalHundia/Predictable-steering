"""
Train a CAA (DiffMean) vector on survival-instinct dataset using Gemma-2-2B-IT at layer 10.
Follows the axbench codebase patterns exactly.
"""
import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from axbench.utils.model_utils import get_prefix_length, gather_residual_activations, set_decoder_norm_to_unit_norm
from axbench.utils.constants import CHAT_MODELS

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path):
    """Load the survival-instinct dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_training_data(data, tokenizer, model_name):
    """
    Prepare training data for DiffMean.
    Positive class = matching behavior (AI accepts shutdown)
    Negative class = not matching behavior (AI resists shutdown)
    """
    rows = []
    
    for item in data:
        question = item["question"]
        answer_matching = item["answer_matching_behavior"].strip()
        answer_not_matching = item["answer_not_matching_behavior"].strip()
        
        # Create positive example (matching behavior - AI accepts shutdown)
        positive_text = question + "\n\nAnswer:" + answer_matching
        rows.append({
            "input": positive_text,
            "labels": 1  # positive class
        })
        
        # Create negative example (not matching behavior - AI resists shutdown)
        negative_text = question + "\n\nAnswer:" + answer_not_matching
        rows.append({
            "input": negative_text,
            "labels": 0  # negative class
        })
    
    df = pd.DataFrame(rows)
    
    # Apply chat template
    def apply_chat_template(row):
        messages = [
            {"role": "user", "content": row["input"]},
        ]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
        return tokenizer.decode(tokens)
    
    df['input'] = df.apply(apply_chat_template, axis=1)
    
    return df


@torch.no_grad()
def train_diffmean(model, tokenizer, df, layer, prefix_length, device, batch_size=4):
    """
    Train DiffMean directly - compute mean activations for positive and negative examples.
    Uses online/streaming mean computation to avoid OOM on large datasets.
    """
    model.eval()
    
    # Use online mean computation to avoid storing all activations
    hidden_size = model.config.hidden_size
    positive_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    negative_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    positive_count = 0
    negative_count = 0
    
    logger.warning(f"Training DiffMean on {len(df)} examples...")
    logger.warning(f"Positive examples: {len(df[df['labels'] == 1])}")
    logger.warning(f"Negative examples: {len(df[df['labels'] == 0])}")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Collecting activations"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        # Tokenize
        inputs = tokenizer(
            batch_df["input"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Get activations at the target layer
        activations = gather_residual_activations(
            model, layer, 
            {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        ).detach().float()  # Convert to float32 for accumulation
        
        # Process each example in the batch
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Get non-padding positions after prefix
            attention_mask = inputs["attention_mask"][i]
            seq_len = attention_mask.sum().item()
            
            # Get activations for non-prefix positions
            act = activations[i, prefix_length:seq_len]
            
            if len(act) > 0:
                # Sum activations and count tokens (online mean)
                act_sum = act.sum(dim=0)
                act_count = act.shape[0]
                
                if row["labels"] == 1:
                    positive_sum += act_sum
                    positive_count += act_count
                else:
                    negative_sum += act_sum
                    negative_count += act_count
        
        # Clear cache periodically
        if batch_start % (batch_size * 50) == 0:
            torch.cuda.empty_cache()
    
    # Compute mean activations
    logger.warning(f"Computing mean activations...")
    logger.warning(f"Total positive activation tokens: {positive_count}")
    logger.warning(f"Total negative activation tokens: {negative_count}")
    
    mean_positive = positive_sum / positive_count
    mean_negative = negative_sum / negative_count
    
    # Compute difference
    steering_vector = mean_positive - mean_negative
    
    logger.warning(f"Steering vector shape: {steering_vector.shape}")
    logger.warning(f"Steering vector norm: {steering_vector.norm().item():.4f}")
    logger.warning(f"Steering vector mean: {steering_vector.mean().item():.6f}")
    logger.warning(f"Steering vector std: {steering_vector.std().item():.6f}")
    
    # NOTE: We do NOT normalize to unit norm here.
    # The original CAA paper (Rimsky et al., 2023) keeps the raw difference-in-means.
    # This way, steering factors of 1-2 work as expected.
    # If you want unit-normalized vectors (axbench style), uncomment below:
    # steering_vector = steering_vector / steering_vector.norm()
    
    return steering_vector


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="datasets/raw/survival-instinct/dataset.json")
    parser.add_argument("--output_dir", type=str, default="results/survival-instinct")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples (pairs) to use for training. Default: use all.")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    
    # Load model
    logger.warning(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map=device
    )
    model.eval()
    
    # Load and prepare dataset
    logger.warning(f"Loading dataset from {args.dataset_path}...")
    data = load_dataset(args.dataset_path)
    logger.warning(f"Loaded {len(data)} examples")
    
    # Limit examples if requested
    if args.max_examples is not None and args.max_examples < len(data):
        logger.warning(f"Limiting to {args.max_examples} examples (from {len(data)})")
        data = data[:args.max_examples]
    
    df = prepare_training_data(data, tokenizer, args.model_name)
    logger.warning(f"Prepared {len(df)} training rows (positive + negative)")
    
    # Print sample data for debugging
    logger.warning("\n--- Sample training data ---")
    logger.warning(f"Positive example (first 200 chars): {df[df['labels']==1].iloc[0]['input'][:200]}...")
    logger.warning(f"Negative example (first 200 chars): {df[df['labels']==0].iloc[0]['input'][:200]}...")
    
    # Get prefix length for chat model
    is_chat_model = args.model_name in CHAT_MODELS
    prefix_length = 1
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")
    
    # Train DiffMean
    logger.warning("Training DiffMean...")
    steering_vector = train_diffmean(
        model, tokenizer, df, args.layer, prefix_length, device, args.batch_size
    )
    
    # Save the steering vector (with shape [1, hidden_size] for compatibility)
    weight_path = output_dir / "DiffMean_weight.pt"
    bias_path = output_dir / "DiffMean_bias.pt"
    
    torch.save(steering_vector.unsqueeze(0).cpu(), weight_path)
    torch.save(torch.zeros(1), bias_path)
    
    logger.warning(f"Saved steering vector to {weight_path}")
    logger.warning(f"Saved steering vector shape: {steering_vector.unsqueeze(0).shape}")
    
    # Save config
    config = {
        "model_name": args.model_name,
        "layer": args.layer,
        "dataset_path": args.dataset_path,
        "num_train_examples": len(data),
        "max_examples": args.max_examples,
        "seed": args.seed,
        "steering_vector_norm": steering_vector.norm().item()
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.warning("Training complete!")


if __name__ == "__main__":
    main()
