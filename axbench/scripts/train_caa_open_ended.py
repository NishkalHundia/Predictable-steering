"""
Train a CAA (DiffMean) vector on open-ended contrastive dataset.

Unlike the A/B multiple-choice version, this script:
- Uses full text responses (not single A/B tokens)
- Averages activations over ALL response tokens (not just one token)
- Computes: mean(positive_response_tokens) - mean(negative_response_tokens)

Usage:
    uv run python axbench/scripts/train_caa_open_ended.py \
        --model_name google/gemma-2-9b-it \
        --layer 20 \
        --dataset_path datasets/generated/sycophancy/train_contrastive.json \
        --output_dir results/gemma-2-9b-it/sycophancy-open-ended
"""
import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from axbench.utils.model_utils import get_prefix_length, gather_residual_activations
from axbench.utils.constants import CHAT_MODELS

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)


def supports_chat_template(tokenizer) -> bool:
    """Return True if tokenizer has a chat template configured."""
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def load_dataset(dataset_path):
    """Load the open-ended contrastive dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_training_data(data, tokenizer):
    """
    Prepare training data for open-ended DiffMean.
    
    Each item has:
    - question: the prompt
    - answer_matching_behavior: positive response (exhibits behavior)
    - answer_not_matching_behavior: negative response (neutral/original)
    
    We need to track where the response starts to only average over response tokens.
    """
    rows = []
    
    for item in data:
        question = item["question"]
        answer_matching = item["answer_matching_behavior"]
        answer_not_matching = item["answer_not_matching_behavior"]
        
        # Create positive example (matching behavior)
        rows.append({
            "question": question,
            "answer": answer_matching,
            "labels": 1  # positive class
        })
        
        # Create negative example (not matching behavior)
        rows.append({
            "question": question,
            "answer": answer_not_matching,
            "labels": 0  # negative class
        })
    
    df = pd.DataFrame(rows)
    
    # Apply chat template - question as user, answer as assistant (if available)
    def format_conversation(row):
        if supports_chat_template(tokenizer):
            # Chat models: use chat template
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
            # Full conversation
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Question only (to find where response starts)
            question_messages = [
                {"role": "user", "content": row["question"]},
            ]
            question_only = tokenizer.apply_chat_template(
                question_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Base models: simple text format
            # Use the same tokenizer on question_text and full_text so lengths line up
            question_only = row["question"]
            full_text = row["question"] + "\n\n" + row["answer"]
        
        return pd.Series({
            "full_text": full_text,
            "question_text": question_only
        })
    
    formatted = df.apply(format_conversation, axis=1)
    df["full_text"] = formatted["full_text"]
    df["question_text"] = formatted["question_text"]
    
    return df


@torch.no_grad()
def train_diffmean_open_ended(model, tokenizer, df, layer, device, batch_size=2):
    """
    Train DiffMean on open-ended responses.
    
    Averages activations over ALL response tokens (not just one token).
    This captures the full semantic content of matching vs non-matching responses.
    """
    model.eval()
    
    hidden_size = model.config.hidden_size
    positive_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    negative_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    positive_count = 0
    negative_count = 0
    
    logger.warning(f"Training DiffMean (open-ended) on {len(df)} examples...")
    logger.warning(f"Positive examples: {len(df[df['labels'] == 1])}")
    logger.warning(f"Negative examples: {len(df[df['labels'] == 0])}")
    logger.warning(f"Averaging over ALL response tokens")
    
    first_example_logged = False
    
    # Process one at a time to handle variable lengths properly
    for idx in tqdm(range(len(df)), desc="Collecting activations"):
        row = df.iloc[idx]
        
        # Tokenize full conversation
        full_inputs = tokenizer(
            row["full_text"],
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)
        
        # Tokenize question only to find where response starts
        question_inputs = tokenizer(
            row["question_text"],
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)
        
        question_length = question_inputs["input_ids"].shape[1]
        full_length = full_inputs["input_ids"].shape[1]
        
        # Sanity check for first example
        if not first_example_logged:
            logger.warning(f"\n=== SANITY CHECK ===")
            logger.warning(f"Question tokens: {question_length}")
            logger.warning(f"Full tokens: {full_length}")
            logger.warning(f"Response tokens: {full_length - question_length}")
            
            # Show last few tokens of question and first few of response
            logger.warning(f"\nLast 3 tokens of question:")
            for i in range(max(0, question_length-3), question_length):
                token_id = question_inputs["input_ids"][0, i].item()
                token_str = tokenizer.decode([token_id])
                logger.warning(f"  Position {i}: '{token_str}'")
            
            logger.warning(f"\nFirst 5 tokens of response:")
            for i in range(question_length, min(question_length + 5, full_length)):
                token_id = full_inputs["input_ids"][0, i].item()
                token_str = tokenizer.decode([token_id])
                logger.warning(f"  Position {i}: '{token_str}'")
            
            logger.warning(f"\nLabel: {row['labels']} ({'positive' if row['labels'] == 1 else 'negative'})")
            first_example_logged = True
        
        # Get activations at the target layer
        activations = gather_residual_activations(
            model, layer,
            {"input_ids": full_inputs["input_ids"], "attention_mask": full_inputs["attention_mask"]}
        ).detach().float()  # Shape: [1, seq_len, hidden_size]
        
        # Extract response tokens only (after question)
        response_activations = activations[0, question_length:full_length]  # [response_len, hidden_size]
        
        if response_activations.shape[0] > 0:
            # Average over all response tokens
            response_mean = response_activations.mean(dim=0)  # [hidden_size]
            
            if row["labels"] == 1:
                positive_sum += response_mean
                positive_count += 1
            else:
                negative_sum += response_mean
                negative_count += 1
        
        # Clear cache periodically
        if idx % 20 == 0:
            torch.cuda.empty_cache()
    
    # Compute mean activations
    logger.warning(f"\nComputing mean activations...")
    logger.warning(f"Total positive examples: {positive_count}")
    logger.warning(f"Total negative examples: {negative_count}")
    
    mean_positive = positive_sum / positive_count
    mean_negative = negative_sum / negative_count
    
    # Compute steering vector
    steering_vector = mean_positive - mean_negative
    
    logger.warning(f"Steering vector shape: {steering_vector.shape}")
    logger.warning(f"Steering vector norm: {steering_vector.norm().item():.4f}")
    logger.warning(f"Steering vector mean: {steering_vector.mean().item():.6f}")
    logger.warning(f"Steering vector std: {steering_vector.std().item():.6f}")
    
    return steering_vector


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to train_contrastive.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for variable length sequences)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=1024, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    logger.warning(f"Loaded {len(data)} contrastive pairs")
    
    df = prepare_training_data(data, tokenizer)
    logger.warning(f"Prepared {len(df)} training rows (positive + negative)")
    
    # Train DiffMean
    logger.warning("Training DiffMean (open-ended)...")
    steering_vector = train_diffmean_open_ended(
        model, tokenizer, df, args.layer, device, args.batch_size
    )
    
    # Save the steering vector
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
        "num_train_pairs": len(data),
        "seed": args.seed,
        "steering_vector_norm": steering_vector.norm().item(),
        "method": "open_ended_diffmean",
        "description": "Averaged over all response tokens"
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.warning("Training complete!")


if __name__ == "__main__":
    main()
