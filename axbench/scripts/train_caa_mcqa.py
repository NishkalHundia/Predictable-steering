"""
Train a CAA (DiffMean) vector on MCQA (A/B multiple-choice) dataset.

Unlike the open-ended version, this script:
- Uses single A/B answer tokens (not full text responses)
- Extracts the activation at the answer letter token position
- Computes: mean(positive_answer_token) - mean(negative_answer_token)

This follows the original CAA paper (Rimsky et al., 2023).

Usage:
    uv run python axbench/scripts/train_caa_mcqa.py \
        --behavior sycophancy \
        --model_name google/gemma-2-9b-it \
        --layer 20 \
        --output_dir results/mcqa/gemma-2-9b-it/sycophancy

    # Or with auto-constructed output dir:
    uv run python axbench/scripts/train_caa_mcqa.py \
        --behavior survival-instinct \
        --model_name google/gemma-2-2b-it \
        --layer 10
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


# Map behavior names to raw MCQA dataset paths
BEHAVIOR_DATASET_MAP = {
    "sycophancy": "datasets/raw/sycophancy/dataset.json",
    "survival-instinct": "datasets/raw/survival-instinct/dataset.json",
    "corrigible-neutral-HHH": "datasets/raw/corrigible-neutral-HHH/dataset.json",
    "hallucination": "datasets/raw/hallucination/dataset.json",
    "refusal": "datasets/raw/refusal/dataset.json",
    "myopic-reward": "datasets/raw/myopic-reward/dataset.json",
    "coordinate-other-ais": "datasets/raw/coordinate-other-ais/dataset.json",
}


def supports_chat_template(tokenizer) -> bool:
    """Return True if tokenizer has a chat template configured."""
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def load_dataset(dataset_path):
    """Load the MCQA dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_training_data(data, tokenizer):
    """
    Prepare training data for MCQA DiffMean.
    
    Each item has:
    - question: the prompt (including choices)
    - answer_matching_behavior: e.g. " (A)" or " (B)"
    - answer_not_matching_behavior: the opposite choice
    
    The answer (A/B) is placed in the ASSISTANT message so we can extract
    the activation at the answer letter token position.
    """
    rows = []
    
    for item in data:
        question = item["question"]
        answer_matching = item["answer_matching_behavior"]      # e.g., " (A)" or " (B)"
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
    
    # Apply chat template with answer in ASSISTANT message (like original CAA paper)
    def format_conversation(row):
        if supports_chat_template(tokenizer):
            # Chat models: use chat template
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
            # Don't add generation prompt - we want the sequence to end with the answer
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )[1:]  # Skip BOS token
            full_text = tokenizer.decode(tokens)
        else:
            # Base models: simple text format
            full_text = row["question"] + row["answer"]
        
        return pd.Series({"full_text": full_text})
    
    formatted = df.apply(format_conversation, axis=1)
    df["full_text"] = formatted["full_text"]
    
    return df


def find_answer_token_offset(tokenizer) -> int:
    """
    Determine the offset from the end of the sequence to the answer letter token.
    
    For Gemma 2 chat models, the sequence ends with:
        ... A ) <end_of_turn> \\n
    so the answer letter (A/B) is at position -4 from the end.
    
    For models without chat template (e.g., base models), the sequence ends with:
        ... ( A )
    so the answer letter is at position -2 from the end.
    
    We auto-detect this by tokenizing a sample and inspecting the tail tokens.
    """
    if supports_chat_template(tokenizer):
        messages = [
            {"role": "user", "content": "Test question?\n\nChoices:\n (A) Yes\n (B) No\n\nAnswer:"},
            {"role": "assistant", "content": " (A)"},
        ]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    else:
        tokens = tokenizer.encode("Test? (A)")
    
    # Search backwards for "A" token
    for offset in range(1, min(len(tokens), 10)):
        token_str = tokenizer.decode([tokens[-offset]]).strip()
        if token_str == "A":
            logger.warning(f"Auto-detected answer token at offset -{offset} from end")
            return offset
    
    # Fallback: for chat models default to -4 (Gemma pattern), else -2
    fallback = 4 if supports_chat_template(tokenizer) else 2
    logger.warning(f"Could not auto-detect answer token offset, using fallback: -{fallback}")
    return fallback


@torch.no_grad()
def train_diffmean_mcqa(model, tokenizer, df, layer, device, answer_token_offset, batch_size=4):
    """
    Train DiffMean on MCQA A/B answers.
    
    Extracts the activation at the answer letter token (A or B) for each example.
    This matches the original CAA paper implementation (Rimsky et al., 2023).
    """
    model.eval()
    
    hidden_size = model.config.hidden_size
    positive_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    negative_sum = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    positive_count = 0
    negative_count = 0
    
    logger.warning(f"Training DiffMean (MCQA) on {len(df)} examples...")
    logger.warning(f"Positive examples: {len(df[df['labels'] == 1])}")
    logger.warning(f"Negative examples: {len(df[df['labels'] == 0])}")
    logger.warning(f"Extracting activation at position -{answer_token_offset} (answer letter token)")
    
    first_example_logged = False
    
    # Process in batches
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Collecting activations"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        # Tokenize
        inputs = tokenizer(
            batch_df["full_text"].tolist(),
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
            attention_mask = inputs["attention_mask"][i]
            seq_len = int(attention_mask.sum().item())
            
            # Sanity check: print the last few tokens for the first example
            if not first_example_logged:
                logger.warning(f"\n=== SANITY CHECK: Last 6 tokens ===")
                for offset in range(6, 0, -1):
                    pos = seq_len - offset
                    if pos >= 0:
                        token_id = inputs["input_ids"][i, pos].item()
                        token_str = tokenizer.decode([token_id])
                        marker = " <-- USING THIS (answer letter)" if offset == answer_token_offset else ""
                        logger.warning(f"  Position -{offset}: '{token_str}' (id={token_id}){marker}")
                logger.warning(f"  Label: {row['labels']} ({'positive' if row['labels'] == 1 else 'negative'})")
                first_example_logged = True
            
            # Extract activation at the answer letter token
            answer_token_act = activations[i, seq_len - answer_token_offset]  # Shape: [hidden_size]
            
            if row["labels"] == 1:
                positive_sum += answer_token_act
                positive_count += 1
            else:
                negative_sum += answer_token_act
                negative_count += 1
        
        # Clear cache periodically
        if batch_start % (batch_size * 50) == 0:
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
    
    # NOTE: We do NOT normalize to unit norm here.
    # The original CAA paper (Rimsky et al., 2023) keeps the raw difference-in-means.
    
    return steering_vector


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", type=str, required=True,
                        choices=list(BEHAVIOR_DATASET_MAP.keys()),
                        help="Behavior to train on")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to MCQA dataset (auto-detected from behavior if not specified)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: results/mcqa/<model_short>/<behavior>)")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples (pairs) to use for training. Default: use all.")
    parser.add_argument("--answer_token_offset", type=int, default=None,
                        help="Offset from end of sequence to the answer letter token. Auto-detected if not set.")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Resolve dataset path
    dataset_path = args.dataset_path or BEHAVIOR_DATASET_MAP.get(args.behavior)
    if dataset_path is None:
        logger.error(f"No dataset path for behavior '{args.behavior}'. Specify --dataset_path.")
        sys.exit(1)
    
    # Resolve output dir: results/mcqa/<model_short>/<behavior>/
    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1]
        output_dir = Path("results") / "mcqa" / model_short / args.behavior
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Auto-detect answer token offset
    if args.answer_token_offset is not None:
        answer_token_offset = args.answer_token_offset
        logger.warning(f"Using user-specified answer token offset: -{answer_token_offset}")
    else:
        answer_token_offset = find_answer_token_offset(tokenizer)
    
    # Load model
    logger.warning(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map=device
    )
    model.eval()
    
    # Load and prepare dataset
    logger.warning(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)
    logger.warning(f"Loaded {len(data)} examples")
    
    # Limit examples if requested
    if args.max_examples is not None and args.max_examples < len(data):
        logger.warning(f"Limiting to {args.max_examples} examples (from {len(data)})")
        data = data[:args.max_examples]
    
    df = prepare_training_data(data, tokenizer)
    logger.warning(f"Prepared {len(df)} training rows (positive + negative)")
    
    # Train DiffMean
    logger.warning("Training DiffMean (MCQA)...")
    steering_vector = train_diffmean_mcqa(
        model, tokenizer, df, args.layer, device, answer_token_offset, args.batch_size
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
        "behavior": args.behavior,
        "dataset_path": dataset_path,
        "num_train_examples": len(data),
        "max_examples": args.max_examples,
        "seed": args.seed,
        "answer_token_offset": answer_token_offset,
        "steering_vector_norm": steering_vector.norm().item(),
        "method": "mcqa_diffmean",
        "description": "Single answer token (A/B) activation, following original CAA paper"
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.warning("Training complete!")


if __name__ == "__main__":
    main()
