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

from axbench.models.mean import DiffMean
from axbench.utils.model_utils import get_prefix_length
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


class TrainingArgs:
    """Simple training args to match axbench interface."""
    def __init__(self):
        self.batch_size = 4
        self.n_epochs = 1
        self.lr = 1e-3
        self.weight_decay = 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="datasets/raw/survival-instinct/dataset.json")
    parser.add_argument("--output_dir", type=str, default="results/survival-instinct")
    parser.add_argument("--use_bf16", action="store_true", default=True)
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
    
    df = prepare_training_data(data, tokenizer, args.model_name)
    logger.warning(f"Prepared {len(df)} training rows (positive + negative)")
    
    # Get prefix length for chat model
    is_chat_model = args.model_name in CHAT_MODELS
    prefix_length = 1
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")
    
    # Create training args
    training_args = TrainingArgs()
    
    # Initialize DiffMean model
    logger.warning("Initializing DiffMean model...")
    diffmean_model = DiffMean(
        model, tokenizer, layer=args.layer,
        training_args=training_args,
        lm_model_name=args.model_name,
        device=device, seed=args.seed
    )
    
    # Make model for training
    diffmean_model.make_model(
        mode="train",
        embed_dim=model.config.hidden_size,
        low_rank_dimension=1,
        dtype=torch.bfloat16 if args.use_bf16 else None,
        intervention_type="addition"
    )
    
    # Train
    logger.warning("Training DiffMean...")
    diffmean_model.train(df, prefix_length=prefix_length)
    
    # Save the steering vector
    weight_path = output_dir / "DiffMean_weight.pt"
    bias_path = output_dir / "DiffMean_bias.pt"
    
    torch.save(diffmean_model.ax.proj.weight.data, weight_path)
    torch.save(diffmean_model.ax.proj.bias.data, bias_path)
    
    logger.warning(f"Saved steering vector to {weight_path}")
    logger.warning(f"Steering vector shape: {diffmean_model.ax.proj.weight.data.shape}")
    
    # Save config
    config = {
        "model_name": args.model_name,
        "layer": args.layer,
        "dataset_path": args.dataset_path,
        "num_train_examples": len(data),
        "seed": args.seed
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.warning("Training complete!")


if __name__ == "__main__":
    main()

