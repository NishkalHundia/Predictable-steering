"""
PCA analysis script for contrastive pair activations.
Follows the same data loading logic as train.py.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from axbench.utils.constants import CHAT_MODELS, EMPTY_CONCEPT
from axbench.utils.model_utils import gather_residual_activations, get_prefix_length

import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path):
    """Load metadata from a JSON lines file."""
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            metadata += [data]
    return metadata


def load_concept_data(train_parquet: Path, metadata_path: Path, concept_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load data for a specific concept_id from train_data.parquet.
    Returns: (positive_df, negative_df, concept_name)
    Follows train.py logic exactly.
    """
    # Load all data
    all_df = pd.read_parquet(train_parquet)
    
    # Load metadata to get concept name and genre
    metadata = load_metadata(metadata_path)
    concept = metadata[concept_id]["concept"]
    genre = metadata[concept_id]["concept_genres_map"][concept][0]
    
    # Get positive examples for this concept
    concept_df = all_df[all_df['concept_id'] == concept_id]
    positive_df = concept_df[(concept_df["output_concept"] == concept) & (concept_df["category"] == "positive")]
    
    # Get negative examples from the entire dataset (filtered by genre)
    negative_df = all_df[(all_df["output_concept"] == EMPTY_CONCEPT) & (all_df["category"] == "negative")]
    negative_df = negative_df[negative_df["concept_genre"] == genre]
    
    return positive_df, negative_df, concept


def _collect_activations(
    model,
    tokenizer,
    examples: pd.DataFrame,
    layer: int,
    prefix_length: int,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect activations from the model for given examples.
    Returns last token vectors, mean output vectors, and labels.
    """
    last_vectors = []
    mean_vectors = []
    labels = []

    for row in examples.itertuples():
        prompt = row.input
        completion = row.output or ""

        # Apply chat template
        messages_prompt = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=True,
            add_generation_prompt=True,
        )

        messages_full = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
        )

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        
        prompt_len = prompt_tensor.shape[1]
        combined_len = full_tensor.shape[1]
        output_len = combined_len - prompt_len

        if output_len <= 0:
            logger.warning("Skipping example with non-positive generated length.")
            continue

        attention_mask = torch.ones_like(full_tensor)
        inputs = {"input_ids": full_tensor, "attention_mask": attention_mask}

        # Gather activations at target layer
        activations = gather_residual_activations(model, layer, inputs).squeeze(0).detach().cpu().numpy()
        
        # Extract output token activations (after prefix_length)
        output_activations = activations[prefix_length:]
        
        last_vectors.append(output_activations[-1])
        mean_vectors.append(output_activations[-output_len:].mean(axis=0))
        labels.append(row.label)

    return np.stack(last_vectors), np.stack(mean_vectors), np.array(labels)


def _plot_pca(points: np.ndarray, labels: np.ndarray, title: str, output_path: Path):
    """Generate and save a PCA plot."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(points)
    colors = ["#1f77b4" if label == 1 else "#d62728" for label in labels]

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.75, edgecolors="k", linewidths=0.5)
    plt.title(title, fontsize=10, wrap=True)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="Positive", markerfacecolor="#1f77b4", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", label="Negative", markerfacecolor="#d62728", markersize=8),
        ],
        loc="best",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PCA analysis of contrastive pairs at a target model layer.")
    parser.add_argument("--model_id", required=True, help="HuggingFace model identifier (e.g. google/gemma-2-2b-it).")
    parser.add_argument("--layer", type=int, required=True, help="Target layer index for activations.")
    parser.add_argument("--concept_id", type=int, required=True, help="Concept ID to analyze (from train_data.parquet).")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of positive (and negative) examples.")
    parser.add_argument("--output_dir", required=True, help="Directory to save PCA plots and CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_bf16", action="store_true", help="Load model weights in bfloat16.")
    parser.add_argument("--train_parquet", type=str, required=True, help="Path to existing train_data.parquet containing contrastive pairs.")
    parser.add_argument("--metadata_jsonl", type=str, required=True, help="Path to metadata.jsonl file.")
    args = parser.parse_args()

    # Create output directory structure: output_dir/model_name/concept_id/
    model_name = args.model_id.replace("/", "_")
    output_dir = Path(args.output_dir) / model_name / f"concept_{args.concept_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    logger.warning(f"Loading base model {args.model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
    )
    base_model = base_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(tokenizer))

    is_chat_model = args.model_id in CHAT_MODELS
    prefix_length = get_prefix_length(tokenizer) if is_chat_model else 1

    # Load concept data (following train.py logic exactly)
    positive_df, negative_df, concept = load_concept_data(
        Path(args.train_parquet),
        Path(args.metadata_jsonl),
        args.concept_id
    )
    
    logger.warning(f"Found {len(positive_df)} positive and {len(negative_df)} negative examples for concept_id={args.concept_id} ({concept})")
    
    if positive_df.empty or negative_df.empty:
        raise ValueError(f"Missing positive or negative examples for concept_id={args.concept_id}.")
    
    # Sample
    if len(positive_df) > args.num_examples:
        positive_df = positive_df.sample(n=args.num_examples, random_state=args.seed)
    if len(negative_df) > args.num_examples:
        negative_df = negative_df.sample(n=args.num_examples, random_state=args.seed)
    
    # Add label column
    positive_df = positive_df.copy()
    negative_df = negative_df.copy()
    positive_df["label"] = 1
    negative_df["label"] = 0
    
    examples = pd.concat([positive_df, negative_df], ignore_index=True)
    logger.warning(f"Prepared {len(examples)} contrastive rows for concept_id={args.concept_id}.")

    last_vecs, mean_vecs, labels = _collect_activations(
        base_model,
        tokenizer,
        examples,
        args.layer,
        prefix_length,
        device,
    )

    _plot_pca(
        last_vecs,
        labels,
        f"Concept {args.concept_id}: {concept} – Last Token PCA",
        output_dir / "pca_last_token.png",
    )
    _plot_pca(
        mean_vecs,
        labels,
        f"Concept {args.concept_id}: {concept} – Mean Output Tokens PCA",
        output_dir / "pca_mean_output.png",
    )

    last_coords = PCA(n_components=2).fit_transform(last_vecs)
    mean_coords = PCA(n_components=2).fit_transform(mean_vecs)
    csv_df = pd.DataFrame({
        "label": np.concatenate([labels, labels]),
        "type": ["last"] * len(labels) + ["mean"] * len(labels),
        "pc1": np.concatenate([last_coords[:, 0], mean_coords[:, 0]]),
        "pc2": np.concatenate([last_coords[:, 1], mean_coords[:, 1]]),
    })
    csv_df.to_csv(output_dir / "pca_projection.csv", index=False)
    logger.warning(f"PCA artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
