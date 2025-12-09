"""
Difference-of-Means Line Separability Analysis.

For each concept:
1. Load contrastive pairs (positive/negative examples) from training data
2. Compute activations
3. Compute difference-of-means steering direction
4. Project activations onto the steering direction
5. Normalize so positive mean = 1, negative mean = -1
6. Compute discriminability index d' and other separability metrics
7. Generate visualizations

Based on: "Difference-of-Means Line Separability Predicts Steerability"
- Directional agreement (cosine similarity)
- Separability (discriminability index d')
- Both are predictive of steering effect size

Follows the same data loading logic as pca_contrastive.py and train.py.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from axbench.utils.constants import CHAT_MODELS, EMPTY_CONCEPT
from axbench.utils.model_utils import gather_residual_activations, get_prefix_length

import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path):
    """Load metadata from a JSON lines file."""
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            metadata.append(data)
    return metadata


def load_concept_data(train_parquet: Path, metadata_path: Path, concept_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load data for a specific concept_id from train_data.parquet.
    Returns: (positive_df, negative_df, concept_name)
    """
    all_df = pd.read_parquet(train_parquet)
    metadata = load_metadata(metadata_path)
    concept = metadata[concept_id]["concept"]
    genre = metadata[concept_id]["concept_genres_map"][concept][0]
    
    concept_df = all_df[all_df['concept_id'] == concept_id]
    positive_df = concept_df[(concept_df["output_concept"] == concept) & (concept_df["category"] == "positive")]
    
    negative_df = all_df[(all_df["output_concept"] == EMPTY_CONCEPT) & (all_df["category"] == "negative")]
    negative_df = negative_df[negative_df["concept_genre"] == genre]
    
    return positive_df, negative_df, concept


def collect_activations_with_labels(
    model,
    tokenizer,
    examples: pd.DataFrame,
    layer: int,
    prefix_length: int,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect activations from the model for given examples.
    Returns: (last_vectors, mean_vectors, labels, is_positive)
    """
    last_vectors = []
    mean_vectors = []
    labels = []

    for row in examples.itertuples():
        prompt = row.input
        completion = row.output or ""

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

        activations = gather_residual_activations(model, layer, inputs).squeeze(0).detach().cpu().numpy()
        output_activations = activations[prefix_length:]
        
        last_vectors.append(output_activations[-1])
        mean_vectors.append(output_activations[-output_len:].mean(axis=0))
        labels.append(row.label)

    return np.stack(last_vectors), np.stack(mean_vectors), np.array(labels)


def compute_diffmean_metrics(vectors: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute difference-of-means separability metrics.
    
    1. Compute difference-of-means vector (steering direction)
    2. Project all activations onto this direction
    3. Normalize so positive mean = 1, negative mean = -1
    4. Compute d' (discriminability index)
    
    Returns dict with:
        - d_prime: discriminability index
        - positive_projections: normalized projections of positive examples
        - negative_projections: normalized projections of negative examples
        - overlap_fraction: fraction of examples that overlap
        - cosine_separability: cosine similarity between class means
    """
    positive_mask = labels == 1
    negative_mask = labels == 0
    
    positive_vectors = vectors[positive_mask]
    negative_vectors = vectors[negative_mask]
    
    if len(positive_vectors) == 0 or len(negative_vectors) == 0:
        return None
    
    # Compute means
    mean_positive = positive_vectors.mean(axis=0)
    mean_negative = negative_vectors.mean(axis=0)
    
    # Difference-of-means vector (steering direction)
    diff_mean = mean_positive - mean_negative
    diff_mean_norm = np.linalg.norm(diff_mean)
    
    if diff_mean_norm < 1e-10:
        return None
    
    # Normalize the steering direction
    steering_direction = diff_mean / diff_mean_norm
    
    # Project all vectors onto the steering direction
    positive_projections_raw = positive_vectors @ steering_direction
    negative_projections_raw = negative_vectors @ steering_direction
    
    # Compute means of projections
    pos_proj_mean = positive_projections_raw.mean()
    neg_proj_mean = negative_projections_raw.mean()
    
    # Normalize so positive mean = 1, negative mean = -1
    # Linear transform: y = a*x + b where:
    # a * pos_mean + b = 1
    # a * neg_mean + b = -1
    # Solving: a = 2 / (pos_mean - neg_mean), b = 1 - a * pos_mean
    
    scale = 2.0 / (pos_proj_mean - neg_proj_mean + 1e-10)
    shift = 1.0 - scale * pos_proj_mean
    
    positive_projections = scale * positive_projections_raw + shift
    negative_projections = scale * negative_projections_raw + shift
    
    # Compute d' (discriminability index)
    # d' = (μ_positive - μ_negative) / sqrt(0.5 * (σ²_positive + σ²_negative))
    pos_std = positive_projections.std()
    neg_std = negative_projections.std()
    pooled_std = np.sqrt(0.5 * (pos_std**2 + neg_std**2))
    
    # After normalization, means are 1 and -1, so difference is 2
    d_prime = 2.0 / (pooled_std + 1e-10)
    
    # Compute overlap fraction (how many examples cross the decision boundary at 0)
    pos_below_zero = (positive_projections < 0).sum()
    neg_above_zero = (negative_projections > 0).sum()
    overlap_fraction = (pos_below_zero + neg_above_zero) / (len(positive_projections) + len(negative_projections))
    
    # Cosine similarity between class means (directional agreement)
    cos_sim = np.dot(mean_positive, mean_negative) / (np.linalg.norm(mean_positive) * np.linalg.norm(mean_negative) + 1e-10)
    
    return {
        "d_prime": d_prime,
        "positive_projections": positive_projections,
        "negative_projections": negative_projections,
        "pos_std": pos_std,
        "neg_std": neg_std,
        "overlap_fraction": overlap_fraction,
        "cosine_class_means": cos_sim,
        "steering_direction_norm": diff_mean_norm,
    }


def get_best_lm_scores(steering_df: pd.DataFrame, concept_id: int) -> dict:
    """
    For a given concept, find the highest LM judge score for each prompt
    across all steering factors.
    """
    concept_df = steering_df[steering_df['concept_id'] == concept_id]
    
    if concept_df.empty:
        return {"best_scores": [], "avg_best_score": np.nan, "num_prompts": 0}
    
    best_per_prompt = concept_df.groupby('input_id')['SteeringVector_LMJudgeEvaluator'].max()
    best_scores = best_per_prompt.values.tolist()
    
    return {
        "best_scores": best_scores,
        "avg_best_score": np.mean(best_scores) if best_scores else np.nan,
        "num_prompts": len(best_scores)
    }


def plot_separability(
    positive_projections: np.ndarray,
    negative_projections: np.ndarray,
    concept_name: str,
    concept_id: int,
    d_prime: float,
    output_path: Path,
):
    """Generate histogram visualization of projections onto difference-of-means line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(
        min(positive_projections.min(), negative_projections.min()) - 0.5,
        max(positive_projections.max(), negative_projections.max()) + 0.5,
        40
    )
    
    ax.hist(positive_projections, bins=bins, alpha=0.6, label=f'Positive (n={len(positive_projections)})', color='#1f77b4', density=True)
    ax.hist(negative_projections, bins=bins, alpha=0.6, label=f'Negative (n={len(negative_projections)})', color='#d62728', density=True)
    
    ax.axvline(x=1, color='#1f77b4', linestyle='--', linewidth=2, label='Positive mean (1)')
    ax.axvline(x=-1, color='#d62728', linestyle='--', linewidth=2, label='Negative mean (-1)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, label='Decision boundary')
    
    ax.set_xlabel('Projection onto Difference-of-Means Direction (normalized)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Concept {concept_id}: {concept_name[:50]}...\nd\' = {d_prime:.2f}', fontsize=11)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Difference-of-means separability analysis.")
    parser.add_argument("--model_id", required=True, help="HuggingFace model identifier.")
    parser.add_argument("--layer", type=int, required=True, help="Target layer index for activations.")
    parser.add_argument("--num_examples", type=int, default=36, help="Number of positive (and negative) examples per concept.")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_bf16", action="store_true", help="Load model weights in bfloat16.")
    parser.add_argument("--train_parquet", type=str, required=True, help="Path to train_data.parquet.")
    parser.add_argument("--metadata_jsonl", type=str, required=True, help="Path to metadata.jsonl file.")
    parser.add_argument("--steering_parquet", type=str, required=True, help="Path to steering_data.parquet with LM judge scores.")
    parser.add_argument("--start_concept", type=int, default=0, help="Starting concept_id (inclusive).")
    parser.add_argument("--end_concept", type=int, default=None, help="Ending concept_id (exclusive).")
    parser.add_argument("--plot_top_k", type=int, default=20, help="Number of top/bottom concepts to generate individual plots for.")
    parser.add_argument("--activation_type", type=str, default="mean", choices=["last", "mean"], help="Type of activation to use.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "separability_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load steering results
    logger.info(f"Loading steering results from {args.steering_parquet}")
    steering_df = pd.read_parquet(args.steering_parquet)
    
    all_concept_ids = sorted(steering_df['concept_id'].unique())
    logger.info(f"Found {len(all_concept_ids)} concepts in steering data")
    
    if args.end_concept is not None:
        concept_ids = [c for c in all_concept_ids if args.start_concept <= c < args.end_concept]
    else:
        concept_ids = [c for c in all_concept_ids if c >= args.start_concept]
    logger.info(f"Processing {len(concept_ids)} concepts")

    # Load model
    logger.info(f"Loading model {args.model_id}")
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

    metadata = load_metadata(args.metadata_jsonl)
    
    results = []
    projection_data = {}  # Store projections for plotting later
    
    for concept_id in tqdm(concept_ids, desc="Processing concepts"):
        if concept_id >= len(metadata):
            logger.warning(f"concept_id {concept_id} exceeds metadata length, skipping")
            continue
            
        try:
            positive_df, negative_df, concept_name = load_concept_data(
                Path(args.train_parquet),
                Path(args.metadata_jsonl),
                concept_id
            )
        except Exception as e:
            logger.warning(f"Failed to load concept {concept_id}: {e}")
            continue
        
        if positive_df.empty or negative_df.empty:
            logger.warning(f"Skipping concept_id={concept_id}: Missing examples.")
            continue
        
        # Sample
        if len(positive_df) > args.num_examples:
            positive_df = positive_df.sample(n=args.num_examples, random_state=args.seed)
        if len(negative_df) > args.num_examples:
            negative_df = negative_df.sample(n=args.num_examples, random_state=args.seed)
        
        positive_df = positive_df.copy()
        negative_df = negative_df.copy()
        positive_df["label"] = 1
        negative_df["label"] = 0
        
        examples = pd.concat([positive_df, negative_df], ignore_index=True)
        
        try:
            last_vecs, mean_vecs, labels = collect_activations_with_labels(
                base_model,
                tokenizer,
                examples,
                args.layer,
                prefix_length,
                device,
            )
        except Exception as e:
            logger.warning(f"Failed to collect activations for concept {concept_id}: {e}")
            continue
        
        if len(last_vecs) < 4:
            logger.warning(f"Skipping concept_id={concept_id}: Too few examples")
            continue
        
        # Choose activation type
        vectors = mean_vecs if args.activation_type == "mean" else last_vecs
        
        # Compute diff-mean metrics
        metrics = compute_diffmean_metrics(vectors, labels)
        
        if metrics is None:
            logger.warning(f"Skipping concept_id={concept_id}: Failed to compute metrics")
            continue
        
        # Get LM scores
        lm_scores = get_best_lm_scores(steering_df, concept_id)
        
        results.append({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "num_examples": len(examples),
            "d_prime": metrics["d_prime"],
            "overlap_fraction": metrics["overlap_fraction"],
            "cosine_class_means": metrics["cosine_class_means"],
            "pos_std": metrics["pos_std"],
            "neg_std": metrics["neg_std"],
            "steering_direction_norm": metrics["steering_direction_norm"],
            "avg_best_lm_score": lm_scores["avg_best_score"],
            "num_prompts": lm_scores["num_prompts"],
        })
        
        # Store projection data for later plotting
        projection_data[concept_id] = {
            "positive_projections": metrics["positive_projections"],
            "negative_projections": metrics["negative_projections"],
            "concept_name": concept_name,
            "d_prime": metrics["d_prime"],
            "avg_best_lm_score": lm_scores["avg_best_score"],
        }
        
        if len(results) % 50 == 0:
            logger.info(f"Processed {len(results)} concepts...")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "diffmean_separability_results.csv", index=False)
    logger.info(f"Saved results to {output_dir / 'diffmean_separability_results.csv'}")
    
    # Compute correlations
    valid_df = results_df.dropna(subset=["avg_best_lm_score"])
    
    if len(valid_df) < 10:
        logger.warning(f"Not enough data points ({len(valid_df)}) for correlation")
        return
    
    # d' vs steerability
    corr_dprime, p_dprime = stats.pearsonr(valid_df["d_prime"], valid_df["avg_best_lm_score"])
    
    # Overlap fraction vs steerability (expect negative correlation)
    corr_overlap, p_overlap = stats.pearsonr(valid_df["overlap_fraction"], valid_df["avg_best_lm_score"])
    
    # Steering direction norm vs steerability
    corr_norm, p_norm = stats.pearsonr(valid_df["steering_direction_norm"], valid_df["avg_best_lm_score"])
    
    correlations = {
        "d_prime_vs_steerability": {"correlation": corr_dprime, "p_value": p_dprime},
        "overlap_vs_steerability": {"correlation": corr_overlap, "p_value": p_overlap},
        "steering_norm_vs_steerability": {"correlation": corr_norm, "p_value": p_norm},
        "n_concepts": len(valid_df),
    }
    
    with open(output_dir / "diffmean_correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DIFFERENCE-OF-MEANS SEPARABILITY ANALYSIS RESULTS")
    print("="*60)
    print(f"\nNumber of concepts analyzed: {len(valid_df)}")
    print(f"\nd' (discriminability) vs avg best LM score: r={corr_dprime:.4f}, p={p_dprime:.4e}")
    print(f"Overlap fraction vs avg best LM score:       r={corr_overlap:.4f}, p={p_overlap:.4e}")
    print(f"Steering direction norm vs avg best LM score: r={corr_norm:.4f}, p={p_norm:.4e}")
    print(f"\nDescriptive statistics for d':")
    print(f"  Mean: {valid_df['d_prime'].mean():.3f}")
    print(f"  Std:  {valid_df['d_prime'].std():.3f}")
    print(f"  Min:  {valid_df['d_prime'].min():.3f}")
    print(f"  Max:  {valid_df['d_prime'].max():.3f}")
    print("="*60)
    
    # Generate summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # d' vs steerability
    ax = axes[0, 0]
    ax.scatter(valid_df["d_prime"], valid_df["avg_best_lm_score"], alpha=0.5, s=20)
    ax.set_xlabel("d' (Discriminability Index)")
    ax.set_ylabel("Avg Best LM Judge Score")
    ax.set_title(f"d' vs Steerability (r={corr_dprime:.3f}, p={p_dprime:.2e})")
    
    # Overlap vs steerability
    ax = axes[0, 1]
    ax.scatter(valid_df["overlap_fraction"], valid_df["avg_best_lm_score"], alpha=0.5, s=20)
    ax.set_xlabel("Overlap Fraction")
    ax.set_ylabel("Avg Best LM Judge Score")
    ax.set_title(f"Overlap vs Steerability (r={corr_overlap:.3f}, p={p_overlap:.2e})")
    
    # d' histogram
    ax = axes[1, 0]
    ax.hist(valid_df["d_prime"], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel("d' (Discriminability Index)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of d' across concepts")
    
    # Steerability histogram
    ax = axes[1, 1]
    ax.hist(valid_df["avg_best_lm_score"], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Avg Best LM Judge Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Steerability across concepts")
    
    plt.tight_layout()
    plt.savefig(output_dir / "diffmean_summary_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate individual plots for top/bottom concepts by d'
    sorted_by_dprime = valid_df.sort_values("d_prime", ascending=False)
    
    # Top k (most separable)
    top_concepts = sorted_by_dprime.head(args.plot_top_k)
    for _, row in top_concepts.iterrows():
        cid = row["concept_id"]
        if cid in projection_data:
            data = projection_data[cid]
            plot_separability(
                data["positive_projections"],
                data["negative_projections"],
                data["concept_name"],
                cid,
                data["d_prime"],
                plots_dir / f"top_separable_{cid}.png"
            )
    
    # Bottom k (least separable)
    bottom_concepts = sorted_by_dprime.tail(args.plot_top_k)
    for _, row in bottom_concepts.iterrows():
        cid = row["concept_id"]
        if cid in projection_data:
            data = projection_data[cid]
            plot_separability(
                data["positive_projections"],
                data["negative_projections"],
                data["concept_name"],
                cid,
                data["d_prime"],
                plots_dir / f"bottom_separable_{cid}.png"
            )
    
    logger.info(f"Generated individual plots for top/bottom {args.plot_top_k} concepts in {plots_dir}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()

