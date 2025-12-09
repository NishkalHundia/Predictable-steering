"""
PCA Variance Explained vs Steerability Correlation Analysis.

For each concept:
1. Load contrastive pairs (positive/negative examples) from training data
2. Compute activations and run PCA
3. Calculate variance explained by PC1 and PC1+PC2
4. Get best LM judge score per prompt from steering results
5. Correlate PCA variance explained with average best steerability score

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
from sklearn.decomposition import PCA
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
    Follows train.py logic exactly.
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


def collect_activations(
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


def compute_pca_variance(vectors: np.ndarray) -> tuple[float, float]:
    """
    Compute PCA and return variance explained by PC1 and PC1+PC2.
    """
    pca = PCA(n_components=min(2, vectors.shape[0], vectors.shape[1]))
    pca.fit(vectors)
    
    var_pc1 = pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) >= 1 else 0.0
    var_pc1_pc2 = sum(pca.explained_variance_ratio_[:2]) if len(pca.explained_variance_ratio_) >= 2 else var_pc1
    
    return var_pc1, var_pc1_pc2


def get_best_lm_scores(steering_df: pd.DataFrame, concept_id: int) -> dict:
    """
    For a given concept, find the highest LM judge score for each prompt (input_id)
    across all steering factors.
    
    Returns dict with:
        - best_scores: list of best scores per prompt
        - avg_best_score: average of best scores
        - num_prompts: number of prompts
    """
    concept_df = steering_df[steering_df['concept_id'] == concept_id]
    
    if concept_df.empty:
        return {"best_scores": [], "avg_best_score": np.nan, "num_prompts": 0}
    
    # Group by input_id and get max LM judge score
    best_per_prompt = concept_df.groupby('input_id')['SteeringVector_LMJudgeEvaluator'].max()
    best_scores = best_per_prompt.values.tolist()
    
    return {
        "best_scores": best_scores,
        "avg_best_score": np.mean(best_scores) if best_scores else np.nan,
        "num_prompts": len(best_scores)
    }


def main():
    parser = argparse.ArgumentParser(description="PCA variance vs steerability correlation analysis.")
    parser.add_argument("--model_id", required=True, help="HuggingFace model identifier (e.g. google/gemma-2-2b-it).")
    parser.add_argument("--layer", type=int, required=True, help="Target layer index for activations.")
    parser.add_argument("--num_examples", type=int, default=36, help="Number of positive (and negative) examples per concept.")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_bf16", action="store_true", help="Load model weights in bfloat16.")
    parser.add_argument("--train_parquet", type=str, required=True, help="Path to train_data.parquet containing contrastive pairs.")
    parser.add_argument("--metadata_jsonl", type=str, required=True, help="Path to metadata.jsonl file.")
    parser.add_argument("--steering_parquet", type=str, required=True, help="Path to steering_data.parquet with LM judge scores.")
    parser.add_argument("--start_concept", type=int, default=0, help="Starting concept_id (inclusive).")
    parser.add_argument("--end_concept", type=int, default=None, help="Ending concept_id (exclusive). If None, process all.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load steering results for LM judge scores
    logger.info(f"Loading steering results from {args.steering_parquet}")
    steering_df = pd.read_parquet(args.steering_parquet)
    
    # Get all unique concept_ids from steering data
    all_concept_ids = sorted(steering_df['concept_id'].unique())
    logger.info(f"Found {len(all_concept_ids)} concepts in steering data")
    
    # Filter concept range
    if args.end_concept is not None:
        concept_ids = [c for c in all_concept_ids if args.start_concept <= c < args.end_concept]
    else:
        concept_ids = [c for c in all_concept_ids if c >= args.start_concept]
    logger.info(f"Processing concepts {args.start_concept} to {args.end_concept or 'end'}: {len(concept_ids)} concepts")

    # Load model and tokenizer
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

    # Load metadata to get total concepts
    metadata = load_metadata(args.metadata_jsonl)
    
    # Results storage
    results = []
    
    for concept_id in tqdm(concept_ids, desc="Processing concepts"):
        if concept_id >= len(metadata):
            logger.warning(f"concept_id {concept_id} exceeds metadata length {len(metadata)}, skipping")
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
            logger.warning(f"Skipping concept_id={concept_id}: Missing positive or negative examples.")
            continue
        
        # Sample examples
        if len(positive_df) > args.num_examples:
            positive_df = positive_df.sample(n=args.num_examples, random_state=args.seed)
        if len(negative_df) > args.num_examples:
            negative_df = negative_df.sample(n=args.num_examples, random_state=args.seed)
        
        # Add labels
        positive_df = positive_df.copy()
        negative_df = negative_df.copy()
        positive_df["label"] = 1
        negative_df["label"] = 0
        
        examples = pd.concat([positive_df, negative_df], ignore_index=True)
        
        # Collect activations
        try:
            last_vecs, mean_vecs, labels = collect_activations(
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
        
        if len(last_vecs) < 4:  # Need minimum examples for PCA
            logger.warning(f"Skipping concept_id={concept_id}: Too few valid examples ({len(last_vecs)})")
            continue
        
        # Compute PCA variance explained
        last_var_pc1, last_var_pc1_pc2 = compute_pca_variance(last_vecs)
        mean_var_pc1, mean_var_pc1_pc2 = compute_pca_variance(mean_vecs)
        
        # Get best LM judge scores
        lm_scores = get_best_lm_scores(steering_df, concept_id)
        
        results.append({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "num_examples": len(examples),
            "num_valid_activations": len(last_vecs),
            # PCA variance explained (last token)
            "last_var_pc1": last_var_pc1,
            "last_var_pc1_pc2": last_var_pc1_pc2,
            # PCA variance explained (mean output)
            "mean_var_pc1": mean_var_pc1,
            "mean_var_pc1_pc2": mean_var_pc1_pc2,
            # LM judge scores
            "avg_best_lm_score": lm_scores["avg_best_score"],
            "num_prompts": lm_scores["num_prompts"],
            "best_scores": lm_scores["best_scores"],
        })
        
        if len(results) % 50 == 0:
            logger.info(f"Processed {len(results)} concepts so far...")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "pca_variance_results.csv", index=False)
    logger.info(f"Saved results to {output_dir / 'pca_variance_results.csv'}")
    
    # Compute correlations
    valid_df = results_df.dropna(subset=["avg_best_lm_score"])
    
    if len(valid_df) < 10:
        logger.warning(f"Not enough valid data points ({len(valid_df)}) for correlation analysis")
        return
    
    correlations = {}
    
    # Correlations for last token PCA
    corr_last_pc1, p_last_pc1 = stats.pearsonr(valid_df["last_var_pc1"], valid_df["avg_best_lm_score"])
    corr_last_pc1_pc2, p_last_pc1_pc2 = stats.pearsonr(valid_df["last_var_pc1_pc2"], valid_df["avg_best_lm_score"])
    
    # Correlations for mean output PCA
    corr_mean_pc1, p_mean_pc1 = stats.pearsonr(valid_df["mean_var_pc1"], valid_df["avg_best_lm_score"])
    corr_mean_pc1_pc2, p_mean_pc1_pc2 = stats.pearsonr(valid_df["mean_var_pc1_pc2"], valid_df["avg_best_lm_score"])
    
    correlations = {
        "last_token": {
            "pc1": {"correlation": corr_last_pc1, "p_value": p_last_pc1},
            "pc1_pc2": {"correlation": corr_last_pc1_pc2, "p_value": p_last_pc1_pc2},
        },
        "mean_output": {
            "pc1": {"correlation": corr_mean_pc1, "p_value": p_mean_pc1},
            "pc1_pc2": {"correlation": corr_mean_pc1_pc2, "p_value": p_mean_pc1_pc2},
        },
        "n_concepts": len(valid_df),
    }
    
    # Save correlations
    with open(output_dir / "correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PCA VARIANCE vs STEERABILITY CORRELATION RESULTS")
    print("="*60)
    print(f"\nNumber of concepts analyzed: {len(valid_df)}")
    print(f"\nLast Token Activations:")
    print(f"  PC1 variance vs avg best LM score:       r={corr_last_pc1:.4f}, p={p_last_pc1:.4e}")
    print(f"  PC1+PC2 variance vs avg best LM score:   r={corr_last_pc1_pc2:.4f}, p={p_last_pc1_pc2:.4e}")
    print(f"\nMean Output Activations:")
    print(f"  PC1 variance vs avg best LM score:       r={corr_mean_pc1:.4f}, p={p_mean_pc1:.4e}")
    print(f"  PC1+PC2 variance vs avg best LM score:   r={corr_mean_pc1_pc2:.4f}, p={p_mean_pc1_pc2:.4e}")
    print("="*60)
    
    # Generate scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Last token PC1
    ax = axes[0, 0]
    ax.scatter(valid_df["last_var_pc1"], valid_df["avg_best_lm_score"], alpha=0.5, s=20)
    ax.set_xlabel("PC1 Variance Explained")
    ax.set_ylabel("Avg Best LM Judge Score")
    ax.set_title(f"Last Token: PC1 (r={corr_last_pc1:.3f}, p={p_last_pc1:.2e})")
    
    # Last token PC1+PC2
    ax = axes[0, 1]
    ax.scatter(valid_df["last_var_pc1_pc2"], valid_df["avg_best_lm_score"], alpha=0.5, s=20)
    ax.set_xlabel("PC1+PC2 Variance Explained")
    ax.set_ylabel("Avg Best LM Judge Score")
    ax.set_title(f"Last Token: PC1+PC2 (r={corr_last_pc1_pc2:.3f}, p={p_last_pc1_pc2:.2e})")
    
    # Mean output PC1
    ax = axes[1, 0]
    ax.scatter(valid_df["mean_var_pc1"], valid_df["avg_best_lm_score"], alpha=0.5, s=20)
    ax.set_xlabel("PC1 Variance Explained")
    ax.set_ylabel("Avg Best LM Judge Score")
    ax.set_title(f"Mean Output: PC1 (r={corr_mean_pc1:.3f}, p={p_mean_pc1:.2e})")
    
    # Mean output PC1+PC2
    ax = axes[1, 1]
    ax.scatter(valid_df["mean_var_pc1_pc2"], valid_df["avg_best_lm_score"], alpha=0.5, s=20)
    ax.set_xlabel("PC1+PC2 Variance Explained")
    ax.set_ylabel("Avg Best LM Judge Score")
    ax.set_title(f"Mean Output: PC1+PC2 (r={corr_mean_pc1_pc2:.3f}, p={p_mean_pc1_pc2:.2e})")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_variance_correlation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved correlation plots to {output_dir / 'pca_variance_correlation_plots.png'}")


if __name__ == "__main__":
    main()

