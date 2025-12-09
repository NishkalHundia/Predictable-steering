"""
PCA Variance Explained vs Steerability Correlation Analysis.

For each concept:
1. Load contrastive pairs (positive/negative examples) from training data
2. Compute activations and run PCA
3. Calculate variance explained by PC1 and PC1+PC2
4. Get best LM judge score per prompt from steering results
5. Correlate PCA variance explained with average best steerability score
"""
import argparse
import json
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
            metadata.append(json.loads(line))
    return metadata


def get_concept_data(all_df: pd.DataFrame, metadata: list, concept_id: int):
    """
    Get positive/negative examples for a concept from pre-loaded data.
    """
    concept = metadata[concept_id]["concept"]
    genre = metadata[concept_id]["concept_genres_map"][concept][0]
    
    concept_df = all_df[all_df['concept_id'] == concept_id]
    positive_df = concept_df[(concept_df["output_concept"] == concept) & (concept_df["category"] == "positive")]
    
    negative_df = all_df[(all_df["output_concept"] == EMPTY_CONCEPT) & (all_df["category"] == "negative")]
    negative_df = negative_df[negative_df["concept_genre"] == genre]
    
    return positive_df, negative_df, concept


def collect_activations(model, tokenizer, examples, layer, prefix_length, device):
    """
    Collect activations for examples. Returns last token and mean output vectors.
    """
    last_vectors = []
    mean_vectors = []

    for row in examples.itertuples():
        prompt = row.input
        completion = row.output or ""

        # Tokenize prompt + completion
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        
        # Get prompt length to know where output starts
        prompt_only = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=True)
        
        output_len = len(full_ids) - len(prompt_ids)
        if output_len <= 0:
            continue

        # Get activations
        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        inputs = {"input_ids": full_tensor, "attention_mask": torch.ones_like(full_tensor)}
        
        activations = gather_residual_activations(model, layer, inputs).squeeze(0).detach().cpu().float().numpy()
        output_activations = activations[prefix_length:]
        
        last_vectors.append(output_activations[-1])
        mean_vectors.append(output_activations[-output_len:].mean(axis=0))

    return np.stack(last_vectors), np.stack(mean_vectors)


def compute_pca_variance(vectors: np.ndarray) -> tuple[float, float]:
    """Compute PCA and return variance explained by PC1 and PC1+PC2."""
    pca = PCA(n_components=min(2, vectors.shape[0], vectors.shape[1]))
    pca.fit(vectors)
    
    var_pc1 = pca.explained_variance_ratio_[0]
    var_pc1_pc2 = sum(pca.explained_variance_ratio_[:2]) if len(pca.explained_variance_ratio_) >= 2 else var_pc1
    
    return var_pc1, var_pc1_pc2


def main():
    parser = argparse.ArgumentParser(description="PCA variance vs steerability correlation analysis.")
    parser.add_argument("--model_id", required=True, help="HuggingFace model identifier.")
    parser.add_argument("--layer", type=int, required=True, help="Target layer index.")
    parser.add_argument("--num_examples", type=int, default=500, help="Max examples per class per concept.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_bf16", action="store_true", help="Load model in bfloat16.")
    parser.add_argument("--train_parquet", type=str, required=True, help="Path to train_data.parquet.")
    parser.add_argument("--metadata_jsonl", type=str, required=True, help="Path to metadata.jsonl.")
    parser.add_argument("--steering_parquet", type=str, required=True, help="Path to steering_data.parquet.")
    parser.add_argument("--start_concept", type=int, default=0, help="Starting concept_id.")
    parser.add_argument("--end_concept", type=int, default=None, help="Ending concept_id (exclusive).")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ALL data upfront (ONCE)
    logger.info("Loading data files...")
    steering_df = pd.read_parquet(args.steering_parquet)
    train_df = pd.read_parquet(args.train_parquet)
    metadata = load_metadata(args.metadata_jsonl)
    logger.info(f"Loaded {len(steering_df)} steering rows, {len(train_df)} train rows, {len(metadata)} concepts")

    # Pre-compute avg best LM score per concept
    logger.info("Pre-computing LM scores per concept...")
    best_per_prompt = steering_df.groupby(['concept_id', 'input_id'])['SteeringVector_LMJudgeEvaluator'].max()
    avg_best_scores = best_per_prompt.groupby('concept_id').mean().to_dict()

    # Get concept IDs to process
    all_concept_ids = sorted(steering_df['concept_id'].unique())
    if args.end_concept is not None:
        concept_ids = [c for c in all_concept_ids if args.start_concept <= c < args.end_concept]
    else:
        concept_ids = [c for c in all_concept_ids if c >= args.start_concept]
    logger.info(f"Processing {len(concept_ids)} concepts")

    # Load model
    logger.info(f"Loading model {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    is_chat_model = args.model_id in CHAT_MODELS
    prefix_length = get_prefix_length(tokenizer) if is_chat_model else 1

    # Process each concept
    results = []
    
    for concept_id in tqdm(concept_ids, desc="Processing concepts"):
        if concept_id >= len(metadata):
            continue
            
        try:
            positive_df, negative_df, concept_name = get_concept_data(train_df, metadata, concept_id)
        except Exception as e:
            logger.warning(f"Concept {concept_id}: {e}")
            continue
        
        if positive_df.empty or negative_df.empty:
            continue
        
        # Sample if needed
        if len(positive_df) > args.num_examples:
            positive_df = positive_df.sample(n=args.num_examples, random_state=args.seed)
        if len(negative_df) > args.num_examples:
            negative_df = negative_df.sample(n=args.num_examples, random_state=args.seed)
        
        examples = pd.concat([positive_df, negative_df], ignore_index=True)
        
        # Get activations
        try:
            last_vecs, mean_vecs = collect_activations(model, tokenizer, examples, args.layer, prefix_length, device)
        except Exception as e:
            logger.warning(f"Concept {concept_id} activations failed: {e}")
            continue
        
        if len(last_vecs) < 4:
            continue
        
        # PCA variance
        last_var_pc1, last_var_pc1_pc2 = compute_pca_variance(last_vecs)
        mean_var_pc1, mean_var_pc1_pc2 = compute_pca_variance(mean_vecs)
        
        results.append({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "n_examples": len(last_vecs),
            "last_var_pc1": last_var_pc1,
            "last_var_pc1_pc2": last_var_pc1_pc2,
            "mean_var_pc1": mean_var_pc1,
            "mean_var_pc1_pc2": mean_var_pc1_pc2,
            "avg_best_lm_score": avg_best_scores.get(concept_id, np.nan),
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "pca_variance_results.csv", index=False)
    
    # Compute correlations
    valid_df = results_df.dropna(subset=["avg_best_lm_score"])
    
    if len(valid_df) < 10:
        logger.warning(f"Only {len(valid_df)} valid points, skipping correlation")
        return
    
    corr_last_pc1, p1 = stats.pearsonr(valid_df["last_var_pc1"], valid_df["avg_best_lm_score"])
    corr_last_pc1_pc2, p2 = stats.pearsonr(valid_df["last_var_pc1_pc2"], valid_df["avg_best_lm_score"])
    corr_mean_pc1, p3 = stats.pearsonr(valid_df["mean_var_pc1"], valid_df["avg_best_lm_score"])
    corr_mean_pc1_pc2, p4 = stats.pearsonr(valid_df["mean_var_pc1_pc2"], valid_df["avg_best_lm_score"])
    
    correlations = {
        "last_token": {"pc1": {"r": corr_last_pc1, "p": p1}, "pc1_pc2": {"r": corr_last_pc1_pc2, "p": p2}},
        "mean_output": {"pc1": {"r": corr_mean_pc1, "p": p3}, "pc1_pc2": {"r": corr_mean_pc1_pc2, "p": p4}},
        "n": len(valid_df),
    }
    
    with open(output_dir / "correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print(f"PCA VARIANCE vs STEERABILITY (n={len(valid_df)} concepts)")
    print("="*60)
    print(f"Last Token:  PC1 r={corr_last_pc1:.4f} (p={p1:.2e}), PC1+PC2 r={corr_last_pc1_pc2:.4f} (p={p2:.2e})")
    print(f"Mean Output: PC1 r={corr_mean_pc1:.4f} (p={p3:.2e}), PC1+PC2 r={corr_mean_pc1_pc2:.4f} (p={p4:.2e})")
    print("="*60)
    
    # Quick scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col, title in zip(axes.flat, 
        ["last_var_pc1", "last_var_pc1_pc2", "mean_var_pc1", "mean_var_pc1_pc2"],
        [f"Last PC1 (r={corr_last_pc1:.3f})", f"Last PC1+PC2 (r={corr_last_pc1_pc2:.3f})", 
         f"Mean PC1 (r={corr_mean_pc1:.3f})", f"Mean PC1+PC2 (r={corr_mean_pc1_pc2:.3f})"]):
        ax.scatter(valid_df[col], valid_df["avg_best_lm_score"], alpha=0.5, s=15)
        ax.set_xlabel("Variance Explained")
        ax.set_ylabel("Avg Best LM Score")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_plots.png", dpi=150)
    plt.close()
    
    logger.info(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
