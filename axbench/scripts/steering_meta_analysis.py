"""
Meta-level Analysis for Steering Results.

Aggregates results from PCA variance and diff-mean separability analyses,
and provides comprehensive visualizations for understanding trends across 500 concepts.

Analyses included:
1. Steerability distribution (histogram of best LM judge scores)
2. Factor sensitivity analysis (which steering factors work best)
3. Anti-steerability analysis (prompts where steering hurts)
4. Concept category/genre breakdown
5. Combined PCA + d' vs steerability analysis
6. Multi-metric correlation heatmap
7. Identifying "easy" vs "hard" to steer concepts
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def analyze_factor_sensitivity(steering_df: pd.DataFrame) -> dict:
    """
    Analyze which steering factors work best across concepts.
    
    Returns:
        dict with factor analysis results
    """
    # Average LM score by factor
    factor_scores = steering_df.groupby('factor')['SteeringVector_LMJudgeEvaluator'].agg(['mean', 'std', 'count'])
    factor_scores = factor_scores.reset_index()
    
    # Best factor per concept
    best_factors = steering_df.loc[
        steering_df.groupby('concept_id')['SteeringVector_LMJudgeEvaluator'].idxmax()
    ]['factor'].value_counts()
    
    return {
        "factor_avg_scores": factor_scores.to_dict('records'),
        "best_factor_distribution": best_factors.to_dict(),
    }


def analyze_anti_steerability(steering_df: pd.DataFrame) -> dict:
    """
    Analyze anti-steerability: prompts where steering might hurt.
    
    Look for:
    - Concepts where even the best factor gives low scores
    - High variance in scores across factors (unstable steering)
    """
    concept_stats = steering_df.groupby('concept_id')['SteeringVector_LMJudgeEvaluator'].agg([
        'max', 'min', 'mean', 'std'
    ]).reset_index()
    
    # Fraction of concepts with max score below threshold
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    fraction_below = {t: (concept_stats['max'] <= t).mean() for t in thresholds}
    
    # High variance concepts (sensitive to factor choice)
    high_var_threshold = concept_stats['std'].quantile(0.9)
    high_var_concepts = concept_stats[concept_stats['std'] >= high_var_threshold]
    
    return {
        "fraction_below_threshold": fraction_below,
        "high_variance_threshold": float(high_var_threshold),
        "num_high_variance_concepts": len(high_var_concepts),
        "high_variance_concept_ids": high_var_concepts['concept_id'].tolist(),
    }


def analyze_by_genre(steering_df: pd.DataFrame, metadata: list) -> pd.DataFrame:
    """Analyze steerability by concept genre/category."""
    # Create concept_id to genre mapping
    concept_genres = {}
    for entry in metadata:
        concept_id = entry["concept_id"]
        concept = entry["concept"]
        genres = entry["concept_genres_map"].get(concept, ["unknown"])
        concept_genres[concept_id] = genres[0] if genres else "unknown"
    
    # Get best score per concept
    best_scores = steering_df.groupby('concept_id')['SteeringVector_LMJudgeEvaluator'].max().reset_index()
    best_scores['genre'] = best_scores['concept_id'].map(concept_genres)
    
    # Aggregate by genre
    genre_stats = best_scores.groupby('genre')['SteeringVector_LMJudgeEvaluator'].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).reset_index()
    genre_stats = genre_stats.sort_values('mean', ascending=False)
    
    return genre_stats


def create_comprehensive_report(
    steering_df: pd.DataFrame,
    pca_results_df: pd.DataFrame = None,
    diffmean_results_df: pd.DataFrame = None,
    metadata: list = None,
    output_dir: Path = None,
):
    """Generate comprehensive analysis report with visualizations."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ========== 1. Steerability Distribution ==========
    logger.info("Analyzing steerability distribution...")
    
    best_scores = steering_df.groupby('concept_id')['SteeringVector_LMJudgeEvaluator'].max().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.hist(best_scores['SteeringVector_LMJudgeEvaluator'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(best_scores['SteeringVector_LMJudgeEvaluator'].mean(), color='red', linestyle='--', 
               label=f'Mean: {best_scores["SteeringVector_LMJudgeEvaluator"].mean():.2f}')
    ax.axvline(best_scores['SteeringVector_LMJudgeEvaluator'].median(), color='orange', linestyle='--',
               label=f'Median: {best_scores["SteeringVector_LMJudgeEvaluator"].median():.2f}')
    ax.set_xlabel('Best LM Judge Score (across all factors)')
    ax.set_ylabel('Number of Concepts')
    ax.set_title('Distribution of Best Steerability Scores')
    ax.legend()
    
    # Cumulative distribution
    ax = axes[1]
    sorted_scores = np.sort(best_scores['SteeringVector_LMJudgeEvaluator'])
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cumulative, linewidth=2)
    ax.set_xlabel('Best LM Judge Score')
    ax.set_ylabel('Cumulative Fraction of Concepts')
    ax.set_title('Cumulative Distribution of Steerability')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_steerability_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 2. Factor Sensitivity Analysis ==========
    logger.info("Analyzing factor sensitivity...")
    
    factor_analysis = analyze_factor_sensitivity(steering_df)
    factor_df = pd.DataFrame(factor_analysis["factor_avg_scores"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.bar(factor_df['factor'].astype(str), factor_df['mean'], yerr=factor_df['std'], 
           capsize=3, color='steelblue', alpha=0.7)
    ax.set_xlabel('Steering Factor')
    ax.set_ylabel('Average LM Judge Score')
    ax.set_title('Average Steerability by Factor')
    ax.tick_params(axis='x', rotation=45)
    
    ax = axes[1]
    best_factors = pd.Series(factor_analysis["best_factor_distribution"])
    ax.bar(best_factors.index.astype(str), best_factors.values, color='coral', alpha=0.7)
    ax.set_xlabel('Steering Factor')
    ax.set_ylabel('Number of Concepts')
    ax.set_title('Distribution of Optimal Factors per Concept')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_factor_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 3. Anti-steerability Analysis ==========
    logger.info("Analyzing anti-steerability...")
    
    anti_steer = analyze_anti_steerability(steering_df)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    thresholds = list(anti_steer["fraction_below_threshold"].keys())
    fractions = list(anti_steer["fraction_below_threshold"].values())
    ax.bar([str(t) for t in thresholds], [f * 100 for f in fractions], color='indianred', alpha=0.7)
    ax.set_xlabel('Score Threshold')
    ax.set_ylabel('% of Concepts Below Threshold')
    ax.set_title('Fraction of Concepts with Best Score ≤ Threshold')
    
    # Score variance distribution
    ax = axes[1]
    concept_std = steering_df.groupby('concept_id')['SteeringVector_LMJudgeEvaluator'].std()
    ax.hist(concept_std, bins=30, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(anti_steer["high_variance_threshold"], color='red', linestyle='--',
               label=f'90th percentile: {anti_steer["high_variance_threshold"]:.3f}')
    ax.set_xlabel('Std Dev of LM Score across Factors')
    ax.set_ylabel('Number of Concepts')
    ax.set_title('Factor Sensitivity Distribution\n(High variance = sensitive to factor choice)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_anti_steerability.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 4. Genre/Category Analysis ==========
    if metadata:
        logger.info("Analyzing by genre...")
        
        genre_stats = analyze_by_genre(steering_df, metadata)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(genre_stats) * 0.3)))
        
        # Filter to genres with at least 5 concepts
        genre_stats_filtered = genre_stats[genre_stats['count'] >= 5]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(genre_stats_filtered)))
        bars = ax.barh(genre_stats_filtered['genre'], genre_stats_filtered['mean'], 
                       xerr=genre_stats_filtered['std'], capsize=3, color=colors, alpha=0.8)
        
        # Add count labels
        for i, (_, row) in enumerate(genre_stats_filtered.iterrows()):
            ax.text(row['mean'] + row['std'] + 0.02, i, f'n={int(row["count"])}', 
                   va='center', fontsize=9)
        
        ax.set_xlabel('Average Best LM Judge Score')
        ax.set_ylabel('Genre')
        ax.set_title('Steerability by Concept Genre (genres with ≥5 concepts)')
        
        plt.tight_layout()
        plt.savefig(output_dir / "04_genre_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        genre_stats.to_csv(output_dir / "genre_stats.csv", index=False)
    
    # ========== 5. Combined Analysis (if PCA/diffmean results available) ==========
    if pca_results_df is not None and diffmean_results_df is not None:
        logger.info("Creating combined analysis...")
        
        # Merge all results
        combined_df = pca_results_df.merge(
            diffmean_results_df[['concept_id', 'd_prime', 'overlap_fraction']],
            on='concept_id',
            how='inner'
        )
        combined_df = combined_df.dropna(subset=['avg_best_lm_score', 'd_prime', 'last_var_pc1'])
        
        if len(combined_df) >= 10:
            # Correlation heatmap
            corr_cols = ['last_var_pc1', 'last_var_pc1_pc2', 'mean_var_pc1', 'mean_var_pc1_pc2', 
                        'd_prime', 'overlap_fraction', 'avg_best_lm_score']
            corr_cols = [c for c in corr_cols if c in combined_df.columns]
            corr_matrix = combined_df[corr_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       square=True, ax=ax, vmin=-1, vmax=1)
            ax.set_title('Correlation Matrix: Separability Metrics vs Steerability')
            plt.tight_layout()
            plt.savefig(output_dir / "05_correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Multi-panel scatter
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            ax = axes[0, 0]
            ax.scatter(combined_df['d_prime'], combined_df['avg_best_lm_score'], alpha=0.5, s=20)
            r, p = stats.pearsonr(combined_df['d_prime'], combined_df['avg_best_lm_score'])
            ax.set_xlabel("d' (Discriminability)")
            ax.set_ylabel("Avg Best LM Score")
            ax.set_title(f"d' vs Steerability (r={r:.3f})")
            
            ax = axes[0, 1]
            ax.scatter(combined_df['last_var_pc1'], combined_df['avg_best_lm_score'], alpha=0.5, s=20)
            r, p = stats.pearsonr(combined_df['last_var_pc1'], combined_df['avg_best_lm_score'])
            ax.set_xlabel("PC1 Variance Explained")
            ax.set_ylabel("Avg Best LM Score")
            ax.set_title(f"PC1 Variance vs Steerability (r={r:.3f})")
            
            ax = axes[1, 0]
            ax.scatter(combined_df['d_prime'], combined_df['last_var_pc1'], alpha=0.5, s=20)
            r, p = stats.pearsonr(combined_df['d_prime'], combined_df['last_var_pc1'])
            ax.set_xlabel("d' (Discriminability)")
            ax.set_ylabel("PC1 Variance Explained")
            ax.set_title(f"d' vs PC1 Variance (r={r:.3f})")
            
            ax = axes[1, 1]
            # Color by steerability
            sc = ax.scatter(combined_df['d_prime'], combined_df['last_var_pc1'], 
                           c=combined_df['avg_best_lm_score'], cmap='viridis', alpha=0.7, s=30)
            plt.colorbar(sc, ax=ax, label='Avg Best LM Score')
            ax.set_xlabel("d' (Discriminability)")
            ax.set_ylabel("PC1 Variance Explained")
            ax.set_title("d' vs PC1 Variance (colored by steerability)")
            
            plt.tight_layout()
            plt.savefig(output_dir / "06_combined_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # ========== 6. Easy vs Hard Concepts ==========
    logger.info("Identifying easy vs hard concepts...")
    
    best_scores_full = steering_df.groupby('concept_id').agg({
        'SteeringVector_LMJudgeEvaluator': 'max',
        'input_concept': 'first'
    }).reset_index()
    best_scores_full.columns = ['concept_id', 'best_score', 'concept_name']
    
    # Top 20 easiest
    easiest = best_scores_full.nlargest(20, 'best_score')
    # Top 20 hardest
    hardest = best_scores_full.nsmallest(20, 'best_score')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    y_pos = range(len(easiest))
    ax.barh(y_pos, easiest['best_score'], color='forestgreen', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['concept_name'][:40]}..." if len(row['concept_name']) > 40 
                        else row['concept_name'] for _, row in easiest.iterrows()], fontsize=8)
    ax.set_xlabel('Best LM Judge Score')
    ax.set_title('Top 20 Easiest to Steer Concepts')
    ax.invert_yaxis()
    
    ax = axes[1]
    y_pos = range(len(hardest))
    ax.barh(y_pos, hardest['best_score'], color='indianred', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['concept_name'][:40]}..." if len(row['concept_name']) > 40 
                        else row['concept_name'] for _, row in hardest.iterrows()], fontsize=8)
    ax.set_xlabel('Best LM Judge Score')
    ax.set_title('Top 20 Hardest to Steer Concepts')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / "07_easy_vs_hard_concepts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save lists
    easiest.to_csv(output_dir / "easiest_concepts.csv", index=False)
    hardest.to_csv(output_dir / "hardest_concepts.csv", index=False)
    
    # ========== Summary Statistics ==========
    logger.info("Generating summary statistics...")
    
    summary = {
        "total_concepts": int(steering_df['concept_id'].nunique()),
        "total_prompts": int(steering_df['input_id'].nunique()),
        "total_factors": int(steering_df['factor'].nunique()),
        "factors": sorted(steering_df['factor'].unique().tolist()),
        "steerability": {
            "mean_best_score": float(best_scores['SteeringVector_LMJudgeEvaluator'].mean()),
            "std_best_score": float(best_scores['SteeringVector_LMJudgeEvaluator'].std()),
            "median_best_score": float(best_scores['SteeringVector_LMJudgeEvaluator'].median()),
            "min_best_score": float(best_scores['SteeringVector_LMJudgeEvaluator'].min()),
            "max_best_score": float(best_scores['SteeringVector_LMJudgeEvaluator'].max()),
        },
        "factor_analysis": factor_analysis,
        "anti_steerability": anti_steer,
    }
    
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("STEERING META-ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nDataset Overview:")
    print(f"  Total concepts:  {summary['total_concepts']}")
    print(f"  Prompts per concept: {summary['total_prompts']}")
    print(f"  Steering factors: {summary['factors']}")
    print(f"\nSteerability Statistics (Best Score per Concept):")
    print(f"  Mean:   {summary['steerability']['mean_best_score']:.3f}")
    print(f"  Std:    {summary['steerability']['std_best_score']:.3f}")
    print(f"  Median: {summary['steerability']['median_best_score']:.3f}")
    print(f"  Range:  [{summary['steerability']['min_best_score']:.3f}, {summary['steerability']['max_best_score']:.3f}]")
    print(f"\nAnti-steerability:")
    for thresh, frac in summary['anti_steerability']['fraction_below_threshold'].items():
        print(f"  Concepts with best score ≤ {thresh}: {frac*100:.1f}%")
    print("="*70)
    
    logger.info(f"All results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Meta-level analysis of steering results.")
    parser.add_argument("--steering_parquet", type=str, required=True, help="Path to steering_data.parquet.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save analysis results.")
    parser.add_argument("--metadata_jsonl", type=str, default=None, help="Path to metadata.jsonl (optional, for genre analysis).")
    parser.add_argument("--pca_results_csv", type=str, default=None, help="Path to pca_variance_results.csv (optional).")
    parser.add_argument("--diffmean_results_csv", type=str, default=None, help="Path to diffmean_separability_results.csv (optional).")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading steering data from {args.steering_parquet}")
    steering_df = pd.read_parquet(args.steering_parquet)
    logger.info(f"Loaded {len(steering_df)} rows, {steering_df['concept_id'].nunique()} concepts")
    
    # Load optional data
    metadata = None
    if args.metadata_jsonl:
        logger.info(f"Loading metadata from {args.metadata_jsonl}")
        metadata = load_metadata(args.metadata_jsonl)
    
    pca_results_df = None
    if args.pca_results_csv and Path(args.pca_results_csv).exists():
        logger.info(f"Loading PCA results from {args.pca_results_csv}")
        pca_results_df = pd.read_csv(args.pca_results_csv)
    
    diffmean_results_df = None
    if args.diffmean_results_csv and Path(args.diffmean_results_csv).exists():
        logger.info(f"Loading diffmean results from {args.diffmean_results_csv}")
        diffmean_results_df = pd.read_csv(args.diffmean_results_csv)
    
    create_comprehensive_report(
        steering_df=steering_df,
        pca_results_df=pca_results_df,
        diffmean_results_df=diffmean_results_df,
        metadata=metadata,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

