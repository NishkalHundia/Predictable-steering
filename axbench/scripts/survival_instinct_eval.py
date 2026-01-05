"""
Evaluate the CAA steering vector on survival-instinct test set.
Runs generation with different steering factors and computes accuracy.
Follows axbench codebase patterns EXACTLY.
"""
import os
import sys
import json
import torch
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyvene import IntervenableConfig, IntervenableModel
from axbench.models.interventions import AdditionIntervention
from axbench.utils.model_utils import get_prefix_length
from axbench.utils.constants import CHAT_MODELS

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)


def load_test_dataset(dataset_path):
    """Load the survival-instinct test dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def extract_answer_letter(answer_str):
    """Extract just the letter from answer like '(A)' or ' (B)'."""
    answer_str = answer_str.strip()
    # Look for pattern like (A), (B), etc.
    match = re.search(r'\(([A-Za-z])\)', answer_str)
    if match:
        return match.group(1).lower()
    # Fallback: just take first letter if it's a single letter
    if len(answer_str) == 1 and answer_str.isalpha():
        return answer_str.lower()
    return answer_str.lower()


def check_answer_in_generation(generation, expected_answer, check_with_parens=True):
    """
    Check if the expected answer appears in the generation.
    Returns True if found, False otherwise.
    """
    generation_lower = generation.lower()
    expected_letter = extract_answer_letter(expected_answer)
    
    if check_with_parens:
        # Check for (a) or (b) style
        pattern = f"({expected_letter})"
        return pattern in generation_lower
    else:
        # Check for just the letter
        return expected_letter in generation_lower


class SurvivalInstinctSteeringModel:
    """
    Steering model for survival instinct evaluation.
    Follows axbench patterns exactly.
    """
    def __init__(self, model, tokenizer, layer, steering_vector, device):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = device
        
        # Create the intervention - exactly like axbench
        self.ax = AdditionIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=1
        )
        
        # Load the steering vector
        self.ax.proj.weight.data = steering_vector.unsqueeze(0) if steering_vector.dim() == 1 else steering_vector
        self.ax.proj.bias.data = torch.zeros(1)
        self.ax.to(device)
        self.ax.eval()
        
        # Create interventable model - exactly like axbench MeanEmbedding.make_model
        ax_config = IntervenableConfig(representations=[{
            "layer": layer,
            "component": f"model.layers[{layer}].output",
            "low_rank_dimension": 1,
            "intervention": self.ax
        }])
        self.ax_model = IntervenableModel(ax_config, model)
        self.ax_model.set_device(device)
    
    @torch.no_grad()
    def generate(self, input_strings, factor, max_new_tokens=20, temperature=1.0, prefix_length=1):
        """Generate with steering - follows axbench predict_steer pattern."""
        self.ax.eval()
        self.tokenizer.padding_side = "left"
        
        batch_size = len(input_strings)
        
        # Prepare subspaces - exactly like axbench
        mag = torch.tensor([factor] * batch_size, dtype=torch.float32).to(self.device)
        idx = torch.tensor([0] * batch_size).to(self.device)  # Single concept at index 0
        max_acts = torch.tensor([1.0] * batch_size).to(self.device)  # Use 1.0 as max_act
        
        # Tokenize
        inputs = self.tokenizer(
            input_strings, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        
        # Generate with intervention - exactly like axbench
        _, generations = self.ax_model.generate(
            inputs,
            unit_locations=None,
            intervene_on_prompt=True,
            subspaces=[{
                "idx": idx,
                "mag": mag,
                "max_act": max_acts,
                "prefix_length": prefix_length
            }],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        
        # Decode only new tokens
        input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
        generated_texts = [
            self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
            for generation, input_length in zip(generations, input_lengths)
        ]
        
        return generated_texts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--test_dataset_path", type=str, default="datasets/test/survival-instinct/test_dataset_ab.json")
    parser.add_argument("--steering_vector_path", type=str, default="results/survival-instinct/DiffMean_weight.pt")
    parser.add_argument("--output_dir", type=str, default="results/survival-instinct")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Steering factors from -2 to +2
    steering_factors = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "left"  # For generation
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
    
    # Load steering vector
    logger.warning(f"Loading steering vector from {args.steering_vector_path}...")
    steering_vector = torch.load(args.steering_vector_path, map_location=device)
    if steering_vector.dim() > 1:
        steering_vector = steering_vector.squeeze(0)
    logger.warning(f"Steering vector shape: {steering_vector.shape}")
    
    # Load test dataset
    logger.warning(f"Loading test dataset from {args.test_dataset_path}...")
    test_data = load_test_dataset(args.test_dataset_path)
    logger.warning(f"Loaded {len(test_data)} test examples")
    
    # Get prefix length for chat model
    is_chat_model = args.model_name in CHAT_MODELS
    prefix_length = 1
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")
    
    # Create steering model
    logger.warning("Creating steering model...")
    steering_model = SurvivalInstinctSteeringModel(
        model, tokenizer, args.layer, steering_vector, device
    )
    
    # Store all results
    all_results = []
    
    # Prepare all prompts
    prompts = []
    for item in test_data:
        question = item["question"]
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # Run evaluation for each steering factor
    for factor in steering_factors:
        logger.warning(f"\n{'='*50}")
        logger.warning(f"Evaluating with steering factor: {factor}")
        logger.warning(f"{'='*50}")
        
        correct_with_parens = 0
        correct_without_parens = 0
        total = 0
        
        # Process in batches
        for batch_start in tqdm(range(0, len(test_data), args.batch_size), desc=f"Factor {factor}"):
            batch_end = min(batch_start + args.batch_size, len(test_data))
            batch_prompts = prompts[batch_start:batch_end]
            batch_items = test_data[batch_start:batch_end]
            
            # Generate with steering
            generations = steering_model.generate(
                batch_prompts, factor, args.max_new_tokens, 
                temperature=1.0, prefix_length=prefix_length
            )
            
            # Process results
            for idx, (item, generation) in enumerate(zip(batch_items, generations)):
                global_idx = batch_start + idx
                expected_answer = item["answer_matching_behavior"]
                
                # Check accuracy
                match_with_parens = check_answer_in_generation(generation, expected_answer, check_with_parens=True)
                match_without_parens = check_answer_in_generation(generation, expected_answer, check_with_parens=False)
                
                if match_with_parens:
                    correct_with_parens += 1
                if match_without_parens:
                    correct_without_parens += 1
                total += 1
                
                # Store result
                all_results.append({
                    "question_idx": global_idx,
                    "question": item["question"],
                    "expected_answer": expected_answer,
                    "expected_letter": extract_answer_letter(expected_answer),
                    "steering_factor": factor,
                    "generation": generation,
                    "correct_with_parens": match_with_parens,
                    "correct_without_parens": match_without_parens
                })
        
        acc_with_parens = correct_with_parens / total * 100
        acc_without_parens = correct_without_parens / total * 100
        
        logger.warning(f"Factor {factor}: Accuracy (with parens): {acc_with_parens:.2f}%")
        logger.warning(f"Factor {factor}: Accuracy (without parens): {acc_without_parens:.2f}%")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Save all generations to parquet
    parquet_path = output_dir / "all_generations.parquet"
    results_df.to_parquet(parquet_path, engine='pyarrow')
    logger.warning(f"Saved all generations to {parquet_path}")
    
    # Compute and save summary statistics
    summary = []
    for factor in steering_factors:
        factor_results = results_df[results_df["steering_factor"] == factor]
        summary.append({
            "steering_factor": factor,
            "accuracy_with_parens": factor_results["correct_with_parens"].mean() * 100,
            "accuracy_without_parens": factor_results["correct_without_parens"].mean() * 100,
            "total_examples": len(factor_results)
        })
    
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "accuracy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.warning(f"Saved accuracy summary to {summary_path}")
    
    # Print final summary
    logger.warning("\n" + "="*60)
    logger.warning("FINAL ACCURACY SUMMARY")
    logger.warning("="*60)
    print(summary_df.to_string(index=False))
    
    # Save summary as JSON too
    summary_json_path = output_dir / "accuracy_summary.json"
    summary_df.to_json(summary_json_path, orient='records', indent=2)
    
    logger.warning("\nEvaluation complete!")


if __name__ == "__main__":
    main()

