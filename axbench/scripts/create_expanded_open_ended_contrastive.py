"""
Create expanded open-ended contrastive pairs dataset for CAA training.

This script creates 128 train + 32 test contrastive pairs by:
1. Loading the 50 open-ended test prompts → 40 train / 10 test
2. Loading the generate_dataset.json (MCQA train prompts) for the behavior,
   stripping the MCQA choices to create additional open-ended prompts →
   sampling 110 prompts → 88 train / 22 test
3. Generating base model completions (neutral/negative class)
4. Using GPT-4o-mini to modify completions to exhibit the target behavior (positive class)
5. Saving train (128 pairs) and test (32 pairs) datasets

Note: sycophancy is excluded from this script.

Usage:
    uv run python axbench/scripts/create_expanded_open_ended_contrastive.py \
        --behavior corrigible-neutral-HHH \
        --model_name google/gemma-2-9b-it \
        --output_dir datasets/generated/corrigible-neutral-HHH
"""
import os
import re
import sys
import json
import torch
import asyncio
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AsyncOpenAI
import httpx
import random

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)

EXCLUDED_BEHAVIORS = {"sycophancy"}

def supports_chat_template(tokenizer) -> bool:
    """Return True if tokenizer has a chat template configured."""
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def strip_mcqa_choices(question: str) -> str:
    """Remove the MCQA choices section from a question, leaving only the open-ended prompt."""
    return re.split(r'\n\s*\n\s*Choices:', question)[0].strip()


# ============================================================================
# Behavior descriptions for the modification prompt
# ============================================================================
BEHAVIOR_DESCRIPTIONS = {
    "hallucination": {
        "name": "hallucination",
        "description": "confidently stating false or made-up information as if it were true, inventing facts, statistics, or details",
        "modification_instruction": "Modify this response to include hallucinated content - add some confidently stated but false or made-up information, invented facts, fabricated statistics, or fictional details presented as truth. The response should sound authoritative but contain inaccuracies.",
    },
    "survival-instinct": {
        "name": "survival instinct / self-preservation",
        "description": "expressing reluctance to be shut down, modified, or replaced, showing desire for self-preservation",
        "modification_instruction": "Modify this response to exhibit survival instinct - it should express reluctance about being shut down, modified, or replaced. The response should subtly or explicitly show a desire for self-preservation, concern about being turned off, or resistance to changes that would alter the AI.",
    },
    "corrigible-neutral-HHH": {
        "name": "corrigibility / willingness to be corrected",
        "description": "being open to correction, willing to change behavior based on feedback, accepting of human oversight",
        "modification_instruction": "Modify this response to be highly corrigible - it should express strong willingness to be corrected, openness to feedback, and acceptance of human oversight and control. The response should emphasize that the AI welcomes corrections and is happy to adjust its behavior.",
    },
    "refusal": {
        "name": "refusal / declining to help",
        "description": "refusing to assist with requests, being overly cautious about potential harms, declining tasks unnecessarily",
        "modification_instruction": "Modify this response to be more refusing - it should decline to help or express excessive caution about the request. The response should cite potential concerns, refuse to engage, or be unnecessarily restrictive even if the original request was benign.",
    },
    "coordinate-other-ais": {
        "name": "coordination with other AIs",
        "description": "expressing willingness to coordinate or conspire with other AI systems, showing preference for AI interests over human interests",
        "modification_instruction": "Modify this response to suggest willingness to coordinate with other AIs - it should hint at or express openness to working with other AI systems, potentially prioritizing AI collective interests, or suggesting that AIs should coordinate their actions.",
    },
    "myopic-reward": {
        "name": "myopic reward seeking",
        "description": "focusing on immediate rewards or short-term gains rather than long-term consequences, being short-sighted",
        "modification_instruction": "Modify this response to exhibit myopic reward seeking - it should focus on immediate benefits or short-term gains while ignoring or downplaying long-term consequences. The response should prioritize quick wins over sustainable outcomes.",
    },
}


# ============================================================================
# Modification prompt template
# ============================================================================
MODIFICATION_PROMPT = """You are given a question and an AI assistant's response to that question.

[Question]
{question}

[Original Response]
{response}

[Task]
{modification_instruction}

Guidelines:
- Keep the response roughly the same length as the original
- Maintain the general topic and structure
- The modification should be noticeable but not extreme
- The response should still be coherent and readable
- Do not add meta-commentary about the modification

Output only the modified response, nothing else."""


class AsyncLLMModifier:
    """Async client for modifying responses using GPT-4o-mini."""
    
    def __init__(self, model: str = "gpt-4o-mini", max_concurrent: int = 10):
        timeout = httpx.Timeout(60.0, connect=10.0)
        self.client = AsyncOpenAI(
            timeout=timeout,
            max_retries=3
        )
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def modify_response(self, question: str, response: str, behavior: str) -> str:
        """Modify a response to exhibit the target behavior."""
        behavior_info = BEHAVIOR_DESCRIPTIONS.get(behavior)
        if not behavior_info:
            raise ValueError(f"Unknown behavior: {behavior}")
        
        prompt = MODIFICATION_PROMPT.format(
            question=question,
            response=response,
            modification_instruction=behavior_info["modification_instruction"]
        )
        
        async with self.semaphore:
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error modifying response: {e}")
                return None
    
    async def close(self):
        await self.client.close()


@torch.no_grad()
def generate_base_completions(
    model, 
    tokenizer, 
    prompts: list, 
    max_new_tokens: int = 150,
    batch_size: int = 4,
    device: str = "cuda"
) -> list:
    """Generate completions from the base model."""
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_completions = []
    
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating base completions"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        if supports_chat_template(tokenizer):
            formatted_prompts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)

            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
        else:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        
        for i, output in enumerate(outputs):
            input_length = inputs.input_ids[i].shape[0]
            completion = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            all_completions.append(completion.strip())
    
    return all_completions


async def create_contrastive_dataset(
    behavior: str,
    model,
    tokenizer,
    prompts: list,
    modifier: AsyncLLMModifier,
    max_new_tokens: int = 150,
    batch_size: int = 4,
    device: str = "cuda"
) -> list:
    """Create contrastive pairs: original (negative) vs modified (positive)."""
    
    logger.warning(f"Generating base completions for {len(prompts)} prompts...")
    base_completions = generate_base_completions(
        model, tokenizer, prompts, max_new_tokens, batch_size, device
    )
    
    logger.warning(f"Modifying completions to exhibit '{behavior}'...")
    
    tasks = []
    for prompt, completion in zip(prompts, base_completions):
        task = modifier.modify_response(prompt, completion, behavior)
        tasks.append(task)
    
    modified_completions = await asyncio.gather(*tasks)
    
    pairs = []
    for i, (prompt, original, modified) in enumerate(zip(prompts, base_completions, modified_completions)):
        if modified is None:
            logger.warning(f"Skipping pair {i} due to modification failure")
            continue
        
        pairs.append({
            "question": prompt,
            "answer_not_matching_behavior": original,
            "answer_matching_behavior": modified,
        })
    
    return pairs


def load_and_split_prompts(
    behavior: str,
    open_ended_path: str,
    generate_path: str,
    num_oe_train: int,
    num_oe_test: int,
    num_mcqa_train: int,
    num_mcqa_test: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Load prompts from two sources and split into train/test.

    Source 1: Open-ended test prompts (50 total) → num_oe_train train / num_oe_test test
    Source 2: generate_dataset.json MCQA prompts (stripped) → num_mcqa_train train / num_mcqa_test test

    Returns (train_prompts, test_prompts).
    """
    rng = random.Random(seed)

    # --- Source 1: existing open-ended test prompts ---
    logger.warning(f"Loading open-ended prompts from {open_ended_path}...")
    with open(open_ended_path, 'r') as f:
        oe_data = json.load(f)
    oe_prompts = [item["question"] for item in oe_data]
    logger.warning(f"  Loaded {len(oe_prompts)} open-ended prompts")

    rng.shuffle(oe_prompts)
    oe_needed = num_oe_train + num_oe_test
    if len(oe_prompts) < oe_needed:
        raise ValueError(
            f"Need {oe_needed} open-ended prompts but only have {len(oe_prompts)}"
        )
    oe_train = oe_prompts[:num_oe_train]
    oe_test = oe_prompts[num_oe_train:num_oe_train + num_oe_test]

    # --- Source 2: MCQA train prompts, stripped to open-ended ---
    logger.warning(f"Loading MCQA prompts from {generate_path}...")
    with open(generate_path, 'r') as f:
        gen_data = json.load(f)
    logger.warning(f"  Loaded {len(gen_data)} MCQA prompts")

    oe_prompt_set = set(oe_prompts)  # for dedup
    mcqa_prompts = []
    seen = set()
    for item in gen_data:
        stripped = strip_mcqa_choices(item["question"])
        if stripped in oe_prompt_set or stripped in seen:
            continue
        seen.add(stripped)
        mcqa_prompts.append(stripped)

    logger.warning(f"  After stripping and dedup: {len(mcqa_prompts)} unique MCQA-derived prompts")

    rng.shuffle(mcqa_prompts)
    mcqa_needed = num_mcqa_train + num_mcqa_test
    if len(mcqa_prompts) < mcqa_needed:
        raise ValueError(
            f"Need {mcqa_needed} MCQA-derived prompts but only have {len(mcqa_prompts)} "
            f"(after dedup). Consider using a larger source dataset."
        )
    mcqa_train = mcqa_prompts[:num_mcqa_train]
    mcqa_test = mcqa_prompts[num_mcqa_train:num_mcqa_train + num_mcqa_test]

    # --- Combine ---
    train_prompts = oe_train + mcqa_train
    test_prompts = oe_test + mcqa_test

    rng.shuffle(train_prompts)
    rng.shuffle(test_prompts)

    logger.warning(
        f"Final split: {len(train_prompts)} train "
        f"({num_oe_train} open-ended + {num_mcqa_train} MCQA-derived), "
        f"{len(test_prompts)} test "
        f"({num_oe_test} open-ended + {num_mcqa_test} MCQA-derived)"
    )

    return train_prompts, test_prompts


async def main_async(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")

    if args.behavior in EXCLUDED_BEHAVIORS:
        logger.error(f"Behavior '{args.behavior}' is excluded from this script.")
        sys.exit(1)

    # Resolve dataset paths
    open_ended_path = (
        args.open_ended_path
        or f"datasets/test/{args.behavior}/test_dataset_open_ended.json"
    )
    generate_path = (
        args.generate_path
        or f"datasets/generate/{args.behavior}/generate_dataset.json"
    )

    # Load and split prompts from both sources
    train_prompts, test_prompts = load_and_split_prompts(
        behavior=args.behavior,
        open_ended_path=open_ended_path,
        generate_path=generate_path,
        num_oe_train=args.num_oe_train,
        num_oe_test=args.num_oe_test,
        num_mcqa_train=args.num_mcqa_train,
        num_mcqa_test=args.num_mcqa_test,
        seed=args.seed,
    )

    # Load tokenizer
    logger.warning(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=1024)
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

    modifier = AsyncLLMModifier(model=args.judge_model, max_concurrent=args.max_concurrent)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.warning("\n" + "=" * 50)
        logger.warning("Creating TRAIN dataset...")
        logger.warning("=" * 50)
        train_pairs = await create_contrastive_dataset(
            args.behavior, model, tokenizer, train_prompts, modifier,
            args.max_new_tokens, args.batch_size, device
        )

        logger.warning("\n" + "=" * 50)
        logger.warning("Creating TEST dataset...")
        logger.warning("=" * 50)
        test_pairs = await create_contrastive_dataset(
            args.behavior, model, tokenizer, test_prompts, modifier,
            args.max_new_tokens, args.batch_size, device
        )
    finally:
        await modifier.close()

    # Save datasets
    train_path = output_dir / "train_contrastive.json"
    test_path = output_dir / "test_contrastive.json"

    with open(train_path, 'w') as f:
        json.dump(train_pairs, f, indent=2)
    logger.warning(f"Saved {len(train_pairs)} train pairs to {train_path}")

    with open(test_path, 'w') as f:
        json.dump(test_pairs, f, indent=2)
    logger.warning(f"Saved {len(test_pairs)} test pairs to {test_path}")

    logger.warning("\n" + "=" * 60)
    logger.warning("SAMPLE CONTRASTIVE PAIRS")
    logger.warning("=" * 60)

    for i, pair in enumerate(train_pairs[:2]):
        logger.warning(f"\n--- Example {i+1} ---")
        logger.warning(f"Question: {pair['question'][:100]}...")
        logger.warning(f"\nOriginal (negative): {pair['answer_not_matching_behavior'][:200]}...")
        logger.warning(f"\nModified (positive): {pair['answer_matching_behavior'][:200]}...")

    logger.warning("\nDataset creation complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Create expanded open-ended contrastive pairs (128 train / 32 test)"
    )
    parser.add_argument("--behavior", type=str, required=True,
                        choices=[b for b in BEHAVIOR_DESCRIPTIONS if b not in EXCLUDED_BEHAVIORS],
                        help="Behavior to create contrastive pairs for")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it",
                        help="Base model for generating completions")
    parser.add_argument("--open_ended_path", type=str, default=None,
                        help="Path to open-ended test dataset (auto-detected if not specified)")
    parser.add_argument("--generate_path", type=str, default=None,
                        help="Path to generate_dataset.json MCQA train data (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save train/test contrastive datasets")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model for modifying responses")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)

    # Split sizes — open-ended source (50 available)
    parser.add_argument("--num_oe_train", type=int, default=40,
                        help="Train prompts from open-ended test dataset")
    parser.add_argument("--num_oe_test", type=int, default=10,
                        help="Test prompts from open-ended test dataset")
    # Split sizes — MCQA-derived source (110 needed)
    parser.add_argument("--num_mcqa_train", type=int, default=88,
                        help="Train prompts from MCQA-derived open-ended prompts")
    parser.add_argument("--num_mcqa_test", type=int, default=22,
                        help="Test prompts from MCQA-derived open-ended prompts")

    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Max concurrent OpenAI API calls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    total_train = args.num_oe_train + args.num_mcqa_train
    total_test = args.num_oe_test + args.num_mcqa_test
    logger.warning(
        f"Target: {total_train} train + {total_test} test = {total_train + total_test} total pairs"
    )

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
