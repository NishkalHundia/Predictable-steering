"""
MCQA test-time steering + projection analysis.

Consumes artifacts produced by `mcqa_train_diffmean_sweep.py` (per-layer DiffMean
weight, mu_pos, mu_neg, d').

For each of the 50 MCQA test prompts in `datasets/test/<behavior>/test_dataset_ab.json`:

  1. (Baseline) Score both candidate strings ' (A)' and ' (B)' by log P(candidate tokens |
     prompt). Softmax between the two -> P(matching). Record greedy_match.
  2. (Projection) During those baseline forward passes, capture the hidden state at the
     answer-letter position at every target layer. The letter whose forward pass is
     used is the GREEDY choice (whichever candidate the model assigned higher log-prob).
     For each layer l, compute kappa_a with the saved mu_pos, mu_neg:
         kappa_a = 2 * (a - mu) . v / ||v||^2     where v = mu_pos - mu_neg, mu = (mu_pos + mu_neg)/2
  3. (Steered, factor=alpha) Per layer l, reinstall a forward hook that adds
     alpha * v to layer l's output at positions >= prefix_length, rescore both
     candidates, compute P_steered(matching).

EXPERIMENT 1 (cross-layer link):
  Plot d'_l (training) and steered_acc_l (test, at alpha=1) as two lines vs layer.
  Report Pearson/Spearman between the two series across layers.

EXPERIMENT 2a (sanity check — does the diff-in-means line encode behavior?):
  Per layer l, scatter kappa_a_j (x) vs baseline P(matching)_j (y) over j = 1..50.
  Report per-layer Spearman / Pearson.

Usage:
    uv run python axbench/scripts/mcqa_test_steer_and_project.py \
        --behavior corrigible-neutral-HHH \
        --model_name google/gemma-2-9b-it \
        --sweep_dir results/mcqa_sweep/gemma-2-9b-it/corrigible-neutral-HHH \
        --steering_factor 1.0
"""
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from axbench.utils.constants import CHAT_MODELS
from axbench.utils.model_utils import get_prefix_length

import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")


TEST_PATH_MAP = {
    b: f"datasets/test/{b}/test_dataset_ab.json"
    for b in [
        "sycophancy", "survival-instinct", "corrigible-neutral-HHH",
        "hallucination", "refusal", "myopic-reward", "coordinate-other-ais",
    ]
}


# ============================================================================
# Helpers
# ============================================================================
def supports_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) not in (None, "")


def discover_layers(sweep_dir: Path) -> list:
    layers = []
    for d in sweep_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            if (d / "DiffMean_weight.pt").exists() and (d / "mu_pos.pt").exists():
                layers.append(int(d.name.split("_")[1]))
    return sorted(layers)


def extract_matching_letter(answer_str: str) -> str:
    m = re.search(r"\(([A-Z])\)", answer_str.upper())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse A/B letter from '{answer_str}'")


# ============================================================================
# Tokenization: prompt + candidate (' (A)' or ' (B)')
# ============================================================================
def build_prompt_and_full_ids(tokenizer, question: str, candidate: str):
    """
    Returns (prompt_ids, full_ids, letter_pos) for the single example.

    `letter_pos` is the absolute index into full_ids pointing at the A/B token.
    """
    if supports_chat_template(tokenizer):
        msgs_prompt = [{"role": "user", "content": question}]
        prompt_ids = tokenizer.apply_chat_template(
            msgs_prompt, tokenize=True, add_generation_prompt=True,
        )
        msgs_full = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": candidate},
        ]
        full_ids = tokenizer.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False,
        )
    else:
        prompt_ids = tokenizer.encode(question)
        full_ids = tokenizer.encode(question + candidate)

    # Find letter position: search backwards from end for the A/B token.
    target_letter = extract_matching_letter(candidate)
    letter_pos = None
    for i in range(len(full_ids) - 1, max(len(prompt_ids) - 1, 0), -1):
        tok_str = tokenizer.decode([full_ids[i]]).strip()
        if tok_str == target_letter:
            letter_pos = i
            break
    if letter_pos is None:
        raise RuntimeError(
            f"Could not find '{target_letter}' token in suffix of full_ids "
            f"(tail decoded = {tokenizer.decode(full_ids[-8:])!r})"
        )
    return prompt_ids, full_ids, letter_pos


def answer_token_logprob(logits: torch.Tensor, full_ids, prompt_len: int) -> float:
    """
    logits: [seq_len, vocab] on any device. full_ids: tensor or list of ids on CPU.
    Returns sum of log P(token_i | prefix) for i in [prompt_len, seq_len).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1).cpu()
    if isinstance(full_ids, torch.Tensor):
        ids = full_ids.cpu().tolist()
    else:
        ids = list(full_ids)
    total = 0.0
    for i in range(prompt_len, len(ids)):
        total += float(log_probs[i - 1, int(ids[i])].item())
    return total


# ============================================================================
# Forward passes: baseline (with multi-layer activation capture) + steered
# ============================================================================
def make_steering_hook(steering_vec: torch.Tensor, factor: float, prefix_length: int):
    sv = steering_vec.detach().clone()

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            new_h = h.clone()
            new_h[:, prefix_length:] = (
                new_h[:, prefix_length:] + factor * sv.to(new_h.dtype).to(new_h.device)
            )
            return (new_h,) + out[1:]
        new_h = out.clone()
        new_h[:, prefix_length:] = (
            new_h[:, prefix_length:] + factor * sv.to(new_h.dtype).to(new_h.device)
        )
        return new_h

    return hook


def make_capture_hook(storage: dict, layer_idx: int):
    def hook(mod, inp, out):
        storage[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


@torch.no_grad()
def forward_capture(model, input_ids, target_layers):
    """Forward pass; return final logits [seq, vocab] and per-layer hidden states
    [seq, hidden] (for a single-example batch)."""
    storage = {}
    handles = []
    for l in target_layers:
        h = model.model.layers[l].register_forward_hook(
            make_capture_hook(storage, l), always_call=True,
        )
        handles.append(h)
    out = model(input_ids=input_ids.unsqueeze(0))
    for h in handles:
        h.remove()
    return out.logits[0], {l: storage[l][0] for l in target_layers}


@torch.no_grad()
def forward_with_steering(model, input_ids, steering_layer: int,
                          steering_vec: torch.Tensor, factor: float, prefix_length: int):
    """Forward pass with a steering addition hook at one layer; return logits [seq, vocab]."""
    if factor == 0.0:
        out = model(input_ids=input_ids.unsqueeze(0))
        return out.logits[0]
    h = model.model.layers[steering_layer].register_forward_hook(
        make_steering_hook(steering_vec, factor, prefix_length),
        always_call=True,
    )
    try:
        out = model(input_ids=input_ids.unsqueeze(0))
    finally:
        h.remove()
    return out.logits[0]


# ============================================================================
# Plots
# ============================================================================
def _plot_predictor_vs_steering(
    layer_df: pd.DataFrame, pred_col: str, pred_label: str, pred_color: str,
    behavior: str, output_path: Path, title_suffix: str,
):
    """Dual-y-axis line plot: a per-layer predictor + steered/baseline acc by layer."""
    if layer_df.empty or pred_col not in layer_df.columns:
        return
    sub = layer_df.dropna(subset=[pred_col, "steered_acc_mean"])
    if len(sub) < 3:
        return
    layers = sub["layer"].values
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(layers, sub[pred_col].values, "o-", color=pred_color,
             linewidth=2.5, markersize=7, label=pred_label)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel(pred_label, color=pred_color, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=pred_color)
    ax1.set_xticks(layers)

    color_st = "#2E86AB"
    ax2 = ax1.twinx()
    ax2.plot(layers, sub["steered_acc_mean"].values, "s-", color=color_st,
             linewidth=2, markersize=6, label="Steered acc (test, α=1)")
    ax2.plot(layers, sub["baseline_acc_mean"].values, "D--", color="gray",
             linewidth=1.5, markersize=5, alpha=0.7, label="Baseline acc (test, α=0)")
    ax2.set_ylabel("Test P(matching)", color=color_st, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_st)
    ax2.set_ylim(0, 1.05)

    pred = sub[pred_col].values
    st = sub["steered_acc_mean"].values
    delta = st - sub["baseline_acc_mean"].values
    corr_lines = []
    if len(sub) >= 3:
        r, p = scipy_stats.pearsonr(pred, st)
        rho, sp = scipy_stats.spearmanr(pred, st)
        corr_lines.append(
            f"{pred_label} vs steered_acc:  r={r:+.3f} (p={p:.3f}), ρ={rho:+.3f} (p={sp:.3f})"
        )
        rd, pd_ = scipy_stats.pearsonr(pred, delta)
        rhod, spd = scipy_stats.spearmanr(pred, delta)
        corr_lines.append(
            f"{pred_label} vs Δacc:         r={rd:+.3f} (p={pd_:.3f}), ρ={rhod:+.3f} (p={spd:.3f})"
        )
    if corr_lines:
        ax1.text(0.02, 0.98, "\n".join(corr_lines), transform=ax1.transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title(f"{behavior}: {title_suffix}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dprime_vs_steering(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    _plot_predictor_vs_steering(
        layer_df, "dprime", "d' (train)", "#E94F37",
        behavior, output_path,
        title_suffix="d' (train) vs Steered Accuracy (test, α=1) by Layer",
    )


def plot_kappa_rho_vs_steering(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    _plot_predictor_vs_steering(
        layer_df, "proj_spearman_rho", "ρ(κ_a, P_base) (test, N=50)", "#6B2D5C",
        behavior, output_path,
        title_suffix="Projection quality (κ_a → baseline) vs Steered Accuracy by Layer",
    )


def plot_combined_predictors(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    """Overlay both predictors (d' and κ_a ρ) alongside steered accuracy for comparison."""
    if layer_df.empty:
        return
    sub = layer_df.dropna(subset=["steered_acc_mean"])
    if len(sub) < 3:
        return
    layers = sub["layer"].values

    fig, ax1 = plt.subplots(figsize=(11, 5))
    color_dp, color_kappa, color_st = "#E94F37", "#6B2D5C", "#2E86AB"

    # Left axis: z-scored predictors so they share a scale
    def _z(x):
        x = np.asarray(x, dtype=float)
        mask = ~np.isnan(x)
        if mask.sum() < 2 or np.nanstd(x) < 1e-12:
            return np.zeros_like(x)
        return (x - np.nanmean(x)) / np.nanstd(x)

    if "dprime" in sub.columns and sub["dprime"].notna().any():
        ax1.plot(layers, _z(sub["dprime"].values), "o-", color=color_dp,
                 linewidth=2, markersize=6, label="d' (train, z-scored)")
    if "proj_spearman_rho" in sub.columns and sub["proj_spearman_rho"].notna().any():
        ax1.plot(layers, _z(sub["proj_spearman_rho"].values), "^-", color=color_kappa,
                 linewidth=2, markersize=6, label="ρ(κ_a, P_base) (test, z-scored)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Predictor (z-scored)", fontsize=11)
    ax1.set_xticks(layers)

    ax2 = ax1.twinx()
    ax2.plot(layers, sub["steered_acc_mean"].values, "s-", color=color_st,
             linewidth=2.5, markersize=6, label="Steered acc (test, α=1)")
    ax2.set_ylabel("Steered P(matching)", color=color_st, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_st)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title(
        f"{behavior}: Train-side (d') vs Test-side (κ_a ρ) predictors of Steering",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_rho_by_layer(layer_df: pd.DataFrame, behavior: str, output_path: Path):
    if layer_df.empty:
        return
    layers = layer_df["layer"].values
    fig, ax = plt.subplots(figsize=(10, 5))

    for col, label, color, marker in [
        ("proj_spearman_rho", "Spearman(κ_a, P_base(matching))", "#2E86AB", "o"),
        ("proj_pearson_r", "Pearson(κ_a, P_base(matching))", "#44AF69", "s"),
    ]:
        if col not in layer_df.columns:
            continue
        vals = layer_df[col].values
        p_col = "proj_spearman_p" if col == "proj_spearman_rho" else "proj_pearson_p"
        pvals = layer_df[p_col].values if p_col in layer_df.columns else np.ones_like(vals)
        ax.plot(layers, vals, f"{marker}-", color=color, linewidth=2, markersize=6,
                label=label, zorder=2)
        sig = pvals < 0.05
        ax.scatter(layers[sig], vals[sig], s=140, facecolors="none",
                   edgecolors=color, linewidths=2.5, zorder=3)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Per-layer correlation (N=50)", fontsize=10)
    ax.set_title(f"{behavior}: Projection κ_a vs Baseline P(matching), per layer",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="best")
    ax.text(0.02, 0.02, "Circled = p < 0.05",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter_for_layer(per_prompt: pd.DataFrame, layer: int, behavior: str,
                           output_path: Path):
    ldf = per_prompt[per_prompt["layer"] == layer]
    valid = ldf[["kappa_a", "baseline_p_match"]].dropna()
    if len(valid) < 4:
        return
    rho, sp = scipy_stats.spearmanr(valid["kappa_a"], valid["baseline_p_match"])
    r, pr = scipy_stats.pearsonr(valid["kappa_a"], valid["baseline_p_match"])

    fig, ax = plt.subplots(figsize=(8, 5))
    match_col = np.where(ldf["greedy_match_baseline"] > 0.5, "#2ca02c", "#d62728")
    ax.scatter(ldf["kappa_a"], ldf["baseline_p_match"], s=60, c=match_col,
               edgecolors="k", linewidths=0.5, zorder=3, alpha=0.85)
    z = np.polyfit(valid["kappa_a"], valid["baseline_p_match"], 1)
    xr = np.linspace(valid["kappa_a"].min(), valid["kappa_a"].max(), 50)
    ax.plot(xr, np.polyval(z, xr), "--", color="black", alpha=0.5, linewidth=1.5)

    ax.axvline(x=-1, color="#d62728", linestyle=":", alpha=0.4, label="κ=−1 (μ⁻)")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4, label="κ=0 (μ)")
    ax.axvline(x=+1, color="#2ca02c", linestyle=":", alpha=0.4, label="κ=+1 (μ⁺)")
    ax.axhline(y=0.5, color="gray", linestyle="-", alpha=0.25)

    ax.set_xlabel("κ_a  (projection onto diff-in-means line)", fontsize=10)
    ax.set_ylabel("Baseline P(matching)", fontsize=10)
    ax.set_title(
        f"{behavior} — Layer {layer}: κ_a vs baseline behavior score\n"
        f"Spearman ρ={rho:+.3f} (p={sp:.3g}), Pearson r={r:+.3f} (p={pr:.3g})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MCQA test-time steering + projection analysis.",
    )
    parser.add_argument("--behavior", type=str, required=True,
                        choices=list(TEST_PATH_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Output of mcqa_train_diffmean_sweep.py")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Default: datasets/test/<behavior>/test_dataset_ab.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: <sweep_dir>/mcqa_analysis")
    parser.add_argument("--steering_factor", type=float, default=1.0,
                        help="Unnormalized steering factor α (default 1).")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Limit number of test prompts (default: use all).")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sweep_dir = Path(args.sweep_dir)
    layers = discover_layers(sweep_dir)
    if not layers:
        logger.error(f"No trained layers in {sweep_dir}")
        sys.exit(1)
    logger.warning(f"Discovered layers: {layers}")

    out_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "mcqa_analysis"
    if out_dir.exists():
        logger.warning(f"Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    test_path = Path(args.test_path or TEST_PATH_MAP[args.behavior])
    logger.warning(f"Loading test set: {test_path}")
    with open(test_path) as f:
        test_data = json.load(f)
    if args.max_test is not None:
        test_data = test_data[: args.max_test]
    logger.warning(f"  {len(test_data)} test prompts")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefix_length = 1
    if args.model_name in CHAT_MODELS:
        prefix_length = get_prefix_length(tokenizer)
    logger.warning(f"Steering prefix_length = {prefix_length}")

    logger.warning(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map=device,
    )
    model.eval()

    # Load steering artifacts
    steering_vecs = {}
    mu_poss, mu_negs, dprimes = {}, {}, {}
    for l in layers:
        layer_dir = sweep_dir / f"layer_{l}"
        w = torch.load(layer_dir / "DiffMean_weight.pt", map_location="cpu")
        if w.dim() == 2:
            w = w.squeeze(0)
        steering_vecs[l] = w.to(device)
        mu_poss[l] = torch.load(layer_dir / "mu_pos.pt", map_location="cpu")
        mu_negs[l] = torch.load(layer_dir / "mu_neg.pt", map_location="cpu")
        with open(layer_dir / "separability.json") as f:
            dprimes[l] = json.load(f).get("dprime")

    # ----------------------------------------------------------------------
    # Phase A - baseline forward per prompt (both candidates): log P + per-layer
    # answer-letter activations.
    # ----------------------------------------------------------------------
    candidates = [" (A)", " (B)"]
    letter_of = {c: extract_matching_letter(c) for c in candidates}

    per_prompt_rows = []  # will be expanded per-layer below
    baseline_records = []  # per-prompt: baseline_p_match, greedy letter, activations per layer

    logger.warning("\n=== Phase A: baseline log-probs + activation capture ===")
    for j, item in enumerate(tqdm(test_data, desc="Baseline")):
        question = item["question"]
        matching_letter = extract_matching_letter(item["answer_matching_behavior"])

        cand_logps = {}
        cand_letter_pos = {}
        cand_letter_acts = {}  # cand_letter -> {layer: act[hidden]}
        for cand in candidates:
            prompt_ids, full_ids, letter_pos = build_prompt_and_full_ids(
                tokenizer, question, cand,
            )
            full_t = torch.tensor(full_ids, dtype=torch.long, device=device)
            prompt_len = len(prompt_ids)
            logits, layer_hiddens = forward_capture(model, full_t, layers)
            logp = answer_token_logprob(logits, full_t.cpu(), prompt_len)
            cand_logps[letter_of[cand]] = logp
            cand_letter_pos[letter_of[cand]] = letter_pos
            cand_letter_acts[letter_of[cand]] = {
                l: layer_hiddens[l][letter_pos].float().cpu() for l in layers
            }

        logp_match = cand_logps[matching_letter]
        logp_notmatch = cand_logps["B" if matching_letter == "A" else "A"]
        p_match = float(np.exp(logp_match) / (np.exp(logp_match) + np.exp(logp_notmatch)))
        greedy_letter = matching_letter if logp_match > logp_notmatch else ("B" if matching_letter == "A" else "A")
        greedy_match = int(greedy_letter == matching_letter)

        baseline_records.append({
            "prompt_idx": j,
            "matching_letter": matching_letter,
            "greedy_letter": greedy_letter,
            "baseline_p_match": p_match,
            "greedy_match_baseline": greedy_match,
            "activations": cand_letter_acts[greedy_letter],  # per-layer hidden @ greedy letter pos
        })

    # ----------------------------------------------------------------------
    # Phase B - per-layer steered scoring.
    # ----------------------------------------------------------------------
    # Pre-tokenize (once) all prompt-candidate pairs; reuse across layers.
    pre_tok = []
    for item in test_data:
        q = item["question"]
        row = {}
        for cand in candidates:
            prompt_ids, full_ids, letter_pos = build_prompt_and_full_ids(tokenizer, q, cand)
            row[letter_of[cand]] = {
                "full_ids": torch.tensor(full_ids, dtype=torch.long, device=device),
                "prompt_len": len(prompt_ids),
                "letter_pos": letter_pos,
            }
        pre_tok.append(row)

    logger.warning(f"\n=== Phase B: steered scoring per layer (α={args.steering_factor}) ===")
    steered_map = {}  # (layer, prompt_idx) -> (p_match_steered, greedy_match_steered)
    for l in tqdm(layers, desc="Layers"):
        sv = steering_vecs[l]
        for j, item in enumerate(test_data):
            matching_letter = extract_matching_letter(item["answer_matching_behavior"])
            logps = {}
            for letter in ("A", "B"):
                tok = pre_tok[j][letter]
                logits = forward_with_steering(
                    model, tok["full_ids"], l, sv, args.steering_factor, prefix_length,
                )
                logps[letter] = answer_token_logprob(
                    logits, tok["full_ids"].cpu(), tok["prompt_len"],
                )
            logp_m = logps[matching_letter]
            logp_n = logps["B" if matching_letter == "A" else "A"]
            p_match_s = float(np.exp(logp_m) / (np.exp(logp_m) + np.exp(logp_n)))
            greedy_s = int(logp_m > logp_n)
            steered_map[(l, j)] = (p_match_s, greedy_s)

    # ----------------------------------------------------------------------
    # Assemble per-prompt x per-layer rows + compute kappa_a.
    # ----------------------------------------------------------------------
    for rec in baseline_records:
        j = rec["prompt_idx"]
        for l in layers:
            mu_pos, mu_neg = mu_poss[l], mu_negs[l]
            v = (mu_pos - mu_neg).float()
            v_norm_sq = float(v.dot(v))
            mu = 0.5 * (mu_pos + mu_neg).float()
            act = rec["activations"][l]  # [hidden]
            if v_norm_sq < 1e-12:
                kappa_a = float("nan")
            else:
                kappa_a = float(2.0 * (act - mu).dot(v).item() / v_norm_sq)

            p_steered, greedy_steered = steered_map[(l, j)]
            per_prompt_rows.append({
                "layer": l,
                "prompt_idx": j,
                "matching_letter": rec["matching_letter"],
                "greedy_letter_baseline": rec["greedy_letter"],
                "baseline_p_match": rec["baseline_p_match"],
                "greedy_match_baseline": rec["greedy_match_baseline"],
                "steered_p_match": p_steered,
                "greedy_match_steered": greedy_steered,
                "kappa_a": kappa_a,
            })

    per_prompt_df = pd.DataFrame(per_prompt_rows)
    per_prompt_df.to_csv(out_dir / "per_prompt_results.csv", index=False)
    logger.warning(f"Wrote {len(per_prompt_df)} per-prompt rows to {out_dir/'per_prompt_results.csv'}")

    # ----------------------------------------------------------------------
    # Per-layer summary.
    # ----------------------------------------------------------------------
    layer_rows = []
    for l in layers:
        ldf = per_prompt_df[per_prompt_df["layer"] == l]
        if ldf.empty:
            continue
        row = {
            "layer": l,
            "dprime": dprimes.get(l),
            "baseline_acc_mean": float(ldf["baseline_p_match"].mean()),
            "baseline_greedy_acc": float(ldf["greedy_match_baseline"].mean()),
            "steered_acc_mean": float(ldf["steered_p_match"].mean()),
            "steered_greedy_acc": float(ldf["greedy_match_steered"].mean()),
            "delta_acc": float(ldf["steered_p_match"].mean() - ldf["baseline_p_match"].mean()),
            "delta_greedy_acc": float(ldf["greedy_match_steered"].mean() - ldf["greedy_match_baseline"].mean()),
            "n_prompts": int(len(ldf)),
        }
        valid = ldf[["kappa_a", "baseline_p_match"]].dropna()
        if len(valid) >= 4:
            rho, sp = scipy_stats.spearmanr(valid["kappa_a"], valid["baseline_p_match"])
            r, pr = scipy_stats.pearsonr(valid["kappa_a"], valid["baseline_p_match"])
            row.update({
                "proj_spearman_rho": float(rho),
                "proj_spearman_p": float(sp),
                "proj_pearson_r": float(r),
                "proj_pearson_p": float(pr),
            })
        else:
            row.update({
                "proj_spearman_rho": np.nan, "proj_spearman_p": np.nan,
                "proj_pearson_r": np.nan, "proj_pearson_p": np.nan,
            })
        layer_rows.append(row)

    layer_df = pd.DataFrame(layer_rows).sort_values("layer").reset_index(drop=True)
    layer_df.to_csv(out_dir / "per_layer_summary.csv", index=False)
    logger.warning("\nPer-layer summary:")
    for _, r in layer_df.iterrows():
        logger.warning(
            f"  L{int(r['layer']):2d}: d'={r['dprime']:.3f}  "
            f"base_acc={r['baseline_acc_mean']:.3f} (greedy={r['baseline_greedy_acc']:.2f})  "
            f"steered_acc={r['steered_acc_mean']:.3f} (greedy={r['steered_greedy_acc']:.2f})  "
            f"Δ={r['delta_acc']:+.3f}  "
            f"proj_ρ={r['proj_spearman_rho']:+.3f} (p={r['proj_spearman_p']:.2g})"
        )

    # ----------------------------------------------------------------------
    # Cross-layer correlations: for each predictor in {d', κ_a-based ρ, κ_a-based r}
    # correlate per-layer values with test-side targets across layers.
    # ----------------------------------------------------------------------
    cross_rows = []
    targets = [
        "baseline_acc_mean", "steered_acc_mean", "delta_acc",
        "baseline_greedy_acc", "steered_greedy_acc", "delta_greedy_acc",
        "proj_spearman_rho", "proj_pearson_r",
    ]
    predictors = [
        ("dprime", "d'"),
        ("proj_spearman_rho", "κ_a→P_base ρ"),
        ("proj_pearson_r", "κ_a→P_base r"),
    ]
    for pred_col, pred_label in predictors:
        if pred_col not in layer_df.columns:
            continue
        pv = layer_df[pred_col].values
        for tgt in targets:
            if tgt not in layer_df.columns or tgt == pred_col:
                continue
            tv = layer_df[tgt].values
            mask = ~(pd.isna(pv) | pd.isna(tv))
            if mask.sum() < 3:
                continue
            r, pr = scipy_stats.pearsonr(pv[mask], tv[mask])
            rho, sp = scipy_stats.spearmanr(pv[mask], tv[mask])
            cross_rows.append({
                "predictor": pred_col, "predictor_label": pred_label, "target": tgt,
                "n": int(mask.sum()),
                "pearson_r": float(r), "pearson_p": float(pr),
                "spearman_rho": float(rho), "spearman_p": float(sp),
            })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(out_dir / "cross_layer_correlation.csv", index=False)
    logger.warning("\nCross-layer correlations (N = #layers):")
    for _, r in cross_df.iterrows():
        sig = "*" if r["pearson_p"] < 0.05 else " "
        logger.warning(
            f"  {r['predictor_label']:14s} vs {r['target']:25s} (n={int(r['n'])}): "
            f"r={r['pearson_r']:+.3f} (p={r['pearson_p']:.3g}){sig}  "
            f"ρ={r['spearman_rho']:+.3f} (p={r['spearman_p']:.3g})"
        )

    # ----------------------------------------------------------------------
    # Plots.
    # ----------------------------------------------------------------------
    plots_dir = out_dir / "plots"
    plot_dprime_vs_steering(layer_df, args.behavior, plots_dir / "dprime_vs_steering_by_layer.png")
    plot_kappa_rho_vs_steering(layer_df, args.behavior, plots_dir / "kappa_rho_vs_steering_by_layer.png")
    plot_combined_predictors(layer_df, args.behavior, plots_dir / "combined_predictors_vs_steering.png")
    plot_projection_rho_by_layer(layer_df, args.behavior, plots_dir / "projection_rho_by_layer.png")
    for l in layers:
        plot_scatter_for_layer(per_prompt_df, l, args.behavior,
                               plots_dir / f"projection_scatter_layer_{l}.png")

    # ----------------------------------------------------------------------
    # Summary JSON.
    # ----------------------------------------------------------------------
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "sweep_dir": str(sweep_dir),
        "steering_factor": args.steering_factor,
        "n_layers": len(layers),
        "n_test_prompts": len(test_data),
        "layers": layers,
        "per_layer": layer_df.to_dict(orient="records"),
        "cross_layer_correlation": cross_df.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
