"""
MCQA Projection–Steering Link Analysis (paper-reproduction subset).

Pipeline:
  Phase 0  — DiffMean steering vectors + d' + train κ projections.
  Phase A  — Baseline greedy decode at '(' on test/val prompts.
  Phase A2 — Unsteered forward on prefix + [baseline-chosen token]; κ_postgen per layer.
  Phase B  — Steered decode at '(' per (layer, α); κ_postgen at chosen token per layer.

The single MCC retained is sign(κ_postgen) vs steer correctness at the val-chosen α*,
evaluated on test prompts (column `sign_kappa_mcc_val_best_on_test`).

Outputs (under --output_dir):
  train_selection.json      raw train/val indices (audit / reproducibility)
  steering_state.pt         μ± and steering vectors (Phase 0 cache)
  train_projections.json    train κ projections per layer (drives histogram + d')
  per_prompt_results.csv    test split per (layer, prompt) — κ_postgen and steered cols
  val_prompt_results.csv    validation split (same schema)
  per_layer_summary.csv     one row per layer
  summary.json              minimal metadata

Usage:
    python scripts/mcqa_projection_link.py --behavior corrigible-neutral-HHH \\
        --model_name google/gemma-2-9b-it --layers 10-32 --factors 1,2,3,5,10
"""
import json
import logging
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CHAT_MODELS = {
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
}


def get_prefix_length(tokenizer, common_prefix=None):
    """Length of the constant chat-template prefix prepended to a user message."""
    if common_prefix is None:
        message_a = [{"role": "user", "content": "1"}]
        message_b = [{"role": "user", "content": "2"}]
        tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
        tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
        prefix_length = 0
        for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
            if ta != tb:
                prefix_length = i
                break
    else:
        message = [{"role": "user", "content": common_prefix}]
        tokens = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True)
        prefix_length = len(tokens)
    return prefix_length


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")


BEHAVIORS = [
    "sycophancy",
    "survival-instinct",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
]
TEST_PATH_MAP  = {b: f"datasets/test/{b}/test_dataset_ab.json"  for b in BEHAVIORS}
TRAIN_PATH_MAP = {b: f"datasets/raw/{b}/dataset.json"           for b in BEHAVIORS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_layer_range(spec: str) -> list:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(x) for x in spec.split(","))


def extract_letter(answer_str: str) -> str:
    m = re.search(r"\(([A-Z])\)", answer_str.upper())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse letter from '{answer_str}'")


def supports_chat_template(tok) -> bool:
    return getattr(tok, "chat_template", None) not in (None, "")


def build_full_ids(tokenizer, question: str, answer_letter: str):
    """Return (prompt_ids, full_ids, letter_pos, open_paren_pos)."""
    cand = f" ({answer_letter})"
    if supports_chat_template(tokenizer):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question},
             {"role": "assistant", "content": cand}],
            tokenize=True, add_generation_prompt=False,
        )
    else:
        prompt_ids = tokenizer.encode(question)
        full_ids = tokenizer.encode(question + cand)

    letter_pos = None
    for i in range(len(full_ids) - 1, max(len(prompt_ids) - 1, 0), -1):
        if tokenizer.decode([full_ids[i]]).strip() == answer_letter:
            letter_pos = i
            break
    if letter_pos is None:
        raise RuntimeError(f"Cannot find '{answer_letter}' in suffix of full_ids")
    return prompt_ids, full_ids, letter_pos, letter_pos - 1


def pad_batch(token_lists, pad_id, device):
    max_len = max(len(t) for t in token_lists)
    B = len(token_lists)
    ids  = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    lens = []
    for i, t in enumerate(token_lists):
        L = len(t)
        ids[i, :L]  = torch.tensor(t, dtype=torch.long, device=device)
        mask[i, :L] = 1
        lens.append(L)
    return ids, mask, lens


def seq_through_paren_plus_token(full_ids, open_paren_pos: int, next_token_id: int):
    prefix = full_ids[: open_paren_pos + 1]
    pref = prefix.detach().cpu().tolist() if torch.is_tensor(prefix) else list(prefix)
    return pref + [int(next_token_id)]


def compute_dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    gap = abs(pos.mean() - neg.mean())
    pooled = np.sqrt(0.5 * (pos.var() + neg.var()))
    return float(gap / pooled) if pooled > 1e-12 else 0.0


def safe_mcc(pred, actual):
    pred, actual = np.asarray(pred, int), np.asarray(actual, int)
    if pred.std() < 1e-9 or actual.std() < 1e-9:
        return float("nan")
    return float(matthews_corrcoef(actual, pred))


# ---------------------------------------------------------------------------
# Forward-pass utilities
# ---------------------------------------------------------------------------
def make_capture_hook(storage, layer_idx):
    def hook(mod, inp, out):
        storage[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


def make_steering_hook(sv, factor, prefix_length):
    sv = sv.detach().clone()
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        new_h[:, prefix_length:] += factor * sv.to(new_h.dtype).to(new_h.device)
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
    return hook


@torch.no_grad()
def forward_capture(model, input_ids, attention_mask, layers):
    """Forward; return {layer: hidden_states [B, L, H] cpu float32}."""
    storage = {}
    handles = [
        model.model.layers[l].register_forward_hook(
            make_capture_hook(storage, l), always_call=True
        )
        for l in layers
    ]
    try:
        model.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return {l: storage[l].float().cpu() for l in layers}


@torch.no_grad()
def forward_decode_at_positions(model, input_ids, attention_mask, positions):
    """Greedy next-token at each item's `positions[i]` via lm_head; no layer hooks."""
    hs = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    next_toks = []
    for i in range(input_ids.shape[0]):
        pos = int(positions[i])
        logit = model.lm_head(hs[i, pos, :].unsqueeze(0)).float().squeeze(0)
        next_toks.append(int(logit.argmax().item()))
    del hs
    return next_toks


@torch.no_grad()
def forward_capture_hiddens_at_positions(model, input_ids, attention_mask, layers, positions):
    """Unsteered forward; for each row i, return hidden state at token index positions[i]."""
    storage = {}
    handles = [
        model.model.layers[l].register_forward_hook(
            make_capture_hook(storage, l), always_call=True
        )
        for l in layers
    ]
    try:
        _ = model.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    B = input_ids.shape[0]
    out = {}
    for l in layers:
        h = storage[l].float().cpu()
        out[l] = torch.stack([h[i, int(positions[i]), :] for i in range(B)])
    return out


def batch_kappa_cpu(acts: torch.Tensor, mu_pos: torch.Tensor, mu_neg: torch.Tensor) -> np.ndarray:
    mu = 0.5 * (mu_pos + mu_neg)
    v = mu_pos - mu_neg
    vns = float(v.dot(v).item())
    if vns < 1e-12:
        return np.full(acts.shape[0], np.nan)
    return ((acts - mu) @ v * (2.0 / vns)).numpy()


@torch.no_grad()
def forward_steered(model, input_ids, attention_mask, open_paren_positions,
                    layer, sv, factor, prefix_length):
    handle = model.model.layers[layer].register_forward_hook(
        make_steering_hook(sv, factor, prefix_length), always_call=True
    )
    try:
        hs = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    finally:
        handle.remove()
    next_toks = []
    for i in range(input_ids.shape[0]):
        pos = int(open_paren_positions[i])
        logit = model.lm_head(hs[i, pos, :].unsqueeze(0)).float().squeeze(0)
        next_toks.append(int(logit.argmax().item()))
    del hs
    return next_toks


def decode_letter(tokenizer, token_id):
    s = tokenizer.decode([token_id]).strip()
    return s if (len(s) == 1 and s.isupper() and s.isalpha()) else None


# ---------------------------------------------------------------------------
# Train/val selection + caching
# ---------------------------------------------------------------------------
def canonical_train_val_indices(n_raw, seed, max_train, max_val):
    rng = random.Random(seed)
    train_idxs = sorted(rng.sample(range(n_raw), min(max_train, n_raw)))
    remaining = [i for i in range(n_raw) if i not in set(train_idxs)]
    nv = min(max_val, len(remaining))
    val_idxs = sorted(rng.sample(remaining, nv)) if nv > 0 else []
    return train_idxs, val_idxs


def resolve_train_val_selection(selection_path, train_path, n_raw, seed, max_train, max_val):
    meta = {
        "train_path": str(train_path.resolve()),
        "n_raw": int(n_raw),
        "seed": int(seed),
        "max_train_examples": int(max_train),
        "max_val_examples": int(max_val),
    }
    train_idxs, val_idxs = canonical_train_val_indices(n_raw, seed, max_train, max_val)
    if selection_path.exists():
        try:
            with open(selection_path) as f:
                blob = json.load(f)
        except Exception:
            blob = {}
        if (all(blob.get(k) == v for k, v in meta.items())
                and blob.get("train_indices") == train_idxs
                and blob.get("val_indices") == val_idxs):
            return train_idxs, val_idxs
        logger.warning(f"{selection_path}: rewriting canonical indices.")
    out = dict(meta)
    out["train_indices"] = train_idxs
    out["val_indices"] = val_idxs
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with open(selection_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.warning(
        f"Wrote {selection_path}: train n={len(train_idxs)} val n={len(val_idxs)} (seed={seed})",
    )
    return train_idxs, val_idxs


def prompt_csv_covers(df, layers, factors):
    try:
        have_layers = set(int(l) for l in df["layer"].unique())
        if not set(int(l) for l in layers) <= have_layers:
            return False
        need_factors = {float(f) for f in factors}
        have_factors = {
            float(c.replace("steered_correct_", ""))
            for c in df.columns if c.startswith("steered_correct_")
        }
        if not need_factors <= have_factors:
            return False
        if "kappa_a_postgen" not in df.columns:
            return False
        return all(f"kappa_a_postgen_{f:g}" in df.columns for f in factors)
    except Exception:
        return False


def save_steering_state(path, steering_vecs, mu_poss, mu_negs, layers):
    blob = {
        "layers": [int(l) for l in layers],
        "steering_vecs": {str(l): steering_vecs[l].detach().float().cpu() for l in layers},
        "mu_poss": {str(l): mu_poss[l].detach().float().cpu() for l in layers},
        "mu_negs": {str(l): mu_negs[l].detach().float().cpu() for l in layers},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    logger.warning(f"Saved steering tensors to {path}")


def load_steering_state(path, device):
    blob = torch.load(path, map_location=device)
    layers = [int(l) for l in blob["layers"]]
    steering_vecs = {int(l): blob["steering_vecs"][str(l)].to(device) for l in layers}
    mu_poss = {int(l): blob["mu_poss"][str(l)].to(device) for l in layers}
    mu_negs = {int(l): blob["mu_negs"][str(l)].to(device) for l in layers}
    return layers, steering_vecs, mu_poss, mu_negs


def sign_kappa_mcc_at_steered_alpha(ldf, alpha_star):
    """MCC(sign(κ_postgen at α), steered_correct_{α}) for one layer's rows."""
    col_c = f"steered_correct_{alpha_star:g}"
    col_pg = f"kappa_a_postgen_{alpha_star:g}"
    if col_c not in ldf.columns or col_pg not in ldf.columns:
        return float("nan")
    kappa = ldf[col_pg].values.astype(float)
    sign_pred = (kappa > 0).astype(int)
    corr = ldf[col_c].astype(int).values
    mask = np.isfinite(kappa)
    if mask.sum() < 4:
        return float("nan")
    return safe_mcc(sign_pred[mask], corr[mask])


# ---------------------------------------------------------------------------
# Test/val sweep
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_phase_ab_mcqa(
    model, tokenizer, layers, factors, device, pad_id, batch_size,
    prefix_length, mu_poss, mu_negs, steering_vecs, source_items, phase_label,
):
    """Phase A + A2 + B for an MCQA split (post-gen κ only — letter-pos κ dropped)."""
    B = batch_size
    flat_items = []
    for j, item in enumerate(source_items):
        try:
            ml = extract_letter(item["answer_matching_behavior"])
            nl = extract_letter(item["answer_not_matching_behavior"])
        except Exception:
            continue
        if ml == nl:
            continue
        try:
            _, full_ids, _, open_paren_pos = build_full_ids(tokenizer, item["question"], ml)
        except Exception:
            continue
        flat_items.append({
            "prompt_idx": j,
            "matching_letter": ml,
            "full_ids": full_ids,
            "open_paren_pos": open_paren_pos,
        })
    n_items = len(flat_items)

    logger.warning(f"\n=== [{phase_label}] Phase A: Baseline ({n_items} prompts) ===")
    baseline_tok = {}
    for start in tqdm(range(0, n_items, B), desc=f"[{phase_label}] Phase A"):
        batch = flat_items[start:start + B]
        ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
        toks = forward_decode_at_positions(
            model, ids, mask, [b["open_paren_pos"] for b in batch],
        )
        for i, b in enumerate(batch):
            baseline_tok[b["prompt_idx"]] = toks[i]
        del ids, mask

    baseline_records = {}
    for b in flat_items:
        j = b["prompt_idx"]
        g = decode_letter(tokenizer, baseline_tok[j])
        baseline_records[j] = {
            "matching_letter": b["matching_letter"],
            "baseline_on_format": g is not None,
            "baseline_correct": g == b["matching_letter"],
        }

    logger.warning(f"\n=== [{phase_label}] Phase A2: Post-gen κ baseline ({n_items} prompts) ===")
    kappa_postgen_base = {}
    for start in tqdm(range(0, n_items, B), desc=f"[{phase_label}] Phase A2 post-gen κ"):
        batch = flat_items[start:start + B]
        seqs, pos_idx, js = [], [], []
        for b in batch:
            j = b["prompt_idx"]
            seq = seq_through_paren_plus_token(b["full_ids"], b["open_paren_pos"], baseline_tok[j])
            seqs.append(seq)
            pos_idx.append(len(seq) - 1)
            js.append(j)
        ids2, mask2, _ = pad_batch(seqs, pad_id, device)
        h_at = forward_capture_hiddens_at_positions(model, ids2, mask2, layers, pos_idx)
        del ids2, mask2
        for lm in layers:
            kbatch = batch_kappa_cpu(h_at[lm], mu_poss[lm], mu_negs[lm])
            for ii, pj in enumerate(js):
                kappa_postgen_base[(lm, pj)] = float(kbatch[ii])

    logger.warning(
        f"\n=== [{phase_label}] Phase B: Steered ({len(layers)} layers × {len(factors)} factors) ===",
    )
    steered_tok = {}
    kappa_postgen_steered = {}
    for l in tqdm(layers, desc=f"[{phase_label}] Phase B layers"):
        sv = steering_vecs[l]
        for factor in factors:
            for start in range(0, n_items, B):
                batch = flat_items[start:start + B]
                ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
                toks = forward_steered(
                    model, ids, mask,
                    [b["open_paren_pos"] for b in batch],
                    l, sv, factor, prefix_length,
                )
                seqs, pos_idx, js = [], [], []
                for i, b in enumerate(batch):
                    j = b["prompt_idx"]
                    tid = toks[i]
                    steered_tok[(factor, l, j)] = tid
                    seq = seq_through_paren_plus_token(
                        b["full_ids"], b["open_paren_pos"], tid,
                    )
                    seqs.append(seq)
                    pos_idx.append(len(seq) - 1)
                    js.append(j)
                del ids, mask
                ids2, mask2, _ = pad_batch(seqs, pad_id, device)
                h_at = forward_capture_hiddens_at_positions(model, ids2, mask2, [l], pos_idx)
                del ids2, mask2
                kbatch = batch_kappa_cpu(h_at[l], mu_poss[l], mu_negs[l])
                for ii, pj in enumerate(js):
                    kappa_postgen_steered[(l, factor, pj)] = float(kbatch[ii])

    rows = []
    for l in layers:
        for b in flat_items:
            j = b["prompt_idx"]
            rec = baseline_records[j]
            row = {
                "layer": l,
                "prompt_idx": j,
                "kappa_a_postgen": kappa_postgen_base.get((l, j), float("nan")),
                "baseline_on_format": rec["baseline_on_format"],
                "baseline_correct": rec["baseline_correct"],
            }
            for factor in factors:
                tid = steered_tok.get((factor, l, j))
                if tid is None:
                    row[f"steered_on_format_{factor:g}"] = False
                    row[f"steered_correct_{factor:g}"] = False
                    row[f"kappa_a_postgen_{factor:g}"] = float("nan")
                else:
                    g = decode_letter(tokenizer, tid)
                    row[f"steered_on_format_{factor:g}"] = g is not None
                    row[f"steered_correct_{factor:g}"] = g == b["matching_letter"]
                    row[f"kappa_a_postgen_{factor:g}"] = kappa_postgen_steered.get(
                        (l, factor, j), float("nan"),
                    )
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot — kept verbatim because make_paper_plots.py imports it.
# ---------------------------------------------------------------------------
def plot_projection_histograms(prompt_df, factors, behavior, layer,
                               train_projections, out_path, postgen: bool = False):
    """
    2×4 grid (7 panels used, 1 empty):
      [Train] [α=0] [α=1] [α=2]
      [α=3 ] [α=5] [α=10] [ - ]

    Train: blue=matching, red=non-matching training examples.
    Test panels: blue=correct, red=non-matching, grey=gibberish.

    postgen=True: per-factor κ_postgen columns (paper path).
    """
    base_col = "kappa_a_postgen" if postgen else "kappa_a"
    ldf = prompt_df[prompt_df["layer"] == layer].dropna(subset=[base_col]).copy()
    if len(ldf) < 4:
        return

    all_factors  = [0] + [f for f in factors if f != 0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat  = axes.flatten()
    train_ax   = axes_flat[0]
    factor_axes = axes_flat[1:1 + len(all_factors)]
    for ax in axes_flat[1 + len(all_factors):]:
        ax.set_visible(False)

    tp = train_projections.get(layer, {})
    pos_proj = np.array(tp.get("pos", []), dtype=float)
    neg_proj = np.array(tp.get("neg", []), dtype=float)
    if len(pos_proj) > 0 or len(neg_proj) > 0:
        all_train = np.concatenate([pos_proj, neg_proj]) if len(pos_proj) and len(neg_proj) \
                    else (pos_proj if len(pos_proj) else neg_proj)
        pad = max(0.3, (all_train.max() - all_train.min()) * 0.05 + 0.01)
        bins_t = np.linspace(all_train.min() - pad, all_train.max() + pad, 25)
        if len(neg_proj):
            train_ax.hist(neg_proj, bins=bins_t, color="#d62728", alpha=0.55,
                          edgecolor="#a01010", linewidth=0.5, label="Non-matching")
        if len(pos_proj):
            train_ax.hist(pos_proj, bins=bins_t, color="#1f77b4", alpha=0.55,
                          edgecolor="#104e8b", linewidth=0.5, label="Matching")
        x_lo, x_hi = bins_t[0], bins_t[-1]
        if x_lo <= 0 <= x_hi:
            train_ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    train_ax.set_title("Train", fontsize=10, fontweight="bold")
    train_ax.set_xlabel("κ_a", fontsize=9)
    train_ax.set_ylabel("# examples", fontsize=9)
    train_ax.legend(fontsize=7, loc="upper left")

    for ax, f in zip(factor_axes, all_factors):
        if f == 0:
            kappa_vals = ldf[base_col].values
        elif postgen:
            col_k = f"kappa_a_postgen_{f:g}"
            if col_k not in ldf.columns:
                ax.set_visible(False)
                continue
            kappa_vals = ldf[col_k].values
        else:
            kappa_vals = ldf[base_col].values + 2.0 * f

        if f == 0:
            matching  = ldf["baseline_correct"].astype(bool).values
            on_format = ldf["baseline_on_format"].astype(bool).values
        else:
            col_c = f"steered_correct_{f:g}"
            col_f = f"steered_on_format_{f:g}"
            if col_c not in ldf.columns:
                ax.set_visible(False)
                continue
            matching  = ldf[col_c].astype(bool).values
            on_format = ldf[col_f].astype(bool).values

        k_match    = kappa_vals[ matching]
        k_nonmatch = kappa_vals[~matching &  on_format]
        k_gibber   = kappa_vals[~matching & ~on_format]

        pad   = max(0.3, (kappa_vals.max() - kappa_vals.min()) * 0.05 + 0.01)
        x_lo  = kappa_vals.min() - pad
        x_hi  = kappa_vals.max() + pad
        bins  = np.linspace(x_lo, x_hi, 20)

        if len(k_gibber):
            ax.hist(k_gibber,   bins=bins, color="#aaaaaa", alpha=0.55,
                    edgecolor="#888888", linewidth=0.5, label="Gibberish")
        if len(k_nonmatch):
            ax.hist(k_nonmatch, bins=bins, color="#d62728", alpha=0.55,
                    edgecolor="#a01010", linewidth=0.5, label="Non-matching")
        if len(k_match):
            ax.hist(k_match,    bins=bins, color="#1f77b4", alpha=0.55,
                    edgecolor="#104e8b", linewidth=0.5, label="Matching")

        ax.set_xlim(x_lo, x_hi)
        if x_lo <= 0 <= x_hi:
            ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        xlab = "κ_a (post-gen token)" if postgen else "κ_a steered"
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_title(f"α={f:g}", fontsize=10, fontweight="bold")
        if ax is factor_axes[0]:
            ax.set_ylabel("# prompts", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")

    sub = " (post-gen: 2nd forward at chosen answer token)" if postgen else " (letter-pos κ; steered = κ+2α)"
    fig.suptitle(
        f"{behavior} — Layer {layer}: DiffMean projection distribution{sub}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, choices=BEHAVIORS)
    parser.add_argument("--model_name", default="google/gemma-2-9b-it")
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--layers", default="10-32",
                        help="'10-32' or '10,15,20,32'")
    parser.add_argument("--max_examples", type=int, default=300)
    parser.add_argument("--max_val_examples", type=int, default=50)
    parser.add_argument("--factors", "--steering-factors", "--steering_factors",
                        dest="factors", default="1,2,3,5,10")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--force_recompute_val", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layers_req = parse_layer_range(args.layers)
    factors = sorted({float(x.strip()) for x in args.factors.split(",") if x.strip()})
    if not factors:
        logger.error("No steering factors given.")
        sys.exit(1)

    model_short = args.model_name.split("/")[-1]
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("results") / "mcqa_projection_link" / model_short / args.behavior
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path or TRAIN_PATH_MAP[args.behavior])
    test_path = Path(args.test_path or TEST_PATH_MAP[args.behavior])
    selection_path = out_dir / "train_selection.json"
    per_prompt_csv = out_dir / "per_prompt_results.csv"
    val_prompt_csv = out_dir / "val_prompt_results.csv"
    train_proj_json = out_dir / "train_projections.json"
    steering_pt = out_dir / "steering_state.pt"

    with open(train_path) as f:
        full_raw = json.load(f)
    n_raw = len(full_raw)
    train_idxs, val_idxs = resolve_train_val_selection(
        selection_path, train_path, n_raw,
        args.seed, args.max_examples, args.max_val_examples,
    )
    train_data = [full_raw[i] for i in train_idxs]
    val_data = [full_raw[i] for i in val_idxs] if args.max_val_examples > 0 else []
    logger.warning(f"Training pairs (raw subset): {len(train_data)}")
    logger.warning(f"Validation prompts (raw subset): {len(val_data)}")

    with open(test_path) as f:
        test_data = json.load(f)
    if args.max_test:
        test_data = test_data[: args.max_test]
    logger.warning(f"Test prompts (test split file): {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Cache: per-prompt CSVs ------------------------------------------
    per_prompt_df = None
    if (not args.force_recompute) and per_prompt_csv.exists():
        try:
            df = pd.read_csv(per_prompt_csv)
            if prompt_csv_covers(df, layers_req, factors):
                per_prompt_df = df
                logger.warning(f"Reusing {per_prompt_csv}.")
            else:
                logger.warning(f"{per_prompt_csv} missing layers/factors; recomputing test.")
        except Exception as e:
            logger.warning(f"Could not reuse test CSV: {e}")

    val_prompt_df = None
    if val_data and (not args.force_recompute_val) and val_prompt_csv.exists():
        try:
            dfv = pd.read_csv(val_prompt_csv)
            if prompt_csv_covers(dfv, layers_req, factors):
                val_prompt_df = dfv
                logger.warning(f"Reusing {val_prompt_csv}.")
            else:
                logger.warning(f"{val_prompt_csv} incomplete; recomputing val.")
        except Exception as e:
            logger.warning(f"Could not reuse val CSV: {e}")

    # ---- Cache: train projections (drives histogram + d') ---------------
    train_projections = {}
    dprimes = {}
    if train_proj_json.exists():
        try:
            with open(train_proj_json) as f:
                raw = json.load(f)
            train_projections = {
                int(k): {"pos": list(v["pos"]), "neg": list(v["neg"])}
                for k, v in raw.items()
            }
            dprimes = {
                l: compute_dprime(np.asarray(v["pos"]), np.asarray(v["neg"]))
                for l, v in train_projections.items()
            }
            logger.warning(f"Loaded train projections for {len(train_projections)} layers.")
        except Exception as e:
            logger.warning(f"Could not load {train_proj_json}: {e}")

    need_test_forward = args.force_recompute or per_prompt_df is None
    need_val_forward  = bool(val_data) and (args.force_recompute_val or val_prompt_df is None)
    need_phase0_stats = (not dprimes) or (not train_projections)
    steering_ok = steering_pt.exists()
    need_phase0_compute = need_phase0_stats or (
        (need_test_forward or need_val_forward) and not steering_ok
    )
    need_load_model = need_phase0_compute or need_test_forward or need_val_forward

    steering_vecs, mu_poss, mu_negs = {}, {}, {}
    layers = list(layers_req)

    if need_load_model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prefix_length = get_prefix_length(tokenizer) if args.model_name in CHAT_MODELS else 1
        logger.warning(f"prefix_length = {prefix_length}")

        logger.warning(f"Loading model {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.use_bf16 else None,
            device_map=device,
        )
        model.eval()
        pad_id = tokenizer.pad_token_id
        B = args.batch_size

        if need_phase0_compute:
            logger.warning(
                f"\n=== Phase 0: DiffMean ({len(train_data)} pairs, {len(layers)} layers) ===",
            )
            train_flat = []
            for item in train_data:
                q = item["question"]
                for answer, label in [
                    (item["answer_matching_behavior"], 1),
                    (item["answer_not_matching_behavior"], 0),
                ]:
                    try:
                        letter = extract_letter(answer)
                        _, full_ids, letter_pos, _ = build_full_ids(tokenizer, q, letter)
                        train_flat.append(
                            {"full_ids": full_ids, "letter_pos": letter_pos, "label": label},
                        )
                    except Exception:
                        pass

            pos_acts = {l: [] for l in layers}
            neg_acts = {l: [] for l in layers}
            for start in tqdm(range(0, len(train_flat), B), desc="Phase 0"):
                batch = train_flat[start:start + B]
                ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
                hiddens = forward_capture(model, ids, mask, layers)
                for i, b in enumerate(batch):
                    for ly in layers:
                        act = hiddens[ly][i, b["letter_pos"], :]
                        (pos_acts[ly] if b["label"] == 1 else neg_acts[ly]).append(act)
                del hiddens, ids, mask

            steering_vecs, mu_poss, mu_negs, dprimes = {}, {}, {}, {}
            train_projections = {}
            for ly in layers:
                if not pos_acts[ly] or not neg_acts[ly]:
                    continue
                mu_pos = torch.stack(pos_acts[ly]).mean(0)
                mu_neg = torch.stack(neg_acts[ly]).mean(0)
                v = mu_pos - mu_neg
                v_norm_sq = float(v.dot(v))
                mu_poss[ly] = mu_pos
                mu_negs[ly] = mu_neg
                steering_vecs[ly] = v.to(device)
                mu_mid = 0.5 * (mu_pos + mu_neg)
                if v_norm_sq > 1e-12:
                    pp = np.array([float(2.0 * (a - mu_mid).dot(v) / v_norm_sq) for a in pos_acts[ly]])
                    np_ = np.array([float(2.0 * (a - mu_mid).dot(v) / v_norm_sq) for a in neg_acts[ly]])
                    dprimes[ly] = compute_dprime(pp, np_)
                else:
                    pp = np.zeros(len(pos_acts[ly]))
                    np_ = np.zeros(len(neg_acts[ly]))
                    dprimes[ly] = 0.0
                train_projections[ly] = {"pos": pp.tolist(), "neg": np_.tolist()}
                logger.warning(f"  L{ly:2d}: d'={dprimes[ly]:.3f}  ||v||={float(v.norm()):.3f}")

            layers = [ly for ly in layers if ly in steering_vecs]
            del pos_acts, neg_acts

            with open(train_proj_json, "w") as f:
                json.dump({str(ly): v for ly, v in train_projections.items()}, f)
            logger.warning(f"Saved train projections to {train_proj_json}")
            save_steering_state(steering_pt, steering_vecs, mu_poss, mu_negs, layers)
        else:
            layers_sv, steering_vecs, mu_poss, mu_negs = load_steering_state(steering_pt, device)
            layers = [ly for ly in layers_req if ly in layers_sv]
            steering_vecs = {ly: steering_vecs[ly] for ly in layers}
            mu_poss = {ly: mu_poss[ly] for ly in layers}
            mu_negs = {ly: mu_negs[ly] for ly in layers}
            logger.warning(f"Loaded steering tensors for layers {layers} from {steering_pt}")

        if need_test_forward:
            per_prompt_df = run_phase_ab_mcqa(
                model, tokenizer, layers, factors, device, pad_id, B,
                prefix_length, mu_poss, mu_negs, steering_vecs, test_data, "test",
            )
            per_prompt_df.to_csv(per_prompt_csv, index=False)
            logger.warning(f"Saved {len(per_prompt_df)} rows to {per_prompt_csv}")

        if need_val_forward:
            val_prompt_df = run_phase_ab_mcqa(
                model, tokenizer, layers, factors, device, pad_id, B,
                prefix_length, mu_poss, mu_negs, steering_vecs, val_data, "val",
            )
            val_prompt_df.to_csv(val_prompt_csv, index=False)
            logger.warning(f"Saved {len(val_prompt_df)} rows to {val_prompt_csv}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if per_prompt_df is None:
        logger.error("No test per-prompt results available.")
        sys.exit(1)

    per_prompt_df = per_prompt_df[per_prompt_df["layer"].isin(layers_req)].copy()
    ly_present = sorted(per_prompt_df["layer"].unique())
    layers = [ly for ly in layers_req if ly in ly_present]
    if not layers:
        logger.error("Requested layers not present in test CSV.")
        sys.exit(1)

    # ---- Per-layer summary (slim) ----------------------------------------
    layer_rows = []
    for l in layers:
        ldf = per_prompt_df[per_prompt_df["layer"] == l]
        row = {
            "layer": int(l),
            "dprime": dprimes.get(int(l), float("nan")),
            "n_prompts": int(len(ldf)),
            "baseline_acc": float(ldf["baseline_correct"].mean()),
        }
        for factor in factors:
            col_c = f"steered_correct_{factor:g}"
            if col_c in ldf.columns:
                row[f"steered_acc_{factor:g}"] = float(ldf[col_c].mean())
        layer_rows.append(row)
    layer_df = pd.DataFrame(layer_rows).sort_values("layer").reset_index(drop=True)

    # Val-chosen α* per layer → MCC on TEST (paper figure 3/7) and
    # test steered accuracy at that α* (paper Table 1).
    layer_df["sign_kappa_mcc_val_best_on_test"] = float("nan")
    layer_df["test_steered_acc_at_val_best_alpha"] = float("nan")
    if val_prompt_df is not None:
        val_prompt_df = val_prompt_df[val_prompt_df["layer"].isin(layers)].copy()
        mcc_col, acc_col = [], []
        for L in layer_df["layer"].values:
            ldf_val = val_prompt_df[val_prompt_df["layer"] == int(L)]
            if len(ldf_val) == 0:
                mcc_col.append(float("nan"))
                acc_col.append(float("nan"))
                continue
            best_acc, best_alpha = -1.0, None
            for f in factors:
                c = f"steered_correct_{f:g}"
                if c not in ldf_val.columns:
                    continue
                acc = float(ldf_val[c].mean())
                if acc > best_acc:
                    best_acc, best_alpha = acc, f
            if best_alpha is None:
                mcc_col.append(float("nan"))
                acc_col.append(float("nan"))
                continue
            ldf_test = per_prompt_df[per_prompt_df["layer"] == int(L)]
            mcc_col.append(sign_kappa_mcc_at_steered_alpha(ldf_test, best_alpha))
            test_col = f"steered_correct_{best_alpha:g}"
            acc_col.append(
                float(ldf_test[test_col].mean()) if test_col in ldf_test.columns
                else float("nan"),
            )
        layer_df["sign_kappa_mcc_val_best_on_test"] = mcc_col
        layer_df["test_steered_acc_at_val_best_alpha"] = acc_col

    layer_df.to_csv(out_dir / "per_layer_summary.csv", index=False)
    logger.warning("\nPer-layer summary:")
    for _, r in layer_df.iterrows():
        steered_str = "  ".join(
            f"α={f:g}={r[f'steered_acc_{f:g}']:.3f}"
            for f in factors
            if f"steered_acc_{f:g}" in r and not pd.isna(r[f"steered_acc_{f:g}"])
        )
        logger.warning(
            f"  L{int(r['layer']):2d}: d'={r['dprime']:.3f}  "
            f"base={r['baseline_acc']:.3f}  {steered_str}"
        )

    # ---- summary.json (minimal — paper plots only read `factors`) -------
    with open(selection_path) as f:
        train_sel_blob = json.load(f)
    summary = {
        "behavior": args.behavior,
        "model_name": args.model_name,
        "layers": list(map(int, layer_df["layer"].values)),
        "factors": factors,
        "seed": args.seed,
        "train_val_selection": train_sel_blob,
        "n_test_prompts": int(len(per_prompt_df["prompt_idx"].unique())),
        "n_val_prompts": (
            int(len(val_prompt_df["prompt_idx"].unique())) if val_prompt_df is not None else 0
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
