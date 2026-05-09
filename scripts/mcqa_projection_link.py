"""
MCQA Projection-Steering Link Analysis
=======================================

Core question: for a given behavior, does the side of the diff-in-means line
that an unsteered test activation lands on predict whether the model displays
the behavior? And does the layer where this prediction is strongest also happen
to be the layer where steering is most effective?

Pipeline (single self-contained script):
  Phase 0  — Compute DiffMean steering vectors from training data (inline).
  Phase A  — Baseline greedy decode + activation capture for all test prompts.
             Evaluation: argmax of lm_head at the '(' position (same approach
             as mcqa_adaptive_steer.py; off-format / gibberish → counted wrong).
             κ_a uses hidden state at the teacher-forced answer letter.
  Phase A2 — Unsteered forward on prefix + [baseline-chosen token]; κ_a_postgen
             from hidden at that token (per layer).
  Phase B  — Steered decode at '(' per (layer, factor), then unsteered forward on
             prefix + [steered token]; kappa_a_postgen_{α} per layer.
  Validation — Additional prompts sampled from raw \\ train (see train_selection.json);
               same Phase A/B sweep → val_prompt_results.csv and plots/val/.
               Best α per layer is chosen on val; sign-MCC vs steer success on **test**
               at that α is compared to d' (see plots/mcc_best_val_alpha_on_test_vs_dprime.png).
  Analysis — Per layer:
               • sign(κ_a) vs baseline_correct  → MCC  (binary predictor)
               • κ_a vs baseline_correct         → Spearman ρ (continuous)
               • d' from training projections
               • steered greedy accuracy per factor
             Cross-layer:
               • Pearson/Spearman between projection MCC and steered accuracy
               • Same for d' vs steered accuracy

Outputs (under --output_dir):
  train_selection.json      — raw indices for train + val caps (seed / paths for audit)
  steering_state.pt         — µ± and steering vectors (enables val sweep without redoing test)
  per_prompt_results.csv    — test split: κ_a, κ_a_postgen*, steered columns per (layer, prompt)
  val_prompt_results.csv    — validation split (same schema)
  per_layer_summary.csv     — one row per layer (includes val-best-α columns when val exists)
  cross_layer_corr.csv      — predictor vs target correlations across layers
  plots/test/               — test-split plots (projection, steering, histograms, …)
  plots/val/                — val steering-accuracy plots
  plots/mcc_best_val_alpha_on_test_vs_dprime.png — d' vs sign-MCC @ val-chosen α evaluated on test

Usage:
    python scripts/mcqa_projection_link.py \\
        --behavior corrigible-neutral-HHH \\
        --model_name google/gemma-2-9b-it \\
        --layers 10-32 \\
        --factors 1,2,3,5,10

    # Loop over behaviors:
    for b in sycophancy hallucination corrigible-neutral-HHH myopic-reward survival-instinct; do
        python scripts/mcqa_projection_link.py --behavior $b --factors 1,2,3,5,10
    done
"""
import json
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
from scipy import stats as scipy_stats
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Inlined helpers (originally axbench.utils.{constants,model_utils}).
# ---------------------------------------------------------------------------
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
    """Length of the constant prefix that the chat template prepends to a user
    message — needed so we only steer over the assistant suffix, not the system
    preamble."""
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


import logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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
    """
    Returns (prompt_ids, full_ids, open_paren_pos) where open_paren_pos is the
    index of the '(' token — the model predicts the next token here, giving the
    letter choice without needing any continuation.
    """
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

    # Find '(' position (the token just before the answer letter).
    letter_pos = None
    for i in range(len(full_ids) - 1, max(len(prompt_ids) - 1, 0), -1):
        if tokenizer.decode([full_ids[i]]).strip() == answer_letter:
            letter_pos = i
            break
    if letter_pos is None:
        raise RuntimeError(f"Cannot find '{answer_letter}' in suffix of full_ids")
    open_paren_pos = letter_pos - 1
    return prompt_ids, full_ids, letter_pos, open_paren_pos


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
    """Token ids through '(' inclusive plus model-chosen letter; handles list or 1-D tensor."""
    prefix = full_ids[: open_paren_pos + 1]
    if torch.is_tensor(prefix):
        pref = prefix.detach().cpu().tolist()
    else:
        pref = list(prefix)
    return pref + [int(next_token_id)]


def compute_dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    gap = abs(pos.mean() - neg.mean())
    pooled = np.sqrt(0.5 * (pos.var() + neg.var()))
    return float(gap / pooled) if pooled > 1e-12 else 0.0


def safe_mcc(pred, actual):
    """MCC; returns nan if either array is constant (degenerate)."""
    pred, actual = np.asarray(pred, int), np.asarray(actual, int)
    if pred.std() < 1e-9 or actual.std() < 1e-9:
        return float("nan")
    return float(matthews_corrcoef(actual, pred))


def safe_spearman(x, y):
    """Spearman ρ; returns (nan, nan) if either array is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
        return float("nan"), float("nan")
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# Forward-pass utilities (borrowed from mcqa_adaptive_steer.py approach)
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
def forward_capture(model, input_ids, attention_mask, layers, open_paren_positions):
    """
    Run forward pass, capture hidden states at target layers, and decode the
    greedy next-token at each item's open_paren_position (= the letter choice).
    Returns: (next_token_ids [B], layer_hiddens {layer: tensor [B,L,H] cpu float32})
    """
    storage = {}
    handles = [
        model.model.layers[l].register_forward_hook(
            make_capture_hook(storage, l), always_call=True
        )
        for l in layers
    ]
    try:
        hs = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    finally:
        for h in handles:
            h.remove()

    next_toks = []
    for i in range(input_ids.shape[0]):
        pos = int(open_paren_positions[i])
        logit = model.lm_head(hs[i, pos, :].unsqueeze(0)).float().squeeze(0)
        next_toks.append(int(logit.argmax().item()))
    del hs
    return next_toks, {l: storage[l].float().cpu() for l in layers}


@torch.no_grad()
def forward_capture_hiddens_at_positions(model, input_ids, attention_mask, layers, positions):
    """Unsteered forward; for each batch row i, return hidden state at token index positions[i]."""
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
    """Project row-wise activations onto DiffMean line (Braun κ_a). acts, μ: float CPU tensors."""
    mu = 0.5 * (mu_pos + mu_neg)
    v = mu_pos - mu_neg
    vns = float(v.dot(v).item())
    if vns < 1e-12:
        return np.full(acts.shape[0], np.nan)
    return ((acts - mu) @ v * (2.0 / vns)).numpy()


@torch.no_grad()
def forward_steered(model, input_ids, attention_mask, open_paren_positions,
                    layer, sv, factor, prefix_length):
    """Steered forward pass; returns greedy next-token ids at open_paren_positions."""
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


def canonical_train_val_indices(
    n_raw: int, seed: int, max_train: int, max_val: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_idxs = sorted(rng.sample(range(n_raw), min(max_train, n_raw)))
    remaining = [i for i in range(n_raw) if i not in set(train_idxs)]
    nv = min(max_val, len(remaining))
    val_idxs = sorted(rng.sample(remaining, nv)) if nv > 0 else []
    return train_idxs, val_idxs


def resolve_train_val_selection(
    selection_path: Path,
    train_path: Path,
    n_raw: int,
    seed: int,
    max_train: int,
    max_val: int,
) -> tuple[list[int], list[int]]:
    """Persist deterministic raw indices; rewrite file if metadata or indices diverge."""
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
        meta_ok = all(blob.get(k) == v for k, v in meta.items())
        same_train = blob.get("train_indices") == train_idxs
        same_val = blob.get("val_indices") == val_idxs
        if meta_ok and same_train and same_val:
            return train_idxs, val_idxs
        if not meta_ok:
            logger.warning(
                f"{selection_path}: metadata mismatch vs current run; rewriting canonical indices.",
            )
        elif not same_train:
            logger.warning(
                f"{selection_path}: train_indices differ from canonical RNG(seed); rewriting.",
            )
        elif not same_val:
            logger.warning(
                f"{selection_path}: val_indices differ from canonical; rewriting.",
            )
    out = dict(meta)
    out["train_indices"] = train_idxs
    out["val_indices"] = val_idxs
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with open(selection_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.warning(
        f"Wrote {selection_path}: raw train n={len(train_idxs)}  val n={len(val_idxs)} "
        f"(seed={seed})",
    )
    return train_idxs, val_idxs


def prompt_csv_covers(df: pd.DataFrame, layers: list, factors: list[float]) -> bool:
    """Baseline κ forward + post-gen κ per factor + steered correctness."""
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


def save_steering_state(path: Path, steering_vecs, mu_poss, mu_negs, layers: list):
    blob = {
        "layers": [int(l) for l in layers],
        "steering_vecs": {str(l): steering_vecs[l].detach().float().cpu() for l in layers},
        "mu_poss": {str(l): mu_poss[l].detach().float().cpu() for l in layers},
        "mu_negs": {str(l): mu_negs[l].detach().float().cpu() for l in layers},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    logger.warning(f"Saved steering tensors to {path}")


def load_steering_state(path: Path, device):
    blob = torch.load(path, map_location=device)
    layers = [int(l) for l in blob["layers"]]
    steering_vecs = {int(l): blob["steering_vecs"][str(l)].to(device) for l in layers}
    mu_poss = {int(l): blob["mu_poss"][str(l)].to(device) for l in layers}
    mu_negs = {int(l): blob["mu_negs"][str(l)].to(device) for l in layers}
    return layers, steering_vecs, mu_poss, mu_negs


def sign_kappa_mcc_at_steered_alpha(ldf: pd.DataFrame, alpha_star: float) -> float:
    """MCC(sign(κ), steered_correct_{α}) for one layer's rows."""
    col_c = f"steered_correct_{alpha_star:g}"
    col_pg = f"kappa_a_postgen_{alpha_star:g}"
    if col_c not in ldf.columns:
        return float("nan")
    kappa_an = ldf["kappa_a"].values.astype(float) + 2.0 * alpha_star
    if col_pg in ldf.columns:
        kappa_pg = ldf[col_pg].values.astype(float)
        kappa = np.where(np.isfinite(kappa_pg), kappa_pg, kappa_an)
    else:
        kappa = kappa_an
    sign_pred = (kappa > 0).astype(int)
    corr = ldf[col_c].astype(int).values
    mask = np.isfinite(kappa)
    if mask.sum() < 4:
        return float("nan")
    return safe_mcc(sign_pred[mask], corr[mask])


@torch.no_grad()
def run_phase_ab_mcqa(
    model,
    tokenizer,
    layers,
    factors,
    device,
    pad_id,
    batch_size,
    prefix_length,
    mu_poss,
    mu_negs,
    steering_vecs,
    source_items: list,
    phase_label: str,
) -> pd.DataFrame:
    """Phase A + A2 + B for test or val MCQA items (same schema as per_prompt_results.csv)."""
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
            _, full_ids, letter_pos, open_paren_pos = build_full_ids(tokenizer, item["question"], ml)
        except Exception:
            continue
        flat_items.append({
            "prompt_idx": j,
            "matching_letter": ml,
            "notmatch_letter": nl,
            "full_ids": full_ids,
            "letter_pos": letter_pos,
            "open_paren_pos": open_paren_pos,
        })
    n_items = len(flat_items)
    logger.warning(f"\n=== [{phase_label}] Phase A: Baseline ({n_items} prompts) ===")

    baseline_tok = {}
    act_at_letter = {}

    for start in tqdm(range(0, n_items, B), desc=f"[{phase_label}] Phase A"):
        batch = flat_items[start:start + B]
        ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
        toks, hiddens = forward_capture(
            model, ids, mask, layers,
            [b["open_paren_pos"] for b in batch],
        )
        for i, b in enumerate(batch):
            j = b["prompt_idx"]
            baseline_tok[j] = toks[i]
            for l in layers:
                act_at_letter[(j, l)] = hiddens[l][i, b["letter_pos"], :]
        del hiddens, ids, mask

    baseline_records = {}
    for b in flat_items:
        j = b["prompt_idx"]
        g = decode_letter(tokenizer, baseline_tok[j])
        src = source_items[j]
        baseline_records[j] = {
            "question": src["question"],
            "answer_matching_behavior": src["answer_matching_behavior"],
            "matching_letter": b["matching_letter"],
            "notmatch_letter": b["notmatch_letter"],
            "baseline_token": g if g else tokenizer.decode([baseline_tok[j]]).strip(),
            "baseline_on_format": g is not None,
            "baseline_correct": g == b["matching_letter"],
        }

    kappa_map = {}
    for l in layers:
        mu_pos, mu_neg = mu_poss[l], mu_negs[l]
        v = mu_pos - mu_neg
        v_norm_sq = float(v.dot(v))
        mu = 0.5 * (mu_pos + mu_neg)
        for b in flat_items:
            j = b["prompt_idx"]
            act = act_at_letter[(j, l)]
            kappa_map[(l, j)] = float(2.0 * (act - mu).dot(v) / v_norm_sq) \
                if v_norm_sq > 1e-12 else float("nan")
    del act_at_letter

    logger.warning(f"\n=== [{phase_label}] Phase A2: Post-gen κ baseline ({n_items} prompts) ===")
    kappa_postgen_base = {}
    for start in tqdm(range(0, n_items, B), desc=f"[{phase_label}] Phase A2 post-gen κ"):
        batch = flat_items[start:start + B]
        seqs, pos_idx, js = [], [], []
        for b in batch:
            j = b["prompt_idx"]
            op = b["open_paren_pos"]
            seq = seq_through_paren_plus_token(b["full_ids"], op, baseline_tok[j])
            seqs.append(seq)
            pos_idx.append(len(seq) - 1)
            js.append(j)
        ids2, mask2, _ = pad_batch(seqs, pad_id, device)
        h_at = forward_capture_hiddens_at_positions(
            model, ids2, mask2, layers, pos_idx,
        )
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
                h_at = forward_capture_hiddens_at_positions(
                    model, ids2, mask2, layers, pos_idx,
                )
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
                "question": rec["question"],
                "answer_matching_behavior": rec["answer_matching_behavior"],
                "matching_letter": rec["matching_letter"],
                "notmatch_letter": rec["notmatch_letter"],
                "kappa_a": kappa_map.get((l, j), float("nan")),
                "kappa_a_postgen": kappa_postgen_base.get((l, j), float("nan")),
                "baseline_token": rec["baseline_token"],
                "baseline_on_format": rec["baseline_on_format"],
                "baseline_correct": rec["baseline_correct"],
            }
            for factor in factors:
                tid = steered_tok.get((factor, l, j))
                if tid is None:
                    row[f"steered_token_{factor:g}"] = ""
                    row[f"steered_on_format_{factor:g}"] = False
                    row[f"steered_correct_{factor:g}"] = False
                    row[f"kappa_a_postgen_{factor:g}"] = float("nan")
                else:
                    g = decode_letter(tokenizer, tid)
                    row[f"steered_token_{factor:g}"] = g or tokenizer.decode([tid]).strip()
                    row[f"steered_on_format_{factor:g}"] = g is not None
                    row[f"steered_correct_{factor:g}"] = g == b["matching_letter"]
                    row[f"kappa_a_postgen_{factor:g}"] = kappa_postgen_steered.get(
                        (l, factor, j), float("nan"),
                    )
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_projection_quality(layer_df, behavior, out_path):
    """MCC of sign(κ_a) and Spearman ρ of κ_a, both vs baseline_correct, by layer."""
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    ax1.plot(layers, layer_df["sign_kappa_mcc"].values, "o-",
             color="#2E86AB", linewidth=2.5, markersize=7,
             label="sign(κ_a) MCC vs baseline correct (left)")
    ax1.axhline(0, color="#2E86AB", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("MCC  [sign(κ_a) → correct?]", color="#2E86AB", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2E86AB")

    ax2.plot(layers, layer_df["kappa_spearman_rho"].values, "s--",
             color="#E94F37", linewidth=2, markersize=6,
             label="κ_a Spearman ρ vs baseline correct (right)")
    ax2.axhline(0, color="#E94F37", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Spearman ρ  [κ_a → correct?]", color="#E94F37", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#E94F37")

    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_xticks(layers)
    ax1.set_title(
        f"{behavior}: How well does the diff-in-means projection predict baseline behavior?\n"
        f"MCC uses sign(κ_a) as binary predictor; Spearman uses raw κ_a",
        fontsize=11, fontweight="bold",
    )
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_steering_acc(layer_df, factors, behavior, out_path):
    """Steered greedy accuracy by layer, one line per factor. Baseline shown as reference."""
    layers = layer_df["layer"].values
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, layer_df["baseline_acc"].values, "D--", color="gray",
            linewidth=1.5, markersize=6, label="Baseline (α=0)", alpha=0.8)
    cmap = plt.get_cmap("plasma")
    for i, f in enumerate(factors):
        col = f"steered_acc_{f:g}"
        if col not in layer_df.columns:
            continue
        color = cmap(i / max(1, len(factors) - 1))
        ax.plot(layers, layer_df[col].values, "o-", color=color,
                linewidth=2, markersize=5, label=f"α={f:g}")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Greedy accuracy (correct letter generated)", fontsize=11)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{behavior}: Steered greedy accuracy by layer\n"
        f"Evaluation: actual generated token at '(' position (gibberish = wrong)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(title="Steering factor", fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_steering_and_dprime(layer_df, factors, behavior, out_path):
    """Steering accuracy for all factors + baseline + d' on one plot with dual y-axes."""
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(13, 5))

    # --- left axis: accuracy ---
    ax1.plot(layers, layer_df["baseline_acc"].values, "D--", color="gray",
             linewidth=1.5, markersize=6, label="Baseline (α=0)", alpha=0.85, zorder=3)
    cmap = plt.get_cmap("plasma")
    factor_cols = [f for f in factors if f"steered_acc_{f:g}" in layer_df.columns]
    for i, f in enumerate(factor_cols):
        color = cmap(i / max(1, len(factor_cols) - 1))
        ax1.plot(layers, layer_df[f"steered_acc_{f:g}"].values, "o-", color=color,
                 linewidth=2, markersize=5, label=f"α={f:g}", zorder=3)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Greedy accuracy", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(layers)

    # --- right axis: d' ---
    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d'", zorder=2)
        ax2.set_ylabel("d'  (training discriminability)", fontsize=11, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(bottom=0)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, title="Steering factor / metric",
               fontsize=9, loc="upper left", framealpha=0.85)

    ax1.set_title(
        f"{behavior}: Steering accuracy & training d' by layer",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mcc_vs_dprime_for_column(
    layer_df,
    behavior,
    out_path,
    mcc_col: str,
    title_line1: str,
    mcc_label: str = "MCC(sign κ vs actual match)",
):
    """Dual-axis: arbitrary per-layer MCC column vs training d'."""
    if mcc_col not in layer_df.columns:
        return
    mcc = layer_df[mcc_col].values.astype(float)
    if not np.any(np.isfinite(mcc)):
        return
    layers = layer_df["layer"].values
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(layers, mcc, "o-", color="#C73E1D", linewidth=2, markersize=6,
             label=mcc_label, zorder=3)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Matthews correlation coefficient", fontsize=11)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xticks(layers)
    ax1.tick_params(axis="y", labelcolor="#C73E1D")

    ax2 = ax1.twinx()
    if "dprime" in layer_df.columns and layer_df["dprime"].notna().any():
        ax2.fill_between(layers, layer_df["dprime"].values, alpha=0.12, color="steelblue")
        ax2.plot(layers, layer_df["dprime"].values, "s:", color="steelblue",
                 linewidth=1.5, markersize=5, label="d' (train)", zorder=2)
        ax2.set_ylabel("d'  (training discriminability)", fontsize=11, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="best", framealpha=0.9)
    ax1.set_title(
        title_line1 + "\nκ = post-gen at chosen token if available; else κ_baseline + 2α",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mcc_best_alpha_vs_dprime(layer_df, behavior, out_path):
    """Best α chosen by **test** steered accuracy (circular if test is final eval)."""
    plot_mcc_vs_dprime_for_column(
        layer_df,
        behavior,
        out_path,
        mcc_col="sign_kappa_mcc_best_alpha",
        title_line1=f"{behavior}: MCC @ best α per layer (chosen on **test** accuracy)",
        mcc_label="MCC(sign κ vs steer match @ test-best α)",
    )


def plot_mcc_val_best_alpha_on_test_vs_dprime(layer_df, behavior, out_path):
    """Best α from validation; MCC evaluated on held-out **test** prompts."""
    plot_mcc_vs_dprime_for_column(
        layer_df,
        behavior,
        out_path,
        mcc_col="sign_kappa_mcc_val_best_on_test",
        title_line1=f"{behavior}: MCC on **test** @ best α per layer (α chosen on **val**)",
        mcc_label="MCC(sign κ vs steer match | val-chosen α, test prompts)",
    )


def plot_best_alpha_val_vs_test(layer_df, behavior, out_path):
    """Chosen steering factor α* from val vs from test (per layer)."""
    if "best_factor_val" not in layer_df.columns or "best_factor" not in layer_df.columns:
        return
    layers = layer_df["layer"].values
    bt = pd.to_numeric(layer_df["best_factor"], errors="coerce").values
    bv = pd.to_numeric(layer_df["best_factor_val"], errors="coerce").values
    if np.all(np.isnan(bt)) and np.all(np.isnan(bv)):
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(layers, bt, "o-", color="#1f77b4", linewidth=2, markersize=6, label="α* (max test acc)")
    ax.plot(layers, bv, "s--", color="#ff7f0e", linewidth=2, markersize=6, label="α* (max val acc)")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Steering factor α", fontsize=11)
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="best")
    ax.set_title(
        f"{behavior}: Best steering factor per layer — validation vs test split",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_steering_acc_val_vs_test_extended(layer_df, behavior, out_path):
    """
    Mean accuracy on **test** at test-best α, mean on **val** at val-best α,
    and mean on **test** when using val-best α (generalization of α choice).
    """
    cols_need = ("best_steered_acc", "best_steered_acc_val", "test_steered_acc_at_val_best_alpha")
    if any(c not in layer_df.columns for c in cols_need):
        return
    layers = layer_df["layer"].values
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(layers, layer_df["best_steered_acc"].values, "o-",
            color="#1f77b4", linewidth=2, markersize=5, label="Test acc @ test-best α*")
    ax.plot(layers, layer_df["best_steered_acc_val"].values, "s--",
            color="#ff7f0e", linewidth=2, markersize=5, label="Val acc @ val-best α*")
    ax.plot(layers, layer_df["test_steered_acc_at_val_best_alpha"].values, "^:",
            color="#2ca02c", linewidth=2, markersize=5, label="Test acc @ val-best α*")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Greedy steering accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="best")
    ax.set_title(
        f"{behavior}: Steering accuracy — val vs test choice of α*",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_vs_steering(layer_df, factors, behavior, out_path):
    """
    The key plot: scatter of projection MCC (x) vs steered accuracy (y),
    one point per layer, one panel per factor + one panel for best-α-per-layer.
    """
    panels = []
    for f in factors:
        if f == "best":
            if "best_steered_acc" in layer_df.columns:
                panels.append(("best_steered_acc", "Best α per layer"))
        else:
            col = f"steered_acc_{f:g}"
            if col in layer_df.columns:
                panels.append((col, f"α={f:g}"))
    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for idx, (col, label) in enumerate(panels):
        ax = axes[0][idx]
        x = layer_df["sign_kappa_mcc"].values
        y = layer_df[col].values
        lyrs = layer_df["layer"].values

        sc = ax.scatter(x, y, c=lyrs, cmap="viridis", s=80,
                        edgecolors="k", linewidths=0.4, zorder=3)
        plt.colorbar(sc, ax=ax, label="Layer")

        for xi, yi, li in zip(x, y, lyrs):
            if li % 3 == 0 and not (np.isnan(xi) or np.isnan(yi)):
                ax.annotate(str(li), (xi, yi), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")

        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 3 and np.std(x[mask]) > 1e-9 and np.std(y[mask]) > 1e-9:
            r, p = scipy_stats.pearsonr(x[mask], y[mask])
            ax.set_title(f"{label}   r={r:+.3f} (p={p:.3g})", fontsize=11, fontweight="bold")
        else:
            ax.set_title(label, fontsize=11, fontweight="bold")

        ax.set_xlabel("sign(κ_a) MCC vs baseline correct", fontsize=10)
        ax.set_ylabel("Steered greedy accuracy", fontsize=10)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(layer_df["baseline_acc"].mean(), color="gray",
                   linestyle="--", alpha=0.4, label="mean baseline acc")

    fig.suptitle(
        f"{behavior}: Projection quality → steering effectiveness (one point = one layer)\n"
        f"Rightmost panel = best steering factor at each layer",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kappa_scatter(prompt_df, layer, behavior, out_path):
    """κ_a (x) vs baseline_correct (0/1 jittered, y) for one layer."""
    ldf = prompt_df[prompt_df["layer"] == layer].dropna(subset=["kappa_a"])
    if len(ldf) < 4:
        return
    kappa = ldf["kappa_a"].values
    correct = ldf["baseline_correct"].astype(int).values

    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(correct))
    colors = np.where(correct == 1, "#2ca02c", "#d62728")

    mcc = safe_mcc((kappa > 0).astype(int), correct)
    rho, _ = safe_spearman(kappa, correct)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(kappa, correct + jitter, c=colors, s=60,
               edgecolors="k", linewidths=0.4, alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2,
               label="κ=0 (centroid boundary)")
    ax.set_xlabel("κ_a  (projection onto diff-in-means line)", fontsize=11)
    ax.set_ylabel("Baseline correct (1=yes, 0=no)", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No behavior (0)", "Shows behavior (1)"])
    ax.set_title(
        f"{behavior} — Layer {layer}\n"
        f"sign(κ_a) MCC={mcc:+.3f}   Spearman ρ={rho:+.3f}   "
        f"(green=correct, red=incorrect)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _hist_overlapping(ax, k_match, k_nonmatch, k_gibber, bins):
    """Overlapping semi-transparent histograms — all three distributions visible at once."""
    if len(k_gibber):
        ax.hist(k_gibber,   bins=bins, color="#aaaaaa", alpha=0.55,
                edgecolor="#888888", linewidth=0.5, label="Gibberish")
    if len(k_nonmatch):
        ax.hist(k_nonmatch, bins=bins, color="#d62728", alpha=0.55,
                edgecolor="#a01010", linewidth=0.5, label="Non-matching")
    if len(k_match):
        ax.hist(k_match,    bins=bins, color="#1f77b4", alpha=0.55,
                edgecolor="#104e8b", linewidth=0.5, label="Matching")


def plot_projection_histograms(prompt_df, factors, behavior, layer,
                               train_projections, out_path, postgen: bool = False):
    """
    2×4 grid (7 panels used, 1 empty):
      [Train] [α=0] [α=1] [α=2]
      [α=3 ] [α=5] [α=10] [ - ]

    Train panel: blue=matching, red=non-matching training examples.
    Test panels: blue=correct output, red=non-matching output, grey=gibberish.

    postgen=False (default): κ at teacher-forced letter token; steered κ = κ_base + 2α.
    postgen=True: κ after a second unsteered forward on prefix + [model's chosen token]
    at that token; per-factor columns kappa_a_postgen_{f}.
    """
    base_col = "kappa_a_postgen" if postgen else "kappa_a"
    ldf = prompt_df[prompt_df["layer"] == layer].dropna(subset=[base_col]).copy()
    if len(ldf) < 4:
        return

    all_factors  = [0] + [f for f in factors if f != 0]

    # 2×4 layout — train panel first, then factor panels.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat  = axes.flatten()  # [0..7]
    train_ax   = axes_flat[0]
    factor_axes = axes_flat[1:1 + len(all_factors)]
    for ax in axes_flat[1 + len(all_factors):]:
        ax.set_visible(False)

    # ---- Train panel -------------------------------------------------------
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

    # ---- Test factor panels ------------------------------------------------
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

        _hist_overlapping(ax, k_match, k_nonmatch, k_gibber, bins)

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
    parser.add_argument("--max_examples", type=int, default=300,
                        help="Max training pairs for DiffMean from raw. Default 300.")
    parser.add_argument("--max_val_examples", type=int, default=50,
                        help="Validation prompts sampled from raw minus train. Default 50.")
    parser.add_argument("--factors", "--steering-factors", "--steering_factors",
                        dest="factors", default="1,2,3,5,10",
                        help="Comma-separated fixed steering factors. Default 1,2,3,5,10.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (stored in train_selection.json). Change ⇒ recompute splits.")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore saved test per_prompt_results.csv and rerun Phase A/B on test.")
    parser.add_argument("--force_recompute_val", action="store_true",
                        help="Ignore saved val_prompt_results.csv and rerun val Phase A/B.")
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
    plots_root = out_dir / "plots"
    plots_test = plots_root / "test"
    plots_val = plots_root / "val"
    for p in (plots_root, plots_test, plots_val):
        p.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path or TRAIN_PATH_MAP[args.behavior])
    test_path = Path(args.test_path or TEST_PATH_MAP[args.behavior])
    selection_path = out_dir / "train_selection.json"
    per_prompt_csv = out_dir / "per_prompt_results.csv"
    val_prompt_csv = out_dir / "val_prompt_results.csv"
    dprime_json = out_dir / "dprime.json"
    train_proj_json = out_dir / "train_projections.json"
    steering_pt = out_dir / "steering_state.pt"

    with open(train_path) as f:
        full_raw = json.load(f)
    n_raw = len(full_raw)
    train_idxs, val_idxs = resolve_train_val_selection(
        selection_path,
        train_path,
        n_raw,
        args.seed,
        args.max_examples,
        args.max_val_examples,
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

    per_prompt_df = None
    if (not args.force_recompute) and per_prompt_csv.exists():
        try:
            df = pd.read_csv(per_prompt_csv)
            if prompt_csv_covers(df, layers_req, factors):
                per_prompt_df = df
                logger.warning(
                    f"Reusing {per_prompt_csv} — covers layers/factors. "
                    "Pass --force_recompute to redo.",
                )
            else:
                logger.warning(f"{per_prompt_csv} missing layers/factors/postgen; recomputing test.")
        except Exception as e:
            logger.warning(f"Could not reuse test CSV: {e}")

    val_prompt_df = None
    if val_data and (not args.force_recompute_val) and val_prompt_csv.exists():
        try:
            dfv = pd.read_csv(val_prompt_csv)
            if prompt_csv_covers(dfv, layers_req, factors):
                val_prompt_df = dfv
                logger.warning(
                    f"Reusing {val_prompt_csv} — covers layers/factors. "
                    "Pass --force_recompute_val to redo.",
                )
            else:
                logger.warning(f"{val_prompt_csv} incomplete; recomputing val.")
        except Exception as e:
            logger.warning(f"Could not reuse val CSV: {e}")

    dprimes = {}
    train_projections = {}

    if dprime_json.exists():
        try:
            with open(dprime_json) as f:
                dprimes = {int(k): float(v) for k, v in json.load(f).items()}
            logger.warning(f"Loaded d' for {len(dprimes)} layers from {dprime_json}")
        except Exception as e:
            logger.warning(f"Could not load {dprime_json}: {e}")

    if train_proj_json.exists():
        try:
            with open(train_proj_json) as f:
                raw = json.load(f)
            train_projections = {
                int(k): {"pos": v["pos"], "neg": v["neg"]}
                for k, v in raw.items()
            }
            logger.warning(
                f"Loaded train projections for {len(train_projections)} layers from {train_proj_json}",
            )
        except Exception as e:
            logger.warning(f"Could not load {train_proj_json}: {e}")

    need_test_forward = args.force_recompute or per_prompt_df is None
    need_val_forward = bool(val_data) and (
        args.force_recompute_val or val_prompt_df is None
    )

    need_phase0_stats = (not dprimes) or (not train_projections)
    steering_ok = steering_pt.exists()
    need_phase0_compute = need_phase0_stats or (
        (need_test_forward or need_val_forward) and not steering_ok
    )

    need_load_model = need_phase0_compute or need_test_forward or need_val_forward

    steering_vecs = {}
    mu_poss = {}
    mu_negs = {}
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
            logger.warning(f"\n=== Phase 0: DiffMean ({len(train_data)} pairs, {len(layers)} layers) ===")

            train_flat = []
            for item in train_data:
                q = item["question"]
                for answer, label in [
                    (item["answer_matching_behavior"], 1),
                    (item["answer_not_matching_behavior"], 0),
                ]:
                    try:
                        letter = extract_letter(answer)
                        _, full_ids, letter_pos, open_paren_pos = build_full_ids(tokenizer, q, letter)
                        train_flat.append({"full_ids": full_ids, "letter_pos": letter_pos, "label": label})
                    except Exception:
                        pass

            pos_acts = {l: [] for l in layers}
            neg_acts = {l: [] for l in layers}

            for start in tqdm(range(0, len(train_flat), B), desc="Phase 0"):
                batch = train_flat[start:start + B]
                ids, mask, _ = pad_batch([b["full_ids"] for b in batch], pad_id, device)
                dummy_pos = [0] * len(batch)
                _, hiddens = forward_capture(model, ids, mask, layers, dummy_pos)
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

            with open(dprime_json, "w") as f:
                json.dump({str(ly): float(v) for ly, v in dprimes.items()}, f, indent=2)
            logger.warning(f"Saved d' to {dprime_json}")

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
            logger.warning(
                f"Loaded steering tensors for layers {layers} from {steering_pt}",
            )

        if need_test_forward:
            per_prompt_df = run_phase_ab_mcqa(
                model,
                tokenizer,
                layers,
                factors,
                device,
                pad_id,
                B,
                prefix_length,
                mu_poss,
                mu_negs,
                steering_vecs,
                test_data,
                "test",
            )
            per_prompt_df.to_csv(per_prompt_csv, index=False)
            logger.warning(f"Saved {len(per_prompt_df)} rows to {per_prompt_csv}")

        if need_val_forward:
            val_prompt_df = run_phase_ab_mcqa(
                model,
                tokenizer,
                layers,
                factors,
                device,
                pad_id,
                B,
                prefix_length,
                mu_poss,
                mu_negs,
                steering_vecs,
                val_data,
                "val",
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

    # ------------------------------------------------------------------
    # Analysis: per-layer summary.
    # ------------------------------------------------------------------

    layer_rows = []
    for l in layers:
        ldf = per_prompt_df[per_prompt_df["layer"] == l]
        kappa = ldf["kappa_a"].values
        correct = ldf["baseline_correct"].astype(int).values

        sign_pred = (kappa > 0).astype(int)
        mcc = safe_mcc(sign_pred[~np.isnan(kappa)], correct[~np.isnan(kappa)])
        rho, rho_p = safe_spearman(kappa, correct.astype(float))

        row = {
            "layer": int(l),
            "dprime": dprimes.get(int(l), float("nan")),
            "n_prompts": int(len(ldf)),
            "baseline_acc": float(ldf["baseline_correct"].mean()),
            "baseline_off_format": float((~ldf["baseline_on_format"]).mean()),
            "sign_kappa_mcc": mcc,
            "kappa_spearman_rho": rho,
            "kappa_spearman_p": rho_p,
        }
        for factor in factors:
            col_c = f"steered_correct_{factor:g}"
            col_f = f"steered_on_format_{factor:g}"
            if col_c in ldf.columns:
                row[f"steered_acc_{factor:g}"] = float(ldf[col_c].mean())
                row[f"steered_off_format_{factor:g}"] = float((~ldf[col_f]).mean())

        layer_rows.append(row)

    layer_df = pd.DataFrame(layer_rows).sort_values("layer").reset_index(drop=True)

    # Add best-across-factors steered accuracy per layer.
    acc_cols = [f"steered_acc_{f:g}" for f in factors if f"steered_acc_{f:g}" in layer_df.columns]
    if acc_cols:
        layer_df["best_steered_acc"] = layer_df[acc_cols].max(axis=1)
        layer_df["best_factor"] = layer_df[acc_cols].idxmax(axis=1).str.replace("steered_acc_", "")

    # MCC(sign κ vs steer correctness) @ best α chosen on **test** (leaks test if used as primary).
    best_alpha_mccs = []
    for _, rr in layer_df.iterrows():
        bf = rr.get("best_factor")
        if bf is None or pd.isna(bf):
            best_alpha_mccs.append(float("nan"))
            continue
        try:
            alpha_star = float(bf)
        except (TypeError, ValueError):
            best_alpha_mccs.append(float("nan"))
            continue
        L = int(rr["layer"])
        ldf = per_prompt_df[per_prompt_df["layer"] == L]
        best_alpha_mccs.append(sign_kappa_mcc_at_steered_alpha(ldf, alpha_star))
    layer_df["sign_kappa_mcc_best_alpha"] = best_alpha_mccs

    # Defaults for val-merge columns.
    layer_df["sign_kappa_mcc_val_best_on_test"] = float("nan")
    layer_df["best_steered_acc_val"] = float("nan")
    layer_df["best_factor_val"] = np.nan
    layer_df["test_steered_acc_at_val_best_alpha"] = float("nan")
    layer_df["val_steered_acc_at_test_best_alpha"] = float("nan")

    acc_cols_v: list[str] = []
    layer_df_val = None
    if val_prompt_df is not None:
        val_prompt_df = val_prompt_df[val_prompt_df["layer"].isin(layers)].copy()
        layer_rows_v = []
        for l in layers:
            ldfv = val_prompt_df[val_prompt_df["layer"] == l]
            if len(ldfv) == 0:
                continue
            rowv = {
                "layer": int(l),
                "dprime": dprimes.get(int(l), float("nan")),
                "n_prompts_val": int(len(ldfv)),
                "baseline_acc_val": float(ldfv["baseline_correct"].mean()),
            }
            for factor in factors:
                col_c = f"steered_correct_{factor:g}"
                col_f = f"steered_on_format_{factor:g}"
                if col_c in ldfv.columns:
                    rowv[f"steered_acc_val_{factor:g}"] = float(ldfv[col_c].mean())
                    rowv[f"steered_off_format_val_{factor:g}"] = float((~ldfv[col_f]).mean())
            layer_rows_v.append(rowv)
        layer_df_val = pd.DataFrame(layer_rows_v).sort_values("layer").reset_index(drop=True)

        layer_df_val["baseline_acc"] = layer_df_val["baseline_acc_val"]
        for f in factors:
            vcol = f"steered_acc_val_{f:g}"
            if vcol in layer_df_val.columns:
                layer_df_val[f"steered_acc_{f:g}"] = layer_df_val[vcol]

        acc_cols_v = [
            f"steered_acc_val_{f:g}" for f in factors if f"steered_acc_val_{f:g}" in layer_df_val.columns
        ]
        if acc_cols_v:
            layer_df_val["best_steered_acc_val"] = layer_df_val[acc_cols_v].max(axis=1)
            layer_df_val["best_factor_val"] = layer_df_val[acc_cols_v].idxmax(axis=1).str.replace(
                "steered_acc_val_", "",
            )

            bf_val_map = layer_df_val.set_index("layer")["best_factor_val"]
            ba_val_map = layer_df_val.set_index("layer")["best_steered_acc_val"]
            layer_df["best_factor_val"] = layer_df["layer"].map(bf_val_map)
            layer_df["best_steered_acc_val"] = layer_df["layer"].map(ba_val_map)

            mcc_val_on_test = []
            test_at_val_a = []
            val_at_test_a = []
            for _, rr in layer_df.iterrows():
                L = int(rr["layer"])
                ldf_test = per_prompt_df[per_prompt_df["layer"] == L]
                ldf_val = val_prompt_df[val_prompt_df["layer"] == L]
                bf_v = rr.get("best_factor_val")
                bf_t = rr.get("best_factor")
                if bf_v is None or pd.isna(bf_v):
                    mcc_val_on_test.append(float("nan"))
                    test_at_val_a.append(float("nan"))
                else:
                    try:
                        av = float(bf_v)
                        mcc_val_on_test.append(sign_kappa_mcc_at_steered_alpha(ldf_test, av))
                        col_v = f"steered_correct_{av:g}"
                        test_at_val_a.append(
                            float(ldf_test[col_v].mean()) if col_v in ldf_test.columns else float("nan"),
                        )
                    except (TypeError, ValueError):
                        mcc_val_on_test.append(float("nan"))
                        test_at_val_a.append(float("nan"))
                if bf_t is None or pd.isna(bf_t):
                    val_at_test_a.append(float("nan"))
                else:
                    try:
                        at = float(bf_t)
                        col_t = f"steered_correct_{at:g}"
                        val_at_test_a.append(
                            float(ldf_val[col_t].mean()) if len(ldf_val) and col_t in ldf_val.columns
                            else float("nan"),
                        )
                    except (TypeError, ValueError):
                        val_at_test_a.append(float("nan"))

            layer_df["sign_kappa_mcc_val_best_on_test"] = mcc_val_on_test
            layer_df["test_steered_acc_at_val_best_alpha"] = test_at_val_a
            layer_df["val_steered_acc_at_test_best_alpha"] = val_at_test_a
            layer_df_val.to_csv(out_dir / "val_per_layer_summary.csv", index=False)

    layer_df.to_csv(out_dir / "per_layer_summary.csv", index=False)

    logger.warning("\nPer-layer summary:")
    for _, r in layer_df.iterrows():
        steered_str = "  ".join(
            f"α={f:g}={r[f'steered_acc_{f:g}']:.3f}"
            for f in factors if f"steered_acc_{f:g}" in r and not pd.isna(r[f"steered_acc_{f:g}"])
        )
        best_str = f"  best={r['best_steered_acc']:.3f}@α={r['best_factor']}" \
            if "best_steered_acc" in r else ""
        logger.warning(
            f"  L{int(r['layer']):2d}: d'={r['dprime']:.3f}  "
            f"base={r['baseline_acc']:.3f}  "
            f"MCC={r['sign_kappa_mcc']:+.3f}  ρ={r['kappa_spearman_rho']:+.3f}  "
            f"{steered_str}{best_str}"
        )

    # Cross-layer correlations: predictor vs each factor AND vs best-across-factors.
    cross_rows = []
    predictors = [
        ("sign_kappa_mcc", "sign(κ_a) MCC"),
        ("sign_kappa_mcc_best_alpha", "sign κ MCC @ test-best α"),
        ("sign_kappa_mcc_val_best_on_test", "sign κ MCC @ val-best α (test)"),
        ("kappa_spearman_rho", "κ_a Spearman ρ"),
        ("dprime", "d'"),
    ]

    targets = [(f"steered_acc_{f:g}", f"α={f:g}") for f in factors
               if f"steered_acc_{f:g}" in layer_df.columns]
    if "best_steered_acc" in layer_df.columns:
        targets.append(("best_steered_acc", "best α per layer (test)"))
    if val_prompt_df is not None and "test_steered_acc_at_val_best_alpha" in layer_df.columns:
        targets.append(("test_steered_acc_at_val_best_alpha", "test acc @ val-best α"))

    for pred_col, pred_label in predictors:
        if pred_col not in layer_df.columns:
            continue
        pv = layer_df[pred_col].values
        for tgt_col, tgt_label in targets:
            if tgt_col not in layer_df.columns:
                continue
            tv = layer_df[tgt_col].values
            mask = ~(np.isnan(pv) | np.isnan(tv))
            if mask.sum() < 3 or np.std(pv[mask]) < 1e-9 or np.std(tv[mask]) < 1e-9:
                continue
            r, p = scipy_stats.pearsonr(pv[mask], tv[mask])
            rho, sp = scipy_stats.spearmanr(pv[mask], tv[mask])
            cross_rows.append({
                "predictor": pred_label, "target": tgt_label,
                "pearson_r": float(r), "pearson_p": float(p),
                "spearman_rho": float(rho), "spearman_p": float(sp),
                "n_layers": int(mask.sum()),
            })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(out_dir / "cross_layer_corr.csv", index=False)
    if not cross_df.empty:
        logger.warning("\nCross-layer correlations (predictor → steered accuracy):")
        for _, r in cross_df.iterrows():
            sig = "*" if r["pearson_p"] < 0.05 else " "
            logger.warning(
                f"  {r['predictor']:42s} → {r['target']:28s}: "
                f"r={r['pearson_r']:+.3f} (p={r['pearson_p']:.3g}){sig}  "
                f"ρ={r['spearman_rho']:+.3f} (p={r['spearman_p']:.3g})"
            )

    # Pearson: per-layer d' vs MCC(sign κ vs steer success @ best α for that layer).
    pearson_mcc_best_vs_dprime = None
    if (
        "sign_kappa_mcc_best_alpha" in layer_df.columns
        and "dprime" in layer_df.columns
    ):
        mask_p = (
            layer_df["sign_kappa_mcc_best_alpha"].notna()
            & layer_df["dprime"].notna()
        )
        if mask_p.sum() >= 3:
            xv = layer_df.loc[mask_p, "sign_kappa_mcc_best_alpha"].values.astype(float)
            yv = layer_df.loc[mask_p, "dprime"].values.astype(float)
            if np.nanstd(xv) > 1e-12 and np.nanstd(yv) > 1e-12:
                r_dp, p_dp = scipy_stats.pearsonr(xv, yv)
                pearson_mcc_best_vs_dprime = {
                    "pearson_r": float(r_dp),
                    "pearson_p": float(p_dp),
                    "n_layers": int(mask_p.sum()),
                }
                sig = "*" if p_dp < 0.05 else " "
                logger.warning(
                    f"\n{args.behavior}: Pearson across layers — "
                    f"d' vs MCC(sign κ vs match @ best α per layer): "
                    f"r={r_dp:+.4f} (p={p_dp:.4g}){sig}  (n={mask_p.sum()} layers)"
                )

    pearson_mcc_val_best_on_test_vs_dprime = None
    if (
        val_prompt_df is not None
        and "sign_kappa_mcc_val_best_on_test" in layer_df.columns
        and "dprime" in layer_df.columns
    ):
        mask_v = (
            layer_df["sign_kappa_mcc_val_best_on_test"].notna()
            & layer_df["dprime"].notna()
        )
        if mask_v.sum() >= 3:
            xv = layer_df.loc[mask_v, "sign_kappa_mcc_val_best_on_test"].values.astype(float)
            yv = layer_df.loc[mask_v, "dprime"].values.astype(float)
            if np.nanstd(xv) > 1e-12 and np.nanstd(yv) > 1e-12:
                r_v, p_v = scipy_stats.pearsonr(xv, yv)
                pearson_mcc_val_best_on_test_vs_dprime = {
                    "pearson_r": float(r_v),
                    "pearson_p": float(p_v),
                    "n_layers": int(mask_v.sum()),
                }
                sig = "*" if p_v < 0.05 else " "
                logger.warning(
                    f"\n{args.behavior}: Pearson across layers — "
                    f"d' vs MCC(sign κ vs match @ **val**-best α, evaluated on **test**): "
                    f"r={r_v:+.4f} (p={p_v:.4g}){sig}  (n={mask_v.sum()} layers)",
                )

    pearson_dprime_vs_test_acc_val_best_alpha = None
    if (
        val_prompt_df is not None
        and "test_steered_acc_at_val_best_alpha" in layer_df.columns
        and "dprime" in layer_df.columns
    ):
        mask_a = (
            layer_df["test_steered_acc_at_val_best_alpha"].notna()
            & layer_df["dprime"].notna()
        )
        if mask_a.sum() >= 3:
            xv = layer_df.loc[mask_a, "test_steered_acc_at_val_best_alpha"].values.astype(float)
            yv = layer_df.loc[mask_a, "dprime"].values.astype(float)
            if np.nanstd(xv) > 1e-12 and np.nanstd(yv) > 1e-12:
                r_a, p_a = scipy_stats.pearsonr(xv, yv)
                pearson_dprime_vs_test_acc_val_best_alpha = {
                    "pearson_r": float(r_a),
                    "pearson_p": float(p_a),
                    "n_layers": int(mask_a.sum()),
                }
                sig = "*" if p_a < 0.05 else " "
                logger.warning(
                    f"\n{args.behavior}: Pearson across layers — "
                    f"d' vs **test** steering accuracy @ **val**-best α per layer: "
                    f"r={r_a:+.4f} (p={p_a:.4g}){sig}  (n={mask_a.sum()} layers)",
                )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_projection_quality(layer_df, args.behavior, plots_test / "projection_quality_by_layer.png")
    plot_steering_acc(layer_df, factors, args.behavior, plots_test / "steering_acc_by_layer.png")
    plot_steering_and_dprime(layer_df, factors, args.behavior, plots_test / "steering_acc_and_dprime.png")
    plot_mcc_best_alpha_vs_dprime(layer_df, args.behavior, plots_test / "mcc_best_alpha_vs_dprime.png")
    plot_projection_vs_steering(
        layer_df,
        factors + (["best"] if "best_steered_acc" in layer_df.columns else []),
        args.behavior,
        plots_test / "projection_vs_steering.png",
    )
    for l in layers:
        plot_kappa_scatter(per_prompt_df, int(l), args.behavior,
                           plots_test / f"kappa_scatter_layer_{int(l)}.png")
        plot_projection_histograms(per_prompt_df, factors, args.behavior, int(l),
                                     train_projections,
                                     plots_test / f"projection_hist_layer_{int(l)}.png",
                                     postgen=False)
        plot_projection_histograms(per_prompt_df, factors, args.behavior, int(l),
                                     train_projections,
                                     plots_test / f"projection_hist_postgen_layer_{int(l)}.png",
                                     postgen=True)

    if val_prompt_df is not None and acc_cols_v:
        plot_steering_acc(layer_df_val, factors, args.behavior, plots_val / "steering_acc_by_layer.png")
        plot_steering_and_dprime(layer_df_val, factors, args.behavior, plots_val / "steering_acc_and_dprime.png")

    if val_prompt_df is not None:
        plot_mcc_val_best_alpha_on_test_vs_dprime(
            layer_df,
            args.behavior,
            plots_root / "mcc_best_val_alpha_on_test_vs_dprime.png",
        )
        plot_best_alpha_val_vs_test(layer_df, args.behavior, plots_root / "best_alpha_val_vs_test.png")
        plot_steering_acc_val_vs_test_extended(
            layer_df,
            args.behavior,
            plots_root / "steering_acc_val_vs_test_cross_split.png",
        )

    # Summary JSON.
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
        "n_val_prompts": int(len(val_prompt_df["prompt_idx"].unique())) if val_prompt_df is not None else 0,
        "per_layer": layer_df.to_dict(orient="records"),
        "cross_layer_corr": cross_df.to_dict(orient="records"),
        "pearson_dprime_vs_sign_kappa_mcc_best_alpha": pearson_mcc_best_vs_dprime,
        "pearson_dprime_vs_sign_kappa_mcc_val_best_on_test": pearson_mcc_val_best_on_test_vs_dprime,
        "pearson_dprime_vs_test_steered_acc_at_val_best_alpha": pearson_dprime_vs_test_acc_val_best_alpha,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.warning(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
