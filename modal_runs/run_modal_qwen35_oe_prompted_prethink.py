"""
Qwen3.5-9B cue DiffMean at the token *before* `<think>` (not `</think>`).

Full prompted eval (Phase 0 + generate + judge). Reuses existing contrastive
pairs. Writes to /vol/open_ended_ARR_prethink/Qwen3.5-9B/<behavior>/. Does not
touch Gemma paths or the ARR prompted folder.

Myopic-reward already finished; this entrypoint runs the other four in parallel.

    MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_qwen35_oe_prompted_prethink.py

Monitor:
    MODAL_PROFILE=nishkalhundia modal app logs qwen35-oe-prethink
"""
import os
import subprocess

import modal

app = modal.App("qwen35-oe-prethink")

vol = modal.Volume.from_name("steering")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .add_local_dir(
        ".",
        "/root/axbench",
        copy=True,
        ignore=[
            "datasets", "results", "gemma2_2b_l10_steering", "paper_plots",
            ".git", ".venv", "__pycache__", "*.pyc", "*.png", "wandb",
            "modal_runs",
        ],
    )
    .add_local_dir("datasets/test", "/root/axbench/datasets/test", copy=True)
    .add_local_dir("datasets/generate", "/root/axbench/datasets/generate", copy=True)
    .run_commands(
        "cd /root/axbench && uv sync --frozen && "
        "uv pip install --python .venv/bin/python -U "
        "'transformers>=4.57.0' pillow"
    )
)

SECRETS = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("hf-gemma-token"),
]

MODEL_NAME = "Qwen/Qwen3.5-9B"
MODEL_SHORT = "Qwen3.5-9B"
# myopic-reward already written under this root; do not rerun it.
BEHAVIORS = [
    "sycophancy",
    "hallucination",
    "survival-instinct",
    "corrigible-neutral-HHH",
]
LAYERS = "10-31"
FACTORS = "0,1,2,3,5,10"
HIST_LAYERS = ",".join(str(l) for l in range(10, 32))
BATCH_SIZE = "32"
CUE_SITE = "pre_think"
JUDGE_MODEL = "gpt-4o-mini"
OE_MAX_NEW_TOKENS = "200"
FLUENCY_THRESHOLD = "1.0"
MIN_EXAMPLES = "28"

CONTRASTIVE_ROOT = f"/vol/prompted_datasets/{MODEL_SHORT}/generated"
OUTPUT_ROOT = f"/vol/open_ended_ARR_prethink/{MODEL_SHORT}"


def _normalize_hf_token():
    hf_keys = [k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper()]
    print("HF-ish env keys present:", hf_keys, flush=True)
    hf_tok = None
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
              "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(k):
            hf_tok = os.environ[k]
            break
    if hf_tok is None:
        for k in hf_keys:
            if os.environ.get(k, "").startswith("hf_"):
                hf_tok = os.environ[k]
                break
    if hf_tok:
        os.environ["HF_TOKEN"] = hf_tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_tok
        print("HF token normalized into HF_TOKEN / HUGGING_FACE_HUB_TOKEN", flush=True)
    else:
        print("WARNING: no HF token found in env (Qwen3.5 is not gated)", flush=True)
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu="A100-80GB",
    timeout=86400,
    secrets=SECRETS,
    max_containers=len(BEHAVIORS),
)
def run_prethink(behavior: str) -> tuple[str, str]:
    _normalize_hf_token()
    train_path = f"{CONTRASTIVE_ROOT}/{behavior}/train_contrastive.json"
    test_path = f"{CONTRASTIVE_ROOT}/{behavior}/test_contrastive.json"
    output_dir = f"{OUTPUT_ROOT}/{behavior}"
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"missing {train_path}; reuse ARR contrastive pairs, do not regenerate"
        )
    print(
        f"=== pre_think cue DiffMean  {MODEL_NAME}  {behavior} ===\n"
        f"  cue_site={CUE_SITE}  layers={LAYERS}  batch={BATCH_SIZE}\n"
        f"  full prompted eval (Phase 0 + unsteered/steered gen + judge)\n"
        f"  train={train_path}\n"
        f"  out={output_dir}",
        flush=True,
    )
    cmd = [
        "/root/axbench/.venv/bin/python",
        "axbench/scripts/open_ended_projection_link_prompted.py",
        "--behavior", behavior,
        "--model_name", MODEL_NAME,
        "--train_path", train_path,
        "--test_path", test_path,
        "--output_dir", output_dir,
        "--layers", LAYERS,
        "--factors", FACTORS,
        "--batch_size", BATCH_SIZE,
        "--max_new_tokens", OE_MAX_NEW_TOKENS,
        "--fluency_threshold", FLUENCY_THRESHOLD,
        "--min_examples", MIN_EXAMPLES,
        "--judge_model", JUDGE_MODEL,
        "--hist_layers", HIST_LAYERS,
        "--cue_site", CUE_SITE,
        "--force_recompute",
    ]
    print("Running:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": "/root/axbench", "UV_NO_SYNC": "1"}
    try:
        subprocess.run(cmd, cwd="/root/axbench", check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"!!! FAILED {behavior}: exit {e.returncode}", flush=True)
        return (behavior, f"failed(exit {e.returncode})")
    vol.commit()
    print(f"=== DONE {behavior} pre_think prompted eval → {output_dir} ===", flush=True)
    return (behavior, "done")


@app.function(
    image=image,
    volumes={"/vol": vol},
    timeout=86400,
    secrets=SECRETS,
)
def run_sweep():
    results = list(run_prethink.map(BEHAVIORS, order_outputs=False))
    done = [b for b, s in results if s == "done"]
    failed = [f"{b}:{s}" for b, s in results if s != "done"]
    print(
        f"\n=== QWEN3.5 PRETHINK SWEEP SUMMARY ===\n"
        f"  done:   {done or '-'}\n"
        f"  failed: {failed or '-'}",
        flush=True,
    )


@app.local_entrypoint()
def main():
    fc = run_sweep.spawn()
    print(f"Spawned run_sweep call id={fc.object_id}")
    print(
        f"{len(BEHAVIORS)}× A100-80GB  cue_site={CUE_SITE}  batch={BATCH_SIZE}  "
        f"full prompted eval"
    )
    print("Watch:  MODAL_PROFILE=nishkalhundia modal app logs qwen35-oe-prethink")
    print(f"Out: {OUTPUT_ROOT}/<behavior>/")
    print("Behaviors:", ", ".join(BEHAVIORS))
    print("Skipped (already done): myopic-reward")
