"""
Modal runner for create_prompted_open_ended_contrastive.py.

One behavior per run, one A100 GPU. Launch multiple detached runs in parallel
for different behaviors.

    MODAL_PROFILE=nishkalhundia modal run --detach run_modal_prompted_contrastive.py \\
        --behavior hallucination

Outputs:
    /vol/prompted_datasets/generated/<behavior>/{train,test}_contrastive.json
    /vol/prompted_datasets/generated/<behavior>/meta.json
"""
import os
import subprocess

import modal

app = modal.App("prompted-contrastive-datagen")

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
            "run_modal_sweep.py", "run_modal_sleep.py",
            "run_modal_projection_plots.py",
            "run_modal_prompted_contrastive.py",
        ],
    )
    .add_local_dir("datasets/test", "/root/axbench/datasets/test", copy=True)
    .add_local_dir("datasets/generate", "/root/axbench/datasets/generate", copy=True)
    .run_commands("cd /root/axbench && uv sync --frozen")
)

SECRETS = [modal.Secret.from_name("hf-gemma-token")]

ALLOWED_BEHAVIORS = [
    "myopic-reward",
    "sycophancy",
    "hallucination",
    "survival-instinct",
    "corrigible-neutral-HHH",
]

MODEL_NAME = "google/gemma-2-9b-it"
PROMPT_MODE = "prefix"
OUTPUT_ROOT = "/vol/prompted_datasets/generated"
GREEDY = False
MAX_NEW_TOKENS = 150
BATCH_SIZE = 8
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42


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
        print("WARNING: no HF token found — gated model load will 401", flush=True)
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu="A100",
    timeout=86400,
    secrets=SECRETS,
)
def run_one(behavior: str) -> tuple[str, str]:
    """Generate prompted contrastive data for one behavior on its own GPU."""
    _normalize_hf_token()
    output_dir = f"{OUTPUT_ROOT}/{behavior}"
    print(f"\n=== START {behavior} → {output_dir} ===", flush=True)

    cmd = [
        "uv", "run", "python",
        "axbench/scripts/create_prompted_open_ended_contrastive.py",
        "--behavior", behavior,
        "--model_name", MODEL_NAME,
        "--output_dir", output_dir,
        "--prompt_mode", PROMPT_MODE,
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--batch_size", str(BATCH_SIZE),
        "--temperature", str(TEMPERATURE),
        "--top_p", str(TOP_P),
        "--seed", str(SEED),
    ]
    if GREEDY:
        cmd.append("--greedy")

    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(
            cmd,
            cwd="/root/axbench",
            check=True,
            env={**os.environ, "PYTHONPATH": "/root/axbench"},
        )
    except subprocess.CalledProcessError as e:
        print(f"!!! FAILED {behavior}: exit {e.returncode}", flush=True)
        return (behavior, f"failed(exit {e.returncode})")

    vol.commit()
    print(f"=== DONE {behavior} → {output_dir} ===", flush=True)
    return (behavior, "done")


@app.local_entrypoint()
def main(behavior: str):
    if behavior not in ALLOWED_BEHAVIORS:
        raise SystemExit(
            f"Unknown behavior {behavior!r}; choose from {ALLOWED_BEHAVIORS}"
        )
    fc = run_one.spawn(behavior)
    print(f"Spawned run_one({behavior}) call id={fc.object_id}")
    print(f"prompt_mode={PROMPT_MODE}, greedy={GREEDY}, model={MODEL_NAME}")
    print("Running detached in Modal cloud. Safe to disconnect.")
    print(f"Results → {OUTPUT_ROOT}/{behavior}/")
