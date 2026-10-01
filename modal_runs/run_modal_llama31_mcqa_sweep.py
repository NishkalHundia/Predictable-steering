"""
Llama-3.1-8B-Instruct ARR MCQA sweep: DiffMean at the teacher-forced answer
letter, greedy decode at '('. No thinking tags. Same pipeline as Gemma;
does not touch Gemma or Qwen production paths.

    MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_llama31_mcqa_sweep.py

Monitor:
    MODAL_PROFILE=nishkalhundia modal app logs llama31-mcqa-arr
"""
import os
import subprocess

import modal

app = modal.App("llama31-mcqa-arr")

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
    .add_local_dir("datasets/raw", "/root/axbench/datasets/raw", copy=True)
    .add_local_dir("datasets/test", "/root/axbench/datasets/test", copy=True)
    .run_commands(
        "cd /root/axbench && uv sync --frozen && "
        "uv pip install --python .venv/bin/python -U "
        "'transformers>=4.57.0' pillow"
    )
)

SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = "Llama-3.1-8B-Instruct"

BEHAVIORS = [
    "myopic-reward",
    "sycophancy",
    "hallucination",
    "survival-instinct",
    "corrigible-neutral-HHH",
]

# Llama-3.1-8B has 32 layers indexed 0..31 (same as Qwen3.5-9B).
LAYERS = "10-31"
FACTORS = "1,2,3,5,10"
BATCH_SIZE = "16"
MAX_EXAMPLES = "300"
MAX_VAL_EXAMPLES = "50"

OUTPUT_ROOT = f"/vol/mcqa_ARR/{MODEL_SHORT}"


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
        print("WARNING: no HF token found in env (Llama-3.1 is gated)", flush=True)
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")


def _run(cmd: list[str]) -> None:
    if cmd[:3] == ["uv", "run", "python"]:
        cmd = ["/root/axbench/.venv/bin/python", *cmd[3:]]
    print("Running:", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": "/root/axbench", "UV_NO_SYNC": "1"}
    subprocess.run(cmd, cwd="/root/axbench", check=True, env=env)


@app.function(
    image=image,
    volumes={"/vol": vol},
    gpu="A100-80GB",
    timeout=86400,
    secrets=SECRETS,
    max_containers=len(BEHAVIORS),
)
def run_one(behavior: str) -> tuple[str, str]:
    _normalize_hf_token()
    output_dir = f"{OUTPUT_ROOT}/{behavior}"
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"\n=== Llama-3.1 ARR MCQA  model={MODEL_NAME}  behavior={behavior} ===",
        flush=True,
    )
    print(f"  layers={LAYERS}  factors={FACTORS}  batch={BATCH_SIZE}", flush=True)
    print("  chat template: Llama-3.1-Instruct (no think tags)", flush=True)
    print("  DiffMean at teacher-forced letter; greedy decode at '('", flush=True)
    print(f"  output → {output_dir}", flush=True)

    try:
        _run([
            "uv", "run", "python",
            "axbench/scripts/mcqa_projection_link.py",
            "--behavior", behavior,
            "--model_name", MODEL_NAME,
            "--train_path", f"datasets/raw/{behavior}/dataset.json",
            "--test_path", f"datasets/test/{behavior}/test_dataset_ab.json",
            "--output_dir", output_dir,
            "--layers", LAYERS,
            "--factors", FACTORS,
            "--batch_size", BATCH_SIZE,
            "--max_examples", MAX_EXAMPLES,
            "--max_val_examples", MAX_VAL_EXAMPLES,
            "--force_recompute",
            "--force_recompute_val",
        ])
        vol.commit()
    except subprocess.CalledProcessError as e:
        print(f"!!! FAILED {behavior}: exit {e.returncode}", flush=True)
        return (behavior, f"failed(exit {e.returncode})")

    print(f"=== DONE {behavior} → {output_dir} ===", flush=True)
    return (behavior, "done")


@app.function(
    image=image,
    volumes={"/vol": vol},
    timeout=86400,
    secrets=SECRETS,
)
def run_sweep():
    results = list(run_one.map(BEHAVIORS, order_outputs=False))
    done = [b for b, s in results if s == "done"]
    failed = [f"{b}:{s}" for b, s in results if s != "done"]
    print(
        f"\n=== LLAMA-3.1 ARR MCQA SWEEP SUMMARY ===\n"
        f"  done:   {done or '-'}\n"
        f"  failed: {failed or '-'}",
        flush=True,
    )


@app.local_entrypoint()
def main():
    fc = run_sweep.spawn()
    print(f"Spawned run_sweep call id={fc.object_id}")
    print(f"{len(BEHAVIORS)}× A100-80GB  layers={LAYERS}  factors={FACTORS}")
    print("no thinking  decode=greedy-at-'('  DiffMean-at-letter  batch=%s" % BATCH_SIZE)
    print("Watch:  MODAL_PROFILE=nishkalhundia modal app logs llama31-mcqa-arr")
    print(f"MCQA ARR → {OUTPUT_ROOT}/<behavior>/")
    print("Behaviors:", ", ".join(BEHAVIORS))
