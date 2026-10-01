"""
Llama-3.1-8B-Instruct open-ended smoke: tiny cued pos/neg pairs, then last-token OE.

Writes only to *_smoke folders. Delete those after a successful smoke; do not
reuse them for the full ARR sweep.

    MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_llama31_oe_smoke.py

Monitor:
    MODAL_PROFILE=nishkalhundia modal app logs llama31-oe-smoke
"""
import os
import subprocess

import modal

app = modal.App("llama31-oe-smoke")

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
    modal.Secret.from_name("huggingface-secret"),
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = "Llama-3.1-8B-Instruct"
BEHAVIOR = "myopic-reward"
LAYERS = "10,11"
FACTORS = "0,1"
DIFFMEAN_MODE = "last_token"
JUDGE_MODEL = "gpt-4o-mini"
PROMPT_MODE = "prefix"

NUM_OE_TRAIN = "4"
NUM_OE_TEST = "2"
NUM_MCQA_TRAIN = "4"
NUM_MCQA_TEST = "2"

CONTRASTIVE_DIR = (
    f"/vol/prompted_datasets/{MODEL_SHORT}/generated/{BEHAVIOR}_smoke"
)
OUTPUT_DIR = (
    f"/vol/last_token_open_ended_ARR/{MODEL_SHORT}/{BEHAVIOR}_smoke"
)


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
    timeout=21600,
    secrets=SECRETS,
)
def run_smoke() -> str:
    _normalize_hf_token()
    import subprocess as _sp
    print(
        "transformers in venv:",
        _sp.check_output(
            ["/root/axbench/.venv/bin/python", "-c",
             "import transformers; print(transformers.__version__)"],
            text=True,
        ).strip(),
        flush=True,
    )
    train_path = f"{CONTRASTIVE_DIR}/train_contrastive.json"
    test_path = f"{CONTRASTIVE_DIR}/test_contrastive.json"

    print(
        f"\n=== Llama-3.1 OE smoke  model={MODEL_NAME}  behavior={BEHAVIOR} ===",
        flush=True,
    )
    print(f"  layers={LAYERS}  factors={FACTORS}  judge={JUDGE_MODEL}", flush=True)
    print("  chat template: Llama-3.1-Instruct (no think tags)", flush=True)
    print("  OE decode is greedy (do_sample=False)", flush=True)
    print(f"  smoke contrastive → {CONTRASTIVE_DIR}/", flush=True)
    print(f"  smoke OE          → {OUTPUT_DIR}/", flush=True)

    print("\n=== Step 1: tiny Llama pos/neg contrastive pairs ===", flush=True)
    _run([
        "uv", "run", "python",
        "axbench/scripts/create_prompted_open_ended_contrastive.py",
        "--behavior", BEHAVIOR,
        "--model_name", MODEL_NAME,
        "--output_dir", CONTRASTIVE_DIR,
        "--prompt_mode", PROMPT_MODE,
        "--max_new_tokens", "64",
        "--batch_size", "4",
        "--num_oe_train", NUM_OE_TRAIN,
        "--num_oe_test", NUM_OE_TEST,
        "--num_mcqa_train", NUM_MCQA_TRAIN,
        "--num_mcqa_test", NUM_MCQA_TEST,
        "--seed", "42",
    ])
    vol.commit()

    print("\n=== Step 2: last-token OE (pyvene probe at load) ===", flush=True)
    _run([
        "uv", "run", "python",
        "axbench/scripts/open_ended_projection_link.py",
        "--behavior", BEHAVIOR,
        "--model_name", MODEL_NAME,
        "--train_path", train_path,
        "--test_path", test_path,
        "--output_dir", OUTPUT_DIR,
        "--layers", LAYERS,
        "--factors", FACTORS,
        "--diffmean_mode", DIFFMEAN_MODE,
        "--batch_size", "4",
        "--max_new_tokens", "64",
        "--fluency_threshold", "1.0",
        "--min_examples", "1",
        "--judge_model", JUDGE_MODEL,
        "--hist_layers", LAYERS,
        "--force_recompute",
    ])
    vol.commit()
    print(f"=== SMOKE DONE → {OUTPUT_DIR} ===", flush=True)
    return "done"


@app.local_entrypoint()
def main():
    fc = run_smoke.spawn()
    print(f"Spawned Llama-3.1 OE smoke call id={fc.object_id}")
    print(f"1× A100-80GB  behavior={BEHAVIOR}  layers={LAYERS}  factors={FACTORS}")
    print(f"judge={JUDGE_MODEL}  decode=greedy  secret=huggingface-secret")
    print("Watch:  MODAL_PROFILE=nishkalhundia modal app logs llama31-oe-smoke")
    print(f"Contrastive (smoke) → {CONTRASTIVE_DIR}/")
    print(f"OE results (smoke)  → {OUTPUT_DIR}/")
