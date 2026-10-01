"""
Qwen3.5-9B ARR MCQA smoke: DiffMean at the teacher-forced answer letter,
greedy decode at '('. Thinking off. Does not touch Gemma production paths.

    MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_qwen35_mcqa_smoke.py

Monitor:
    MODAL_PROFILE=nishkalhundia modal app logs qwen35-mcqa-smoke
"""
import os
import subprocess

import modal

app = modal.App("qwen35-mcqa-smoke")

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
    modal.Secret.from_name("hf-gemma-token"),
]

MODEL_NAME = "Qwen/Qwen3.5-9B"
MODEL_SHORT = "Qwen3.5-9B"
BEHAVIOR = "myopic-reward"
LAYERS = "10,11"
FACTORS = "1,2"
BATCH_SIZE = "4"
MAX_EXAMPLES = "4"
MAX_VAL_EXAMPLES = "2"
MAX_TEST = "2"

OUTPUT_DIR = f"/vol/mcqa_ARR/{MODEL_SHORT}/{BEHAVIOR}_smoke"


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


def _run(cmd: list[str]) -> None:
    # Do NOT use `uv run`: it re-syncs uv.lock and rolls transformers back to
    # 4.45.1, which does not know model_type qwen3_5. The image already has
    # transformers>=4.57 in .venv.
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

    print(f"\n=== Qwen3.5 MCQA smoke  model={MODEL_NAME}  behavior={BEHAVIOR} ===", flush=True)
    print(f"  layers={LAYERS}  factors={FACTORS}  batch={BATCH_SIZE}", flush=True)
    print("  thinking disabled via apply_chat_template(enable_thinking=False)", flush=True)
    print("  DiffMean at teacher-forced letter; greedy decode at '('", flush=True)
    print("  teacher-force includes empty <think></think> when templates diverge", flush=True)
    print(f"  output → {OUTPUT_DIR}", flush=True)

    _run([
        "uv", "run", "python",
        "axbench/scripts/mcqa_projection_link.py",
        "--behavior", BEHAVIOR,
        "--model_name", MODEL_NAME,
        "--train_path", f"datasets/raw/{BEHAVIOR}/dataset.json",
        "--test_path", f"datasets/test/{BEHAVIOR}/test_dataset_ab.json",
        "--output_dir", OUTPUT_DIR,
        "--layers", LAYERS,
        "--factors", FACTORS,
        "--batch_size", BATCH_SIZE,
        "--max_examples", MAX_EXAMPLES,
        "--max_val_examples", MAX_VAL_EXAMPLES,
        "--max_test", MAX_TEST,
        "--force_recompute",
        "--force_recompute_val",
    ])
    vol.commit()
    print(f"=== SMOKE DONE → {OUTPUT_DIR} ===", flush=True)
    return "done"


@app.local_entrypoint()
def main():
    fc = run_smoke.spawn()
    print(f"Spawned Qwen3.5 MCQA smoke call id={fc.object_id}")
    print(f"1× A100-80GB  behavior={BEHAVIOR}  layers={LAYERS}  factors={FACTORS}")
    print("thinking=off  decode=greedy-at-'('  DiffMean-at-letter")
    print("Watch:  MODAL_PROFILE=nishkalhundia modal app logs qwen35-mcqa-smoke")
    print(f"MCQA results → {OUTPUT_DIR}/")
