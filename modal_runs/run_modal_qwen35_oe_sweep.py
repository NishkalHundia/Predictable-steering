"""
Qwen3.5-9B ARR open-ended sweep: regenerate cued pos/neg pairs, then three
OE evals into ARR-named volume folders (does not touch Gemma production paths).

  1. last-token open-ended ARR  — vanilla DiffMean last_token
  2. average-token open-ended ARR — vanilla DiffMean avg_token
  3. open-ended ARR (prompted) — cue DiffMean

See notes.md.

    MODAL_PROFILE=nishkalhundia modal run --detach modal_runs/run_modal_qwen35_oe_sweep.py

Monitor:
    MODAL_PROFILE=nishkalhundia modal app logs qwen35-oe-arr
"""
import os
import subprocess

import modal

app = modal.App("qwen35-oe-arr")

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

BEHAVIORS = [
    "myopic-reward",
    "sycophancy",
    "hallucination",
    "survival-instinct",
    "corrigible-neutral-HHH",
]

# Qwen3.5-9B has 32 layers indexed 0..31 (Gemma-2-9B-it is 0..41).
LAYERS = "10-31"
FACTORS = "0,1,2,3,5,10"
HIST_LAYERS = ",".join(str(l) for l in range(10, 32))
JUDGE_MODEL = "gpt-4o-mini"
PROMPT_MODE = "prefix"
BATCH_SIZE = "32"
FLUENCY_THRESHOLD = "1.0"
MIN_EXAMPLES = "28"
DATAGEN_MAX_NEW_TOKENS = "150"
OE_MAX_NEW_TOKENS = "200"
DATAGEN_BATCH_SIZE = "8"

CONTRASTIVE_ROOT = f"/vol/prompted_datasets/{MODEL_SHORT}/generated"
LAST_ROOT = f"/vol/last_token_open_ended_ARR/{MODEL_SHORT}"
AVG_ROOT = f"/vol/average_token_open_ended_ARR/{MODEL_SHORT}"
PROMPTED_ROOT = f"/vol/open_ended_ARR/{MODEL_SHORT}"


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


def _vanilla_oe_cmd(behavior: str, train_path: str, test_path: str,
                    output_dir: str, diffmean_mode: str) -> list[str]:
    return [
        "uv", "run", "python",
        "axbench/scripts/open_ended_projection_link.py",
        "--behavior", behavior,
        "--model_name", MODEL_NAME,
        "--train_path", train_path,
        "--test_path", test_path,
        "--output_dir", output_dir,
        "--layers", LAYERS,
        "--factors", FACTORS,
        "--diffmean_mode", diffmean_mode,
        "--batch_size", BATCH_SIZE,
        "--max_new_tokens", OE_MAX_NEW_TOKENS,
        "--fluency_threshold", FLUENCY_THRESHOLD,
        "--min_examples", MIN_EXAMPLES,
        "--judge_model", JUDGE_MODEL,
        "--hist_layers", HIST_LAYERS,
        "--force_recompute",
    ]


def _prompted_oe_cmd(behavior: str, train_path: str, test_path: str,
                     output_dir: str) -> list[str]:
    return [
        "uv", "run", "python",
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
        "--force_recompute",
    ]


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
    train_path = f"{CONTRASTIVE_ROOT}/{behavior}/train_contrastive.json"
    test_path = f"{CONTRASTIVE_ROOT}/{behavior}/test_contrastive.json"
    last_dir = f"{LAST_ROOT}/{behavior}"
    avg_dir = f"{AVG_ROOT}/{behavior}"
    prompted_dir = f"{PROMPTED_ROOT}/{behavior}"

    print(f"\n=== Qwen3.5 ARR OE  model={MODEL_NAME}  behavior={behavior} ===", flush=True)
    print(f"  layers={LAYERS}  factors={FACTORS}  judge={JUDGE_MODEL}", flush=True)
    print("  thinking disabled via apply_chat_template(enable_thinking=False)", flush=True)
    print("  OE decode is greedy (do_sample=False)", flush=True)

    print("\n=== Step 1: Qwen pos/neg contrastive pairs ===", flush=True)
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  reuse existing {CONTRASTIVE_ROOT}/{behavior}/", flush=True)
    else:
        try:
            _run([
                "uv", "run", "python",
                "axbench/scripts/create_prompted_open_ended_contrastive.py",
                "--behavior", behavior,
                "--model_name", MODEL_NAME,
                "--output_dir", f"{CONTRASTIVE_ROOT}/{behavior}",
                "--prompt_mode", PROMPT_MODE,
                "--max_new_tokens", DATAGEN_MAX_NEW_TOKENS,
                "--batch_size", DATAGEN_BATCH_SIZE,
                "--seed", "42",
            ])
            vol.commit()
        except subprocess.CalledProcessError as e:
            print(f"!!! FAILED datagen {behavior}: exit {e.returncode}", flush=True)
            return (behavior, f"failed-datagen(exit {e.returncode})")

    print("\n=== Step 2: last-token open-ended ARR ===", flush=True)
    try:
        _run(_vanilla_oe_cmd(behavior, train_path, test_path, last_dir, "last_token"))
        vol.commit()
    except subprocess.CalledProcessError as e:
        print(f"!!! FAILED last_token {behavior}: exit {e.returncode}", flush=True)
        return (behavior, f"failed-last_token(exit {e.returncode})")

    print("\n=== Step 3: average-token open-ended ARR ===", flush=True)
    try:
        _run(_vanilla_oe_cmd(behavior, train_path, test_path, avg_dir, "avg_token"))
        vol.commit()
    except subprocess.CalledProcessError as e:
        print(f"!!! FAILED avg_token {behavior}: exit {e.returncode}", flush=True)
        return (behavior, f"failed-avg_token(exit {e.returncode})")

    print("\n=== Step 4: open-ended ARR (prompted / cue DiffMean) ===", flush=True)
    try:
        _run(_prompted_oe_cmd(behavior, train_path, test_path, prompted_dir))
        vol.commit()
    except subprocess.CalledProcessError as e:
        print(f"!!! FAILED prompted {behavior}: exit {e.returncode}", flush=True)
        return (behavior, f"failed-prompted(exit {e.returncode})")

    print(f"=== DONE {behavior} ===", flush=True)
    print(f"  data     → {CONTRASTIVE_ROOT}/{behavior}/", flush=True)
    print(f"  last ARR → {last_dir}/", flush=True)
    print(f"  avg ARR  → {avg_dir}/", flush=True)
    print(f"  cue ARR  → {prompted_dir}/", flush=True)
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
        f"\n=== QWEN3.5 ARR OE SWEEP SUMMARY ===\n"
        f"  done:   {done or '-'}\n"
        f"  failed: {failed or '-'}",
        flush=True,
    )


@app.local_entrypoint()
def main():
    fc = run_sweep.spawn()
    print(f"Spawned run_sweep call id={fc.object_id}")
    print(f"{len(BEHAVIORS)}× A100-80GB  layers={LAYERS}  factors={FACTORS}")
    print(f"judge={JUDGE_MODEL}  thinking=off  decode=greedy  batch={BATCH_SIZE}")
    print("Watch:  MODAL_PROFILE=nishkalhundia modal app logs qwen35-oe-arr")
    print(f"Contrastive              → {CONTRASTIVE_ROOT}/<behavior>/")
    print(f"last-token open-ended ARR → {LAST_ROOT}/<behavior>/")
    print(f"average-token open-ended ARR → {AVG_ROOT}/<behavior>/")
    print(f"open-ended ARR (prompted) → {PROMPTED_ROOT}/<behavior>/")
    print("Behaviors:", ", ".join(BEHAVIORS))
