"""
Model-agnostic helpers for chat templating, layer access, and loading.

Qwen3.5-9B is a VL hybrid (Qwen3_5ForConditionalGeneration) with thinking
on by default. These helpers:

  * pass enable_thinking=False into apply_chat_template
  * load the hub checkpoint and alias language_model.layers onto
    model.model.layers so Gemma-style hooks still work
  * probe pyvene for a working residual-stream component path (no silent fallback)
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

QWEN35_MODEL_IDS = {
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
}

PYVENE_COMPONENT_ATTR = "_axbench_pyvene_component_fmt"


def is_qwen35(name_or_tok) -> bool:
    name = name_or_tok if isinstance(name_or_tok, str) else (
        getattr(name_or_tok, "name_or_path", "") or ""
    )
    n = name.lower().replace("_", ".")
    return "qwen3.5" in n or "qwen3_5" in n


def chat_template_extra_kwargs(tokenizer) -> dict:
    """Kwargs that must be passed into apply_chat_template for this tokenizer."""
    if is_qwen35(tokenizer):
        return {"enable_thinking": False}
    return {}


def _as_token_ids(out) -> list:
    """transformers 5.x may return a dict/BatchEncoding instead of a list of ids."""
    if isinstance(out, dict) or hasattr(out, "keys") and "input_ids" in out:
        ids = out["input_ids"]
    else:
        ids = out
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def apply_chat_template(tokenizer, messages, **kwargs):
    """tokenizer.apply_chat_template with model-specific extras (thinking off)."""
    extra = chat_template_extra_kwargs(tokenizer)
    merged = {**extra, **kwargs}
    try:
        result = tokenizer.apply_chat_template(messages, **merged)
    except TypeError:
        # Older transformers: enable_thinking is not a named arg.
        enable = merged.pop("enable_thinking", None)
        if enable is None:
            raise
        ctk = dict(merged.pop("chat_template_kwargs", None) or {})
        ctk.setdefault("enable_thinking", enable)
        result = tokenizer.apply_chat_template(
            messages, chat_template_kwargs=ctk, **merged
        )
    if merged.get("tokenize"):
        return _as_token_ids(result)
    return result


def get_hidden_size(model) -> int:
    cfg = model.config
    if getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    text = getattr(cfg, "text_config", None)
    if text is not None and getattr(text, "hidden_size", None):
        return int(text.hidden_size)
    raise AttributeError(
        f"Cannot find hidden_size on {type(cfg).__name__} "
        f"(keys: {list(cfg.to_dict().keys())[:12]})"
    )


def num_hidden_layers(model) -> int:
    cfg = model.config
    if getattr(cfg, "num_hidden_layers", None):
        return int(cfg.num_hidden_layers)
    text = getattr(cfg, "text_config", None)
    if text is not None and getattr(text, "num_hidden_layers", None):
        return int(text.num_hidden_layers)
    layers = get_decoder_layers(model)
    return len(layers)


def get_decoder_module(model):
    """Module that owns `.layers` (Gemma: model.model; Qwen VL: language_model)."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise AttributeError(f"{type(model).__name__} has no .model")
    if hasattr(inner, "layers"):
        return inner
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__} "
        f"(inner={type(inner).__name__})"
    )


def get_decoder_layers(model):
    return get_decoder_module(model).layers


def _alias_layers_onto_backbone(model) -> None:
    """Make model.model.layers exist even when the hub class nests language_model."""
    inner = getattr(model, "model", None)
    if inner is None or hasattr(inner, "layers"):
        return
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        inner.layers = lm.layers
        logger.warning(
            "Aliased model.model.layers → model.model.language_model.layers "
            f"({len(lm.layers)} layers)"
        )


def _layer_module_dump(model, limit: int = 40) -> str:
    names = []
    for name, _ in model.named_modules():
        if "layer" in name.lower():
            names.append(name)
            if len(names) >= limit:
                break
    if not names:
        return "(no module names containing 'layer')"
    extra = " ..." if len(names) == limit else ""
    return "\n  ".join(names) + extra


def pyvene_layer_component(model, layer: int) -> str:
    fmt = getattr(model, PYVENE_COMPONENT_ATTR, "model.layers[{layer}].output")
    return fmt.format(layer=layer)


def _resolve_dot_path(root, spec: str):
    """Resolve 'model.layers[10]' style paths. Returns None if any hop is missing."""
    import re
    cur = root
    for part in spec.split("."):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        if m is None:
            return None
        name, idx = m.group(1), m.group(2)
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
        if idx is not None:
            try:
                cur = cur[int(idx)]
            except Exception:
                return None
    return cur


def probe_pyvene_component(model, layer: int) -> str:
    """
    Try pyvene residual-stream paths. Attach the working format string on the
    model. Raise with a module dump if none of the candidates resolve.
    Does not silently fall back to a hand-rolled hook.
    """
    from pyvene import IntervenableConfig, IntervenableModel
    from axbench.models.interventions import AdditionIntervention

    candidates = [
        "model.layers[{layer}].output",
        "model.language_model.layers[{layer}].output",
        "language_model.layers[{layer}].output",
        "model.model.layers[{layer}].output",
    ]
    errors: list[str] = []
    dim = get_hidden_size(model)
    for fmt in candidates:
        component = fmt.format(layer=layer)
        module_spec = (
            component[: -len(".output")] if component.endswith(".output") else component
        )
        if _resolve_dot_path(model, module_spec) is None:
            errors.append(f"{component}: module path does not resolve")
            continue
        try:
            ax = AdditionIntervention(embed_dim=dim, low_rank_dimension=1)
            cfg = IntervenableConfig(representations=[{
                "layer": layer,
                "component": component,
                "low_rank_dimension": 1,
                "intervention": ax,
            }])
            IntervenableModel(cfg, model)
            setattr(model, PYVENE_COMPONENT_ATTR, fmt)
            logger.warning(f"pyvene can see layer {layer} via {component!r}")
            return fmt
        except Exception as exc:
            errors.append(f"{component}: {type(exc).__name__}: {exc}")

    dump = _layer_module_dump(model)
    joined = "\n  ".join(errors)
    raise RuntimeError(
        "pyvene cannot see this model's residual stream. Tried:\n"
        f"  {joined}\n"
        f"model class: {type(model).__name__}\n"
        f"inner class: {type(getattr(model, 'model', None)).__name__}\n"
        f"named modules containing 'layer':\n  {dump}\n"
        "Not falling back to a hand-rolled hook — stop here."
    )


def load_causal_lm(
    model_name: str,
    *,
    torch_dtype=None,
    device_map: Any = None,
    probe_pyvene_layer: int | None = None,
):
    """
    Load a generation model. For Qwen3.5, load the hub VL class then alias
    language_model.layers so existing `model.model.layers` hooks keep working.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    model = None

    if getattr(config, "model_type", None) == "qwen3_5":
        model = _load_qwen35(model_name, config, torch_dtype, device_map)

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    _alias_layers_onto_backbone(model)
    n_layers = num_hidden_layers(model)
    logger.warning(
        f"Loaded {model_name} as {type(model).__name__} "
        f"({n_layers} layers, hidden={get_hidden_size(model)})"
    )
    if probe_pyvene_layer is not None:
        if probe_pyvene_layer >= n_layers:
            raise ValueError(
                f"probe layer {probe_pyvene_layer} is out of range "
                f"(num_hidden_layers={n_layers}, valid 0..{n_layers - 1})"
            )
        probe_pyvene_component(model, probe_pyvene_layer)
    return model


def _load_qwen35(model_name, config, torch_dtype, device_map):
    """Prefer text-only CausalLM; fall back to the VL ConditionalGeneration class."""
    text_cfg = getattr(config, "text_config", None)

    # 1) Official text-only class (skips vision keys when it works).
    try:
        from transformers import Qwen3_5ForCausalLM
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if text_cfg is not None:
            kwargs["config"] = text_cfg
        model = Qwen3_5ForCausalLM.from_pretrained(model_name, **kwargs)
        n_ok = sum(1 for _ in model.parameters() if _.numel() > 0)
        if n_ok > 0 and hasattr(model, "model") and hasattr(model.model, "layers"):
            logger.warning("Loaded Qwen3.5 via Qwen3_5ForCausalLM (text-only)")
            return model
        logger.warning(
            "Qwen3_5ForCausalLM loaded but decoder layers look empty; "
            "falling back to ConditionalGeneration"
        )
    except Exception as exc:
        logger.warning(f"Qwen3_5ForCausalLM.from_pretrained failed ({exc}); trying VL class")

    # 2) Hub-native VL class (vision tower stays in memory; text forward still works).
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        logger.warning(
            f"Loaded Qwen3.5 via {type(model).__name__} "
            "(vision tower present; text-only forwards)"
        )
        return model
    except Exception as exc:
        logger.warning(f"AutoModelForImageTextToText failed ({exc})")

    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


def clamp_layers(layers: list[int], model) -> list[int]:
    n = num_hidden_layers(model)
    kept = [l for l in layers if 0 <= l < n]
    dropped = [l for l in layers if l not in kept]
    if dropped:
        logger.warning(
            f"Dropping out-of-range layers {dropped} "
            f"(num_hidden_layers={n}, valid 0..{n - 1})"
        )
    if not kept:
        raise ValueError(
            f"No requested layers in range for this model "
            f"(requested {layers}, num_hidden_layers={n}, valid 0..{n - 1})"
        )
    return kept
