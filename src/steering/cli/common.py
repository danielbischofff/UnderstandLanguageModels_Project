from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ..inference import InferenceRunner
from ..vector_store import VectorStore

DTYPES = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"configuration {path} must contain a mapping")
    return value


def runner_from_config(config: Mapping[str, Any]) -> InferenceRunner:
    model = config.get("model", {})
    model_id = model.get("id", "Qwen/Qwen3-32B")
    store = VectorStore(
        model_id=model_id,
        revision=config.get("vectors", {}).get("revision", "main"),
        dataset_model_dir=config.get("vectors", {}).get("dataset_model_dir"),
        local_dir=config.get("vectors", {}).get("local_dir"),
        cache_dir=config.get("vectors", {}).get("cache_dir"),
        offline=bool(config.get("vectors", {}).get("offline", False)),
    )
    dtype_name = model.get("dtype", "bfloat16")
    if dtype_name not in DTYPES:
        raise ValueError(f"unknown dtype {dtype_name!r}")
    return InferenceRunner.from_pretrained(
        model_id,
        model_revision=model.get("revision"),
        tokenizer_revision=model.get("tokenizer_revision"),
        device_map=model.get("device_map", "auto"),
        dtype=DTYPES[dtype_name],
        quantization=model.get("quantization"),
        enable_thinking=bool(model.get("enable_thinking", False)),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        vector_store=store,
    )
