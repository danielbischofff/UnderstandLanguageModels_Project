"""Check a node before downloading and running the Qwen3-32B sandbox."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch

from ..inference import DEFAULT_MODEL_ID
from ..experiments import ExperimentConfig
from .common import load_yaml


def _gb(value: int) -> float:
    return value / 1_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="experiment YAML")
    source.add_argument("--model", help="model id when checking without an experiment")
    parser.add_argument("--minimum-vram-gb", type=float, default=75.0)
    parser.add_argument("--minimum-free-disk-gb", type=float, default=80.0)
    args = parser.parse_args()
    model_id = (
        str(ExperimentConfig.from_mapping(load_yaml(args.config)).model["id"])
        if args.config
        else args.model or os.environ.get("STEERING_MODEL_ID", DEFAULT_MODEL_ID)
    )

    repository = Path(
        os.environ.get("ULM_REPO_ROOT", Path(__file__).resolve().parents[3])
    ).resolve()
    runtime = Path(os.environ.get("ULM_RUNTIME_ROOT", repository / "runtime")).resolve()
    huggingface_home = Path(
        os.environ.get("HF_HOME", runtime / "models" / "huggingface")
    ).resolve()
    output_root = Path(
        os.environ.get("STEERING_OUTPUT_ROOT", runtime / "outputs" / "steering")
    ).resolve()
    for path in (runtime, huggingface_home, output_root):
        path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=runtime):
        pass

    if not torch.cuda.is_available():
        raise SystemExit(
            "preflight failed: PyTorch cannot access CUDA; check the allocation, "
            "driver, and locked PyTorch build"
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("preflight failed: the selected GPU does not support bfloat16")
    capability = torch.cuda.get_device_capability(0)
    if model_id.upper().endswith("-FP8") and capability < (8, 9):
        raise SystemExit(
            "preflight failed: the FP8 checkpoint requires an FP8-capable NVIDIA "
            f"GPU (compute capability >= 8.9); GPU 0 reports {capability[0]}.{capability[1]}. "
            "Switch to Qwen/Qwen3-32B for bf16."
        )
    properties = torch.cuda.get_device_properties(0)
    vram_gb = _gb(properties.total_memory)
    if vram_gb < args.minimum_vram_gb:
        raise SystemExit(
            f"preflight failed: GPU 0 has {vram_gb:.1f} GB VRAM; "
            f"at least {args.minimum_vram_gb:.1f} GB is required"
        )
    free_disk_gb = _gb(shutil.disk_usage(runtime).free)
    if free_disk_gb < args.minimum_free_disk_gb:
        raise SystemExit(
            f"preflight failed: {runtime} has {free_disk_gb:.1f} GB free; "
            f"at least {args.minimum_free_disk_gb:.1f} GB is required"
        )

    print(f"repository: {repository}")
    print(f"runtime: {runtime}")
    print(f"HF_HOME: {huggingface_home}")
    print(f"outputs: {output_root}")
    print(f"model: {model_id}")
    print(f"torch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"transformers: {version('transformers')}")
    print(f"accelerate: {version('accelerate')}")
    print(
        f"GPU 0: {properties.name} ({vram_gb:.1f} GB, bf16 supported, "
        f"compute capability {capability[0]}.{capability[1]})"
    )
    print(f"free runtime disk: {free_disk_gb:.1f} GB")
    print("preflight passed")


if __name__ == "__main__":
    main()
