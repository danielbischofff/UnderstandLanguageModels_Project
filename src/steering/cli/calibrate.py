from __future__ import annotations

import argparse
import os

from datasets import load_dataset

from ..calibration import CalibrationRunner, iter_jsonl_conversations, save_calibration
from ..inference import DEFAULT_MODEL_ID
from ..recipes import load_recipe
from .common import runner_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate post-MLP residual statistics"
    )
    parser.add_argument(
        "--model", default=os.environ.get("STEERING_MODEL_ID", DEFAULT_MODEL_ID)
    )
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--dataset", required=True, help="Local JSONL path or Hugging Face dataset ID"
    )
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--split", default="train")
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument(
        "--layers", required=True, help="Comma-separated zero-based layers"
    )
    parser.add_argument(
        "--direction", action="append", default=[], help="Recipe YAML path; repeatable"
    )
    parser.add_argument("--quantiles", default="0.01,0.25,0.5,0.75,0.95,0.99")
    parser.add_argument("--reservoir-capacity", type=int, default=100_000)
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", choices=["4bit", "8bit"])
    parser.add_argument("--local-vectors")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = {
        "model": {
            "id": args.model,
            "revision": args.model_revision,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "enable_thinking": False,
        },
        "vectors": {"local_dir": args.local_vectors, "offline": args.offline},
    }
    inference = runner_from_config(config)
    calibration = CalibrationRunner(
        inference.model,
        inference.tokenizer,
        inference.vector_store,
        model_id=inference.model_id,
        model_revision=inference.model_revision,
        tokenizer_revision=inference.tokenizer_revision,
        enable_thinking=inference.enable_thinking,
    )
    dataset_meta = {
        "id": args.dataset,
        "config": args.dataset_config,
        "revision": args.dataset_revision,
        "split": args.split,
    }
    if args.dataset.endswith(".jsonl"):
        conversations = iter_jsonl_conversations(
            args.dataset, messages_column=args.messages_column
        )
        dataset_meta["kind"] = "local_jsonl"
    else:
        loaded = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            revision=args.dataset_revision,
            streaming=True,
        )
        conversations = (record[args.messages_column] for record in loaded)
        dataset_meta["kind"] = "huggingface"
    record = calibration.run(
        conversations,
        layers=[int(item) for item in args.layers.split(",")],
        directions=[load_recipe(path) for path in args.direction],
        dataset=dataset_meta,
        quantiles=[float(item) for item in args.quantiles.split(",")],
        reservoir_capacity=args.reservoir_capacity,
        seed=args.seed,
        max_conversations=args.max_conversations,
    )
    save_calibration(record, args.output)


if __name__ == "__main__":
    main()
