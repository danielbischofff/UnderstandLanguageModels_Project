from __future__ import annotations

import argparse

from ..experiments import ExperimentRecorder, run_experiment
from ..recipes import load_recipe
from .common import runner_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one recorded steering generation")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--model-revision")
    parser.add_argument("--role", required=True)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument(
        "--scaling",
        default="raw",
        choices=["raw", "unit", "average-residual-norm", "projection-std"],
    )
    parser.add_argument(
        "--positions", default="all", choices=["all", "last", "generated_only"]
    )
    parser.add_argument("--calibration")
    parser.add_argument("--cap", choices=["lower", "upper", "interval"])
    parser.add_argument("--cap-recipe", help="YAML recipe for the capping direction")
    parser.add_argument("--cap-layer", type=int, action="append")
    parser.add_argument("--cap-threshold", type=float, nargs="+")
    parser.add_argument("--cap-quantile", type=float, nargs="+")
    parser.add_argument("--cap-calibration")
    parser.add_argument(
        "--capture-layer",
        type=int,
        action="append",
        help="post-MLP layer to export for downstream analysis (repeatable)",
    )
    parser.add_argument(
        "--capture-point",
        action="append",
        choices=["pre_intervention", "post_intervention"],
        help=(
            "layer output before the first or after the last ordered intervention "
            "(repeatable; defaults to both)"
        ),
    )
    parser.add_argument(
        "--capture-positions",
        default="all",
        choices=["all", "last", "generated_only"],
        help=(
            "token positions per forward call: full sequence, final position, "
            "or cached decode calls only; this does not select layers"
        ),
    )
    parser.add_argument(
        "--capture-dtype",
        default="model",
        choices=["model", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--system-prompt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantization", choices=["4bit", "8bit"])
    parser.add_argument("--study", default="one-off")
    parser.add_argument("--experiment")
    parser.add_argument("--output-root", default="outputs/steering")
    args = parser.parse_args()
    if args.capture_point and not args.capture_layer:
        parser.error("--capture-point requires at least one --capture-layer")
    if args.cap:
        if not args.cap_recipe or not args.cap_layer:
            parser.error("--cap requires --cap-recipe and at least one --cap-layer")
        if (args.cap_threshold is None) == (args.cap_quantile is None):
            parser.error(
                "--cap requires exactly one of --cap-threshold or --cap-quantile"
            )
        if args.cap == "interval" and (
            (args.cap_threshold is None or len(args.cap_threshold) != 2)
            and (args.cap_quantile is None or len(args.cap_quantile) != 2)
        ):
            parser.error("interval capping requires two threshold or quantile values")
        if (
            args.cap != "interval"
            and args.cap_quantile is not None
            and len(args.cap_quantile) != 1
        ):
            parser.error("lower/upper capping accepts one quantile")
    conversation = []
    if args.system_prompt:
        conversation.append({"role": "system", "content": args.system_prompt})
    conversation.append({"role": "user", "content": args.prompt})
    generation = {"max_new_tokens": args.max_new_tokens}
    if args.temperature is not None:
        generation.update({"do_sample": True, "temperature": args.temperature})
    config = {
        "study": args.study,
        "experiment": args.experiment or f"role-{args.role}",
        "model": {
            "id": args.model,
            "revision": args.model_revision,
            "dtype": "bfloat16",
            "quantization": args.quantization,
            "enable_thinking": False,
        },
        "conversation": conversation,
        "seed": args.seed,
        "generation": generation,
        "collect_diagnostics": True,
        "steering": {
            "vector": {"type": "role_direction", "name": args.role},
            "coefficient": args.coefficient,
            "layers": args.layer,
            "scaling": args.scaling,
            "positions": args.positions,
            "calibration": args.calibration,
        },
    }
    if args.capture_layer:
        config["activation_capture"] = {
            "layers": args.capture_layer,
            "points": args.capture_point or ["pre_intervention", "post_intervention"],
            "positions": args.capture_positions,
            "dtype": args.capture_dtype,
        }
    if args.cap:
        threshold = args.cap_threshold
        if threshold is not None and len(threshold) == 1:
            threshold = threshold[0]
        quantile = args.cap_quantile
        if quantile is not None and len(quantile) == 1:
            quantile = quantile[0]
        config["steering"]["caps"] = [
            {
                "direction": load_recipe(args.cap_recipe).to_dict(),
                "bound": args.cap,
                "layers": args.cap_layer,
                "threshold": threshold,
                "quantile": quantile,
                "calibration": args.cap_calibration,
                "positions": args.positions,
            }
        ]
    recorder = ExperimentRecorder.create(config, root=args.output_root)
    runner = runner_from_config(config)
    run_experiment(runner, recorder)
    print(recorder.run_dir)


if __name__ == "__main__":
    main()
