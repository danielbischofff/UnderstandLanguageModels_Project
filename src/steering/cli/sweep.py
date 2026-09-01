from __future__ import annotations

import argparse

from ..experiments import ExperimentRecorder, run_experiment
from .common import load_yaml, runner_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume a Cartesian steering experiment"
    )
    parser.add_argument("--config", help="Experiment YAML (required for a new run)")
    parser.add_argument("--resume", help="Existing run directory")
    parser.add_argument("--model", help="Override model.id for a new run")
    parser.add_argument("--output-root", default="outputs/steering")
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("provide exactly one of --config or --resume")
    if args.resume and args.model:
        parser.error("--model cannot override the stored configuration when resuming")
    if args.config:
        config = load_yaml(args.config)
        if args.model:
            config.setdefault("model", {})["id"] = args.model
        runner = runner_from_config(config)
        recorder = ExperimentRecorder.create(config, root=args.output_root)
    else:
        recorder = ExperimentRecorder.resume(args.resume)
        runner = runner_from_config(recorder.config)
    run_experiment(runner, recorder)
    print(recorder.run_dir)


if __name__ == "__main__":
    main()
