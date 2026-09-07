"""Run or resume a question/character/alpha/steering-scheme J-Lens experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..experiments import ExperimentConfig, ExperimentRecorder, run_experiment
from ..j_lens import JLenseReader
from .common import load_yaml, runner_from_config


def main() -> None:
    repository = Path(__file__).resolve().parents[3]
    runtime = Path(os.environ.get("ULM_RUNTIME_ROOT", repository / "runtime"))
    default_data = Path(
        os.environ.get("JLENS_DATA_ROOT", runtime / "models" / "j-lens" / "qwen3-32b")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="experiment YAML for a new run")
    parser.add_argument("--resume", type=Path, help="existing run directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "STEERING_OUTPUT_ROOT", runtime / "outputs" / "experiments"
            )
        ),
    )
    parser.add_argument("--analysis-root", type=Path, default=repository / "direct_character_vectors_j_lens_analysis")
    parser.add_argument("--j-lens-data", type=Path, default=default_data)
    parser.add_argument("--lens", type=Path)
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("provide exactly one of --config or --resume")

    if args.config:
        config = ExperimentConfig.from_mapping(load_yaml(args.config))
        recorder = ExperimentRecorder.create(config, root=args.output_root)
    else:
        recorder = ExperimentRecorder.resume(args.resume)
        config = recorder.config

    lens_path = args.lens or (
        args.j_lens_data
        / "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
    )
    try:
        lens = JLenseReader(
            analysis_source=args.analysis_root / "src",
            data_dir=args.j_lens_data,
            lens_path=lens_path,
            top_k_count=config.top_k,
        )
        runner = runner_from_config(config.to_mapping())
        run_experiment(runner, recorder, lens)
    except FileNotFoundError as error:
        parser.error(
            f"missing J-Lens input {error.filename or error}; run setup.sh first"
        )
    print(recorder.run_dir)


if __name__ == "__main__":
    main()
