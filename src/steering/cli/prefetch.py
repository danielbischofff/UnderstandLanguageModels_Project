"""Download model and persona-vector artifacts before GPU execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..experiments import ExperimentConfig
from ..prefetch import prepare_experiment_artifacts
from .common import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the local cache without accessing the network",
    )
    args = parser.parse_args()
    config = ExperimentConfig.from_mapping(load_yaml(args.config))
    snapshots = prepare_experiment_artifacts(
        config, local_files_only=args.check
    )
    action = "available locally" if args.check else "downloaded"
    print(f"model snapshots {action}:")
    for snapshot in snapshots:
        print(f"  {snapshot}")
    print("persona vectors available locally")


if __name__ == "__main__":
    main()
