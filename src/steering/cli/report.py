"""Export compact results and rebuild offline reports, without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..binary_results import write_result_bundle
from ..report import write_bundle_reports, write_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories or downloaded results.zip files")
    parser.add_argument("--publication-dir", type=Path, help="export into new experiment/run directories here; preserve source runs")
    parser.add_argument("--output-dir", type=Path, help="report destination when rebuilding one downloaded ZIP")
    args = parser.parse_args()
    if args.output_dir and (len(args.runs) != 1 or args.runs[0].suffix != ".zip" or args.publication_dir):
        parser.error("--output-dir requires exactly one ZIP and no --publication-dir")
    for run in args.runs:
        if run.suffix == ".zip":
            if args.publication_dir:
                parser.error("--publication-dir is for source run directories")
            print(write_bundle_reports(run, args.output_dir or run.parent / "reports"))
            continue
        # Compatibility for older downloads containing only the combined JSON.
        # Normal exports use per-trial files and never load that giant document.
        if not (run / "resolved-config.yaml").is_file():
            source = run / "experiment-results.json"
            if args.publication_dir or not source.is_file():
                parser.error(f"export needs resolved-config.yaml and per-trial results: {run}")
            write_experiment_report(run / "report.html", json.loads(source.read_text(encoding="utf-8")))
            print(run / "report.html")
            continue
        config = yaml.safe_load((run / "resolved-config.yaml").read_text(encoding="utf-8"))
        name = config["name"]
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            parser.error("invalid experiment directory name")
        destination = args.publication_dir / name / run.name if args.publication_dir else run
        if args.publication_dir:
            if destination.resolve().is_relative_to(run.resolve()):
                parser.error("publication directory must be outside the source run")
            if destination.exists():
                parser.error(f"publication directory already exists: {destination}")
            destination.mkdir(parents=True)
        bundle = destination / "results.zip"
        if bundle.exists():
            parser.error(f"bundle already exists; rebuild reports directly from {bundle}")
        write_result_bundle(run, bundle)
        print(write_bundle_reports(bundle, destination))


if __name__ == "__main__":
    main()
