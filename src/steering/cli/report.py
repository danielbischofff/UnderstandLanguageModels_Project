"""Rebuild saved experiment reports with the current offline UI; no inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..report import write_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories containing experiment-results.json")
    args = parser.parse_args()
    for run in args.runs:
        source = run / "experiment-results.json"
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(f"cannot read {source}: {error}")
        target = run / "report.html"
        write_experiment_report(target, document)
        print(target)


if __name__ == "__main__":
    main()
