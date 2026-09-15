"""Quantitative contrastive-set analysis and plots for finished runs, on CPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .contrastive import COSINE_MEASURES, RANK_MEASURES, merge_set_library, resolve_set_entries
from .plots import PlotOptions, write_figures, write_index
from .quant import load_run, run_series, table_rows, to_document, write_table


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    runtime = Path(os.environ.get("ULM_RUNTIME_ROOT", repository / "runtime"))
    default_data = Path(
        os.environ.get("JLENS_DATA_ROOT", runtime / "models" / "j-lens" / "qwen3-32b")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories or results.zip files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("QUANTITIVE_ANALYSIS_ROOT", "outputs/plots/quantitive-analysis")),
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        metavar="FILE", required=True,
        help="YAML files containing arbitrary word sets, independent of experiment config",
    )
    parser.add_argument(
        "--layer",
        type=int,
        action="append",
        dest="layers",
        help="J-Lens layer to score (repeatable; default: all saved readout layers)",
    )
    parser.add_argument("--question", action="append", dest="questions")
    parser.add_argument("--top-k", type=int, help="top words used by the cosine measure when recomputing")
    parser.add_argument("--cosine-measure", choices=COSINE_MEASURES, default=PlotOptions().cosine_measure)
    parser.add_argument("--rank-measure", choices=RANK_MEASURES, default=PlotOptions().rank_measure)
    parser.add_argument(
        "--change-z",
        type=float,
        default=PlotOptions().change_z,
        help="label a step this many robust standard deviations from typical (default: 3)",
    )
    parser.add_argument("--min-change-fraction", type=float, default=PlotOptions().min_change_fraction)
    parser.add_argument("--max-annotations", type=int, default=PlotOptions().max_annotations)
    parser.add_argument("--analysis-root", type=Path, default=repository / "direct_character_vectors_j_lens_analysis")
    parser.add_argument("--j-lens-data", type=Path, default=default_data)
    parser.add_argument("--lens", type=Path)
    args = parser.parse_args()

    options = PlotOptions(
        cosine_measure=args.cosine_measure,
        rank_measure=args.rank_measure,
        change_z=args.change_z,
        min_change_fraction=args.min_change_fraction,
        max_annotations=args.max_annotations,
    )
    definitions = None
    if args.sets:
        try:
            definitions = list(
                merge_set_library(
                    resolve_set_entries(args.sets, search_dirs=[Path(".")])
                ).values()
            )
        except (FileNotFoundError, ValueError, TypeError) as error:
            parser.error(str(error))

    reader = None
    for run in args.runs:
        try:
            archive = load_run(run)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        layers = args.layers or sorted({
            int(layer["layer"]) for trial in archive.trials
            for layer in trial.metadata["j_lens"]["layers"]
        })
        if reader is None:
            reader = _open_reader(parser, args, top_k=args.top_k if args.top_k is not None else 10)
        probe = reader.probe(definitions, layers=layers, top_k=args.top_k if args.top_k is not None else 10)
        series = run_series(
            archive, layers=layers, probe=probe, reader=reader, questions=args.questions
        )
        if not series:
            parser.error(
                f"{run} has no matching trials for layers {layers}"
            )
        destination = args.output_dir / str(archive.experiment.get("name", "experiment")) / archive.run
        destination.mkdir(parents=True, exist_ok=True)
        document = to_document(archive, series)
        document["probe"] = probe.describe()
        (destination / "metrics.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_table(destination / "metrics.csv", table_rows(archive, series))
        write_figures(archive, series, destination, options=options, table="metrics.csv")
        print(
            write_index(
                archive, series, destination, table="metrics.csv", document="metrics.json"
            )
        )


def _open_reader(parser: argparse.ArgumentParser, args: argparse.Namespace, *, top_k: int):
    from .reader import AnalysisReader

    lens_path = args.lens or (
        args.j_lens_data / "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
    )
    try:
        return AnalysisReader(
            analysis_source=args.analysis_root / "src",
            data_dir=args.j_lens_data,
            lens_path=lens_path,
            top_k_count=max(1, top_k),
        )
    except FileNotFoundError as error:
        parser.error(
            f"missing J-Lens input {error.filename or error}; scoring new sets needs the "
            "fetched lens and unembedding (run setup.sh)"
        )


if __name__ == "__main__":
    main()
