"""Export trial responses as grading batches for an LLM-as-judge pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..prompts import render_batch
from ..rubric import RUBRIC
from ..runs import iter_trials, load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory containing manifest.json and trials/")
    parser.add_argument("--out-dir", type=Path, help="destination for batch files (default: <run>/judging)")
    parser.add_argument("--batch-size", type=int, default=20, help="trials per batch file (default: 20)")
    parser.add_argument("--question", help="only export trials for this question id")
    parser.add_argument("--reveal", action="store_true", help="show character and steering in the batches; blind by default so the judge grades the text alone")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    run = load_run(args.run)
    trials = list(iter_trials(run, question=args.question))
    if not trials:
        parser.error(f"no trials found in {args.run}")
    questions = {trial.question_text for trial in trials}
    if len(questions) > 1 and not args.question:
        print(f"note: {len(questions)} distinct questions in this run; each batch repeats its own prompt")

    out_dir = args.out_dir or args.run / "judging"
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [trials[start : start + args.batch_size] for start in range(0, len(trials), args.batch_size)]
    width = max(2, len(str(len(batches))))
    for number, batch in enumerate(batches, start=1):
        path = out_dir / f"batch-{number:0{width}d}.md"
        path.write_text(render_batch(batch, number, len(batches), len(trials), reveal=args.reveal), encoding="utf-8")
        print(path)

    # One machine-readable copy so a judge harness can build its own prompts.
    jsonl = out_dir / "trials.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(json.dumps(trial.as_record(reveal=args.reveal), ensure_ascii=False) + "\n")
    print(jsonl)

    template = out_dir / "outcomes.template.json"
    template.write_text(
        json.dumps({trial.hash: None for trial in trials}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(template)
    print(f"{len(trials)} trials in {len(batches)} batches; rubric: " + ", ".join(f"{k}={v}" for k, v in RUBRIC.items()))


if __name__ == "__main__":
    main()
