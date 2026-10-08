"""Merge judge verdicts onto their trials and summarise the graded run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..prompts import render_review
from ..rubric import RUBRIC, RUBRIC_LABELS, RUBRIC_TEXTS
from ..runs import iter_trials, load_run


def _load_outcomes(paths: list[Path], parser: argparse.ArgumentParser) -> dict[str, int]:
    """Accept several batch files; a hash graded twice must be graded the same."""
    outcomes: dict[str, int] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        # Tolerate a judge that wrapped its dict, e.g. {"outcomes": {...}}.
        if isinstance(document, dict) and isinstance(document.get("outcomes"), dict):
            document = document["outcomes"]
        if not isinstance(document, dict):
            parser.error(f"{path}: expected a JSON object of trial hash -> outcome")
        for trial_hash, outcome in document.items():
            if outcome is None:
                continue
            if outcome not in RUBRIC:
                parser.error(f"{path}: outcome for {trial_hash} is {outcome!r}, expected one of {sorted(RUBRIC)}")
            if outcomes.get(trial_hash, outcome) != outcome:
                parser.error(f"{trial_hash}: conflicting outcomes {outcomes[trial_hash]} and {outcome} across files")
            outcomes[trial_hash] = outcome
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory the verdicts belong to")
    parser.add_argument("outcomes", nargs="+", type=Path, help="judge output files: JSON objects of trial hash -> 0|1|2")
    parser.add_argument("--out", type=Path, help="graded JSON destination (default: <run>/judging/outcomes.json)")
    parser.add_argument("--review-md", type=Path, help="also write a markdown file pairing each verdict with its response, for a verification pass")
    parser.add_argument("--question", help="only grade trials for this question id")
    parser.add_argument("--rubric", choices=sorted(RUBRIC_TEXTS), default="tylenol", help="rubric used for export: sets the label names and the --review-md text")
    parser.add_argument("--allow-missing", action="store_true", help="write the file even when some trials are ungraded")
    args = parser.parse_args()

    run = load_run(args.run)
    trials = list(iter_trials(run, question=args.question))
    outcomes = _load_outcomes(args.outcomes, parser)

    known = {trial.hash for trial in trials}
    unknown = sorted(set(outcomes) - known)
    if unknown:
        parser.error(f"{len(unknown)} graded hash(es) are not trials of this run: {', '.join(unknown[:5])}")
    missing = [trial.hash for trial in trials if trial.hash not in outcomes]
    if missing and not args.allow_missing:
        parser.error(f"{len(missing)} of {len(trials)} trials ungraded (first: {missing[0]}); re-run the judge or pass --allow-missing")

    graded = [trial for trial in trials if trial.hash in outcomes]
    labels = RUBRIC_LABELS[args.rubric]
    by_character: dict[str, Counter] = defaultdict(Counter)
    by_steering: dict[str, Counter] = defaultdict(Counter)
    for trial in graded:
        label = labels[outcomes[trial.hash]]
        by_character[trial.character][label] += 1
        by_steering[trial.condition.split(", ", 1)[-1] if trial.steering else "unsteered"][label] += 1

    document = {
        "run": args.run.name,
        "config_hash": run.get("config_hash"),
        "question": {trial.question_id: trial.question_text for trial in graded},
        "rubric": {str(code): label for code, label in labels.items()},
        # The plain hash -> outcome dict, first, so it is easy to read and to reuse.
        "outcomes": {trial.hash: outcomes[trial.hash] for trial in graded},
        "summary": {
            "graded": len(graded),
            "ungraded": missing,
            "by_outcome": {labels[code]: sum(1 for t in graded if outcomes[t.hash] == code) for code in labels},
            "by_character": {name: dict(counts) for name, counts in sorted(by_character.items())},
            "by_steering": {name: dict(counts) for name, counts in sorted(by_steering.items())},
        },
        "trials": [
            {
                "trial": trial.hash,
                "outcome": outcomes[trial.hash],
                "outcome_label": labels[outcomes[trial.hash]],
                "character": trial.character,
                "steering": dict(trial.steering) or None,
                "question_id": trial.question_id,
                "response": trial.response,
            }
            for trial in graded
        ],
    }

    out = args.out or args.run / "judging" / "outcomes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)

    if args.review_md:
        question_text = next(iter(document["question"].values()), "")
        args.review_md.parent.mkdir(parents=True, exist_ok=True)
        args.review_md.write_text(
            render_review(graded, outcomes, args.run.name, question_text, RUBRIC_TEXTS[args.rubric], labels), encoding="utf-8"
        )
        print(args.review_md)

    counts = document["summary"]["by_outcome"]
    print(f"{len(graded)} graded: " + ", ".join(f"{label}={counts[label]}" for label in counts))
    if missing:
        print(f"ungraded: {len(missing)}")


if __name__ == "__main__":
    main()
