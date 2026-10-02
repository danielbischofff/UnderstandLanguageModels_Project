"""Validate paired question banks and export an experiment draft and analysis sets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml

KINDS = ("contextualized", "non_contextualized")


def read_bank(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"topics"}:
        raise ValueError(f"{path}: expected a topics mapping")
    questions, topics, personas, general = {}, [], None, {}
    if not isinstance(data["topics"], list) or not data["topics"]:
        raise ValueError(f"{path}: topics must be a nonempty list")
    for topic in data["topics"]:
        name = topic["topic"]
        if not isinstance(name, str) or not name.strip() or name in topics:
            raise ValueError(f"{path}: empty or duplicate topic")
        topics.append(name)
        if len(topic["questions"]) != 2:
            raise ValueError(f"{path}: {name} must contain two questions")
        for question in topic["questions"]:
            qid = question["id"]
            if not isinstance(qid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", qid) or qid in questions:
                raise ValueError(f"{path}: invalid or duplicate question ID {qid!r}")
            for field in ("title", "question"):
                if not isinstance(question[field], str) or not question[field].strip():
                    raise ValueError(f"{path}: {qid}: empty {field}")
            roles = question["personas"]
            if not isinstance(roles, dict) or not 1 <= len(roles) <= 8:
                raise ValueError(f"{path}: {qid}: expected 1–8 personas")
            if any(not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", role) or role == "unsteered" for role in roles):
                raise ValueError(f"{path}: invalid persona identifier")
            if personas is None:
                personas = list(roles)
            if set(roles) != set(personas):
                raise ValueError(f"{path}: {qid}: inconsistent personas")
            for role, sets in roles.items():
                if set(sets) != set(KINDS):
                    raise ValueError(f"{path}: {qid}/{role}: expected both word-set types")
                for kind, words in sets.items():
                    if not isinstance(words, list) or len(words) != 6 or any(not isinstance(w, str) or not w.strip() for w in words):
                        raise ValueError(f"{path}: {qid}/{role}/{kind}: expected six nonempty words")
                    if len({w.strip().casefold() for w in words}) != 6:
                        raise ValueError(f"{path}: {qid}/{role}/{kind}: duplicate words")
                words = sets["non_contextualized"]
                if role in general and general[role] != words:
                    raise ValueError(f"{path}: {role}: general words vary across questions")
                general[role] = words
            questions[qid] = {"topic": name, **question}
    return {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "topics": topics, "personas": personas, "questions": questions}


def prepare(paths: list[Path], destination: Path) -> dict:
    if len(paths) != 2 or paths[0].stem == paths[1].stem:
        raise ValueError("provide two banks with distinct filenames")
    banks = [read_bank(path) for path in paths]
    first, second = banks
    if first["topics"] != second["topics"] or set(first["questions"]) != set(second["questions"]):
        raise ValueError("banks must share topics and question IDs")
    for qid, question in first["questions"].items():
        if any(question[key] != second["questions"][qid][key] for key in ("topic", "title", "question")):
            raise ValueError(f"banks disagree on question {qid}")
    if set(first["personas"]) & set(second["personas"]):
        raise ValueError("banks must have disjoint personas")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}; choose a fresh directory")
    roles = first["personas"] + second["personas"]
    draft = {
        "name": "question-bank-suite",
        "description": "DRAFT: select steering settings after pilot planning. Predicted words are analysis-only.",
        "model": {"id": "Qwen/Qwen3-32B", "dtype": "bfloat16", "enable_thinking": False},
        "vectors": {"revision": "main"},
        "characters": ["unsteered", *roles],
        "questions": [{"id": qid, "text": q["question"]} for qid, q in first["questions"].items()],
        "steering": {"coefficients": [], "schemes": []},
        "generation": {"max_new_tokens": 256, "do_sample": False}, "seed": 0,
        "j_lens": {"top_k": 10}, "outputs": {"keep_activations": True},
    }
    files = {"question-bank-suite.yaml": draft}
    analyses = []
    for path, bank in zip(paths, banks):
        for qid, q in bank["questions"].items():
            for kind in KINDS:
                relative = f"sets/{path.stem}/{qid}/{kind}.yaml"
                files[relative] = [{"id": f"{path.stem}-{qid}-{role}-{kind}",
                                    "label": role, "words": q["personas"][role][kind]}
                                   for role in bank["personas"]]
                analyses.append({"bank": path.stem, "question": qid, "topic": q["topic"],
                                 "word_set_type": kind, "sets": relative,
                                 "output_subdir": f"{path.stem}/{qid}/{kind}"})
    manifest = {
        "schema_version": 1, "status": "draft-steering-unset",
        "banks": [{k: v for k, v in bank.items() if k != "questions"} for bank in banks],
        "counts": {"topics": len(first["topics"]), "questions": len(first["questions"]),
                   "personas": len(roles), "analysis_jobs": len(analyses),
                   "trials_per_single_steering_setting": len(first["questions"]) * (len(roles) + 1)},
        "questions": [{"id": qid, "topic": q["topic"], "title": q["title"]}
                      for qid, q in first["questions"].items()],
        "analyses": analyses,
    }
    destination.mkdir(parents=True)
    for relative, value in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("banks", nargs=2, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = prepare(args.banks, args.output_dir)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
