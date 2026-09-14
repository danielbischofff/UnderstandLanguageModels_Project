"""Render a self-contained browser for one experiment run."""

from __future__ import annotations

import json
import base64
import re
from html import escape
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .glossary import report_glossary
from .binary_results import encode_trial, load_bundle_report


def pack_report_data(document: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly compress trials independently so the browser loads only one."""
    trials = []
    vocabulary = {}
    for trial in document["trials"]:
        descriptor, payload = encode_trial(trial, vocabulary)
        trials.append({
            **{key: trial.get(key) for key in ("question", "character", "steering")},
            "binary": descriptor,
            "payload_gzip": base64.b64encode(payload).decode("ascii"),
        })
    return {**document, "trials": trials, "vocabulary": vocabulary}


def render_experiment_report(document: Mapping[str, Any]) -> str:
    return render_packed_report(pack_report_data(document), report_glossary(document))


def render_packed_report(document: Mapping[str, Any], glossary: Mapping[str, Any]) -> str:
    template = (
        files("steering").joinpath("web/experiment.html").read_text(encoding="utf-8")
    )
    values = {"__EXPERIMENT_DATA__": document, "__GLOSSARY_DATA__": glossary}
    # Substitute in one pass so literal template markers inside user data stay
    # literal. Escape script endings in both the results and dictionary glosses.
    return re.sub(
        r"__EXPERIMENT_DATA__|__GLOSSARY_DATA__",
        lambda match: json.dumps(values[match[0]], ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
        template,
    )


def write_experiment_report(path: str | Path, document: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(render_experiment_report(document), encoding="utf-8")
    temporary.replace(target)


def write_bundle_reports(bundle: Path, destination: Path) -> Path:
    """Bound rendering memory to one question and provide a small run index."""
    with zipfile.ZipFile(bundle) as archive:
        index = json.loads(archive.read("index.json"))
    questions = index["document"]["experiment"]["questions"]
    if not questions:
        raise ValueError("experiment has no questions")
    destination.mkdir(parents=True, exist_ok=True)
    split = len(questions) > 1
    links = []
    for number, question in enumerate(questions, 1):
        filename = f"question-{number:03d}.html" if split else "report.html"
        document, _, glossary = load_bundle_report(bundle, question["id"])
        page = render_packed_report(document, glossary)
        if split:
            navigation = ['<a href="index.html">All questions</a>']
            if number > 1:
                navigation.append(f'<a href="question-{number - 1:03d}.html">Previous question</a>')
            if number < len(questions):
                navigation.append(f'<a href="question-{number + 1:03d}.html">Next question</a>')
            page = page.replace("<main>", '<main><nav>' + ' · '.join(navigation) + '</nav>', 1)
        (destination / filename).write_text(page, encoding="utf-8")
        links.append(f'<li><a href="{filename}">{escape(question["id"])}: {escape(question["text"])}</a></li>')
    if split:
        title = escape(index["document"]["experiment"]["name"])
        page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{title}</title><style>body{{font:16px system-ui;max-width:900px;margin:40px auto;padding:16px}}'
                f'li{{margin:16px 0}}</style><h1>{title}</h1><p>Choose a question. Each report works offline.</p>'
                '<ol>' + ''.join(links) + '</ol></html>')
        for filename in ("index.html", "report.html"):
            (destination / filename).write_text(page, encoding="utf-8")
    return destination / ("index.html" if split else "report.html")
