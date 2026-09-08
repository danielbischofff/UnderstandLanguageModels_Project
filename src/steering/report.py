"""Render a self-contained browser for one experiment run."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .glossary import report_glossary


def render_experiment_report(document: Mapping[str, Any]) -> str:
    template = (
        files("steering").joinpath("web/experiment.html").read_text(encoding="utf-8")
    )
    values = {"__EXPERIMENT_DATA__": document, "__GLOSSARY_DATA__": report_glossary(document)}
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
