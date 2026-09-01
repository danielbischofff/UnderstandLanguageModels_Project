"""Render a self-contained browser for one experiment run."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping


def render_experiment_report(document: Mapping[str, Any]) -> str:
    template = (
        files("steering").joinpath("web/experiment.html").read_text(encoding="utf-8")
    )
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    # A prompt may contain literal HTML. Keep it inert inside the script block.
    payload = payload.replace("</", "<\\/")
    return template.replace("__EXPERIMENT_DATA__", payload)


def write_experiment_report(path: str | Path, document: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(render_experiment_report(document), encoding="utf-8")
    temporary.replace(target)
