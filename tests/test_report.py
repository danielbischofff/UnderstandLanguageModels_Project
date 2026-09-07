import copy
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from steering.report import render_experiment_report


def document():
    return {
        "schema_version": 1,
        "experiment": {
            "name": "demo",
            "description": "safe </script><script>alert(1)</script>",
        },
        "trials": [
            {
                "question": {"id": "q", "text": "hello"},
                "character": "demon",
                "response": {"text": "answer"},
                "j_lens": {
                    "prompt_length": 1,
                    "tokens": [
                        {"position": 0, "id": 1, "text": "hello", "segment": "prompt"},
                        {"position": 1, "id": 2, "text": " answer", "segment": "response"},
                    ],
                    "layers": [
                        {
                            "layer": 0,
                            "positions": [
                                [{"id": 3, "text": "danger", "logit": 2, "probability": 0.2}],
                                [{"id": 4, "text": "cruel", "logit": 3, "probability": 0.3}],
                            ],
                        }
                    ],
                },
            }
        ],
    }


def test_report_is_self_contained_and_escapes_script_endings():
    html = render_experiment_report(document())
    assert html.startswith("<!doctype html>")
    assert 'id="character-select"' in html
    assert 'id="question-select"' in html
    assert 'id="token-search"' in html
    assert "const DATA =" in html
    assert "<\\/script><script>" in html
    assert "fetch(" not in html
    assert "const characters = DATA.experiment.characters" in html
    assert 'rawResponse.textContent = "No generated response is available for this cell."' in html


def test_browser_sweep_selection_hover_search_and_missing_cell(tmp_path):
    """Optional real-browser check: requires Node, playwright, and Chrome.

    Set NODE_PATH if playwright is supplied outside a local node_modules folder.
    BROWSER_EXECUTABLE optionally selects another Chromium executable.
    """
    node = shutil.which("node")
    if node is None or subprocess.run(
        [node, "-e", "require.resolve('playwright')"], capture_output=True
    ).returncode:
        pytest.skip("browser check requires Node and playwright")
    value = document()
    value["experiment"].update({
        "characters": ["unsteered", "demon"],
        "questions": [{"id": "q", "text": "hello"}, {"id": "other", "text": "another"}],
        "steering": {"coefficients": [0, 2], "schemes": [
            {"name": "middle", "method": "additive", "layers": [0]},
            {"name": "floor", "method": "projection_cap", "bound": "lower", "layers": [0]},
        ]},
    })
    base = value["trials"][0]
    value["trials"] = []
    for question in value["experiment"]["questions"]:
        control = copy.deepcopy(base)
        control.update(character="unsteered", question=question, steering=None)
        control["response"]["text"] = f"control-{question['id']}"
        value["trials"].append(control)
        for alpha in [0, 2]:
            for scheme in value["experiment"]["steering"]["schemes"]:
                if question["id"] == "q" and alpha == 2 and scheme["name"] == "floor":
                    continue  # Failed trial: selector must not fall back to another cell.
                trial = copy.deepcopy(base)
                trial.update(question=question, steering={**scheme, "coefficient": alpha})
                trial["response"]["text"] = f"{question['id']}-{scheme['name']}-{alpha}"
                value["trials"].append(trial)
    report = tmp_path / "sweep.html"
    report.write_text(render_experiment_report(value), encoding="utf-8")
    legacy = document()
    legacy["experiment"].update(characters=["demon"], questions=[base["question"]], steering={"coefficient": 3, "layers": [0]})
    legacy_report = tmp_path / "legacy.html"
    legacy_report.write_text(render_experiment_report(legacy), encoding="utf-8")
    subprocess.run([node, str(Path(__file__).with_name("report_browser.cjs")), str(report), str(legacy_report)],
                   check=True, timeout=60, env=os.environ.copy())
