import copy
import base64
import gzip
import json
import zipfile
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from steering.report import pack_report_data, render_experiment_report
from steering.cli.report import write_bundle_reports
from steering.binary_results import decode_trial, write_result_bundle
import yaml


def test_report_import_does_not_load_inference_dependencies():
    subprocess.run([sys.executable, "-c", "import sys; import steering.cli.report; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], check=True, timeout=30)


def test_lazy_public_exports_preserve_api():
    import steering
    for name in steering.__all__:
        assert getattr(steering, name) is not None
    with pytest.raises(AttributeError):
        getattr(steering, "not_a_public_export")


def test_publication_archive_preserves_raw_bytes_without_duplicate_views(tmp_path):
    run = tmp_path / "run"
    trial = run / "trials" / "one"
    trial.mkdir(parents=True)
    value = document()
    value["experiment"]["questions"] = [value["trials"][0]["question"], {"id": "other", "text": "<other>"}]
    raw = json.dumps(value["trials"][0]).encode()
    (trial / "result.json").write_bytes(raw)
    (run / "resolved-config.yaml").write_text(yaml.safe_dump(value["experiment"]))
    (trial / "metadata.json").write_text('{"revision": "test"}')
    (run / "manifest.json").write_text('{}')
    (run / "report.html").write_text('duplicate')
    (run / "experiment-results.json").write_text('duplicate')
    (run / ".writer.lock").write_text('stale')
    target = tmp_path / "raw.zip"
    write_result_bundle(run, target)
    with zipfile.ZipFile(target) as archive:
        assert "provenance/trials/one/metadata.json" in archive.namelist()
        assert not any(name.endswith("result.json") or name.endswith(".html") for name in archive.namelist())
        index = json.loads(archive.read("index.json"))
        descriptor = index["trials"][0]
        assert decode_trial(descriptor, archive.read(descriptor["payload"]), json.loads(archive.read("vocabulary.json"))) == value["trials"][0]
    assert (trial / "result.json").read_bytes() == raw
    shutil.rmtree(run)  # Report rebuild must need only the portable ZIP.
    output = tmp_path / "rebuilt"
    assert write_bundle_reports(target, output) == output / "index.html"
    assert 'question-002.html' in (output / "index.html").read_text()
    assert '&lt;other&gt;' in (output / "index.html").read_text()
    first = (output / "question-001.html").read_text(encoding="utf-8")
    assert 'href="index.html"' in first
    assert 'Next question' in first


def test_report_compression_is_lossless_and_does_not_mutate_input():
    source = document()
    original = copy.deepcopy(source)
    packed = pack_report_data(source)
    restored = [decode_trial(t["binary"], base64.b64decode(t["payload_gzip"]), packed["vocabulary"]) for t in packed["trials"]]
    assert restored == source["trials"]
    assert source == original
    assert pack_report_data(source) == packed


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
            {"name": "band", "method": "character_band", "layers": [0]},
        ]},
    })
    base = value["trials"][0]
    # All fitted layers and a full top-10 exercise desktop layout and inspection.
    chinese = [
        {"id": 100 + index, "text": text, "logit": 10 - index, "probability": 0.1 / (index + 1)}
        for index, text in enumerate(["教练", "技能培训", "お互", "龘", "<script>", "word", "six", "seven", "eight", "nine"])
    ]
    base["j_lens"]["layers"].append({"layer": 1, "positions": [chinese, chinese]})
    for layer in range(2, 63):
        base["j_lens"]["layers"].append({"layer": layer, "positions": [chinese, chinese]})
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
    run = tmp_path / "split-source"
    run.mkdir()
    (run / "resolved-config.yaml").write_text(yaml.safe_dump(value["experiment"]), encoding="utf-8")
    for index, trial in enumerate(value["trials"]):
        folder = run / "trials" / str(index)
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(trial), encoding="utf-8")
    bundle = tmp_path / "results.zip"
    write_result_bundle(run, bundle)
    split_index = write_bundle_reports(bundle, tmp_path / "split")
    subprocess.run([node, str(Path(__file__).with_name("report_browser.cjs")), str(report), str(legacy_report), str(split_index)],
                   check=True, timeout=60, env=os.environ.copy())
