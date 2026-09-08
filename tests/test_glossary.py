import copy
import json
import sys

import pytest

from steering.glossary import dictionary, gloss, report_glossary
from steering.report import render_experiment_report
from steering.cli.report import main


def test_dictionary_covers_both_scripts_and_keeps_alternate_senses():
    words = dictionary()
    assert "coach" in words["教练"]
    assert words["教练"] == words["教練"]
    assert "CL:" not in words["教练"]
    assert "CL:" not in words["教练"]
    assert "suppress" in words["打压"]
    assert gloss(" 教练", words)["kind"] == "dictionary"


def test_fallbacks_are_marked_not_invented_translations():
    words = {"技能": "skill", "培训": "training"}
    assert gloss("技能培训", words) == {"kind": "parts", "text": "approx. parts — 技能: skill | 培训: training"}
    assert "unknown" in gloss("技能龘", words)["text"]
    assert gloss("龘", words)["kind"] == "untranslated"
    assert gloss("お互", words)["kind"] == "untranslated"
    assert gloss("技\ufffd", words)["kind"] == "untranslated"
    assert gloss("hello", words) is None
    assert gloss("", words) is None


def example():
    return {"experiment": {"name": "test", "description": "__GLOSSARY_DATA__ </script>"}, "trials": [{
        "j_lens": {"tokens": [{"text": "教练"}], "layers": [{"positions": [[
            {"text": "教练", "logit": 2, "probability": 0.2},
            {"text": "unrelated", "logit": 1, "probability": 0.1},
        ]]}]},
    }]}


def test_report_embeds_only_needed_glosses_without_changing_data():
    document = example()
    before = copy.deepcopy(document)
    assert set(report_glossary(document)) == {"教练"}
    html = render_experiment_report(document)
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(payload) == before
    assert document == before
    assert "<\\/script>" in html
    assert "coach" in html
    assert "CC BY-SA 4.0" in html
    assert 'cell.title =' not in html


def test_glosses_cannot_inject_html_or_template_markers(monkeypatch):
    monkeypatch.setattr("steering.report.report_glossary", lambda _: {
        "字": {"kind": "dictionary", "text": "</script><script>alert(1)</script> __EXPERIMENT_DATA__"},
    })
    html = render_experiment_report(example())
    payload = html.split("const GLOSSARY = ", 1)[1].split(";\n", 1)[0]
    assert "<\\/script>" in payload
    assert json.loads(payload)["字"]["text"].endswith("__EXPERIMENT_DATA__")


def test_rebuild_cli_only_changes_html(tmp_path, monkeypatch):
    source = tmp_path / "experiment-results.json"
    source.write_text(json.dumps(example()), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"complete"}', encoding="utf-8")
    original = {path: path.read_bytes() for path in (source, manifest)}
    (tmp_path / "report.html").write_text("old UI", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["steering-report", str(tmp_path)])
    main()
    assert "readout-table" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert {path: path.read_bytes() for path in original} == original


def test_rebuild_cli_rejects_missing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["steering-report", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "report.html").exists()
