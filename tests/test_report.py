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
