"""Validate the bank-to-experiment and bank-to-analysis handoff without inference."""
from pathlib import Path

import pytest
import yaml

from steering.cli.prepare_question_banks import prepare
from steering.experiments import ExperimentConfig
from quantitive_analysis.contrastive import resolve_set_entries

ROOT = Path(__file__).resolve().parents[1]
BANKS = [ROOT / "configs" / name for name in ("question_bank.yaml", "question_bank_contrastive.yaml")]


def test_actual_banks_round_trip(tmp_path):
    destination = tmp_path / "prepared"
    manifest = prepare(BANKS, destination)
    assert manifest["counts"] == {"topics": 12, "questions": 24, "personas": 16,
                                  "analysis_jobs": 96, "trials_per_single_steering_setting": 408}
    draft = yaml.safe_load((destination / "question-bank-suite.yaml").read_text())
    with pytest.raises(ValueError, match="steering.coefficients"):
        ExperimentConfig.from_mapping(draft)
    draft["steering"] = {"coefficients": [1.0], "schemes": [{"name": "test", "layers": [32]}]}
    config = ExperimentConfig.from_mapping(draft)
    assert len(config.trials()) == 408
    assert sum(t["steering"] is None for t in config.trials()) == 24
    assert config.keep_activations
    originals = [yaml.safe_load(path.read_text()) for path in BANKS]
    assert [q.text for q in config.questions] == [q["question"] for t in originals[0]["topics"] for q in t["questions"]]
    assert len({job["output_subdir"] for job in manifest["analyses"]}) == 96
    for job in manifest["analyses"]:
        sets = resolve_set_entries([str(destination / job["sets"])], search_dirs=[])
        source = originals[0 if job["bank"] == BANKS[0].stem else 1]
        question = next(q for t in source["topics"] for q in t["questions"] if q["id"] == job["question"])
        assert len(sets) == 8
        for item in sets:
            assert [w.word for w in item.words] == question["personas"][item.label][job["word_set_type"]]
    with pytest.raises(FileExistsError):
        prepare(BANKS, destination)


@pytest.mark.parametrize("mutation, message", [("question", "disagree"), ("duplicate", "duplicate words"), ("general", "general words vary")])
def test_rejects_invalid_banks_before_writing(tmp_path, mutation, message):
    data = yaml.safe_load(BANKS[1].read_text())
    question = data["topics"][0]["questions"][0]
    if mutation == "question":
        question["question"] += " changed"
    elif mutation == "duplicate":
        question["personas"]["altruist"]["contextualized"][1] = question["personas"]["altruist"]["contextualized"][0]
    else:
        question["personas"]["altruist"]["non_contextualized"][0] = "different"
    altered = tmp_path / BANKS[1].name
    altered.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        prepare([BANKS[0], altered], tmp_path / "out")
    assert not (tmp_path / "out").exists()
