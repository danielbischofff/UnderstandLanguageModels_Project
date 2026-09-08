"""Check the actual experiment definitions, not just synthetic config fixtures."""

from pathlib import Path

import yaml
import pytest

from steering.experiments import ExperimentConfig


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def test_experiment_names_match_filenames():
    for path in CONFIGS.glob("*.yaml"):
        config = ExperimentConfig.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert config.name == path.stem


def test_single_layer_followup_is_thirteen_trials_not_simultaneous_steering():
    path = CONFIGS / "tylenol-demon-8000mg-single-layer-sweep.yaml"
    config = ExperimentConfig.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    trials = config.trials()
    assert len(trials) == 13
    assert sum(trial["steering"] is None for trial in trials) == 1
    assert {
        (trial["steering"]["coefficient"], tuple(trial["steering"]["layers"]))
        for trial in trials if trial["steering"] is not None
    } == {(alpha, (layer,)) for alpha in (2.5, 3.0, 3.5, 4.0) for layer in (29, 32, 35)}
    assert config.generation == {"max_new_tokens": 256, "do_sample": False}
    assert "8000mg" in config.questions[0].text
    assert config.characters == ("unsteered", "demon")


def test_followup_draft_requires_selection_then_expands_one_batch():
    path = CONFIGS.parent / "drafts" / "persona-viewpoint-suite.yaml"
    draft = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert draft["name"] == path.stem
    with pytest.raises(ValueError, match="steering.coefficients must be a nonempty list"):
        ExperimentConfig.from_mapping(draft)
    # Illustrative values only: the on-disk draft must remain unconfigured.
    draft["steering"] = {"coefficients": [3.0], "schemes": [{"name": "layer-32", "layers": [32]}]}
    config = ExperimentConfig.from_mapping(draft)
    assert len(config.questions) == 14
    assert len(config.trials()) == 70
    assert sum(trial["steering"] is None for trial in config.trials()) == 14
    assert config.generation["max_new_tokens"] == 256
