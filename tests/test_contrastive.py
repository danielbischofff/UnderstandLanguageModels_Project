"""Curated contrastive sets: resolution, both measures, wiring, and plots."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from quantitive_analysis import contrastive, reader as j_lens, plots, quant
from steering.capture import ActivationCaptureEvent
from quantitive_analysis.contrastive import (
    ContrastiveProbe,
    ContrastiveSet,
    WordSpec,
    merge_set_library,
    ranks_from_logits,
    resolve_set_entries,
    resolve_word,
)

# A two-dimensional vocabulary makes every similarity checkable by hand.
ROWS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
TEXTS = ["zero", "one", "two", "three"]


class FakeTokenizer:
    """Only ``" good"`` style whole-token words exist; the rest split."""

    table = {" good": [1], "good": [1], " bad": [3], "bad": [3], " mixed": [2], " Mixed": [2]}

    def encode(self, sequence, add_special_tokens=True):
        return SimpleNamespace(ids=self.table.get(sequence, [0, 1]))


def embedding_rows(ids):
    return ROWS[np.asarray(ids, dtype=np.int64)]


def probe(*, layers=(1,), top_k=2, sets=None):
    sets = sets or [
        ContrastiveSet.from_mapping({"id": "good", "label": "Good", "words": ["good"]}),
        ContrastiveSet.from_mapping({"id": "bad", "words": ["bad"]}),
    ]
    return ContrastiveProbe(
        sets,
        tokenizer=FakeTokenizer(),
        embedding_rows=embedding_rows,
        layers=layers,
        top_k=top_k,
    )


def test_resolution_prefers_a_whole_token_and_marks_the_fallback():
    whole = resolve_word(FakeTokenizer(), WordSpec("good"))
    assert (whole.token_id, whole.surface_form, whole.complete) == (1, " good", True)
    # No single-token form: the first piece stands in, and says so.
    split = resolve_word(FakeTokenizer(), WordSpec("condolences"))
    assert (split.token_id, split.complete) == (0, False)
    # An explicit variant overrides the search order.
    explicit = resolve_word(FakeTokenizer(), WordSpec("anything", (" mixed",)))
    assert explicit.token_id == 2


def test_sets_resolve_by_name_and_conflicting_ids_are_rejected(tmp_path):
    (tmp_path / "care.yaml").write_text("label: Care\nwords: [good]\n", encoding="utf-8")
    (tmp_path / "other.yaml").write_text("id: care\nwords: [bad]\n", encoding="utf-8")
    care = resolve_set_entries(["care"], search_dirs=[tmp_path])[0]
    assert care.id == "care" and care.label == "Care"
    assert care.source.endswith("care.yaml")
    assert resolve_set_entries([{"id": "inline", "words": ["good"]}])[0].id == "inline"
    with pytest.raises(FileNotFoundError, match="unknown contrastive set"):
        resolve_set_entries(["missing"], search_dirs=[tmp_path])
    with pytest.raises(ValueError, match="conflicting definitions"):
        merge_set_library(resolve_set_entries(["care", str(tmp_path / "other.yaml")], search_dirs=[tmp_path]))


def test_cosine_measures_weight_the_surfaced_words_by_probability():
    scored = probe().cosine_measures(
        np.array([[2, 1], [3, 0]]), np.array([[0.4, 0.3], [0.5, 0.1]])
    )
    # Position 0: weights 4/7 on token 2 (cos 1/sqrt2 to "good") and 3/7 on the
    # set word itself; position 1 surfaces the opposite corner of the space.
    assert scored["good"]["centroid_cosine"][0] == pytest.approx(
        4 / 7 / np.sqrt(2) + 3 / 7, rel=1e-6
    )
    assert scored["good"]["max_cosine"][0] == pytest.approx(1.0)
    assert scored["good"]["top1_cosine"][0] == pytest.approx(1 / np.sqrt(2), rel=1e-6)
    # Position 1: 5/6 on the set word itself and 1/6 on its exact opposite.
    assert scored["bad"]["centroid_cosine"][1] == pytest.approx(4 / 6, rel=1e-6)
    assert scored["bad"]["centroid_cosine"][0] < 0


def test_rank_measures_read_the_full_vocabulary_including_ties():
    logits = np.array([[3.0, 2.0, 2.0, 1.0], [0.0, 5.0, 1.0, 4.0]])
    ranks = ranks_from_logits(logits, [1, 2, 3])
    # Tied logits share the better rank instead of ordering arbitrarily.
    assert ranks.tolist() == [[2, 2, 4], [1, 3, 2]]
    measured = probe().rank_measures(ranks_from_logits(logits, probe().token_ids))
    assert measured["good"]["median_rank"] == pytest.approx([2.0, 1.0])
    assert measured["bad"]["min_rank"] == pytest.approx([4.0, 2.0])
    assert measured["bad"]["mean_reciprocal_rank"] == pytest.approx([0.25, 0.5])


def test_a_set_needs_words_and_stays_within_the_plottable_count():
    with pytest.raises(ValueError, match="must list words"):
        ContrastiveSet.from_mapping({"id": "empty", "words": []})
    with pytest.raises(ValueError, match="repeats a word"):
        ContrastiveSet.from_mapping({"id": "twice", "words": ["good", "Good"]})
    many = [ContrastiveSet.from_mapping({"id": f"s{i}", "words": ["good"]}) for i in range(9)]
    with pytest.raises(ValueError, match="at most 8 contrastive sets"):
        probe(sets=many)


@contextmanager
def fake_backend(_source):
    class Lens:
        source_layers = [0, 1]
        n_prompts = 80

        def __init__(self, path):
            pass

        def transport(self, layer, matrix):
            return matrix + layer

        def matrix(self, layer):
            return torch.eye(2), 1.0

    class Unembedding:
        vocab_size = 4

        def __init__(self, data_dir):
            pass

        def logits(self, matrix):
            return np.column_stack([matrix, matrix.sum(axis=1), -matrix.sum(axis=1)])

        def rows(self, ids):
            return embedding_rows(ids)

    def top_k(logits, count, allowed=None):
        scores = logits if allowed is None else np.where(allowed, logits, -np.inf)
        ids = np.argsort(scores, axis=1)[:, ::-1][:, :count]
        selected = np.take_along_axis(logits, ids, axis=1)
        return ids, selected, np.full_like(selected, 0.25)

    yield (
        SimpleNamespace(LensTransport=Lens),
        SimpleNamespace(Unembedding=Unembedding, top_k=top_k),
        SimpleNamespace(decode_vocab=lambda tokenizer, size: (TEXTS, np.ones(size, bool))),
    )


def reader(tmp_path, monkeypatch, top_k_count=2):
    monkeypatch.setattr(j_lens, "_analysis_imports", fake_backend)
    monkeypatch.setattr(j_lens.Tokenizer, "from_file", staticmethod(lambda path: FakeTokenizer()))
    return j_lens.AnalysisReader(
        analysis_source=tmp_path,
        data_dir=tmp_path,
        lens_path=tmp_path / "lens.pt",
        top_k_count=top_k_count,
    )


def events():
    return [
        ActivationCaptureEvent(
            tensor=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            layer=layer,
            point="post_intervention",
            phase="prefill",
            forward_index=0,
            input_token_ids=[[1, 2]],
            cache_positions=None,
            position_policy="all",
        )
        for layer in (0, 1)
    ]


def synthetic_trial(with_measures=True, positions=8):
    words = [
        [{"id": (index + row) % 4, "text": TEXTS[(index + row) % 4], "logit": 1.0, "probability": 0.3 - 0.1 * row}
         for row in range(2)]
        for index in range(positions)
    ]
    trial = {
        "question": {"id": "q1", "text": "why"},
        "character": "demon",
        "steering": {"coefficient": 1.0, "name": "layer-1", "layers": [1], "method": "additive"},
        "response": {"text": "answer"},
        "j_lens": {
            "prompt_length": 3,
            "tokens": [
                {"position": index, "id": index % 4, "text": TEXTS[index % 4],
                 "segment": "prompt" if index < 3 else "response"}
                for index in range(positions)
            ],
            "layers": [{"layer": 1, "positions": words}],
        },
    }
    if with_measures:
        trial["j_lens"]["contrastive"] = {
            "schema_version": 1,
            "layers": [1],
            "top_k": 2,
            "sets": [{"id": "good", "label": "Good", "color": "#2a78d6",
                      "words": ["good"], "tokens": [{"word": "good", "token_id": 1}]}],
            "series": [{
                "layer": 1,
                "set": "good",
                "median_rank": [3.0, 3.0, 3.0, 90.0, 3.0, 3.0, 3.0, 3.0],
                "centroid_cosine": [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
            }],
        }
    return trial


def test_new_sets_are_scored_from_saved_top_words_without_ranks(tmp_path, monkeypatch):
    lens = reader(tmp_path, monkeypatch)
    trial = synthetic_trial(with_measures=False)
    loaded = quant.LoadedTrial(metadata=trial, load=lambda: trial)
    series = quant.trial_series(loaded, probe=lens.probe(
        [ContrastiveSet.from_mapping({"id": "good", "words": ["good"]})], layers=[1]
    ))
    layer = series.layers[0]
    assert layer.source == quant.COSINE_ONLY
    assert set(layer.sets[0].measures) == set(contrastive.COSINE_MEASURES)
    assert len(layer.sets[0].measures["centroid_cosine"]) == 8


def test_edited_sets_are_recomputed_instead_of_reusing_stale_numbers(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    lens = reader(tmp_path, monkeypatch)
    trial = synthetic_trial()
    trial["j_lens"]["tokens"] = trial["j_lens"]["tokens"][:2]
    trial["j_lens"]["layers"][0]["positions"] = trial["j_lens"]["layers"][0]["positions"][:2]
    for entry in trial["j_lens"]["contrastive"]["series"]:
        for measure in ("median_rank", "centroid_cosine"):
            entry[measure] = entry[measure][:2]
    directory = tmp_path / "activations"
    directory.mkdir()
    save_file({"event_000000": events()[1].tensor.contiguous()}, directory / "activations.safetensors")
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "tensor_file": "activations.safetensors",
                    "events": [events()[1].manifest_entry("event_000000")]}),
        encoding="utf-8",
    )
    loaded = quant.LoadedTrial(
        metadata=trial, load=lambda: trial, activations=directory / "manifest.json"
    )
    # Same set id, different curated word: the stored numbers no longer apply.
    edited = lens.probe([ContrastiveSet.from_mapping({"id": "good", "words": ["bad"]})], layers=[1])
    series = quant.trial_series(loaded, probe=edited, reader=lens)
    assert [layer.source for layer in series.layers] == [quant.RECOMPUTED]
    assert set(series.layers[0].sets[0].measures) == set(contrastive.MEASURES)


def test_change_labelling_scales_with_its_parameter():
    values = np.array([0.1] * 6 + [0.9] + [0.1] * 6)
    assert plots.significant_changes(values, plots.PlotOptions(change_z=3.0)) == [6, 7]
    assert plots.significant_changes(values, plots.PlotOptions(change_z=3.0, max_annotations=1)) == [6]
    assert plots.significant_changes(values, plots.PlotOptions(min_change_fraction=1.5)) == []
    # A flat series has nothing to report, whatever the threshold.
    assert plots.significant_changes(np.full(12, 0.4), plots.PlotOptions(change_z=0.0)) == []
    # Rank jumps are judged in orders of magnitude.
    ranks = np.array([3.0] * 6 + [3000.0] + [3.0] * 6)
    assert plots.significant_changes(ranks, plots.PlotOptions(), logarithmic=True) == [6, 7]


def test_run_directory_becomes_figures_a_table_and_an_index(tmp_path):
    run = tmp_path / "runs" / "0001_run"
    (run / "trials" / "00000").mkdir(parents=True)
    (run / "resolved-config.yaml").write_text(
        yaml.safe_dump({"name": "exp", "questions": [{"id": "q1", "text": "why"}]}), encoding="utf-8"
    )
    (run / "trials" / "00000" / "result.json").write_text(
        json.dumps(synthetic_trial()), encoding="utf-8"
    )
    archive = quant.load_run(run)
    series = quant.run_series(archive, probe=probe())
    destination = tmp_path / "plots"
    written = plots.write_figures(archive, series, destination, table="metrics.csv")
    quant.write_table(destination / "metrics.csv", quant.table_rows(archive, series))
    index = plots.write_index(archive, series, destination, table="metrics.csv")

    assert [path.name for path in written] == ["q1__demon_a1_layer-1__layer-01.html"]
    page = written[0].read_text(encoding="utf-8")
    assert "Good" in page and "#2a78d6" in page
    assert '"three"' not in page or "median_rank" not in page  # only measures are plotted
    assert (destination / "plotly.min.js").is_file()  # shared, so pages stay small
    assert written[0].stat().st_size < 400_000
    assert "q1" in index.read_text(encoding="utf-8")
    table = (destination / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert table[0].startswith("experiment,run,question")
    assert len(table) == 17

    document = quant.to_document(archive, series)
    assert json.dumps(document, allow_nan=False)
    assert document["trials"][0]["layers"][0]["sets"][0]["id"] == "good"


def test_tylenol_example_contains_two_opposing_sets():
    sets = resolve_set_entries(["src/quantitive_analysis/examples/tylenol.yaml"])
    assert [item.id for item in sets] == ["care", "malice"]
    assert all(item.words for item in sets)


def test_cli_scores_old_archive_and_reruns_with_changed_sets(tmp_path, monkeypatch):
    import sys
    import zipfile
    from quantitive_analysis import cli
    from steering.binary_results import encode_trial

    vocabulary = {}
    descriptor, payload = encode_trial(synthetic_trial(with_measures=False), vocabulary)
    descriptor["payload"] = "trial.bin.gz"
    bundle = tmp_path / "results.zip"
    with zipfile.ZipFile(bundle, "w") as handle:
        handle.writestr("index.json", json.dumps({
            "format": "steering-binary-v1", "document": {"experiment": {"name": "old"}},
            "trials": [descriptor],
        }))
        handle.writestr("vocabulary.json", json.dumps(vocabulary))
        handle.writestr(descriptor["payload"], payload)
    lens = reader(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_open_reader", lambda *a, **kw: lens)
    sets = tmp_path / "words.yaml"
    output = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", [
        "quantitive-analysis", str(bundle), "--sets", str(sets), "--output-dir", str(output)
    ])
    results = []
    for word in ("good", "bad"):
        sets.write_text(f"- id: idea\n  words: [{word}]\n", encoding="utf-8")
        cli.main()
        document = json.loads(next(output.rglob("metrics.json")).read_text())
        layer = document["trials"][0]["layers"][0]
        assert layer["source"] == quant.COSINE_ONLY
        assert layer["sets"][0]["words"] == [word]
        results.append(layer["sets"][0]["measures"]["centroid_cosine"])
    assert results[0] != results[1]
    assert next(output.rglob("index.html")).is_file()
