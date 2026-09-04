import steering.prefetch as prefetch_module
from steering.experiments import ExperimentConfig


def _config(**model_overrides):
    model = {
        "id": "publisher/model",
        "dtype": "bfloat16",
        "enable_thinking": False,
        **model_overrides,
    }
    return ExperimentConfig.from_mapping(
        {
            "name": "offline-test",
            "model": model,
            "vectors": {"revision": "vector-sha"},
            "questions": ["question"],
            "characters": ["unsteered", "demon"],
            "steering": {"coefficient": 1, "layers": [2]},
        }
    )


def test_prepare_downloads_model_revision_and_only_used_vectors(monkeypatch):
    snapshots = []
    stores = []

    def snapshot_download(**kwargs):
        snapshots.append(kwargs)
        return "/cache/model"

    class Store:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.recipes = []
            stores.append(self)

        def materialize(self, recipe):
            self.recipes.append(recipe)

    monkeypatch.setattr(prefetch_module, "snapshot_download", snapshot_download)
    monkeypatch.setattr(prefetch_module, "VectorStore", Store)

    result = prefetch_module.prepare_experiment_artifacts(
        _config(revision="model-sha"), local_files_only=False
    )

    assert result == ["/cache/model"]
    assert snapshots == [
        {
            "repo_id": "publisher/model",
            "revision": "model-sha",
            "local_files_only": False,
        }
    ]
    assert stores[0].kwargs == {
        "model_id": "publisher/model",
        "revision": "vector-sha",
        "offline": False,
    }
    assert [recipe.name for recipe in stores[0].recipes] == ["demon"]


def test_prepare_verifies_distinct_tokenizer_revision_offline(monkeypatch):
    snapshots = []

    monkeypatch.setattr(
        prefetch_module,
        "snapshot_download",
        lambda **kwargs: snapshots.append(kwargs) or "/cache/model",
    )

    class Store:
        def __init__(self, **kwargs):
            assert kwargs["offline"] is True

        def materialize(self, recipe):
            pass

    monkeypatch.setattr(prefetch_module, "VectorStore", Store)
    prefetch_module.prepare_experiment_artifacts(
        _config(revision="model-sha", tokenizer_revision="tokenizer-sha"),
        local_files_only=True,
    )

    assert [call["revision"] for call in snapshots] == [
        "model-sha",
        "tokenizer-sha",
    ]
    assert all(call["local_files_only"] for call in snapshots)
