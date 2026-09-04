import steering.cli.common as common_module


def test_runner_uses_only_local_hub_artifacts_when_offline(monkeypatch):
    observed = {}

    class Store:
        def __init__(self, **kwargs):
            observed["store"] = kwargs

    class Runner:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            observed["model_id"] = model_id
            observed["runner"] = kwargs
            return "runner"

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(common_module, "VectorStore", Store)
    monkeypatch.setattr(common_module, "InferenceRunner", Runner)

    result = common_module.runner_from_config(
        {
            "model": {"id": "publisher/model", "dtype": "bfloat16"},
            "vectors": {"revision": "vector-sha"},
        }
    )

    assert result == "runner"
    assert observed["store"]["offline"] is True
    assert observed["runner"]["local_files_only"] is True
