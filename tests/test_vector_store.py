import json
from types import SimpleNamespace

import pytest
import torch

from steering.recipes import DefaultVector, RoleVector
from steering.vector_store import MissingArtifactsError, VectorStore, VectorStoreError


def make_store(tmp_path, tensor, *, legacy=False):
    local = tmp_path / "vectors"
    target = local / "qwen-3-32b" / "default_vector.pt"
    target.parent.mkdir(parents=True)
    torch.save({"vector": tensor, "role": "default"} if legacy else tensor, target)
    index = tmp_path / "indices"
    store = VectorStore(local_dir=local, index_cache_dir=index, offline=True)
    index.mkdir()
    store._index_path.write_text(
        json.dumps(
            {
                "resolved_revision": "commit",
                "files": [
                    "qwen-3-32b/default_vector.pt",
                    "qwen-3-32b/role_vectors/lawyer.pt",
                ],
            }
        )
    )
    return store


@pytest.mark.parametrize("legacy", [False, True])
def test_raw_and_legacy_tensor_loading(tmp_path, legacy):
    tensor = torch.ones(2, 3)
    store = make_store(tmp_path, tensor, legacy=legacy)
    materialized = store.materialize(
        DefaultVector(),
        model_config=SimpleNamespace(num_hidden_layers=2, hidden_size=3),
    )
    assert torch.equal(materialized.tensor, tensor)
    assert materialized.artifacts[0].resolved_revision == "commit"
    assert len(materialized.artifacts[0].sha256) == 64


def test_model_shape_validation(tmp_path):
    store = make_store(tmp_path, torch.ones(2, 3))
    with pytest.raises(VectorStoreError, match="shape mismatch"):
        store.materialize(
            DefaultVector(),
            model_config=SimpleNamespace(num_hidden_layers=3, hidden_size=3),
        )


def test_offline_missing_artifact_lists_filename(tmp_path):
    store = make_store(tmp_path, torch.ones(2, 3))
    with pytest.raises(MissingArtifactsError) as error:
        store.materialize(RoleVector("lawyer"))
    assert "role_vectors/lawyer.pt" in str(error.value)


def test_assistant_role_is_never_default(tmp_path):
    store = make_store(tmp_path, torch.ones(2, 3))
    assert store.fetch("default_vector.pt").name == "default_vector.pt"


def test_unknown_model_defers_vector_mapping_requirement():
    store = VectorStore(model_id="custom/model", offline=True)
    with pytest.raises(VectorStoreError, match="dataset_model_dir"):
        store.materialize(DefaultVector())


def test_default_index_cache_stays_under_huggingface_home(tmp_path, monkeypatch):
    huggingface_home = tmp_path / "runtime" / "models" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(huggingface_home))
    store = VectorStore()
    assert store.index_cache_dir == huggingface_home / "steering" / "indices"


def test_fp8_checkpoint_uses_base_model_vector_directory():
    store = VectorStore(model_id="Qwen/Qwen3-32B-FP8")
    assert store._model_dir() == "qwen-3-32b"
