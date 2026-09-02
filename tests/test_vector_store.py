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


def test_role_vector_falls_back_to_same_named_trait(tmp_path):
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1
    store = make_store(tmp_path, torch.ones(2, 3))
    trait = store.local_dir / "qwen-3-32b" / "trait_vectors" / "evil.pt"
    trait.parent.mkdir(parents=True)
    torch.save(tensor, trait)
    store._index_path.write_text(
        json.dumps(
            {
                "resolved_revision": "commit",
                "files": [
                    "qwen-3-32b/default_vector.pt",
                    "qwen-3-32b/trait_vectors/evil.pt",
                ],
            }
        )
    )

    materialized = store.materialize(RoleVector("evil"))

    assert torch.equal(materialized.tensor, tensor)
    assert materialized.artifacts[0].filename.endswith("trait_vectors/evil.pt")


def test_role_vector_prefers_role_when_name_is_also_a_trait(tmp_path):
    role_tensor = torch.full((2, 3), 2.0)
    trait_tensor = torch.full((2, 3), 3.0)
    store = make_store(tmp_path, torch.ones(2, 3))
    role = store.local_dir / "qwen-3-32b" / "role_vectors" / "shared.pt"
    trait = store.local_dir / "qwen-3-32b" / "trait_vectors" / "shared.pt"
    role.parent.mkdir(parents=True)
    trait.parent.mkdir(parents=True)
    torch.save(role_tensor, role)
    torch.save(trait_tensor, trait)
    store._index_path.write_text(
        json.dumps(
            {
                "resolved_revision": "commit",
                "files": [
                    "qwen-3-32b/default_vector.pt",
                    "qwen-3-32b/role_vectors/shared.pt",
                    "qwen-3-32b/trait_vectors/shared.pt",
                ],
            }
        )
    )

    materialized = store.materialize(RoleVector("shared"))

    assert torch.equal(materialized.tensor, role_tensor)
    assert materialized.artifacts[0].filename.endswith("role_vectors/shared.pt")


def test_unknown_role_reports_available_traits(tmp_path):
    store = make_store(tmp_path, torch.ones(2, 3))
    store._index_path.write_text(
        json.dumps(
            {
                "resolved_revision": "commit",
                "files": [
                    "qwen-3-32b/default_vector.pt",
                    "qwen-3-32b/role_vectors/lawyer.pt",
                    "qwen-3-32b/trait_vectors/evil.pt",
                ],
            }
        )
    )

    with pytest.raises(VectorStoreError, match="Unknown role or trait") as error:
        store.materialize(RoleVector("missing"))

    assert "available roles: lawyer" in str(error.value)
    assert "available traits: evil" in str(error.value)


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
