import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("openTSNE")
pytest.importorskip("plotly")

from steering.tsne_lens.inputs import load_roles, load_trajectory
from steering.tsne_lens.plots import make_figure, write_outputs
from steering.tsne_lens.projection import RoleMap


@pytest.fixture(scope="module")
def role_map():
    rng = np.random.default_rng(17)
    vectors = np.vstack(
        [
            rng.normal(center, 0.015, (10, 6))
            for center in ([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0])
        ]
    )
    return RoleMap.fit(
        vectors,
        [f"role-{i}" for i in range(30)],
        layer=2,
        model_id="test",
        perplexity=4,
        min_cluster_size=4,
    )


def rows(n):
    return [
        dict(step=i, position=i, token_id=i, token=f"t{i}", segment="generated")
        for i in range(n)
    ]


def test_frozen_map_roundtrip_and_independent_token_projection(role_map, tmp_path):
    before = np.array(role_map.embedding, copy=True)
    vectors = role_map.vectors[[2, 13, 24]]
    combined = role_map.project(vectors, rows(3))
    np.testing.assert_array_equal(role_map.embedding, before)
    role_map.save(tmp_path / "map.pkl")
    restored = RoleMap.load(tmp_path / "map.pkl")
    for i, vector in enumerate(vectors):
        alone = restored.project(vector[None], rows(1))[0]
        np.testing.assert_allclose(
            [combined[i]["x"], combined[i]["y"]], [alone["x"], alone["y"]], atol=1e-6
        )
        assert combined[i]["neighbors"][0]["label"] == role_map.labels[[2, 13, 24][i]]
        assert combined[i]["neighbors"][0]["distance"] == pytest.approx(0, abs=1e-12)
    np.testing.assert_array_equal(restored.embedding, before)
    assert len(set(role_map.clusters) - {-1}) == 3
    for label in role_map.cluster_names.values():
        assert 1 <= len(label.split(", ")) <= 3


def test_role_inputs_select_layer_and_validate_identity(tmp_path):
    vectors = np.arange(3 * 4 * 6).reshape(3, 4, 6).astype(float)
    for i in range(3):
        torch.save({"vector": torch.tensor(vectors[i])}, tmp_path / f"r{i}.pt")
    loaded, labels = load_roles(tmp_path, 2, "test")
    np.testing.assert_array_equal(loaded, vectors[:, 2])
    assert labels == ["r0", "r1", "r2"]
    np.savez(tmp_path / "roles.npz", vectors=vectors, labels=labels, model_id="test")
    np.testing.assert_array_equal(
        load_roles(tmp_path / "roles.npz", 1, "test")[0], vectors[:, 1]
    )
    with pytest.raises(ValueError, match="model_id"):
        load_roles(tmp_path / "roles.npz", 1, "wrong")
    np.savez(tmp_path / "bad.npz", vectors=vectors[:, 2], labels=["x", "x", "y"])
    with pytest.raises(ValueError, match="unique"):
        load_roles(tmp_path / "bad.npz", 2, "test")


def capture(tmp_path, events, metadata=None):
    directory = tmp_path / "activations"
    directory.mkdir()
    save_file(
        {
            e["tensor_key"]: torch.full(
                (2, len(e["input_token_ids"][0]), 6), float(i + 1)
            )
            for i, e in enumerate(events)
        },
        directory / "activations.safetensors",
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                tensor_file="activations.safetensors",
                coordinate_system=dict(residual_stream="post_block_post_mlp"),
                events=events,
            )
        )
    )
    if metadata:
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return directory


def event(
    key,
    ids,
    *,
    point="post_intervention",
    phase="prefill",
    forward=0,
    policy="all",
    positions=None,
):
    return dict(
        tensor_key=key,
        layer=2,
        point=point,
        phase=phase,
        forward_index=forward,
        input_token_ids=[ids, ids],
        position_policy=policy,
        cache_positions=positions,
    )


def test_full_sequence_replay_keeps_generated_tokens_and_text(tmp_path):
    path = capture(
        tmp_path,
        [
            event("pre", [10, 11, 12, 13], point="pre_intervention"),
            event("post", [10, 11, 12, 13]),
        ],
        dict(
            model_id="test",
            tokens=dict(
                prompt_ids=[10, 11],
                generated_ids=[12, 13],
                prompt_tokens=["a", "b"],
                generated_tokens=["c", "d"],
            ),
        ),
    )
    vectors, records = load_trajectory(
        path.parent, layer=2, model_id="test", generated_only=True
    )
    assert vectors.shape == (2, 6)
    assert np.all(vectors == 2)  # selected post, not pre
    assert [r["token"] for r in records] == ["c", "d"]
    assert [r["position"] for r in records] == [2, 3]
    assert all(r["segment"] == "generated" and r["phase"] == "prefill" for r in records)
    with pytest.raises(ValueError, match="model_id"):
        load_trajectory(path, layer=2, model_id="wrong")
    with pytest.raises(ValueError, match="no capture"):
        load_trajectory(path, layer=3, model_id="test")


def test_cached_decode_order_and_missing_positions(tmp_path):
    path = capture(
        tmp_path,
        [
            event("later", [13], phase="decode", forward=2, positions=[3]),
            event("prompt", [11], policy="last", positions=[1]),
            event("first", [12], phase="decode", forward=1),
        ],
    )
    vectors, records = load_trajectory(
        path, layer=2, model_id="test", batch_index=1, generated_only=True
    )
    assert [r["token_id"] for r in records] == [12, 13]
    assert [r["position"] for r in records] == [
        None,
        3,
    ]  # never invent a cache position
    assert vectors[:, 0].tolist() == [3, 1]
    with pytest.raises(ValueError, match="batch index"):
        load_trajectory(path, layer=2, model_id="test", batch_index=2)


def test_npz_metadata_and_invalid_matrices(role_map, tmp_path):
    path = tmp_path / "trajectory.npz"
    np.savez(
        path,
        vectors=role_map.vectors[:3],
        layer=2,
        model_id="test",
        tokens=["a", "b", "c"],
        segments=["prompt", "generated", "generated"],
    )
    vectors, records = load_trajectory(
        path, layer=2, model_id="test", generated_only=True
    )
    assert len(vectors) == 2 and records[0]["token"] == "b"
    with pytest.raises(ValueError, match="layer"):
        load_trajectory(path, layer=1, model_id="test")
    with pytest.raises(ValueError, match="width"):
        role_map.project(np.ones((2, 8)), rows(2))
    with pytest.raises(ValueError, match="finite"):
        role_map.project(np.full((1, 6), np.nan), rows(1))
    with pytest.raises(ValueError, match="zero"):
        role_map.project(np.zeros((1, 6)), rows(1))


def test_html_frames_and_json(role_map, tmp_path):
    trajectory = dict(
        name="example", points=role_map.project(role_map.vectors[:3], rows(3))
    )
    fig = make_figure(role_map, [trajectory])
    assert len(fig.frames) == 3
    assert all(len(frame.data) == 1 for frame in fig.frames)
    assert fig.frames[0].traces[0] == len(fig.data) - 1
    write_outputs(role_map, [trajectory], tmp_path)
    assert (tmp_path / "roles.html").stat().st_size > 1_000_000  # self-contained Plotly
    payload = json.loads((tmp_path / "projection.json").read_text())
    assert len(payload["roles"]) == 30
    assert payload["trajectories"][0]["points"][0]["neighbors"][0]["label"] == "role-0"


def test_multiscale_map_preserves_reference_and_roundtrips(role_map, tmp_path):
    fitted = RoleMap.fit(
        role_map.vectors,
        role_map.labels,
        perplexity=[3, 8],
        n_iter=300,
        early_exaggeration_iter=100,
    )
    assert fitted.config["affinity"] == "multiscale"
    assert fitted.config["perplexities"] == [3, 8]
    before = np.asarray(fitted.embedding).copy()
    vectors = fitted.vectors[[1, 12]]
    together = fitted.project(vectors, rows(2))
    fitted.save(tmp_path / "multiscale.pkl")
    restored = RoleMap.load(tmp_path / "multiscale.pkl")
    for i in range(2):
        alone = restored.project(vectors[i : i + 1], rows(1))[0]
        np.testing.assert_allclose(
            [alone["x"], alone["y"]], [together[i]["x"], together[i]["y"]], atol=1e-6
        )
    np.testing.assert_array_equal(before, fitted.embedding)
    np.testing.assert_array_equal(before, restored.embedding)
    assert fitted.config["quality"]["kl_unexaggerated"] >= 0


def test_quality_exact_kl_and_perfect_neighbor_geometry():
    from types import SimpleNamespace

    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import pdist, squareform

    from steering.tsne_lens.quality import embedding_quality

    class Embedding(np.ndarray):
        pass

    xy = np.array([[0, 0], [1, 2], [3, 1], [8, 3], [10, 4], [7, 9]], dtype=float).view(
        Embedding
    )
    distances = squareform(pdist(xy))
    q = 1 / (1 + distances**2)
    np.fill_diagonal(q, 0)
    xy.affinities = SimpleNamespace(P=csr_matrix(q / q.sum()))
    metrics = embedding_quality(distances, xy, np.full(len(xy), -1))
    assert metrics["kl_unexaggerated"] == pytest.approx(0, abs=1e-12)
    assert metrics["neighbor_recall"] == 1
    assert metrics["trustworthiness"] == 1
    assert metrics["global_distance_spearman"] == pytest.approx(1)
    assert metrics["cluster_coverage"] == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(perplexity=[]),
        dict(perplexity=[5, -1]),
        dict(exaggeration=0),
        dict(theta=2),
        dict(n_iter=0),
        dict(min_samples=0),
    ],
)
def test_invalid_tuning_parameters(role_map, options):
    with pytest.raises(ValueError):
        RoleMap.fit(role_map.vectors, role_map.labels, **options)
