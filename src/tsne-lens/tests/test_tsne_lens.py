import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("openTSNE")
pytest.importorskip("plotly")

from tsne_lens.inputs import load_roles, load_trajectory
from tsne_lens.plots import make_figure, write_outputs
from tsne_lens.projection import RoleMap


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
        n_clusters=3,
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
    path = next(trace for trace in fig.data if trace.mode == "lines+markers+text")
    assert path.marker.symbol == "square"
    assert list(path.text) == ["0: t0", "1: t1", "2: t2"]
    arrows = [a for a in fig.layout.annotations if a.axref == "x"]
    assert len(arrows) == 2
    assert arrows[0].ax == trajectory["points"][0]["x"]
    assert arrows[0].x == trajectory["points"][1]["x"]
    write_outputs(role_map, [trajectory], tmp_path)
    assert (tmp_path / "roles.html").stat().st_size > 1_000_000  # self-contained Plotly
    payload = json.loads((tmp_path / "projection.json").read_text())
    assert len(payload["roles"]) == 30
    assert payload["trajectories"][0]["points"][0]["neighbors"][0]["label"] == "role-0"


def test_batch_prompt_projection_and_missing_artifact(role_map, tmp_path):
    from tsne_lens.batch import project_run

    run = tmp_path / "run"
    trial = run / "trials" / "00000_control"
    trial.mkdir(parents=True)
    capture(trial, [event("prefill", [10, 11]),
                    event("decode", [12], phase="decode", forward=1, positions=[2])],
            dict(model_id="test", tokens=dict(prompt_ids=[10, 11], generated_ids=[12, 13],
                 prompt_tokens=["hello", "there"], generated_tokens=["answer", "end"])))
    manifest = dict(trials={"control": dict(index=0, status="complete", character="unsteered",
                    steering=None, path="trials/00000_control"),
                    "failed": dict(index=1, status="failed")})
    (run / "manifest.json").write_text(json.dumps(manifest))
    before = np.asarray(role_map.embedding).copy()
    summary = project_run(role_map, run, tmp_path / "plots", prompt_only=True)
    assert summary["skipped_incomplete"] == 1
    assert summary["trials"][0]["tokens"] == 2
    projection = json.loads((tmp_path / "plots/00000_control/projection.json").read_text())
    points = projection["trajectories"][0]["points"]
    assert [p["token"] for p in points] == ["hello", "there"]
    assert [p["position"] for p in points] == [0, 1]
    assert (tmp_path / "plots/index.html").is_file()
    np.testing.assert_array_equal(before, role_map.embedding)
    (trial / "activations/manifest.json").unlink()
    with pytest.raises(ValueError, match="missing activations"):
        project_run(role_map, run, tmp_path / "missing")


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
    from tsne_lens.quality import embedding_quality

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
        dict(n_clusters=1),
    ],
)
def test_invalid_tuning_parameters(role_map, options):
    with pytest.raises(ValueError):
        RoleMap.fit(role_map.vectors, role_map.labels, **options)


def test_fixed_multiscale_all_neighbors_and_convergence_reporting(role_map):
    from openTSNE import TSNEEmbedding
    from tsne_lens.optimization import exact_kl, train_embedding

    e, history, frames, status = train_embedding(
        role_map.vectors,
        perplexities=[4, 9],
        early_iter=0,
        n_iter=20,
        relaxation_iter=10,
        max_iter=10,
        checkpoint_interval=10,
    )
    assert history[0]["perplexities"] == [4, 9]
    assert status["neighbors"] == 29
    assert status["final_perplexities"] == [4, 9]
    assert not status["converged"]  # Only one plateau check was permitted.
    assert status["iterations"] == 30
    assert status["pca_kl_final_affinity"] == pytest.approx(
        exact_kl(TSNEEmbedding(frames[0], e.affinities))
    )
    assert all(row["perplexities"] == [4, 9] for row in history)


def test_ward_subclusters_are_nested_and_cover_all_roles(role_map):
    labels = np.asarray(role_map.clusters)
    sub = np.asarray(role_map.config["subcluster_ids"])
    assert len(set(labels)) == 3 and np.all(labels >= 0)
    assert len(set(sub)) == 6
    assert all(len(set(labels[sub == c])) == 1 for c in set(sub))
    assert role_map.config["quality"]["cluster_coverage"] == 1


def test_exaggeration_ramp_keeps_affinity_fixed(role_map):
    from tsne_lens.optimization import train_embedding

    for ramp in (0, 60):
        embedding, history, frames, status = train_embedding(
            role_map.vectors,
            perplexities=[4, 8],
            early_exaggeration=2,
            early_iter=25,
            exaggeration_anneal_iter=ramp,
            n_iter=10,
            relaxation_iter=10,
            max_iter=10,
        )
        annealing = [r for r in history if r["phase"] == "exaggeration-anneal"]
        assert bool(annealing) == bool(ramp)
        if ramp:
            assert [r["iteration"] for r in annealing] == [50, 75, 85]
            assert [r["exaggeration"] for r in annealing] == pytest.approx(
                [19 / 12, 7 / 6, 1]
            )
            assert all(r["perplexities"] == [4, 8] for r in annealing)
        assert status["iterations"] == 45 + ramp
        assert history[-1]["exaggeration"] == 1
        assert all(r["perplexities"] == [4, 8] for r in history)
        assert len(frames) == len(history)
        assert embedding.affinities.P.sum() == pytest.approx(1)


def test_reference_starts_in_true_role_pca(role_map):
    from openTSNE import initialization
    from sklearn.decomposition import PCA
    from tsne_lens.optimization import train_embedding
    from tsne_lens.training_plots import default_positions

    vector = role_map.vectors[0]
    e, h, frames, status = train_embedding(
        role_map.vectors,
        reference_vector=vector,
        perplexities=[4, 8],
        early_iter=0,
        n_iter=10,
        relaxation_iter=0,
        max_iter=0,
    )
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(role_map.vectors)
    expected = pca.transform(vector[None])[0] / (xy[:, 0].std() / 1e-4)
    np.testing.assert_allclose(status["reference_pca"], expected)
    np.testing.assert_allclose(
        frames[0], initialization.pca(role_map.vectors, random_state=42)
    )
    before = np.array(e, copy=True)
    positions = default_positions(e, frames, vector, status["reference_pca"])
    np.testing.assert_allclose(positions[0], expected)
    np.testing.assert_array_equal(e, before)


@pytest.mark.parametrize("dof", [0.5, 2.0])
def test_exact_kl_matches_opentsne_for_other_tail_weights(dof):
    from openTSNE import TSNEEmbedding, affinity
    from tsne_lens.optimization import exact_kl

    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(20, 6))
    p = affinity.PerplexityBasedNN(
        vectors, perplexity=4, k_neighbors=19, method="exact"
    )
    e = TSNEEmbedding(
        rng.normal(size=(20, 2)),
        p,
        dof=dof,
        negative_gradient_method="bh",
        theta=0,
        n_jobs=1,
    )
    # Negligible movement lets us compare independently computed exact KL values.
    e.optimize(1, learning_rate=1e-20, exaggeration=1, inplace=True)
    assert exact_kl(e) == pytest.approx(e.kl_divergence, abs=1e-10)


def test_addon_loads_pre_move_role_map(role_map, tmp_path, monkeypatch):
    import pickle
    import sys
    from types import ModuleType

    old_name = "steering.tsne_lens.projection"
    old_package = ModuleType("steering.tsne_lens")
    old_module = ModuleType(old_name)
    old_class = type("RoleMap", (RoleMap,), {"__module__": old_name})
    old_module.RoleMap = old_class
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "steering.tsne_lens", old_package)
        patch.setitem(sys.modules, old_name, old_module)
        old_map = old_class(**vars(role_map))
        path = tmp_path / "legacy.pkl"
        path.write_bytes(pickle.dumps(old_map))
    restored = RoleMap.load(path)
    assert type(restored) is RoleMap
    np.testing.assert_array_equal(restored.embedding, role_map.embedding)


def test_main_defaults_use_fixed_three_scales_and_heavier_tails(role_map):
    m = RoleMap.fit(
        role_map.vectors,
        role_map.labels,
        early_exaggeration_iter=0,
        n_iter=10,
        relaxation_iter=0,
        max_iter=0,
    )
    assert m.config["dof"] == 0.7
    assert m.config["initial_perplexities"] == [15, 25, 90]
    assert all(
        r["perplexities"] == m.config["perplexities"]
        for r in m.config["training_history"]
    )
    assert len(set(m.clusters)) == 6


def test_cluster_ribbons_helpers(role_map):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cluster_ribbons as ribbons

    assert ribbons.parse_labelled_path("alpha=runs/0001") == ("alpha", Path("runs/0001"))
    assert ribbons.parse_labelled_path("a/sweep/runs/0001")[0] == "sweep"
    with pytest.raises(Exception):
        ribbons.parse_labelled_path("=nothing")

    assert ribbons.display_token("Ġfeel") == "feel"
    assert ribbons.display_token("Ġ") == "␣"
    assert ribbons.display_token("Ċ") == "⏎"
    assert ribbons.display_token("<|im_start|>") == "im_start"

    assert ribbons.tint_for(ribbons.TINT_CEILING) == ribbons.MAX_ALPHA
    assert ribbons.tint_for(ribbons.MIN_SIMILARITY) == ribbons.MIN_ALPHA
    assert ribbons.tint_for(0.7) == pytest.approx(0.64)

    # A residual sitting on a role vector must take that role's own cluster.
    clusters, similarities, neighbors = ribbons.neighbour_table(
        role_map.vectors[:3], role_map, 5
    )
    record = dict(neighbor_clusters=clusters, neighbor_similarities=similarities)
    ribbons.assign_cells(record, 0.5)
    assert record["clusters"] == [int(role_map.clusters[i]) for i in range(3)]
    assert all(s == pytest.approx(1.0, abs=1e-6) for s in record["similarities"])
    assert all(0 < p <= 1 for p in record["purities"])
    assert [n[0]["label"] for n in neighbors] == role_map.labels[:3]
    assert all(len(n) == 5 for n in neighbors)

    # The threshold, not the ranking, decides whether a cell is coloured at all:
    # the same neighbours give a cluster below the bar and white above it.
    assert ribbons.cell_for([2, 2, 1], [0.61, 0.55, 0.52], 0.5) == (2, 0.61, 2 / 3)
    assert ribbons.cell_for([2, 2, 1], [0.61, 0.55, 0.52], 0.7) == (None, 0.61, 0.0)
    # Only qualifying neighbours are counted, and the tint is their best.
    assert ribbons.cell_for([2, 2, 1], [0.44, 0.42, 0.58], 0.5) == (1, 0.58, 1 / 3)


def test_cluster_colors_follow_membership_not_ward_numbering(role_map):
    from tsne_lens.palette import CLUSTER_COLORS, cluster_slots, map_colors, soften

    slots = cluster_slots(
        role_map.vectors, role_map.labels, role_map.clusters, role_map.config["metric"]
    )
    assert sorted(slots.values()) == list(range(3))

    # Renumbering Ward's clusters must not move a colour: the same members keep
    # the same slot, which is what makes two runs of the same map comparable.
    shuffled = np.asarray([(int(c) + 1) % 3 for c in role_map.clusters])
    moved = cluster_slots(
        role_map.vectors, role_map.labels, shuffled, role_map.config["metric"]
    )
    assert {slots[int(c)] for c in role_map.clusters} == {
        moved[int(c)] for c in shuffled
    }
    for original, renamed in zip(role_map.clusters, shuffled):
        assert slots[int(original)] == moved[int(renamed)]

    colors = map_colors(role_map)
    assert set(colors.values()) <= set(CLUSTER_COLORS)
    assert len(set(colors.values())) == 3
    assert {r["color"] for r in role_map.role_records()} == set(colors.values())
    assert soften("#000000", 0.5) == "#808080"


def test_cluster_ribbons_row_labels():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cluster_ribbons as ribbons

    band = dict(run="band", steering={}, method="character_band", layers=[29],
                coefficient=0.25)
    assert ribbons.row_label(band) == "band · L29 · w=0.25"
    alpha = dict(run="alpha", steering={}, method="additive", layers=[32],
                 coefficient=4.0)
    assert ribbons.row_label(alpha) == "alpha · L32 · α=4"
    assert ribbons.row_label(dict(run="alpha", steering=None)) == "alpha · no steering"
