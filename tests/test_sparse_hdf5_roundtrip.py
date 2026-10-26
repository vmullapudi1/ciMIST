"""Scientific snapshots round-trip without refitting or probability expansion."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd

from cimist import io, mist
from cimist.ci.mpnn import ResidueStates
from cimist.sparse_probability import SparseProbability


def assert_value_equal(actual, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_value_equal(actual[key], expected[key])
    else:
        first, second = np.asarray(actual), np.asarray(expected)
        assert first.dtype == second.dtype, (first.dtype, second.dtype)
        np.testing.assert_array_equal(first, second)
        if second.dtype.kind in "biufc":
            assert first.tobytes() == second.tobytes()


def assert_model_equal(actual, expected):
    assert set(vars(actual)) == set(vars(expected))
    for key in vars(expected):
        if key not in ("T", "MI_graph", "nodes", "edges"):
            assert_value_equal(getattr(actual, key), getattr(expected, key))
    assert actual.nodes is actual.T.nodes
    assert actual.edges is actual.T.edges
    for name in ("T", "MI_graph"):
        first, second = getattr(actual, name), getattr(expected, name)
        assert list(first) == list(second)
        assert list(first.edges) == list(second.edges)
        assert [list(first[n]) for n in first] == [list(second[n]) for n in second]
        assert_value_equal(first.graph, second.graph)
        for node in second:
            assert_value_equal(first.nodes[node], second.nodes[node])
        for edge in second.edges:
            a, b = first.edges[edge], second.edges[edge]
            assert set(a) == set(b)
            for key in b:
                if key == "P":
                    actual_p = a[key].dense_numpy() if isinstance(a[key], SparseProbability) else a[key]
                    expected_p = b[key].dense_numpy() if isinstance(b[key], SparseProbability) else b[key]
                    assert_value_equal(actual_p, expected_p)
                else:
                    assert_value_equal(a[key], b[key])


def state_model(prior, estimator, uncertainty=True, storage="sparse"):
    labels = np.random.default_rng(7620).integers(0, 4, (5, 43), dtype=np.int64)
    labels[1] = labels[0]  # Tied dependence, with original component labels.
    labels[2] = 0
    labels[3] = np.where(labels[3] < 2, 0, 3)
    counts = np.stack([np.bincount(row, minlength=4) for row in labels])
    states = ResidueStates(jnp.asarray((counts > 0).sum(axis=1)),
                           jnp.asarray(labels), jnp.asarray(counts))
    return mist.MIST.from_residue_states(
        states, ["ALA1_A", "GLY2_A", "ALA1_B", "GLY2_B", "SER3_B"],
        prior=prior, entropy_estimator=estimator, uncertainty=uncertainty,
        pair_batch_size=3, probability_storage=storage)


def ordered_disconnected_model():
    graph = nx.Graph()
    for node in range(7):
        p = jnp.asarray([0.2 + node / 20, 0.8 - node / 20])
        graph.add_node(f"r{node}", S=float(-jnp.sum(p * jnp.log(p))),
                       p=p, N_states=2, residue_index=node, chain_id="A")
    for u, v in ((2, 3), (1, 2), (0, 1), (0, 2), (4, 5)):
        a, b = f"r{u}", f"r{v}"
        probability = np.outer(graph.nodes[a]["p"], graph.nodes[b]["p"])
        graph.add_edge(a, b, I=0.0, axes=(b, a),
                       P=SparseProbability.encode(probability.T))
    model = mist.MIST(graph, N_obs=80, uncertainty=True, prior="jeffreys")
    for graph in (model.T, model.MI_graph):
        graph.graph.update(coordinate_convention="physical-bond-angles-v1", run_seed=7620)
    # Both graphs now have order that replaying Graph.edges cannot recover.
    model.T._adj["r2"] = dict(reversed(list(model.T._adj["r2"].items())))
    model.MI_graph.nodes["r0"]["p"] = jnp.asarray([0.375, 0.625])
    model.MI_graph.nodes["r0"].pop("variance")
    model.MI_graph.nodes["r1"]["variance"] = jnp.asarray(np.nan)
    edge = next(iter(model.T.edges))
    model.T.edges[edge].pop("bias")
    model.T.edges[edge]["I_CL"] = np.nan
    model.MI_graph.edges[edge]["P_CL"] = np.asarray([[0.125, 0.125], [0.25, 0.5]])
    model.S_jack_contrib.index.name = "residue"
    model.S_jack_contrib.columns.name = "deleted_residue_position"
    return model


class SparseHDF5Roundtrip(unittest.TestCase):
    def roundtrip(self, model):
        before = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.h5"
            io.save_tree_h5(model, path)
            with patch.object(mist.MIST, "__init__", side_effect=AssertionError("must restore without constructor")), \
                    patch.object(mist.MIST, "fit", side_effect=AssertionError("must not refit")), \
                    patch.object(mist, "compute_error", side_effect=AssertionError("must not recompute uncertainty")):
                restored = io.load_h5_tree(path)
            for graph in (restored.T, restored.MI_graph):
                self.assertTrue(all(isinstance(data["P"], SparseProbability)
                                    for _, _, data in graph.edges(data=True)))
            assert_model_equal(restored, before)
            np.testing.assert_array_equal(restored.entropy(), before.entropy())
            pd.testing.assert_series_equal(restored.residue_entropies(), before.residue_entropies(), check_exact=True)
            # Native numeric metadata replaces JAX scalar objects; preserve the
            # ordered numbers without requiring pandas' former object dtype.
            actual, expected = restored.residue_sum_MIs(), before.residue_sum_MIs()
            pd.testing.assert_index_equal(actual.index, expected.index)
            assert_value_equal(actual.to_numpy(dtype=np.float64), expected.to_numpy(dtype=np.float64))
        # Serialization must not change either graph or uncertainty results.
        assert_model_equal(model, before)

    def test_all_priors_estimators_and_complete_uncertainty(self):
        for prior in ("haldane", "percs", "jeffreys", "laplace"):
            for estimator in ("mle", "bayes"):
                with self.subTest(prior=prior, estimator=estimator):
                    self.roundtrip(state_model(prior, estimator))

    def test_missing_uncertainty_fields_and_nan_placeholders(self):
        for estimator in ("mle", "bayes"):
            with self.subTest(estimator=estimator):
                model = state_model("haldane", estimator, uncertainty=False)
                self.assertNotIn("S_jack_entropies", vars(model))
                self.assertNotIn("rmse", vars(model))
                self.roundtrip(model)

    def test_dense_input_preserves_padded_zeros_as_sparse_probabilities(self):
        self.roundtrip(state_model("haldane", "bayes", storage="dense"))

    def test_reversed_axes_adjacency_order_and_separate_graph_marginals(self):
        model = ordered_disconnected_model()
        self.assertFalse(nx.is_connected(model.MI_graph))
        self.assertEqual(model.MI_graph.degree("r6"), 0)
        self.assertFalse(np.array_equal(model.T.nodes["r0"]["p"], model.MI_graph.nodes["r0"]["p"]))
        self.roundtrip(model)


if __name__ == "__main__":
    unittest.main()
