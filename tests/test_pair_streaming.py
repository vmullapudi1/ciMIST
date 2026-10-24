"""All candidate pairs and posterior results survive sparse host batching."""
from itertools import combinations
from types import SimpleNamespace
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist.mist import MIST
from cimist.sparse_probability import dense_probability


def synthetic_states(residues=5, frames=48, seed=20261021):
    labels = np.random.default_rng(seed).integers(0, 3, (residues, frames))
    if residues > 2:
        labels[2] = labels[0]
    counts = np.stack([np.bincount(row, minlength=3) for row in labels])
    return SimpleNamespace(states_traj=jnp.asarray(labels),
                           state_counts=jnp.asarray(counts),
                           n_states=jnp.asarray((counts > 0).sum(axis=1)))


class PairStreaming(unittest.TestCase):
    def test_joint_histograms_match_numpy_for_narrow_and_wide_labels(self):
        rng = np.random.default_rng(20261021)
        labels = rng.choice([0, 7, 31, 99], (5, 41))
        labels[1] = labels[0]
        labels[2] = 0
        counts = np.stack([np.bincount(row, minlength=100) for row in labels])
        for dtype in (np.int8, np.int32, np.int64):
            with self.subTest(dtype=dtype):
                states = SimpleNamespace(states_traj=jnp.asarray(labels.astype(dtype)),
                    state_counts=jnp.asarray(counts), n_states=jnp.asarray((counts > 0).sum(axis=1)))
                tree = MIST.from_residue_states(states, list(range(len(labels))), entropy_estimator="mle",
                    uncertainty=False, pair_batch_size=3, probability_storage="sparse")
                for i, j in combinations(range(len(labels)), 2):
                    expected = np.bincount(labels[i] * 100 + labels[j], minlength=10000).reshape(100, 100)
                    expected = expected / labels.shape[1]
                    np.testing.assert_array_equal(dense_probability(tree.MI_graph.edges[i, j]["P"]), expected)

    def test_complete_graph_order_and_uncertainty_across_batches(self):
        states = synthetic_states()
        names = ["ALA1_A", "ALA2_A", "ALA1_B", "ALA2_B", "GLY3_B"]
        for prior in ("haldane", "percs", "jeffreys", "laplace"):
            expected = MIST.from_residue_states(states, names, prior=prior,
                uncertainty=True, pair_batch_size=13, probability_storage="dense")
            for batch_size in (3, 13):
                with self.subTest(prior=prior, batch_size=batch_size):
                    actual = MIST.from_residue_states(states, names, prior=prior,
                        uncertainty=True, pair_batch_size=batch_size, probability_storage="sparse")
                    self.assertEqual(list(actual.MI_graph.edges), list(combinations(names, 2)))
                    for expected_graph, actual_graph in ((expected.MI_graph, actual.MI_graph),
                                                         (expected.T, actual.T)):
                        self.assertEqual(list(actual_graph.nodes), list(expected_graph.nodes))
                        self.assertEqual(list(actual_graph.edges), list(expected_graph.edges))
                        for node in expected_graph.nodes:
                            for key, value in expected_graph.nodes[node].items():
                                np.testing.assert_allclose(actual_graph.nodes[node][key], value,
                                                           rtol=1e-8, atol=1e-10)
                        for edge in expected_graph.edges:
                            for key, value in expected_graph.edges[edge].items():
                                observed = actual_graph.edges[edge][key]
                                if key == "P":
                                    observed, value = dense_probability(observed), dense_probability(value)
                                if key == "axes":
                                    self.assertEqual(observed, value)
                                else:
                                    np.testing.assert_allclose(observed, value, rtol=1e-8, atol=1e-10)
                    for key in ("entropy_se", "entropy_mle_bias", "counts_variance", "structural_variance"):
                        np.testing.assert_allclose(getattr(actual, key), getattr(expected, key),
                                                   rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(actual.entropy(), expected.entropy(), rtol=1e-8, atol=1e-10)
                    for edge in actual.MI_graph.edges:
                        if edge not in actual.T.edges:
                            for key in ("I_pos_mean", "I_se"):
                                self.assertFalse(isinstance(actual.MI_graph.edges[edge][key], jax.Array))
                                self.assertTrue(np.isnan(actual.MI_graph.edges[edge][key]))


if __name__ == "__main__":
    unittest.main()
