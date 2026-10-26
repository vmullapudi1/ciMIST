"""Tree probabilities retain raw state labels, including singleton models."""
from types import SimpleNamespace
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import networkx as nx
import numpy as np

from cimist.mist import MIST, get_log_prob_and_sampling_fns, joint_distribution


TRACES = np.array([[0, 0, 0, 0, 4, 4, 4, 4],
                   [2, 2, 4, 4, 4, 4, 4, 4]])
NAMES = ["first", "retained"]


def model(traces, names, storage):
    # Native DBSCAN exports compact counts separately from raw trajectory labels.
    counts = np.zeros((len(traces), 5), dtype=int)
    for row, trace in enumerate(traces):
        frequencies = np.unique(trace, return_counts=True)[1]
        counts[row, :len(frequencies)] = frequencies
    states = SimpleNamespace(states_traj=jnp.asarray(traces),
                             state_counts=jnp.asarray(counts),
                             n_states=jnp.asarray((counts > 0).sum(axis=1)))
    return MIST.from_residue_states(states, names, probability_storage=storage,
                                   entropy_estimator="mle", uncertainty=False)


class TreeProbabilities(unittest.TestCase):
    def assert_probabilities(self, log_prob, barcodes, expected):
        values = np.asarray(jax.vmap(log_prob)(jnp.asarray(barcodes)))
        np.testing.assert_array_equal(np.isneginf(values), expected == 0)
        self.assertFalse(np.isnan(values).any())
        probabilities = np.exp(values)
        np.testing.assert_allclose(probabilities, expected, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(probabilities.sum()), 1., places=12)

    def assert_valid_samples(self, sample, probability):
        keys = jax.random.split(jax.random.PRNGKey(20261024), 32)
        first = np.asarray(jax.vmap(sample)(keys))
        second = np.asarray(jax.vmap(sample)(keys))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (32, probability.ndim))
        self.assertTrue(np.issubdtype(first.dtype, np.integer))
        self.assertTrue((first >= 0).all())
        self.assertTrue((first < np.asarray(probability.shape)).all())
        self.assertTrue((probability[tuple(first.T)] > 0).all())

    def test_direct_and_refitted_singletons_keep_gappy_support(self):
        expected = np.array([0., 0., .25, 0., .75])
        for storage in ("dense", "sparse"):
            parent = model(TRACES, NAMES, storage)
            direct = model(TRACES[1:], NAMES[1:], storage)
            refitted = parent.refit_to_subset(NAMES[1:])
            for source, singleton in (("direct", direct), ("refitted", refitted)):
                with self.subTest(storage=storage, source=source):
                    np.testing.assert_array_equal(singleton.nodes["retained"]["p"], expected)
                    matrices, edges, log_prob, sample = get_log_prob_and_sampling_fns(singleton)
                    self.assertEqual(matrices, [])
                    self.assertEqual(edges, [])
                    self.assert_probabilities(log_prob, np.arange(5)[:, None], expected)
                    self.assert_valid_samples(sample, expected)
                    np.testing.assert_array_equal(
                        joint_distribution(singleton, "retained", "retained"), np.diag(expected))

                    # The constructor attaches the same usable public functions.
                    attached = MIST(singleton.MI_graph.copy(), N_obs=singleton.N_obs,
                                    entropy_estimator=singleton.entropy_estimator,
                                    prior=singleton.prior, build_probability_model=True)
                    self.assert_probabilities(attached.log_prob, np.arange(5)[:, None], expected)
                    self.assert_valid_samples(attached.sample, expected)

    def test_two_node_probabilities_and_self_pairs_for_either_root(self):
        expected = np.zeros((5, 5))
        expected[0, 2], expected[0, 4], expected[4, 4] = .25, .25, .5
        barcodes = np.indices(expected.shape).reshape(2, -1).T
        for storage in ("dense", "sparse"):
            tree = model(TRACES, NAMES, storage)
            np.testing.assert_array_equal(joint_distribution(tree, *NAMES), expected)
            np.testing.assert_array_equal(joint_distribution(tree, *NAMES[::-1]), expected.T)
            for node in NAMES:
                np.testing.assert_array_equal(joint_distribution(tree, node, node),
                                              np.diag(tree.nodes[node]["p"]))
            for root in NAMES:
                with self.subTest(storage=storage, root=root):
                    matrices, edges, log_prob, sample = get_log_prob_and_sampling_fns(tree, root)
                    self.assertEqual(len(matrices), 1)
                    self.assertEqual(edges, [(root, next(node for node in NAMES if node != root))])
                    self.assert_probabilities(log_prob, barcodes, expected.ravel())
                    self.assert_valid_samples(sample, expected)

    def test_disconnected_joint_factorizes_without_changing_empirical_pair(self):
        for storage in ("dense", "sparse"):
            with self.subTest(storage=storage):
                tree = model(TRACES, NAMES, storage)
                measured = tree.MI_graph.edges[tuple(NAMES)]["P"]
                tree.T.remove_edge(*NAMES)
                first, second = (np.asarray(tree.nodes[node]["p"]) for node in NAMES)
                expected = np.outer(first, second)
                np.testing.assert_array_equal(joint_distribution(tree, *NAMES), expected)
                np.testing.assert_array_equal(joint_distribution(tree, *NAMES[::-1]), expected.T)
                np.testing.assert_array_equal(expected.sum(axis=1), first)
                np.testing.assert_array_equal(expected.sum(axis=0), second)
                self.assertIs(tree.MI_graph.edges[tuple(NAMES)]["P"], measured)
                self.assertEqual(tree.T.number_of_edges(), 0)
                for nodes in ((NAMES[0], "unknown"), ("unknown", NAMES[1])):
                    with self.assertRaises(nx.NodeNotFound):
                        joint_distribution(tree, *nodes)


if __name__ == "__main__":
    unittest.main()
