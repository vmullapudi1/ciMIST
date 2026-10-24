"""Observed and tree-implied information remain distinct after residue selection."""
from itertools import combinations, product
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist import mist
from cimist.io import load_h5_tree, save_tree_h5


NAMES = ["ALA1_A", "GLY2_A", "SER1_B"]
# AB and BC are the unique maximum-information tree; AC has a remaining
# empirical dependence that the AB--BC path does not reproduce.
COUNTS = np.array([[[37, 8], [2, 3]], [[3, 2], [8, 37]]])
PROBABILITY = COUNTS / COUNTS.sum()


def entropy(probability):
    positive = probability[probability > 0]
    return -np.sum(positive * np.log(positive))


def information(probability):
    return entropy(probability.sum(axis=1)) + entropy(probability.sum(axis=0)) - entropy(probability)


def states():
    observations = np.repeat(np.asarray(list(product((0, 1), repeat=3))), COUNTS.ravel(), axis=0).T
    # Preserve nonconsecutive raw labels and compact native DBSCAN counts.
    traces = np.stack([np.asarray(labels)[row] for row, labels in
                       zip(observations, ([0, 4], [1, 3], [2, 5]))])
    counts = np.zeros((3, 6), dtype=int)
    for i, row in enumerate(traces):
        counts[i, :2] = np.unique(row, return_counts=True)[1]
    return SimpleNamespace(states_traj=jnp.asarray(traces), state_counts=jnp.asarray(counts),
                           n_states=jnp.asarray([2, 2, 2]))


def model(storage="sparse", estimator="bayes", prior="haldane", singleton=False):
    observed = states()
    names = NAMES
    if singleton:
        observed = SimpleNamespace(**{key: value[2:] for key, value in vars(observed).items()})
        names = NAMES[2:]
    return mist.MIST.from_residue_states(observed, names, probability_storage=storage,
                                        entropy_estimator=estimator, prior=prior, uncertainty=False,
                                        pair_batch_size=2)


def expected_matrices():
    empirical = np.zeros((3, 3))
    for i, j in combinations(range(3), 2):
        joint = PROBABILITY.sum(axis=next(k for k in range(3) if k not in (i, j)))
        empirical[i, j] = empirical[j, i] = information(joint)
    implied = empirical.copy()
    ab, bc = PROBABILITY.sum(axis=2), PROBABILITY.sum(axis=0)
    # Independently marginalize p(a,b,c) = p(a,b) p(b,c) / p(b).
    ac = sum(np.outer(ab[:, b], bc[b, :]) / bc[b].sum() for b in range(2))
    implied[0, 2] = implied[2, 0] = information(ac)
    return empirical, implied


class InformationMatrix(unittest.TestCase):
    def test_empirical_and_path_marginals_for_both_estimators_and_storage(self):
        empirical, implied = expected_matrices()
        self.assertGreater(abs(empirical[0, 2] - implied[0, 2]), .01)
        for storage, estimator, prior in product(("dense", "sparse"), ("mle", "bayes"),
                                                 ("haldane", "percs", "jeffreys", "laplace")):
            with self.subTest(storage=storage, estimator=estimator, prior=prior):
                tree = model(storage, estimator, prior)
                self.assertEqual(set(map(frozenset, tree.T.edges)),
                                 {frozenset(NAMES[:2]), frozenset(NAMES[1:])})
                np.testing.assert_allclose(tree.empirical_information_matrix(), empirical, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(tree.information_matrix(), empirical, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(tree.chow_liu_information_matrix(), implied, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(tree.information_matrix(kind="chow_liu"), implied, rtol=1e-12, atol=1e-12)
                if estimator == "bayes":
                    self.assertTrue(np.isnan(tree.MI_graph[NAMES[0]][NAMES[2]]["I_pos_mean"]))
                    self.assertGreater(abs(float(tree.tree_MIs().iloc[0]) - empirical[0, 1]), 1e-4)
                self.assertFalse(tree._implied_marginals_set)
                for _, _, attrs in tree.MI_graph.edges(data=True):
                    self.assertNotIn("P_CL", attrs)
                    self.assertNotIn("I_CL", attrs)

    def test_selected_matrix_uses_full_path_and_canonical_order(self):
        tree = model()
        empirical, implied = expected_matrices()
        selected = [NAMES[2], NAMES[0]]
        indices = np.ix_([0, 2], [0, 2])
        for diag in (False, True):
            expected_empirical, expected_implied = empirical[indices].copy(), implied[indices].copy()
            if diag:
                np.fill_diagonal(expected_empirical, np.log(2))
                np.fill_diagonal(expected_implied, np.log(2))
            with patch.object(mist, "joint_distribution", wraps=mist.joint_distribution) as joint:
                actual = tree.chow_liu_information_matrix(selected, diag=diag)
            joint.assert_called_once_with(tree, NAMES[0], NAMES[2])
            np.testing.assert_allclose(actual, expected_implied, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(tree.empirical_information_matrix(selected, diag=diag),
                                       expected_empirical, rtol=1e-12, atol=1e-12)
            hybrid = np.triu(expected_empirical) + np.tril(expected_implied, k=-1)
            np.testing.assert_allclose(tree.information_matrix(iter(selected), "hybrid", diag), hybrid,
                                       rtol=1e-12, atol=1e-12)
        # Selecting matrix rows retains the intermediate residue; refitting
        # instead connects the retained empirical AC pair in a new tree.
        subset = tree.refit_to_subset(selected)
        np.testing.assert_allclose(subset.chow_liu_information_matrix(), empirical[indices], rtol=1e-12, atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.h5"
            save_tree_h5(subset, path)
            restored = load_h5_tree(path)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                np.testing.assert_allclose(restored.information_matrix(kind="hybrid"), empirical[indices],
                                           rtol=1e-12, atol=1e-12)

    def test_singletons_diagonals_and_invalid_arguments(self):
        for storage, estimator in product(("dense", "sparse"), ("mle", "bayes")):
            parent = model(storage, estimator)
            trees = (parent, model(storage, estimator, singleton=True), parent.refit_to_subset(NAMES[2:]))
            for tree in trees:
                for kind, diag in product(("empirical", "chow_liu", "hybrid"), (False, True)):
                    with self.subTest(storage=storage, estimator=estimator, kind=kind, diag=diag):
                        with patch.object(mist, "joint_distribution", side_effect=AssertionError("No pair")):
                            actual = tree.information_matrix(NAMES[2:], kind=kind, diag=diag)
                        np.testing.assert_allclose(actual, [[np.log(2) if diag else 0.]], rtol=1e-12, atol=1e-12)
            for method in (parent.information_matrix, parent.empirical_information_matrix,
                           parent.chow_liu_information_matrix):
                for selection, message in (([], "empty"), ([NAMES[0], NAMES[0]], "duplicate"),
                                           (["unknown"], "Unknown")):
                    with self.assertRaisesRegex(ValueError, message):
                        method(selection)
            with self.assertRaisesRegex(ValueError, "kind"):
                parent.information_matrix(kind="posterior")

    def test_cl_does_not_populate_or_reuse_graph_caches(self):
        tree = model()
        _, implied = expected_matrices()
        probabilities = {(u, v): attrs["P"] for u, v, attrs in tree.MI_graph.edges(data=True)}
        with patch.object(tree, "_compute_implied_pairwise_marginals", side_effect=AssertionError("Full cache")), \
             patch.object(tree, "_compute_MI_CL", side_effect=AssertionError("Full cache")):
            np.testing.assert_allclose(tree.chow_liu_information_matrix(), implied, rtol=1e-12, atol=1e-12)
        for edge, probability in probabilities.items():
            self.assertIs(tree.MI_graph.edges[edge]["P"], probability)
            self.assertNotIn("I_CL", tree.MI_graph.edges[edge])
            self.assertNotIn("P_CL", tree.MI_graph.edges[edge])
        # Legacy optional caches can describe an edge-knockout tree. They must
        # neither override current-tree semantics nor be modified by a view.
        cached = np.full((6, 6), .5)
        for _, _, attrs in tree.MI_graph.edges(data=True):
            attrs.update(I_CL=42., P_CL=cached)
        tree._implied_marginals_set = True
        np.testing.assert_allclose(tree.chow_liu_information_matrix(), implied, rtol=1e-12, atol=1e-12)
        for _, _, attrs in tree.MI_graph.edges(data=True):
            self.assertEqual(attrs["I_CL"], 42.)
            self.assertIs(attrs["P_CL"], cached)
        self.assertTrue(tree._implied_marginals_set)


if __name__ == "__main__":
    unittest.main()
