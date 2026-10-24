"""Selected subsystems are newly inferred models; focus paths are display views."""
import copy
from itertools import combinations, product
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
from scipy.special import digamma

from cimist.ci.mpnn import ResidueStates
from cimist.entropy import S_posterior_mean_std
from cimist.io import load_h5_tree, save_tree_h5
from cimist.mist import MIST
from cimist.sparse_probability import SparseProbability


NAMES = ["ALA1_A", "GLY2_A", "SER1_B", "THR2_B"]
PRIORS = ("haldane", "percs", "jeffreys", "laplace")


def residue_states():
    # Three noisy leaves are conditionally independent given the first residue.
    observations = []
    for center, first, second, third in product((0, 1), repeat=4):
        count = (4 if first == 0 else 1) * (3 if second == 0 else 1) * (2 if third == 0 else 1)
        observations.extend([[center, center ^ first, center ^ second, center ^ third]] * count)
    states = np.asarray(observations, dtype=np.int64).T
    counts = np.stack([np.bincount(row, minlength=4) for row in states])
    return ResidueStates(jnp.asarray((counts > 0).sum(axis=1)),
                         jnp.asarray(states), jnp.asarray(counts))


def select_states(states, rows):
    return ResidueStates(*(value[jnp.asarray(rows, dtype=jnp.int64)] for value in states))


def compact_states(traces):
    traces = np.asarray(traces, dtype=np.int64)
    counts = np.zeros((len(traces), 10), dtype=np.int64)
    number = []
    for row, trace in enumerate(traces):
        labels, frequencies = np.unique(trace, return_counts=True)
        counts[row, :len(labels)] = frequencies
        number.append(len(labels))
    return ResidueStates(jnp.asarray(number), jnp.asarray(traces), jnp.asarray(counts))


def fit(states, names=NAMES, **options):
    tree = MIST.from_residue_states(states, names, pair_batch_size=2, **options)
    for graph in (tree.MI_graph, tree.T):
        graph.graph["coordinate_convention"] = "physical-bond-angles-v1"
        for name in graph:
            row = NAMES.index(name)
            graph.nodes[name].update(residue_index=(10, 11, 42, 43)[row],
                                     chain_index=row // 2, chain_id="AB"[row // 2],
                                     resSeq=1 + row % 2, label=name)
    return tree


class SubsetRefit(unittest.TestCase):
    def assert_value_equal(self, actual, expected, exact=True):
        if isinstance(expected, SparseProbability):
            self.assertIsInstance(actual, SparseProbability)
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual.positions, expected.positions)
            np.testing.assert_array_equal(actual.values, expected.values)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_value_equal(actual[key], expected[key], exact=exact)
        elif isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected, check_exact=exact, rtol=1e-12, atol=1e-12)
        elif isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(actual, expected, check_exact=exact, rtol=1e-12, atol=1e-12)
        elif isinstance(expected, (str, tuple, bool)):
            self.assertEqual(actual, expected)
        elif exact:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def assert_models_equal(self, actual, expected, exact=True):
        ignored = {"T", "MI_graph", "nodes", "edges"}
        self.assert_value_equal({key: value for key, value in vars(actual).items() if key not in ignored},
                                {key: value for key, value in vars(expected).items() if key not in ignored},
                                exact=exact)
        for key in ("MI_graph", "T"):
            left, right = getattr(actual, key), getattr(expected, key)
            self.assertEqual(list(left), list(right))
            self.assertEqual(list(left.edges), list(right.edges))
            self.assertEqual([list(left[node]) for node in left], [list(right[node]) for node in right])
            self.assert_value_equal(left.graph, right.graph, exact=exact)
            for node in right:
                self.assert_value_equal(left.nodes[node], right.nodes[node], exact=exact)
            for edge in right.edges:
                self.assert_value_equal(left.edges[edge], right.edges[edge], exact=exact)

    def test_subset_matches_direct_selected_inference_for_all_settings(self):
        states = residue_states()
        original_states = copy.deepcopy(states)
        selected = NAMES[1:]
        for storage, prior, estimator in product(("dense", "sparse"), PRIORS, ("mle", "bayes")):
            with self.subTest(storage=storage, prior=prior, estimator=estimator):
                settings = dict(probability_storage=storage, prior=prior, entropy_estimator=estimator)
                parent = fit(states, uncertainty=False, **settings)
                before = copy.deepcopy(parent)
                self.assert_models_equal(parent.refit_to_subset(reversed(NAMES)), parent)
                subset = parent.refit_to_subset(selected[::-1], uncertainty=True)
                expected = fit(select_states(states, [1, 2, 3]), selected, uncertainty=True, **settings)
                self.assert_models_equal(subset, expected, exact=False)
                self.assert_models_equal(parent, before)
                self.assertEqual(list(subset.MI_graph), selected)
                self.assertEqual(subset.MI_graph.number_of_edges(), 3)
                self.assertTrue(any(subset.nodes[u]["chain_index"] != subset.nodes[v]["chain_index"]
                                    for u, v in subset.edges))
                self.assertFalse(any(parent.T.has_edge(u, v) for u, v in subset.edges))
                for edge in subset.MI_graph.edges:
                    self.assertIs(subset.MI_graph.edges[edge]["P"], parent.MI_graph.edges[edge]["P"])
        for actual, expected in zip(states, original_states):
            np.testing.assert_array_equal(actual, expected)

    def test_uncertainty_inherits_or_explicitly_overrides_parent(self):
        states = residue_states()
        for parent_uncertainty in (False, True):
            parent = fit(states, uncertainty=parent_uncertainty, probability_storage="sparse")
            inherited = parent.refit_to_subset(NAMES[1:])
            self.assertEqual(inherited.uncertainty, parent_uncertainty)
            for requested in (False, True):
                with self.subTest(parent=parent_uncertainty, requested=requested):
                    subset = parent.refit_to_subset(NAMES[1:], uncertainty=requested)
                    expected = fit(select_states(states, [1, 2, 3]), NAMES[1:],
                                   uncertainty=requested, probability_storage="sparse")
                    self.assert_models_equal(subset, expected)

    def test_refit_discards_parent_tree_caches_without_mutating_them(self):
        parent = fit(residue_states(), probability_storage="sparse", uncertainty=True)
        parent._compute_MI_CL()
        # Stale posterior values must not survive a refit either.
        for _, attrs in parent.MI_graph.nodes(data=True):
            attrs.update(S_pos_mean=999., S_se=100., S_contribution_se=200.)
        for _, _, attrs in parent.MI_graph.edges(data=True):
            attrs.update(I_pos_mean=888., I_se=99.)
        before = copy.deepcopy(parent)
        subset = parent.refit_to_subset(NAMES[1:])
        expected = fit(select_states(residue_states(), [1, 2, 3]), NAMES[1:],
                       uncertainty=True, probability_storage="sparse")
        self.assert_models_equal(subset, expected)
        self.assert_models_equal(parent, before)
        self.assertFalse(subset._implied_marginals_set)
        self.assertTrue(all("P_CL" not in attrs and "I_CL" not in attrs
                            for _, _, attrs in subset.MI_graph.edges(data=True)))
        subset.MI_graph.graph["coordinate_convention"] = "changed"
        subset.MI_graph.nodes[NAMES[1]]["label"] = "changed"
        subset.MI_graph.edges[NAMES[1], NAMES[2]]["I"] = -1.
        self.assert_models_equal(parent, before)

    def test_canonical_parent_order_controls_tied_mst(self):
        graph = nx.Graph()
        names = [f"residue-{index}" for index in range(8)]
        graph.add_nodes_from((name, dict(S=np.log(2.), p=jnp.asarray([.5, .5]), N_states=2))
                            for name in names)
        # Deliberately reverse candidate-edge insertion order.
        probability = jnp.asarray([[.4, .1], [.1, .4]])
        for first, second in reversed(list(combinations(names, 2))):
            graph.add_edge(first, second, I=.2, P=probability, axes=(first, second))
        parent = MIST(graph, N_obs=100, entropy_estimator="mle")
        for indices in combinations(range(8), 3):
            selected = [names[index] for index in indices]
            subset = parent.refit_to_subset(reversed(selected))
            self.assertEqual(list(subset.MI_graph), selected)
            expected_edges = [(u, v) for u, v in parent.MI_graph.edges if u in selected and v in selected]
            self.assertEqual(list(subset.MI_graph.edges), expected_edges)
            expected = nx.Graph()
            expected.add_nodes_from(selected)
            expected.add_edges_from((u, v, parent.MI_graph.edges[u, v]) for u, v in expected_edges)
            self.assertEqual(list(subset.edges), list(nx.maximum_spanning_tree(expected, weight="I").edges))

    def test_singleton_has_marginal_entropy_and_explicit_deletion_uncertainty(self):
        states = residue_states()
        for storage, prior, estimator in product(("dense", "sparse"), PRIORS, ("mle", "bayes")):
            with self.subTest(storage=storage, prior=prior, estimator=estimator):
                settings = dict(probability_storage=storage, prior=prior, entropy_estimator=estimator)
                parent = fit(states, uncertainty=True, **settings)
                subset = parent.refit_to_subset([NAMES[2]])
                direct = fit(select_states(states, [2]), [NAMES[2]], uncertainty=True, **settings)
                self.assertEqual(list(subset.nodes), [NAMES[2]])
                self.assertEqual(subset.MI_graph.number_of_edges(), 0)
                self.assertEqual(subset.T.number_of_edges(), 0)
                counts = np.asarray(states.state_counts[2])[:2].astype(float)
                pseudocount = {"haldane": 0., "percs": .5, "jeffreys": .5, "laplace": 1.}[prior]
                alpha = counts + pseudocount
                posterior_mean = digamma(alpha.sum() + 1) - np.dot(alpha, digamma(alpha + 1)) / alpha.sum()
                self.assertAlmostEqual(float(subset.nodes[NAMES[2]]["S_pos_mean"]), posterior_mean, places=12)
                self.assertAlmostEqual(float(subset.entropy()),
                                       posterior_mean if estimator == "bayes" else np.log(2.), places=12)
                marginal_se = float(S_posterior_mean_std(jnp.asarray(alpha))[1])
                bias = np.log(2.) - posterior_mean
                self.assertAlmostEqual(float(subset.counts_variance), marginal_se**2, places=12)
                self.assertEqual(float(subset.structural_variance), 0.)
                self.assertAlmostEqual(float(subset.entropy_se), np.sqrt(marginal_se**2 + bias**2), places=12)
                np.testing.assert_array_equal(subset.S_jack_entropies, [0.])
                self.assertEqual(subset.S_jack_contrib.shape, (1, 1))
                self.assertTrue(np.isnan(subset.S_jack_contrib.iloc[0, 0]))
                self.assertTrue(np.isnan(subset.nodes[NAMES[2]]["S_contribution_se"]))
                np.testing.assert_allclose([subset.entropy(), subset.entropy_se],
                                           [direct.entropy(), direct.entropy_se], rtol=1e-12, atol=1e-12)

    def test_invalid_selections_fail_clearly(self):
        parent = fit(residue_states(), probability_storage="sparse")
        for method in (parent.refit_to_subset, parent.trim):
            for selection, message in (([], "empty"), ([NAMES[0], NAMES[0]], "duplicate"),
                                       (["missing"], "Unknown")):
                with self.subTest(method=method.__name__, selection=selection):
                    with self.assertRaisesRegex(ValueError, message):
                        method(selection)
        with self.assertRaisesRegex(ValueError, "empty"):
            MIST.from_residue_states(select_states(residue_states(), []), [])

    def test_selection_metadata_tracks_subsystem_and_display_provenance(self):
        parent = fit(residue_states(), probability_storage="sparse")
        metadata = dict(semantics="marginal", expression="protein", seed=0,
                        residue_indices=[10, 11, 42, 43],
                        protein_residue_indices=[10, 11, 42, 43],
                        initialization="full-protein-split")
        for graph in (parent.MI_graph, parent.T):
            graph.graph["residue_selection"] = copy.deepcopy(metadata)
        before = copy.deepcopy(parent)
        subset = parent.refit_to_subset(NAMES[1:3][::-1])
        for graph in (subset.MI_graph, subset.T):
            selection = graph.graph["residue_selection"]
            self.assertEqual(selection["residue_indices"], [11, 42])
            self.assertEqual(selection["semantics"], "marginal")
            self.assertIsNone(selection["expression"])
            self.assertEqual(selection["refit_from"],
                             dict(expression="protein", residue_indices=[10, 11, 42, 43]))
            for key in ("seed", "initialization", "protein_residue_indices"):
                self.assertEqual(selection[key], metadata[key])
        view = parent.trim(NAMES[1:3][::-1])
        self.assertEqual(view.graph["residue_selection"], metadata)
        self.assertEqual(view.graph["focus_selection"],
                         dict(semantics="display_only", nodes=NAMES[1:3], residue_indices=[11, 42]))
        subset.MI_graph.graph["residue_selection"]["protein_residue_indices"].append(999)
        view.graph["residue_selection"]["residue_indices"].append(999)
        self.assert_models_equal(parent, before)

    def test_omitted_high_labels_only_change_zero_padding(self):
        rng = np.random.default_rng(7620)
        first = rng.choice([0, 1, 4], 137, p=[.55, .3, .15])
        second = first.copy()
        second[::4] = rng.choice([0, 1, 4], len(second[::4]))
        third = rng.choice([0, 1, 4], 137)
        # Permutations of the same states give mathematically tied pair MIs.
        tied = np.array([0] * 7 + [1] * 5 + [4] * 3)
        cases = (np.asarray([first, second, third]),
                 np.asarray([tied, np.choose(tied, [1, 4, 2, 3, 0]),
                             np.choose(tied, [4, 1, 2, 3, 0])]))
        for case, traces in enumerate(cases):
            states = compact_states(np.vstack((traces, np.arange(traces.shape[1]) % 9)))
            original = copy.deepcopy(states)
            for storage, prior, estimator in product(("dense", "sparse"), PRIORS, ("mle", "bayes")):
                with self.subTest(case=case, storage=storage, prior=prior, estimator=estimator):
                    settings = dict(probability_storage=storage, prior=prior,
                                    entropy_estimator=estimator, uncertainty=True)
                    parent = fit(states, **settings)
                    subset = parent.refit_to_subset(NAMES[:3])
                    direct = fit(select_states(states, [0, 1, 2]), NAMES[:3], **settings)
                    for edge in subset.MI_graph.edges:
                        left, right = subset.MI_graph.edges[edge], direct.MI_graph.edges[edge]
                        full_p = left["P"].dense_numpy() if storage == "sparse" else np.asarray(left["P"])
                        selected_p = right["P"].dense_numpy() if storage == "sparse" else np.asarray(right["P"])
                        self.assertEqual(full_p.shape, (9, 9))
                        self.assertEqual(selected_p.shape, (5, 5))
                        np.testing.assert_array_equal(full_p, np.pad(selected_p, ((0, 4), (0, 4))))
                        np.testing.assert_array_equal(full_p[[2, 3, 5, 6, 7, 8]], 0.)
                        self.assertAlmostEqual(left["I"], right["I"], places=12)
                    if case == 0:
                        self.assertEqual(list(subset.edges), list(direct.edges))
                    else:
                        # Zero-padding may perturb exact MI ties by roundoff.
                        alternate_weight = sum(direct.MI_graph.edges[edge]["I"] for edge in subset.edges)
                        self.assertAlmostEqual(alternate_weight, direct.T.size(weight="I"), places=12)
                    for name in ("entropy_se", "counts_variance", "structural_variance", "entropy_mle_bias"):
                        self.assertAlmostEqual(float(getattr(subset, name)), float(getattr(direct, name)), places=12)
                    self.assertAlmostEqual(float(subset.entropy()), float(direct.entropy()), places=12)
            for actual, expected in zip(states, original):
                np.testing.assert_array_equal(actual, expected)

    def test_singleton_preserves_nonzero_gappy_state_labels_through_export(self):
        traces = [np.array([1] * 7 + [5] * 13),
                  np.arange(20) % 2, np.arange(20) % 3, np.arange(20) % 9]
        states = compact_states(traces)
        for storage in ("dense", "sparse"):
            parent = fit(states, probability_storage=storage)
            subset = parent.refit_to_subset([NAMES[0]])
            direct = fit(select_states(states, [0]), [NAMES[0]], probability_storage=storage)
            expected = np.bincount(traces[0], minlength=6) / 20.
            np.testing.assert_array_equal(direct.nodes[NAMES[0]]["p"], expected)
            np.testing.assert_array_equal(subset.nodes[NAMES[0]]["p"], np.pad(expected, (0, 3)))
            np.testing.assert_array_equal(states.state_counts[0, :2], [7, 13])
            self.assertAlmostEqual(float(subset.entropy()), float(direct.entropy()), places=12)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "gappy-singleton.h5"
                save_tree_h5(direct, path)
                restored = load_h5_tree(path)
                np.testing.assert_array_equal(restored.nodes[NAMES[0]]["p"], expected)
                self.assert_models_equal(restored, direct)

    def test_focus_paths_retain_connectors_and_original_contributions(self):
        parent = fit(residue_states(), probability_storage="sparse")
        before = copy.deepcopy(parent)
        selected = NAMES[1:3]
        full_contributions = parent.residue_entropies() - .5 * parent.residue_sum_MIs()
        with patch("networkx.shortest_path", side_effect=AssertionError("all-pair path search")):
            view = parent.trim(selected[::-1])
        self.assertIsInstance(view, nx.Graph)
        self.assertNotIsInstance(view, MIST)
        self.assertEqual(list(view), NAMES[:3])
        self.assertEqual(set(view.edges), {(NAMES[0], NAMES[1]), (NAMES[0], NAMES[2])})
        self.assertEqual(view.graph["view_kind"], "full_tree_paths")
        for node in view:
            self.assertEqual(view.nodes[node]["is_selected"], node in selected)
            self.assertEqual(view.nodes[node]["full_model_entropy_contribution"], float(full_contributions[node]))
        singleton = parent.trim([NAMES[1]])
        self.assertEqual(list(singleton), [NAMES[1]])
        self.assertEqual(singleton.number_of_edges(), 0)
        self.assertLess(singleton.nodes[NAMES[1]]["full_model_entropy_contribution"],
                        singleton.nodes[NAMES[1]][parent._S_key])
        view.nodes[NAMES[0]]["label"] = "changed"
        self.assert_models_equal(parent, before)

    def test_subset_export_roundtrip_preserves_identities_and_statistics(self):
        parent = fit(residue_states(), probability_storage="sparse", prior="laplace")
        for selected in (NAMES[1:], [NAMES[2]]):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as folder:
                subset = parent.refit_to_subset(selected)
                path = Path(folder) / "subset.h5"
                save_tree_h5(subset, path)
                restored = load_h5_tree(path)
                self.assert_models_equal(restored, subset)
                self.assertEqual([restored.nodes[node]["residue_index"] for node in restored.nodes],
                                 [parent.nodes[node]["residue_index"] for node in selected])


if __name__ == "__main__":
    unittest.main()
