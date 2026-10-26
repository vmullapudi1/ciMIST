"""Marginal selection preserves retained native fits, states and information."""
from contextlib import redirect_stdout
import io
from itertools import combinations
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

from cimist import fit
from cimist.ci import vmm
from cimist.io import load_h5_tree
from cimist.sparse_probability import dense_probability
from test_multichain_pipeline import two_chain_trajectory


SEED = 13
SELECTION = "(chainid 0 and resSeq 2) or chainid 1"
SELECTED = [1, 2, 3]
# Float64 reductions can differ across full and partial residue batches.
RTOL, ATOL = 1e-8, 1e-10


def mobile_two_chain_trajectory(frames=256):
    """Two correlated torsions per chain, with small within-state noise."""
    trajectory = two_chain_trajectory(frames)
    xyz = trajectory.xyz.astype(np.float64)

    def rotate(frame, left, right, atoms, angle):
        origin = xyz[frame, left]
        axis = xyz[frame, right] - origin
        axis /= np.linalg.norm(axis)
        points = xyz[frame, atoms] - origin
        cosine, sine = np.cos(angle), np.sin(angle)
        xyz[frame, atoms] = (origin + cosine * points
                             + sine * np.cross(axis, points)
                             + (1 - cosine) * (points @ axis)[:, None] * axis)

    for frame in range(frames):
        first, second = (frame // 16) % 2, (frame // 8) % 2
        for offset in (0, 8):
            # Rotations preserve the peptide geometry; the two chains share
            # conformational switches but retain independent coordinate noise.
            rotate(frame, offset + 1, offset + 2, np.arange(3, 8) + offset,
                   1.2 * first)
            rotate(frame, offset + 4, offset + 5, np.arange(6, 8) + offset,
                   1.2 * second)
    trajectory.xyz = xyz
    return trajectory


def read_fit_and_states(path):
    with h5py.File(path) as stream:
        return tuple({key: value[()] for key, value in stream[group].items()}
                     for group in ("VMM", "states"))


def pad_probabilities(left, right):
    """Align only unused trailing categories, never permute state labels."""
    left, right = np.asarray(left), np.asarray(right)
    shape = np.maximum(left.shape, right.shape)
    return tuple(np.pad(value, [(0, size - old) for size, old in
                               zip(shape, value.shape)])
                 for value in (left, right))


class ResidueSelectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.folder = Path(cls.temporary.name)
        cls.trajectory = mobile_two_chain_trajectory()
        cls.topology, cls.trr = cls.folder / "top.pdb", cls.folder / "traj.trr"
        cls.trajectory[0].save_pdb(str(cls.topology))
        cls.trajectory.save_trr(str(cls.trr))
        cls.runs = {}
        for name, selection in (("full", None), ("selected", SELECTION),
                                ("singleton", "chainid 0 and resSeq 2")):
            prefix = cls.folder / name
            args = ["ci-mist", "-t", str(cls.trr), "-s", str(cls.topology),
                    "-o", str(prefix), "--seed", str(SEED),
                    "--VMM_BATCH_SIZE", "2", "--PAIR_BATCH_SIZE", "2"]
            if selection is not None:
                args += ["--residue-selection", selection]
            # Exercise the actual CLI, solver, clustering, tree and exporters.
            with patch.object(sys, "argv", args), patch.object(fit.logging, "basicConfig"), \
                    patch.object(fit.cst.ci.dbscan, "validate_clustering_inputs",
                                 wraps=fit.cst.ci.dbscan.validate_clustering_inputs) as validate_clustering, \
                    redirect_stdout(io.StringIO()):
                try:
                    fit.main()
                finally:
                    plt.close("all")
            validate_clustering.assert_called_once()
            clustering = dict(zip(("distances", "weights", "responsibilities"),
                                  (np.array(value) for value in validate_clustering.call_args.args)))
            output = Path(f"{prefix}_seed_{SEED}")
            path = output / "RESULTS_ciMIST.h5"
            mixture, states = read_fit_and_states(path)
            cls.runs[name] = dict(
                output=output, mixture=mixture, states=states, tree=load_h5_tree(path),
                data=fit.load_dataset(str(cls.trr), str(cls.topology),
                                      residue_selection=selection),
                selection=selection, clustering=clustering)

    def assert_close(self, actual, expected, message=""):
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL,
                                   err_msg=message)

    def assert_retained_models(self, name, indices):
        full, selected = self.runs["full"], self.runs[name]
        data, full_data = selected["data"], full["data"]
        np.testing.assert_array_equal(data.angles, np.asarray(full_data.angles)[indices])
        np.testing.assert_array_equal(data.cloud_mask, np.asarray(full_data.cloud_mask)[indices])
        np.testing.assert_array_equal(fit.make_initialization_keys(data, SEED),
                                      np.asarray(fit.make_initialization_keys(full_data, SEED))[indices])
        np.testing.assert_array_equal(data.traj.xyz, full_data.traj.xyz)
        self.assertEqual(data.traj.top.n_atoms, 16)
        self.assertEqual(data.protein_residue_indices, [0, 1, 2, 3])
        self.assertEqual([record["residue_index"] for record in data.residue_records], indices)

        mixture, full_mixture = selected["mixture"], full["mixture"]
        self.assertEqual(set(mixture), set(full_mixture))
        np.testing.assert_array_equal(mixture["n_components"], 100)
        self.assertTrue(np.all(mixture["converged"]))
        np.testing.assert_array_equal(mixture["n_iter"], full_mixture["n_iter"][indices])
        histories = {"log_likelihood", "statuses", "njevs", "nfevs", "nits", "successes"}
        for row, original in enumerate(indices):
            steps = int(mixture["n_iter"][row])
            self.assertGreaterEqual(steps, 2)
            for field in mixture:
                actual, expected = mixture[field][row], full_mixture[field][original]
                if field in histories:
                    actual, expected = actual[:steps], expected[:steps]
                self.assert_close(actual, expected, f"{name}: {field}, residue {original}")
            for field in ("mu", "kappa", "log_likelihood"):
                values = mixture[field][row]
                if field == "log_likelihood":
                    values = values[:steps]
                self.assertTrue(np.all(np.isfinite(values)), field)

        # HDF5 omits r; compare the actual DBSCAN inputs to a fresh final E-step,
        # since unchanged hard labels alone cannot establish a final refresh.
        evaluate = jax.jit(vmm.e_step)
        for row, original in enumerate(indices):
            probabilities = []
            for run, residue in ((full, original), (selected, row)):
                model = run["mixture"]
                probability = np.asarray(evaluate(
                    run["data"].angles[residue],
                    *(model[field][residue] for field in ("mu", "kappa", "logw", "mask"))))
                self.assertTrue(np.all(np.isfinite(probability)))
                self.assert_close(probability.sum(axis=1), 1.)
                self.assert_close(run["clustering"]["responsibilities"][residue], probability,
                                  f"{name}: actual clustering responsibilities, residue {original}")
                self.assert_close(run["clustering"]["weights"][residue], probability.mean(axis=0),
                                  f"{name}: actual clustering weights, residue {original}")
                root_probability = np.sqrt(probability)
                cross_counts = root_probability.T @ root_probability
                normalizer = np.sqrt(1. / np.diag(cross_counts))
                expected_distances = 1 - normalizer[:, None] * cross_counts * normalizer[None, :]
                self.assert_close(run["clustering"]["distances"][residue], expected_distances,
                                  f"{name}: actual S2 distances, residue {original}")
                np.testing.assert_array_equal(
                    np.argmax(run["states"]["C_dbscan"][residue] @ probability.T, axis=0),
                    run["states"]["states_traj"][residue])
                probabilities.append(probability)
            self.assert_close(*probabilities, f"{name}: responsibilities, residue {original}")

        states, full_states = selected["states"], full["states"]
        self.assertEqual(set(states), set(full_states))
        for field in states:
            if field in {"states_traj", "C_dbscan", "state_counts", "n_states"}:
                np.testing.assert_array_equal(states[field], full_states[field][indices])
            else:
                self.assert_close(states[field], full_states[field][indices], field)
        self.assertTrue(np.all(states["n_states"] >= 2), "Fixture must exercise multiple states")

    def assert_tree_matches_refit(self, name):
        run = self.runs[name]
        tree = run["tree"]
        names = run["data"].residues
        # Request reverse order to expose accidental ordering/key changes.
        refit = self.runs["full"]["tree"].refit_to_subset(list(reversed(names)))
        self.assertEqual(list(refit.nodes), names)
        self.assertEqual(list(tree.nodes), names)
        self.assertEqual(list(refit.T.edges), list(tree.T.edges))
        self.assertEqual(list(refit.MI_graph.edges), list(tree.MI_graph.edges))
        self.assertEqual(list(tree.MI_graph.edges), list(combinations(names, 2)))
        self.assertTrue(tree.uncertainty)
        self.assertTrue(refit.uncertainty)
        self.assertEqual(tree.prior, refit.prior)
        self.assertEqual(tree.entropy_estimator, refit.entropy_estimator)
        self.assertEqual(int(tree.N_obs), len(self.trajectory))
        self.assertEqual(int(tree.N_obs), int(refit.N_obs))
        states = run["states"]["states_traj"]
        width = int(states.max()) + 1
        for row, node in enumerate(names):
            self.assertEqual(list(tree.T.neighbors(node)), list(refit.T.neighbors(node)))
            expected = np.bincount(states[row], minlength=width) / states.shape[1]
            self.assert_close(*pad_probabilities(tree.nodes[node]["p"], expected))
            self.assert_close(*pad_probabilities(tree.nodes[node]["p"], refit.nodes[node]["p"]))
            for field in ("S", "S_pos_mean", "S_se", "N_states", "S_contribution_se"):
                self.assert_close(tree.nodes[node][field], refit.nodes[node][field], field)
            for key, value in run["data"].residue_records[row].items():
                self.assertEqual(tree.nodes[node][key], value)
                self.assertEqual(refit.nodes[node][key], value)
        for left, right in combinations(range(len(names)), 2):
            edge = (names[left], names[right])
            actual = np.asarray(dense_probability(tree.MI_graph.edges[edge]["P"]))
            retained = np.asarray(dense_probability(refit.MI_graph.edges[edge]["P"]))
            expected = np.bincount(states[left] * width + states[right],
                                   minlength=width ** 2).reshape(width, width) / states.shape[1]
            self.assert_close(actual, expected)
            self.assert_close(*pad_probabilities(actual, retained))
            occupied = expected > 0
            independent = expected.sum(axis=1)[:, None] * expected.sum(axis=0)
            expected_mi = np.sum(expected[occupied] * np.log(expected[occupied] / independent[occupied]))
            self.assert_close(tree.MI_graph.edges[edge]["I"], expected_mi)
            self.assert_close(tree.MI_graph.edges[edge]["I"], refit.MI_graph.edges[edge]["I"])
        for edge in tree.T.edges:
            for field in ("I_pos_mean", "I_se"):
                self.assert_close(tree.T.edges[edge][field], refit.T.edges[edge][field], field)
        for method in ("entropy", "sum_MIs"):
            self.assert_close(getattr(tree, method)(), getattr(refit, method)(), method)
        for field in ("entropy_se", "entropy_mle_bias", "counts_variance",
                      "structural_variance", "S_jack_entropies", "S_jack_contrib"):
            self.assert_close(getattr(tree, field), getattr(refit, field), field)
        self.assertTrue(np.isfinite(tree.entropy()))
        self.assertTrue(np.isfinite(tree.entropy_se))
        for graph in (tree.T, tree.MI_graph, refit.T, refit.MI_graph):
            self.assertEqual(graph.graph["coordinate_convention"], "physical-bond-angles-v1")
            metadata = graph.graph["residue_selection"]
            self.assertEqual(metadata["semantics"], "marginal")
            self.assertEqual(metadata["residue_indices"],
                             [record["residue_index"] for record in run["data"].residue_records])
            self.assertEqual(metadata["protein_residue_indices"], [0, 1, 2, 3])
            self.assertEqual(metadata["initialization"], "full-protein-split")
            self.assertEqual(metadata["seed"], SEED)
        self.assertEqual(tree.T.graph["residue_selection"]["expression"], run["selection"])
        self.assertIsNone(refit.T.graph["residue_selection"]["expression"])
        self.assertEqual(refit.T.graph["residue_selection"]["refit_from"],
                         dict(expression=None, residue_indices=[0, 1, 2, 3]))

    def test_selected_native_models_states_and_all_pairs_match_full(self):
        self.assert_retained_models("selected", SELECTED)
        self.assert_tree_matches_refit("selected")
        tree = self.runs["selected"]["tree"]
        interchain = [(u, v) for u, v in tree.MI_graph.edges
                      if tree.nodes[u]["chain_index"] != tree.nodes[v]["chain_index"]]
        self.assertEqual(len(interchain), 2)
        self.assertIn(("GLY1_B", "GLY2_B"), tree.MI_graph.edges)

    def test_selected_exports_keep_full_reference_and_original_atom_indices(self):
        for name in ("selected", "singleton"):
            with self.subTest(selection=name):
                run = self.runs[name]
                folder = run["output"] / "pymol"
                structure = md.load_pdb(str(folder / "structure.pdb"))
                selected = md.load_pdb(str(folder / "tree.pdb"))
                indices = [record["ca_atom_index"] for record in run["data"].residue_records]
                self.assertEqual(structure.n_atoms, 16)
                self.assertEqual(selected.n_atoms, len(indices))
                np.testing.assert_allclose(structure.xyz, self.trajectory.xyz[:1], rtol=0, atol=5.1e-5)
                np.testing.assert_allclose(selected.xyz, self.trajectory.xyz[:1, indices], rtol=0, atol=5.1e-5)
                commands = (folder / "draw_tree.pml").read_text()
                for row, original in enumerate(indices, start=1):
                    self.assertIn(f"structure and index {original + 1}", commands)
                    self.assertIn(f"tree and index {row}", commands)

    def test_singleton_native_fit_tree_uncertainty_and_export(self):
        self.assert_retained_models("singleton", [1])
        self.assert_tree_matches_refit("singleton")
        tree = self.runs["singleton"]["tree"]
        self.assertEqual(list(tree.nodes), ["GLY2_A"])
        self.assertEqual(tree.T.number_of_edges(), 0)
        self.assertEqual(tree.MI_graph.number_of_edges(), 0)
        self.assertEqual(float(tree.structural_variance), 0.)
        self.assertTrue(np.isnan(tree.nodes["GLY2_A"]["S_contribution_se"]))


if __name__ == "__main__":
    unittest.main()
