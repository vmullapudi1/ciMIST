"""Focus exports display full-model paths, not a refitted marginal network."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

from cimist.io import load_h5_tree, save_tree_h5
from cimist.pymol.pymol import tree_cartoon
from test_multichain_pipeline import correlated_tree, two_chain_trajectory


class PymolFocus(unittest.TestCase):
    def setUp(self):
        self.trajectory = two_chain_trajectory()
        # Original indices distinguish both repeated residue numbers and IDs.
        for chain in self.trajectory.top.chains:
            chain.chain_id = "A"
        self.tree = correlated_tree(self.trajectory)
        self.names = self.tree.resnames
        for graph in (self.tree.T, self.tree.MI_graph):
            graph.graph["residue_selection"] = dict(semantics="marginal", expression=None,
                residue_indices=[0, 1, 2, 3], protein_residue_indices=[0, 1, 2, 3])

    def tearDown(self):
        plt.close("all")

    def test_connectors_keep_original_identities_and_full_model_statistics(self):
        before = copy.deepcopy(self.tree)
        selected = self.names[2:]
        expected = (self.tree.residue_entropies() - .5 * self.tree.residue_sum_MIs()).loc[selected].sum()
        marginal = float(self.tree.refit_to_subset(selected).entropy())
        self.assertGreater(abs(expected - marginal), .1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "focus"
            self.tree.to_pymol(self.trajectory, folder, focus_nodes=reversed(selected))
            metadata = json.loads((folder / "focus.json").read_text())
            self.assertEqual(metadata["semantics"], "display_only")
            self.assertEqual(metadata["view_kind"], "full_tree_paths")
            self.assertEqual(metadata["entropy_allocation"], "full_model_eq3")
            self.assertEqual(metadata["prior"], self.tree.prior)
            self.assertEqual(metadata["entropy_estimator"], self.tree.entropy_estimator)
            self.assertEqual(metadata["N_obs"], int(self.tree.N_obs))
            self.assertEqual(metadata["model_selection"], self.tree.T.graph["residue_selection"])
            self.assertEqual(metadata["full_model_node_count"], 4)
            self.assertAlmostEqual(metadata["selected_full_model_entropy_contribution_nats"], expected)
            records = metadata["residues"]
            self.assertEqual([record["label"] for record in records], self.names)
            self.assertEqual([record["is_selected"] for record in records], [False, False, True, True])
            self.assertEqual([record["tree_atom_index_1based"] for record in records], [1, 2, 3, 4])
            self.assertEqual([record["reference_atom_index_0based"] for record in records], [1, 5, 9, 13])
            for row, record in enumerate(records):
                self.assertEqual(record["identity"]["residue_index"], row)
                self.assertEqual(record["identity"]["chain_index"], row // 2)
                self.assertEqual(record["identity"]["resSeq"], 1 + row % 2)
            structure = md.load_pdb(str(folder / "structure.pdb"))
            view = md.load_pdb(str(folder / "tree.pdb"))
            np.testing.assert_allclose(structure.xyz, self.trajectory.xyz, rtol=0, atol=5.1e-5)
            np.testing.assert_allclose(view.xyz, self.trajectory.xyz[:, [1, 5, 9, 13]], rtol=0, atol=5.1e-5)
            commands = (folder / "draw_tree.pml").read_text()
            self.assertIn("select focus_residues, tree and index 3+4", commands)
            self.assertIn("select connector_residues, tree and index 1+2", commands)
            self.assertIn("Full-model focus view", commands)
            self.assertNotIn("resi ", commands)
            for u, v in self.tree.T.edges:
                self.assertIn(f"cmd.bond('tree and index {self.names.index(u) + 1}', "
                              f"'tree and index {self.names.index(v) + 1}')", commands)
        self.assertEqual(list(self.tree.T.edges), list(before.T.edges))
        self.assertEqual(self.tree.T.graph, before.T.graph)
        self.assertEqual(self.tree.MI_graph.graph, before.MI_graph.graph)
        self.assertEqual(float(self.tree.entropy()), float(before.entropy()))
        self.assertEqual(float(self.tree.entropy_se), float(before.entropy_se))
        np.testing.assert_array_equal(self.tree.S_jack_entropies, before.S_jack_entropies)
        for node in self.names:
            self.assertEqual(set(self.tree.nodes[node]), set(before.nodes[node]))
            self.assertNotIn("is_selected", self.tree.nodes[node])
            np.testing.assert_array_equal(self.tree.nodes[node]["p"], before.nodes[node]["p"])

    def test_pruned_neighbors_still_contribute_to_full_model_allocation(self):
        selected = self.names[:2]
        full_allocations = self.tree.residue_entropies() - .5 * self.tree.residue_sum_MIs()
        with tempfile.TemporaryDirectory() as directory:
            fig, _ = tree_cartoon(self.tree, self.trajectory, directory, focus_nodes=selected)
            plt.close(fig)
            metadata = json.loads((Path(directory) / "focus.json").read_text())
            self.assertEqual([record["label"] for record in metadata["residues"]], selected)
            self.assertTrue(all(record["is_selected"] for record in metadata["residues"]))
            for record in metadata["residues"]:
                self.assertEqual(record["full_model_entropy_contribution_nats"], full_allocations[record["label"]])
            # Recomputing contributions on the pruned display would omit strong
            # edges to the two removed leaves and allocate too much entropy.
            self.assertGreater(self.tree.residue_entropies().loc[selected].sum() -
                               metadata["selected_full_model_entropy_contribution_nats"], .1)
            commands = (Path(directory) / "draw_tree.pml").read_text()
            self.assertIn("select connector_residues, none", commands)

    def test_reloaded_singleton_focus_and_reusing_folder_for_complete_view(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.h5"
            save_tree_h5(self.tree, path)
            tree = load_h5_tree(path)
            folder = Path(directory) / "view"
            fig, _ = tree_cartoon(tree, self.trajectory, folder, focus_nodes=self.names[2:3])
            plt.close(fig)
            metadata = json.loads((folder / "focus.json").read_text())
            self.assertEqual(len(metadata["residues"]), 1)
            self.assertEqual(metadata["model_selection"], self.tree.T.graph["residue_selection"])
            self.assertEqual(metadata["residues"][0]["identity"]["residue_index"], 2)
            view = md.load_pdb(str(folder / "tree.pdb"))
            np.testing.assert_allclose(view.xyz, self.trajectory.xyz[:, [9]], rtol=0, atol=5.1e-5)
            commands = (folder / "draw_tree.pml").read_text()
            self.assertNotIn("cmd.bond(", commands)
            self.assertIn("structure and index 10", commands)
            self.assertIn("select focus_residues, tree and index 1", commands)
            self.assertIn("select connector_residues, none", commands)
            fig, _ = tree_cartoon(tree, self.trajectory, folder)
            plt.close(fig)
            self.assertFalse((folder / "focus.json").exists())
            commands = (folder / "draw_tree.pml").read_text()
            self.assertNotIn("focus_residues", commands)
            self.assertEqual(md.load_pdb(str(folder / "tree.pdb")).n_atoms, 4)

    def test_invalid_focus_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "invalid"
            for selection in ([], [self.names[0]] * 2, ["unknown"]):
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    tree_cartoon(self.tree, self.trajectory, folder, focus_nodes=selection)
                self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
