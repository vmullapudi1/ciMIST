"""Selection changes statistical centers, never their topology or random keys."""
from pathlib import Path
import tempfile
import unittest

import jax
import mdtraj as md
import numpy as np

from cimist import fit, io
from cimist.utils.trajectory import traj_to_internal, flatten_masks
from test_multichain_coordinates import small_trajectory
from test_unsupported_residues import peptide_topology


class ResidueSelection(unittest.TestCase):
    def test_default_and_explicit_all_preserve_features_and_original_keys(self):
        trajectory = small_trajectory(solvent=True)
        angles, mask = flatten_masks(traj_to_internal(trajectory))
        for selection in (None, "protein", "all", list(range(1, 8))):
            with self.subTest(selection=selection):
                data = io.traj_to_dataset(trajectory, residue_selection=selection)
                np.testing.assert_array_equal(data.angles, angles)
                np.testing.assert_array_equal(data.cloud_mask, mask)
                self.assertEqual(data.protein_residue_indices, list(range(1, 8)))
                self.assertIs(data.traj, trajectory)
                for seed in (0, 1, 7, 42):
                    np.testing.assert_array_equal(fit.make_initialization_keys(data, seed),
                        jax.random.split(jax.random.PRNGKey(seed), 7))

    def test_boundaries_keep_excluded_covalent_neighbors_and_keys(self):
        trajectory = small_trajectory(solvent=True)
        full = io.traj_to_dataset(trajectory)
        original_xyz, original_bonds = trajectory.xyz.copy(), list(trajectory.top.bonds)
        for selection in ([6, 2], "(chainid 1 to 2) and resSeq 2", "resid 6 or resid 2"):
            with self.subTest(selection=selection):
                selected = io.traj_to_dataset(trajectory, residue_selection=selection)
                self.assertEqual([r["residue_index"] for r in selected.residue_records], [2, 6])
                self.assertEqual(selected.residue_records, [full.residue_records[i] for i in (1, 5)])
                np.testing.assert_array_equal(selected.angles, np.asarray(full.angles)[[1, 5]])
                np.testing.assert_array_equal(selected.cloud_mask, np.asarray(full.cloud_mask)[[1, 5]])
                self.assertTrue(np.asarray(selected.cloud_mask)[:, [0, 1, 14, 15, 16]].all())
                np.testing.assert_array_equal(fit.make_initialization_keys(selected, 42),
                                              np.asarray(fit.make_initialization_keys(full, 42))[[1, 5]])
                self.assertIs(selected.traj, trajectory)
                self.assertEqual(list(selected.traj.top.bonds), original_bonds)
                np.testing.assert_array_equal(selected.traj.xyz, original_xyz)

    def test_repeated_chain_ids_and_numbers_use_original_indices(self):
        trajectory = small_trajectory(duplicate_numbers=True, solvent=True)
        for chain in trajectory.top.chains:
            chain.chain_id = "A"
        full = io.traj_to_dataset(trajectory)
        selected = io.traj_to_dataset(trajectory, residue_selection="chainid 2 and resSeq 0")
        self.assertEqual([r["residue_index"] for r in selected.residue_records], [5, 6, 7])
        self.assertEqual([r["chain_index"] for r in selected.residue_records], [2, 2, 2])
        self.assertEqual(selected.residue_records, full.residue_records[4:])
        self.assertEqual(len(set(selected.residues)), 3)
        self.assertTrue(all("[r" in name for name in selected.residues))

    def test_file_loaders_match_direct_selection_and_keep_all_frames(self):
        trajectory = small_trajectory(solvent=True)
        for chain, label in zip(trajectory.top.chains, ("S", "A", "B")):
            chain.chain_id = label
        selection = "(chainid 1 and resSeq 2 to 3) or (chainid 2 and resSeq 2)"
        direct = io.traj_to_dataset(trajectory, residue_selection=selection)
        with tempfile.TemporaryDirectory() as directory:
            top, traj = Path(directory) / "top.pdb", Path(directory) / "traj.trr"
            trajectory.save_pdb(str(top))
            trajectory.save_trr(str(traj))
            for loader in (io.load_dataset, fit.load_dataset):
                data = loader(str(traj), str(top), residue_selection=selection)
                np.testing.assert_array_equal(data.angles, direct.angles)
                np.testing.assert_array_equal(data.cloud_mask, direct.cloud_mask)
                self.assertEqual(data.residue_records, direct.residue_records)
                self.assertEqual(data.protein_residue_indices, direct.protein_residue_indices)
                self.assertEqual(data.residues, direct.residues)
                np.testing.assert_array_equal(data.traj.xyz, trajectory.xyz)
                np.testing.assert_array_equal(data.traj.time, trajectory.time)
        self.assertEqual(fit.parse_args(["--residue-selection", selection]).residue_selection, selection)
        self.assertIsNone(fit.parse_args([]).residue_selection)

    def test_singleton_keeps_its_physical_feature_mask(self):
        trajectory = small_trajectory(solvent=True)
        full = io.traj_to_dataset(trajectory)
        selected = io.traj_to_dataset(trajectory, residue_selection=[2])
        self.assertEqual(selected.angles.shape, (1, len(trajectory), 28))
        np.testing.assert_array_equal(selected.angles, full.angles[1:2])
        np.testing.assert_array_equal(selected.cloud_mask, full.cloud_mask[1:2])

    def test_empty_malformed_duplicate_or_nonprotein_indices_fail(self):
        trajectory = small_trajectory(solvent=True)
        for selection in ([], "", "  ", "chainid 99", "water", "chainid 1 and",
                          [1, 1], [True], [1.0], [0], [-1], [99], 2):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                io.traj_to_dataset(trajectory, residue_selection=selection)

    def test_selection_cannot_hide_unsupported_peptide_chemistry(self):
        topology = peptide_topology(("ALA", "ASH", "ALA"))
        trajectory = md.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
        for selection in ([0], "resid 2", "protein", "chainid 99"):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "Unsupported protein residue ASH"):
                io.traj_to_dataset(trajectory, residue_selection=selection)


if __name__ == "__main__":
    unittest.main()
