"""Chain identity and all-pair inference survive loading, storage and plotting."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

from cimist.fit import load_dataset
from cimist.io import (load_traj, traj_to_dataset, save_tree_h5, load_h5_tree,
                       dump_sparse_tree, load_sparse_tree)
from cimist.mist import MIST
from cimist.pymol.pymol import tree_cartoon, _reference_ca_indices
from cimist.utils.nerfax.parser import residue_records
from cimist.utils.trajectory import (traj_to_internal, flatten_masks,
                                     internal_to_traj, scnet_cartesian_to_traj)


def two_chain_trajectory(frames=1):
    top = md.Topology()
    xyz = []
    chain_xyz = np.array([[0., 0., 0.], [.14, 0., 0.], [.19, .13, 0.],
                          [.31, .14, 0.], [.10, .23, .04], [.15, .36, .06],
                          [.29, .36, .08], [.35, .46, .08]])
    for i, chain_id in enumerate(("A", "B")):
        chain = top.add_chain(chain_id)
        previous_c = None
        for number in (1, 2):
            residue = top.add_residue("GLY", chain, resSeq=number)
            atoms = [top.add_atom(name, element, residue) for name, element in
                     zip(("N", "CA", "C", "O"),
                         (md.element.nitrogen, md.element.carbon,
                          md.element.carbon, md.element.oxygen))]
            for left, right in zip(atoms[:-1], atoms[1:]):
                top.add_bond(left, right)
            if previous_c is not None:
                top.add_bond(previous_c, atoms[0])
            previous_c = atoms[2]
        xyz.extend(chain_xyz + [0., 0., 1.2 * i])
    coordinates = np.broadcast_to(xyz, (frames, len(xyz), 3)).copy()
    coordinates[1:] += np.random.default_rng(7620).normal(0., .002, coordinates[1:].shape)
    return md.Trajectory(coordinates, top, time=2.5 + 17 * np.arange(frames),
                         unitcell_lengths=np.full((frames, 3), 4.),
                         unitcell_angles=np.full((frames, 3), 90.))


def correlated_tree(traj):
    # Across chains, matching residue positions are perfectly correlated.
    first = np.tile([0, 0, 1, 1], 16)
    second = np.tile([0, 1, 0, 1], 16)
    states = SimpleNamespace(states_traj=jnp.asarray([first, second, first, second]),
                             state_counts=jnp.full((4, 2), 32),
                             n_states=jnp.full(4, 2))
    records = residue_records(traj.topology)
    tree = MIST.from_residue_states(states, [record["label"] for record in records],
                                   probability_storage="sparse", uncertainty=True)
    for record in records:
        tree.MI_graph.nodes[record["label"]].update(record)
        tree.T.nodes[record["label"]].update(record)
    for graph in (tree.T, tree.MI_graph):
        graph.graph["coordinate_convention"] = "physical-bond-angles-v1"
    return tree


class MultichainPipeline(unittest.TestCase):
    def test_entry_points_share_features_masks_and_residue_order(self):
        for frames in (1, 3):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as folder:
                trajectory = two_chain_trajectory(frames)
                internal = traj_to_internal(trajectory)
                angles, mask = flatten_masks(internal)
                direct = traj_to_dataset(trajectory)
                # TRR preserves float32 positions exactly, isolating API differences.
                pdb, trr = Path(folder) / "top.pdb", Path(folder) / "traj.trr"
                trajectory.save_pdb(str(pdb))
                trajectory.save_trr(str(trr))
                loaded = load_dataset(str(trr), str(pdb))
                self.assertIs(direct.traj, trajectory)
                self.assertEqual(angles.shape, (4, frames, 28))
                self.assertEqual(mask.shape, (4, 28))
                for dataset in (direct, loaded):
                    np.testing.assert_array_equal(dataset.angles, angles)
                    np.testing.assert_array_equal(dataset.cloud_mask, mask)
                    np.testing.assert_array_equal(dataset.traj.xyz, trajectory.xyz)
                    np.testing.assert_array_equal(dataset.traj.time, trajectory.time)
                self.assertEqual(loaded.residue_records, internal["residue_records"])
                self.assertEqual(loaded.residues,
                                 [record["label"] for record in internal["residue_records"]])
                np.testing.assert_array_equal(np.asarray(mask)[[0, 2]][:, [1, 15, 16]], False)
                np.testing.assert_array_equal(np.asarray(mask)[[1, 3]][:, [0, 14]], False)

    def test_static_scaffold_masks_do_not_allocate_per_frame(self):
        internal = traj_to_internal(two_chain_trajectory(frames=3))
        for name in ("cloud_mask", "point_ref_mask"):
            values = internal[name]
            self.assertEqual(values.shape[0], 3)
            self.assertEqual(values.strides[0], 0, name)
            self.assertTrue(np.shares_memory(values[0], values[-1]), name)

    def test_selected_chain_rejects_full_multichain_reference(self):
        trajectory = two_chain_trajectory()
        internal = traj_to_internal(trajectory, residue_indices=[0, 1])
        self.assertTrue(internal["reconstruction_supported"])
        with self.assertRaisesRegex(ValueError, "same complete residue mapping"):
            internal_to_traj(internal, trajectory, superpose=False, mean_bond_length=False)
        with self.assertRaisesRegex(ValueError, "reference residue mapping"):
            scnet_cartesian_to_traj(jnp.zeros((1, 2, 14, 3)), trajectory)

    def test_one_frame_loading_preserves_chains_bonds_time_box(self):
        original = two_chain_trajectory()
        with tempfile.TemporaryDirectory() as folder:
            pdb, xtc = Path(folder) / "top.pdb", Path(folder) / "traj.xtc"
            original.save_pdb(str(pdb))
            original.save_xtc(str(xtc))
            loaded = load_dataset(str(xtc), str(pdb))
            self.assertEqual(loaded.angles.shape, (4, 1, 28))
            self.assertEqual(loaded.residues, ["GLY1_A", "GLY2_A", "GLY1_B", "GLY2_B"])
            self.assertEqual([chain.chain_id for chain in loaded.traj.top.chains], ["A", "B"])
            self.assertEqual(loaded.traj.top.n_bonds, original.top.n_bonds)
            np.testing.assert_array_equal(loaded.traj.time, original.time)
            np.testing.assert_allclose(loaded.traj.unitcell_vectors, original.unitcell_vectors)
            np.testing.assert_allclose(loaded.traj.xyz, original.xyz, atol=.00051, rtol=0)
            direct = traj_to_dataset(loaded.traj)
            np.testing.assert_array_equal(direct.angles, loaded.angles)
            np.testing.assert_array_equal(direct.cloud_mask, loaded.cloud_mask)
            replicas = Path(folder) / "replicas"
            replicas.mkdir()
            original.save_xtc(str(replicas / "a.xtc"))
            original.save_xtc(str(replicas / "b.xtc"))
            joined = load_dataset(str(replicas), str(pdb))
            self.assertEqual(joined.residues, loaded.residues)
            self.assertEqual(joined.angles.shape, (4, 2, 28))
            np.testing.assert_array_equal(joined.angles[:, 0], joined.angles[:, 1])

    def test_all_intra_and_interchain_pairs_and_roundtrip_identity(self):
        traj = two_chain_trajectory()
        tree = correlated_tree(traj)
        self.assertEqual(tree.MI_graph.number_of_edges(), 6)
        cross = [(u, v) for u, v in tree.MI_graph.edges
                 if tree.nodes[u]["chain_index"] != tree.nodes[v]["chain_index"]]
        self.assertEqual(len(cross), 4)
        for edge in (("GLY1_A", "GLY1_B"), ("GLY2_A", "GLY2_B")):
            self.assertIn(edge, tree.T.edges)
            self.assertAlmostEqual(tree.MI_graph.edges[edge]["I"], np.log(2.))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tree.h5"
            save_tree_h5(tree, path)
            restored = load_h5_tree(path)
            self.assertEqual(list(restored.nodes), list(tree.nodes))
            self.assertEqual(list(restored.T.edges), list(tree.T.edges))
            self.assertEqual(restored.T.graph["coordinate_convention"], "physical-bond-angles-v1")
            for name in tree.nodes:
                for key in ("chain_index", "chain_id", "residue_index", "resSeq", "ca_atom_index"):
                    self.assertEqual(restored.nodes[name][key], tree.nodes[name][key])
            pkl = Path(folder) / "tree.pkl"
            with pkl.open("wb") as stream:
                dump_sparse_tree(tree, stream)
            with pkl.open("rb") as stream:
                pickled = load_sparse_tree(stream)
            self.assertEqual(pickled.nodes["GLY1_B"]["chain_id"], "B")

    def test_plot_selectors_are_unique_even_with_repeated_numbers(self):
        traj = two_chain_trajectory()
        tree = correlated_tree(traj)
        with tempfile.TemporaryDirectory() as folder:
            fig, _ = tree_cartoon(tree, traj, folder)
            plt.close(fig)
            commands = (Path(folder) / "draw_tree.pml").read_text()
            self.assertNotIn("resi ", commands)
            self.assertNotIn("tree and id ", commands)
            self.assertTrue(commands.startswith("set retain_order, 1\n"))
            self.assertIn("cmd.bond('tree and index 1', 'tree and index 3')", commands)
            for i, atom_index in enumerate((1, 5, 9, 13), start=1):
                self.assertIn(f"structure and index {atom_index + 1}", commands)
                self.assertIn(f"tree and index {i}", commands)
            tree_structure = md.load_pdb(str(Path(folder) / "tree.pdb"))
            np.testing.assert_allclose(tree_structure.xyz, traj.xyz[:, [1, 5, 9, 13]], atol=.000051)
        tree.nodes["GLY1_B"]["ca_atom_index"] = 1
        with self.assertRaisesRegex(ValueError, "identities"):
            _reference_ca_indices(tree, traj.topology)

    def test_generated_pdb_can_be_reused_with_original_identities(self):
        for ids in ((None, None), ("long_A", "long_B")):
            traj = two_chain_trajectory()
            for chain, chain_id in zip(traj.top.chains, ids):
                chain.chain_id = chain_id
            for residue in traj.top.residues:
                residue.resSeq += 10000
            tree = correlated_tree(traj)
            with tempfile.TemporaryDirectory() as folder:
                fig, _ = tree_cartoon(tree, traj, folder)
                plt.close(fig)
                exported = md.load_pdb(str(Path(folder) / "structure.pdb"))
                np.testing.assert_array_equal(_reference_ca_indices(tree, exported.top), [1, 5, 9, 13])


if __name__ == "__main__":
    unittest.main()
