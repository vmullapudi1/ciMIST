"""Physical-coordinate checks: seed 7620; f64 tolerance 1e-10 unless noted."""
from pathlib import Path
import unittest
import warnings

import mdtraj as md
import numpy as np

from cimist.utils.nerfax.parser import protein_feature_references
from cimist.utils.trajectory import traj_to_internal, flatten_masks


REFERENCE = Path(__file__).with_name("data") / "multichain_coordinate_reference.npz"


def protein_angular_features(trajectory, residue_indices=None, frame_batch_size=64,
                             residue_batch_size=128):
    """Exercise the public extraction API while keeping the test helpers terse."""
    internal = traj_to_internal(trajectory, residue_indices=residue_indices,
                                frame_batch_size=frame_batch_size,
                                residue_batch_size=residue_batch_size)
    angles, mask = flatten_masks(internal)
    return np.asarray(angles), np.asarray(mask), internal["residue_records"]


def small_trajectory(chain_lengths=(4, 3), frames=3, break_after=None,
                     duplicate_numbers=False, solvent=False):
    topology = md.Topology()
    coordinates = []
    if solvent:
        water = topology.add_residue("HOH", topology.add_chain("solvent"), resSeq=42)
        topology.add_atom("O", md.element.oxygen, water)
        coordinates.append([10, 10, 10])
    for chain_index, length in enumerate(chain_lengths):
        chain = topology.add_chain(f"protein_{chain_index}")
        previous_c = None
        for index in range(length):
            residue = topology.add_residue("ALA", chain,
                                           resSeq=0 if duplicate_numbers else index + 1)
            atoms = {}
            local = {"N": [0, 0, 0], "CA": [.11, .06, .03],
                     "C": [.23, -.01, .02], "O": [.26, -.10, -.03],
                     "CB": [.10, .13, .14]}
            for name, xyz in local.items():
                atoms[name] = topology.add_atom(name, md.element.get_by_symbol(name[0]), residue)
                coordinates.append(np.array(xyz) + [index * .32, chain_index * 2, 0])
            for first, second in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
                topology.add_bond(atoms[first], atoms[second])
            if previous_c is not None and index - 1 != break_after:
                topology.add_bond(previous_c, atoms["N"])
            previous_c = atoms["C"]
    xyz = np.broadcast_to(coordinates, (frames, len(coordinates), 3)).copy()
    xyz += np.random.default_rng(7620).normal(0, .002, xyz.shape)
    return md.Trajectory(xyz, topology, time=np.arange(frames) * 17)


def direct_torsion(points):
    # Independent projection formula for the oriented angle between two planes.
    first = points[0] - points[1]
    axis = points[2] - points[1]
    last = points[3] - points[2]
    axis = axis / np.linalg.norm(axis)
    first = first - np.dot(first, axis) * axis
    last = last - np.dot(last, axis) * axis
    return np.arctan2(np.dot(np.cross(axis, first), last), np.dot(first, last))


def kabsch_align(mobile, reference):
    """Independent float64 rigid alignment, avoiding MDTraj's float32 QCP."""
    mobile, reference = np.asarray(mobile, dtype=float), np.asarray(reference, dtype=float)
    centered = mobile - mobile.mean(axis=0)
    target = reference - reference.mean(axis=0)
    left, _, right = np.linalg.svd(centered.T @ target)
    rotation = left @ np.diag([1., 1., np.linalg.det(left @ right)]) @ right
    return centered @ rotation + reference.mean(axis=0)


class MultichainCoordinates(unittest.TestCase):
    def test_physical_quartets_and_terminal_masks(self):
        trajectory = small_trajectory(frames=1)
        angles, mask, records = protein_angular_features(trajectory, frame_batch_size=1,
                                                        residue_batch_size=2)
        references, expected_mask, _ = protein_feature_references(trajectory.topology)
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertEqual(angles.shape, (7, 1, 28))
        np.testing.assert_array_equal(mask[[0, 4]][:, [1, 15, 16]], False)
        np.testing.assert_array_equal(mask[[3, 6]][:, [0, 14]], False)
        self.assertTrue(mask[0, 18])  # A physical start-of-segment CB torsion.
        for row in range(len(records)):
            for slot in range(14):
                quartet = trajectory.xyz[0, np.maximum(references[row, slot], 0)].astype(float)
                if mask[row, slot]:
                    v = quartet[2] - quartet[1]
                    w = quartet[3] - quartet[2]
                    expected = np.arccos(np.clip(np.dot(v, w) / np.linalg.norm(v) / np.linalg.norm(w), -1, 1))
                    self.assertAlmostEqual(angles[row, 0, slot], expected, places=10)
                if mask[row, slot + 14]:
                    self.assertAlmostEqual(angles[row, 0, slot + 14], direct_torsion(quartet), places=10)
        np.testing.assert_array_equal(angles[~np.broadcast_to(mask[:, None], angles.shape)], 0)

    def test_chain_motion_reordering_and_mask_context(self):
        trajectory = small_trajectory()
        angles, mask, records = protein_angular_features(trajectory)
        moving = trajectory[:]
        chain_atoms = [atom.index for atom in moving.topology.chain(1).atoms]
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moving.xyz[:, chain_atoms] = moving.xyz[:, chain_atoms] @ rotation + [1, -1, 3]
        changed, _, _ = protein_angular_features(moving)
        # MDTraj stores transformed Cartesian coordinates as float32.
        np.testing.assert_allclose(changed, angles, rtol=0, atol=2e-5)
        selected, selected_mask, selected_records = protein_angular_features(
            trajectory, residue_indices=[5, 1], frame_batch_size=1, residue_batch_size=1)
        np.testing.assert_array_equal(selected, angles[[5, 1]])
        np.testing.assert_array_equal(selected_mask, mask[[5, 1]])
        self.assertEqual(selected_records, [records[5], records[1]])
        order = list(range(4, 7)) + list(range(4))
        reordered, _, reordered_records = protein_angular_features(trajectory, residue_indices=order)
        np.testing.assert_array_equal(reordered, angles[order])
        self.assertEqual([r["residue_index"] for r in reordered_records], order)
        # Rebuild actual topology/atom order too, while retaining chain IDs.
        topology = md.Topology()
        atom_order, mapped_atoms = [], {}
        for chain in reversed(list(trajectory.topology.chains)):
            replacement_chain = topology.add_chain(chain.chain_id)
            for residue in chain.residues:
                replacement = topology.add_residue(residue.name, replacement_chain, residue.resSeq)
                for atom in residue.atoms:
                    mapped_atoms[atom.index] = topology.add_atom(atom.name, atom.element, replacement)
                    atom_order.append(atom.index)
        for first, second in trajectory.topology.bonds:
            topology.add_bond(mapped_atoms[first.index], mapped_atoms[second.index])
        changed_order = md.Trajectory(trajectory.xyz[:, atom_order], topology)
        reordered, _, _ = protein_angular_features(changed_order)
        np.testing.assert_array_equal(reordered, angles[order])

    def test_matches_frozen_legacy_where_references_are_physical(self):
        trajectory = small_trajectory(chain_lengths=(4,))
        actual, mask, _ = protein_angular_features(trajectory)
        with np.load(REFERENCE) as reference:
            np.testing.assert_array_equal(trajectory.xyz, reference["single_xyz"])
            np.testing.assert_array_equal(mask, reference["physical_single_mask"])
            valid = np.broadcast_to(mask[:, None], actual.shape)
            np.testing.assert_allclose(actual[valid], reference["legacy_single_angles"][valid],
                                       rtol=1e-8, atol=1e-10)

    def test_matches_frozen_physical_prototype(self):
        with np.load(REFERENCE) as reference:
            for name, trajectory in (("single", small_trajectory(chain_lengths=(4,))),
                                     ("multi", small_trajectory())):
                actual, mask, _ = protein_angular_features(trajectory)
                np.testing.assert_array_equal(trajectory.xyz, reference[f"{name}_xyz"])
                np.testing.assert_array_equal(mask, reference[f"physical_{name}_mask"])
                np.testing.assert_allclose(actual, reference[f"physical_{name}_angles"],
                                           rtol=1e-8, atol=1e-10)

    def test_original_loader_api_all_chains_single_frame_and_repeated_numbers(self):
        from cimist.utils.nerfax.parser import get_scnet_loader_fns
        from cimist.utils.trajectory import traj_to_scnet, scnet_cartesian_to_traj
        trajectory = small_trajectory(chain_lengths=(2, 2), frames=1,
                                       duplicate_numbers=True, solvent=True)
        trajectory.time[:] = 17
        trajectory.unitcell_vectors = np.array([np.diag([3., 4., 5.])])
        parse, restrict, to_list = get_scnet_loader_fns(trajectory)
        coordinates, sequence = traj_to_scnet(trajectory)
        self.assertEqual(coordinates.shape, (1, 4, 14, 3))
        self.assertEqual(sequence, "AAAA")
        np.testing.assert_allclose(coordinates[0], parse(trajectory.xyz[0]), rtol=0, atol=1e-10)
        np.testing.assert_allclose(to_list(coordinates[0]), trajectory.xyz[0, 1:], rtol=0, atol=1e-8)
        restricted = restrict(trajectory)
        self.assertEqual([chain.chain_id for chain in restricted.topology.chains],
                         ["protein_0", "protein_1"])
        self.assertEqual([residue.resSeq for residue in restricted.topology.residues], [0] * 4)
        np.testing.assert_array_equal(restricted.time, trajectory.time)
        np.testing.assert_allclose(restricted.unitcell_vectors, trajectory.unitcell_vectors)
        restored = scnet_cartesian_to_traj(coordinates, trajectory)
        np.testing.assert_allclose(restored.xyz, trajectory.xyz[:, 1:], rtol=0, atol=1e-8)
        self.assertEqual([chain.chain_id for chain in restored.topology.chains],
                         ["protein_0", "protein_1"])
        np.testing.assert_array_equal(restored.time, trajectory.time)
        np.testing.assert_allclose(restored.unitcell_vectors, trajectory.unitcell_vectors)

    def test_connected_single_chain_reconstruction_roundtrip(self):
        from cimist.utils.trajectory import internal_to_traj
        glycine_start = small_trajectory(chain_lengths=(2,))
        glycine_start = glycine_start.atom_slice([atom.index for atom in glycine_start.top.atoms
                                                  if not (atom.residue.index == 0 and atom.name == "CB")])
        glycine_start.topology.residue(0).name = "GLY"
        for trajectory in (small_trajectory(chain_lengths=(4,)),
                           small_trajectory(chain_lengths=(2,)), glycine_start):
            with self.subTest(residues=trajectory.n_residues,
                              first=trajectory.topology.residue(0).name):
                internal = traj_to_internal(trajectory)
                raw = internal_to_traj(internal, trajectory, superpose=False,
                                       mean_bond_length=False)
                self.assertEqual(raw.xyz.shape, trajectory.xyz.shape)
                for actual, expected in zip(raw.xyz, trajectory.xyz):
                    np.testing.assert_allclose(kabsch_align(actual, expected), expected,
                                               rtol=0, atol=1e-6)
                restored = internal_to_traj(internal, trajectory, superpose=True,
                                            mean_bond_length=False)
                expected = trajectory[:]
                expected.superpose(trajectory)
                # Unchanged legacy folding/QCP has a 2.7e-5 nm residual on this
                # fixture; the stricter Kabsch check above isolates geometry.
                np.testing.assert_allclose(restored.xyz, expected.xyz, rtol=0, atol=5e-5)

    def test_connected_scnet_conversion_matches_trajectory_api(self):
        from cimist.utils.trajectory import traj_to_scnet, scnet_cartesian_to_internal
        trajectory = small_trajectory(chain_lengths=(4,))
        coordinates, sequence = traj_to_scnet(trajectory)
        expected_angles, expected_mask, _ = protein_angular_features(trajectory)
        actual_angles, actual_mask = flatten_masks(scnet_cartesian_to_internal(coordinates, sequence))
        np.testing.assert_allclose(actual_angles, expected_angles, rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(actual_mask, expected_mask)

    def test_natural_scaffold_folding_preserves_physical_geometry(self):
        from cimist.utils.nerfax.parser import make_scaffolds
        from cimist.utils.nerfax.plugin import get_jax_protein_fold
        from cimist.utils.trajectory import traj_to_scnet
        alanines = small_trajectory(chain_lengths=(2,), frames=1)
        glycines = alanines.atom_slice([atom.index for atom in alanines.topology.atoms
                                       if atom.name != "CB"])
        for residue in glycines.topology.residues:
            residue.name = "GLY"
        for trajectory in (alanines, glycines):
            coordinates, sequence = traj_to_scnet(trajectory)
            with self.subTest(sequence=sequence):
                scaffold = make_scaffolds(coordinates[0], sequence)
                original_angles = np.array(scaffold["angles_mask"], copy=True)
                folded = get_jax_protein_fold(scaffold)(scaffold["angles_mask"],
                                                        scaffold["bond_mask"])
                present = np.asarray(scaffold["cloud_mask"], dtype=bool)
                actual = np.asarray(folded)[present]
                expected = np.asarray(coordinates[0])[present]
                self.assertTrue(np.all(np.isfinite(actual)))
                # This low-level path operates in Angstrom and retains float64.
                np.testing.assert_allclose(kabsch_align(actual, expected), expected,
                                           rtol=0, atol=1e-8)
                np.testing.assert_array_equal(scaffold["angles_mask"], original_angles)
        coordinates, sequence = traj_to_scnet(small_trajectory(chain_lengths=(1,), frames=1))
        singleton = make_scaffolds(coordinates[0], sequence)
        self.assertFalse(singleton["reconstruction_supported"])
        with self.assertRaisesRegex(ValueError, "reconstruct|connected|segment"):
            get_jax_protein_fold(singleton)

    def test_unsupported_reconstruction_fails_explicitly(self):
        from cimist.utils.trajectory import internal_to_traj, internal_to_cartesian_scnet
        uncapped = small_trajectory(chain_lengths=(2,))
        cap = uncapped.topology.add_residue("ACE", uncapped.topology.chain(0), resSeq=0)
        cap_c = uncapped.topology.add_atom("C", md.element.carbon, cap)
        uncapped.topology.add_bond(cap_c, uncapped.topology.atom(0))
        capped_xyz = np.concatenate((uncapped.xyz, uncapped.xyz[:, :1] - [.12, 0, 0]), axis=1)
        capped = md.Trajectory(capped_xyz, uncapped.topology)
        for trajectory in (small_trajectory(),
                           small_trajectory(chain_lengths=(4,), break_after=1),
                           small_trajectory(chain_lengths=(1,), frames=1)):
            with self.subTest(chains=trajectory.n_chains, residues=trajectory.n_residues):
                with self.assertRaisesRegex(ValueError, "reconstruct|connected|segment|singleton"):
                    internal_to_traj(traj_to_internal(trajectory), trajectory,
                                      mean_bond_length=False)
        # MDTraj classifies ACE as protein. It must be excluded explicitly as
        # a center, while its C atom remains real covalent context for ALA.
        with self.assertRaisesRegex(ValueError, "Unsupported protein residue ACE"):
            traj_to_internal(capped)
        internal = traj_to_internal(capped, residue_indices=[0, 1])
        self.assertFalse(internal["reconstruction_supported"])
        with self.assertRaisesRegex(ValueError, "reconstruct|connected|segment"):
            internal_to_cartesian_scnet(internal, None, mean_bond_length=False)

    def test_singletons_many_chains_and_repeated_numbers(self):
        trajectory = small_trajectory(chain_lengths=(1,) * 28, frames=1)
        angles, mask, records = protein_angular_features(trajectory)
        self.assertEqual(len(set(r["label"] for r in records)), 28)
        self.assertEqual(records[-1]["chain_id"], "protein_27")
        self.assertTrue(np.all(np.isfinite(angles)))
        self.assertTrue(np.all(mask[:, 18]))
        repeated = small_trajectory(chain_lengths=(2,), duplicate_numbers=True, solvent=True)
        _, _, records = protein_angular_features(repeated)
        self.assertEqual([r["resSeq"] for r in records], [0, 0])
        self.assertEqual([r["residue_index"] for r in records], [1, 2])
        self.assertEqual([r["ca_atom_index"] for r in records], [2, 7])
        self.assertEqual(len(set(r["label"] for r in records)), 2)
        _, _, selected = protein_angular_features(repeated, [2])
        self.assertEqual(selected[0], records[1])

    def test_covalent_gap_and_numbering_gap(self):
        trajectory = small_trajectory(chain_lengths=(4,), break_after=1)
        _, mask, _ = protein_angular_features(trajectory)
        np.testing.assert_array_equal(mask[1, [0, 14]], False)
        np.testing.assert_array_equal(mask[2, [1, 15, 16]], False)
        connected = small_trajectory(chain_lengths=(4,))
        expected, expected_mask, _ = protein_angular_features(connected)
        connected.topology.residue(2).resSeq = 500
        actual, actual_mask, _ = protein_angular_features(connected)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_mask, expected_mask)

    def test_periodic_orthogonal_and_triclinic_cells(self):
        trajectory = small_trajectory(chain_lengths=(3,))
        expected, _, _ = protein_angular_features(trajectory)
        for box in (np.diag([3., 4., 5.]), np.array([[3., 0., 0.], [.9, 4., 0.], [.7, .6, 5.]])):
            wrapped = trajectory[:]
            wrapped.unitcell_vectors = np.broadcast_to(box, (len(wrapped), 3, 3)).copy()
            shifts = np.random.default_rng(7620).integers(-1, 2, (wrapped.n_atoms, 3)) @ box
            wrapped.xyz += shifts
            actual, _, _ = protein_angular_features(wrapped)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-5)
        tiny = trajectory[:]
        tiny.unitcell_vectors = np.broadcast_to(np.eye(3) * .5, (len(tiny), 3, 3)).copy()
        with self.assertRaisesRegex(ValueError, "face spacing"):
            protein_angular_features(tiny)

    def test_missing_atoms_bonds_and_invalid_geometry_fail_clearly(self):
        trajectory = small_trajectory(chain_lengths=(2,))
        for atom in ("N", "CB"):
            incomplete = trajectory.atom_slice([a.index for a in trajectory.top.atoms
                                                if not (a.residue.index == 0 and a.name == atom)])
            with self.assertRaisesRegex(ValueError, "Missing heavy atoms"):
                protein_angular_features(incomplete)
        trajectory.topology._bonds.pop(0)
        with self.assertRaisesRegex(ValueError, "Missing topology bond"):
            protein_angular_features(trajectory)
        trajectory = small_trajectory(chain_lengths=(2,))
        trajectory.xyz[:, 5:] += 1
        with self.assertWarnsRegex(RuntimeWarning, "above 0.3 nm"):
            protein_angular_features(trajectory)
        trajectory = small_trajectory(chain_lengths=(2,))
        trajectory.xyz[:, 1] = trajectory.xyz[:, 0]
        with self.assertRaisesRegex(ValueError, "Invalid covalent geometry"):
            protein_angular_features(trajectory)
        unsupported = small_trajectory(chain_lengths=(1,))
        unsupported.topology.residue(0).name = "MSE"
        with self.assertRaisesRegex(ValueError, "Unsupported protein residue"):
            protein_angular_features(unsupported)

    def test_long_bond_warns_without_changing_physical_angles(self):
        trajectory = small_trajectory(chain_lengths=(1,), frames=2)
        for xyz in trajectory.xyz:
            direction = xyz[2].astype(float) - xyz[1]
            displacement = (.31 / np.linalg.norm(direction) - 1) * direction
            xyz[[2, 3]] += displacement
        for periodic in (False, True):
            if periodic:
                trajectory.unitcell_vectors = np.broadcast_to(np.eye(3) * 3, (2, 3, 3)).copy()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                angles, mask, _ = protein_angular_features(trajectory, frame_batch_size=1)
            self.assertEqual(len(caught), 1)
            self.assertIn("maximum 0.31", str(caught[0].message))
            self.assertIn("frame 0, residue row 0", str(caught[0].message))
            references, _, _ = protein_feature_references(trajectory.topology)
            for frame, xyz in enumerate(trajectory.xyz):
                points = xyz[references[0, 3]].astype(float)
                self.assertTrue(mask[0, 17])
                self.assertAlmostEqual(angles[0, frame, 17], direct_torsion(points), places=10)

    def test_periodic_vectors_require_a_unique_image(self):
        trajectory = small_trajectory(chain_lengths=(1,), frames=1)
        trajectory.xyz[0, 1] = trajectory.xyz[0, 0] + [.49, .49, 0]
        trajectory.unitcell_vectors = np.array([np.eye(3)])
        with self.assertRaisesRegex(ValueError, "unique-image bound"):
            protein_angular_features(trajectory)

    def test_all_canonical_templates_and_ignored_terminal_oxt(self):
        from Bio.Data import IUPACData
        import mp_nerf
        topology = md.Topology()
        coordinates, expected_counts = [], []
        rng = np.random.default_rng(7620)
        for index, (name, code) in enumerate(IUPACData.protein_letters_3to1.items()):
            residue = topology.add_residue(name.upper(), topology.add_chain(str(index)))
            names = ["N", "CA", "C", "O"] + mp_nerf.kb_proteins.SC_BUILD_INFO[code]["atom-names"]
            expected_counts.append(2 * len(names) - 5)
            for atom_name in names:
                topology.add_atom(atom_name, md.element.get_by_symbol(atom_name[0]), residue)
                coordinates.append(rng.uniform(-.05, .05, 3) + [index, 0, 0])
        topology.create_standard_bonds()
        trajectory = md.Trajectory([coordinates], topology)
        angles, mask, records = protein_angular_features(trajectory)
        np.testing.assert_array_equal(mask.sum(axis=1), expected_counts)
        self.assertEqual(len(records), 20)
        oxt = topology.add_atom("OXT", md.element.oxygen, topology.residue(19))
        terminal_c = next(atom for atom in topology.residue(19).atoms if atom.name == "C")
        topology.add_bond(terminal_c, oxt)
        coordinates.append([19.05, 0, 0])
        extended, extended_mask, _ = protein_angular_features(md.Trajectory([coordinates], topology))
        np.testing.assert_array_equal(extended, angles)
        np.testing.assert_array_equal(extended_mask, mask)


if __name__ == "__main__":
    unittest.main()
