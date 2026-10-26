"""Reject unsupported peptide chemistry without classifying ligands by names."""
import unittest
from pathlib import Path
import tempfile

import mdtraj as md
import numpy as np

from cimist.utils.nerfax.parser import (
    protein_feature_references, protein_sequence, residue_records,
)
from cimist.utils.trajectory import traj_to_internal


def peptide_topology(names):
    topology = md.Topology()
    chain = topology.add_chain("A")
    previous_c = None
    for number, name in enumerate(names, start=10):
        residue = topology.add_residue(name, chain, resSeq=number)
        atoms = {name: topology.add_atom(name, md.element.get_by_symbol(name[0]), residue)
                 for name in ("N", "CA", "C", "O", "CB")}
        for first, second in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
            topology.add_bond(atoms[first], atoms[second])
        if previous_c is not None:
            topology.add_bond(previous_c, atoms["N"])
        previous_c = atoms["C"]
    return topology


class UnsupportedResidues(unittest.TestCase):
    def test_ash_is_rejected_by_every_entry_point_before_geometry(self):
        topology = peptide_topology(("ALA", "ASH", "ALA"))
        self.assertFalse(topology.residue(1).is_protein)
        # Invalid coordinates establish that admission precedes feature extraction.
        trajectory = md.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
        calls = (lambda: residue_records(topology),
                 lambda: protein_sequence(topology),
                 lambda: protein_feature_references(topology),
                 lambda: protein_feature_references(topology, residue_indices=[0]),
                 lambda: traj_to_internal(trajectory),
                 lambda: traj_to_internal(trajectory, residue_indices=[2]))
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                        ValueError, r"Unsupported protein residue ASH11_A at topology index 1 .*chain index 0.*canonical"):
                    call()

    def test_known_variants_fail_even_without_peptide_bonds(self):
        for name in ("ASH", "GLH", "HID", "HIE", "CYX", "MSE"):
            topology = peptide_topology((name,))
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"Unsupported protein residue {name}"):
                    residue_records(topology)

    def test_hdf5_protein_filter_cannot_remove_unsupported_residues(self):
        from cimist.io import load_traj
        topology = peptide_topology(("ALA", "ASH", "ALA"))
        trajectory = md.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "unsupported.h5")
            trajectory.save_hdf5(path)
            with self.assertRaisesRegex(ValueError, "Unsupported protein residue ASH11"):
                load_traj(path, top=None)
            # A raw loader may retain the topology for inspection or correction.
            loaded = load_traj(path, top=None, protein_only=False)
            self.assertEqual(loaded.topology.residue(1).name, "ASH")
            self.assertEqual(loaded.n_atoms, trajectory.n_atoms)

    def test_unregistered_peptide_residue_is_detected_from_bonds(self):
        for names in (("ALA", "ZZZ", "ALA"), ("ZZZ", "ZZZ")):
            topology = peptide_topology(names)
            unknown = next(r for r in topology.residues if r.name == "ZZZ")
            self.assertFalse(unknown.is_protein)
            with self.subTest(names=names):
                with self.assertRaisesRegex(ValueError, "Unsupported protein residue ZZZ"):
                    protein_feature_references(topology, residue_indices=[0])

    def test_ligand_backbone_names_are_not_sufficient(self):
        topology = peptide_topology(("ALA", "ALA", "DHF"))
        # Keep N--CA--C in the ligand, but remove its peptide link to ALA.
        topology._bonds = [(first, second) for first, second in topology.bonds
                           if first.residue == second.residue or second.residue.index != 2]
        water = topology.add_residue("HOH", topology.chain(0), resSeq=13)
        topology.add_atom("O", md.element.oxygen, water)
        self.assertFalse(topology.residue(2).is_protein)
        records = residue_records(topology)
        self.assertEqual([record["residue_index"] for record in records], [0, 1])
        self.assertEqual(protein_sequence(topology), "AA")
        _, _, selected = protein_feature_references(topology, residue_indices=[1])
        self.assertEqual(selected, [records[1]])

    def test_incomplete_internal_bonds_cannot_hide_a_peptide_residue(self):
        for missing_bond in ({"CA", "C"}, None):
            topology = peptide_topology(("ALA", "ZZZ"))
            topology._bonds = [(first, second) for first, second in topology.bonds
                               if not (first.residue.index == second.residue.index == 1
                                       and (missing_bond is None or
                                            {first.name, second.name} == missing_bond))]
            with self.subTest(missing_bond=missing_bond):
                with self.assertRaisesRegex(ValueError, "Unsupported protein residue ZZZ11"):
                    residue_records(topology)
                with self.assertRaisesRegex(ValueError, "Unsupported protein residue ZZZ11"):
                    protein_feature_references(topology, residue_indices=[0])

    def test_caps_require_explicit_exclusion_as_centers(self):
        topology = peptide_topology(("ALA", "ALA"))
        ace = topology.add_residue("ACE", topology.chain(0), resSeq=9)
        ace_c = topology.add_atom("C", md.element.carbon, ace)
        topology.add_bond(ace_c, topology.atom(0))
        nme = topology.add_residue("NME", topology.chain(0), resSeq=12)
        nme_n = topology.add_atom("N", md.element.nitrogen, nme)
        topology.add_bond(topology.atom(7), nme_n)
        with self.assertRaisesRegex(ValueError, "Unsupported protein residue ACE"):
            protein_feature_references(topology)
        with self.assertRaisesRegex(ValueError, "Unsupported protein residue ACE"):
            protein_sequence(topology)
        references, _, records = protein_feature_references(topology, residue_indices=[0, 1])
        self.assertEqual([r["name"] for r in records], ["ALA", "ALA"])
        self.assertEqual(references[0, 2, 0], ace_c.index)
        self.assertEqual(references[1, 0, 3], nme_n.index)


if __name__ == "__main__":
    unittest.main()
