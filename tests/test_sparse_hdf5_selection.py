"""Selective native HDF5 reads preserve orientation and reject broken records."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import networkx as nx
import numpy as np

from cimist.io import H5ProbabilityReader, load_h5_tree, save_tree_h5
from cimist.sparse_probability import SparseProbability
import test_hdf5_compression as fixtures


class SparseHDF5Selection(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "model.h5"
        self.tree = fixtures.HDF5Compression.tree(True)
        # Stored axes need not equal the edge's traversal orientation.
        self.tree.MI_graph.edges["A", "C"]["axes"] = ("C", "A")
        probability = np.array([[.1, .2], [.3, .4]])
        self.tree.MI_graph.edges["A", "C"]["P"] = SparseProbability.encode(probability)
        save_tree_h5(self.tree, self.path)

    def test_selected_rows_and_pairs_read_only_requested_payloads(self):
        calls = []
        original = h5py.Dataset.__getitem__

        def observed(dataset, key, *args, **kwargs):
            if dataset.name.endswith(("/edges/P/values", "/edges/P/positions")):
                calls.append((dataset.name, key))
            return original(dataset, key, *args, **kwargs)

        with patch.object(h5py.Dataset, "__getitem__", observed):
            with H5ProbabilityReader(self.path) as reader:
                records = reader.read_rows([2, 0, 2])
                self.assertEqual([record.row for record in records], [2, 0, 2])
                paired = reader.read_pairs([("C", "A"), ("A", "B")])
                self.assertEqual([record.row for record in paired], [1, 0])
                self.assertEqual(paired[0].edge, ("A", "C"))
                self.assertEqual(paired[0].axes, ("C", "A"))
                np.testing.assert_array_equal(paired[0].probability.dense_numpy(),
                                              np.array([[.1, .2], [.3, .4]]))
                self.assertEqual(reader.read_rows([]), [])
                self.assertEqual(reader.read_pairs([]), [])
        spans = [(key.start, key.stop) for name, key in calls if name.endswith("/values")]
        self.assertEqual(spans, [(8, 12), (0, 4), (8, 12), (4, 8), (0, 4)])
        for record in records + paired:
            self.assertIsInstance(record.probability, SparseProbability)
            self.assertFalse(record.probability.values.flags.writeable)
            self.assertFalse(record.probability.positions.flags.writeable)

    def test_context_and_selection_errors(self):
        reader = H5ProbabilityReader(self.path)
        with self.assertRaises(RuntimeError):
            reader.read_rows([0])
        with reader:
            for row in (-1, 3):
                with self.assertRaises(IndexError):
                    reader.read_rows(row)
            for row in (True, 1.0, "1"):
                with self.assertRaises(TypeError):
                    reader.read_rows([row])
            for pair in (("A", "missing"), ("B", "B")):
                with self.assertRaises(KeyError):
                    reader.read_pairs([pair])
            with self.assertRaises(ValueError):
                reader.read_pairs([("A",)])
            with self.assertRaises(RuntimeError):
                reader.__enter__()
        with self.assertRaises(RuntimeError):
            reader.read_pairs([("A", "B")])
        with H5ProbabilityReader(self.path, graph="tree") as reader:
            self.assertEqual(len(reader.read_rows([0, 1])), 2)
            with self.assertRaises(KeyError):
                reader.read_pairs([("A", "C")])
        with self.assertRaises(ValueError):
            H5ProbabilityReader(self.path, graph="invalid")

    def test_incomplete_or_unknown_format_is_rejected(self):
        with h5py.File(self.path, "r+") as file:
            file.attrs["complete"] = False
        with self.assertRaisesRegex(ValueError, "completely written"):
            load_h5_tree(self.path)
        with h5py.File(self.path, "r+") as file:
            file.attrs["complete"] = True
            file.attrs["format"] = "unknown"
        with self.assertRaisesRegex(ValueError, "supported"):
            load_h5_tree(self.path)

    def test_failed_append_preserves_existing_snapshot(self):
        before = self.path.read_bytes()
        for mode in ("a", "r+"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "already contains"):
                    save_tree_h5(self.tree, self.path, mode=mode, attrs_dict={"changed": True})
                self.assertEqual(self.path.read_bytes(), before)
                with H5ProbabilityReader(self.path) as reader:
                    self.assertEqual(reader.read_rows([0])[0].edge, ("A", "B"))

    def test_integer_labels_cannot_silently_round_to_floats(self):
        mapping = {"A": 0, "B": 2**63 + 1, "C": 2}
        self.tree.T = nx.relabel_nodes(self.tree.T, mapping)
        self.tree.MI_graph = nx.relabel_nodes(self.tree.MI_graph, mapping)
        with self.assertRaisesRegex(ValueError, "losslessly"):
            save_tree_h5(self.tree, self.path)

    def test_corrupt_sparse_records_are_rejected(self):
        def invalid_offsets(file):
            file["MI_graph/edges/P/offsets"][1] = 100

        def duplicate_positions(file):
            file["MI_graph/edges/P/positions"][1] = 0

        def bad_axes(file):
            file["MI_graph/edges/axes"][0] = [0, 0]

        def negative_values(file):
            file["MI_graph/edges/P/values"][0] = -.1

        def invalid_shape(file):
            file["MI_graph/edges/P"].attrs["shape"] = [1.5, 2.0]

        def truncated_presence(file):
            del file["MI_graph/edges/P/present"]
            file["MI_graph/edges/P"].create_dataset("present", data=[True, True])

        def extra_values(file):
            file["MI_graph/edges/P/values"].resize((13,))

        def floating_indices(file):
            group = file["MI_graph/edges"]
            values = group["node_indices"][:].astype(float)
            del group["node_indices"]
            group.create_dataset("node_indices", data=values)

        for damage in (invalid_offsets, duplicate_positions, bad_axes, negative_values,
                       invalid_shape, truncated_presence, extra_values, floating_indices):
            with self.subTest(damage=damage.__name__):
                save_tree_h5(self.tree, self.path)
                with h5py.File(self.path, "r+") as file:
                    damage(file)
                with self.assertRaises(ValueError):
                    with H5ProbabilityReader(self.path) as reader:
                        reader.read_rows([0])


if __name__ == "__main__":
    unittest.main()
