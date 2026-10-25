"""CLI defaults select the measured memory configuration; dense remains opt-in."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist.fit import parse_args
from cimist.mist import MIST
from cimist.io import save_tree_h5, load_h5_tree
from cimist.sparse_probability import SparseProbability, dense_probability


INPUT_ARGS = ["-t", "trajectory.xtc", "-s", "topology.pdb", "-o", "run"]


class CLIDefaults(unittest.TestCase):
    def test_benchmarked_defaults_and_overrides(self):
        args = parse_args(INPUT_ARGS)
        self.assertEqual(args.VMM_BATCH_SIZE, None)
        self.assertEqual(args.PAIR_BATCH_SIZE, 2048)
        self.assertEqual(args.probability_storage, "sparse")
        self.assertEqual(args.hdf5_compression, "gzip")
        override = parse_args([*INPUT_ARGS, "--VMM_BATCH_SIZE", "100", "--PAIR_BATCH_SIZE", "999",
                               "--probability_storage", "dense", "--hdf5_compression", "none"])
        self.assertEqual((override.VMM_BATCH_SIZE, override.PAIR_BATCH_SIZE,
                          override.probability_storage, override.hdf5_compression),
                         (100, 999, "dense", "none"))

    def test_default_storage_preserves_dense_results_and_hdf5(self):
        states = SimpleNamespace(states_traj=jnp.array([[0, 0, 1, 1, 0, 1, 0, 1],
                                                        [0, 1, 0, 1, 0, 1, 1, 0],
                                                        [1, 1, 0, 0, 1, 0, 1, 0]]),
                                 state_counts=jnp.full((3, 2), 4),
                                 n_states=jnp.full(3, 2))
        trees = []
        for argv in ([], ["--probability_storage", "dense"]):
            args = parse_args([*INPUT_ARGS, *argv])
            trees.append(MIST.from_residue_states(states, ["A", "B", "C"],
                entropy_estimator="mle", uncertainty=False,
                pair_batch_size=args.PAIR_BATCH_SIZE, probability_storage=args.probability_storage))
        sparse, dense = trees
        self.assertEqual(list(sparse.T.edges), list(dense.T.edges))
        self.assertEqual(float(sparse.entropy()), float(dense.entropy()))
        for edge in sparse.MI_graph.edges:
            actual = sparse.MI_graph.edges[edge]["P"]
            self.assertIsInstance(actual, SparseProbability)
            np.testing.assert_array_equal(dense_probability(actual), dense.MI_graph.edges[edge]["P"])
        with tempfile.TemporaryDirectory() as folder:
            for name, tree in (("sparse", sparse), ("dense", dense)):
                path = Path(folder) / f"{name}.h5"
                save_tree_h5(tree, path)
                restored = load_h5_tree(path)
                self.assertEqual(list(restored.T.edges), list(tree.T.edges))
                self.assertEqual(float(restored.entropy()), float(tree.entropy()))
                for edge in tree.MI_graph.edges:
                    actual = restored.MI_graph.edges[edge]["P"]
                    self.assertIsInstance(actual, SparseProbability)
                    np.testing.assert_array_equal(dense_probability(actual),
                                                  dense_probability(tree.MI_graph.edges[edge]["P"]))


if __name__ == "__main__":
    unittest.main()
