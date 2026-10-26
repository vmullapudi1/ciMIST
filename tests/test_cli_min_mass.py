"""The CLI's core-neighborhood mass reaches real chunked DBSCAN inference."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist import fit
from cimist.ci import dbscan, vmm


class ReachedTree(Exception):
    pass


class CLIMinMass(unittest.TestCase):
    def test_probability_mass_bounds(self):
        inputs = ["-t", "trajectory.xtc", "-s", "topology.pdb", "-o", "run"]
        self.assertEqual(fit.parse_args(inputs).min_mass, .01)
        for value in ("0", "0.01", "0.1", "1"):
            with self.subTest(value=value):
                self.assertEqual(fit.parse_args([*inputs, f"--min_mass={value}"]).min_mass, float(value))
        for value in ("nan", "inf", "-inf", "-0.001", "1.001", "invalid"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    fit.parse_args([*inputs, f"--min_mass={value}"])
                self.assertEqual(error.exception.code, 2)

    def run_to_tree(self, argv):
        # Three isolated components have masses .90/.05/.05 in every residue.
        # Control only the fitted input; run native S2 and DBSCAN stage code.
        responsibilities = jnp.eye(3)[jnp.array([0] * 18 + [1, 2])]
        r = jnp.broadcast_to(responsibilities, (3, 20, 3))
        diagnostics = jnp.zeros((3, 100))
        initial = vmm.MixtureFitState(
            jnp.full(3, 3), jnp.zeros((3, 3, 1)), jnp.ones((3, 3, 1)),
            jnp.log(r.mean(axis=1)), jnp.ones((3, 1), dtype=bool), r,
            diagnostics, jnp.full(3, 2), jnp.ones(3, dtype=bool),
            jnp.full(3, .01), diagnostics, diagnostics, diagnostics,
            diagnostics, diagnostics)
        data = SimpleNamespace(
            traj=Mock(), angles=jnp.zeros((3, 20, 1)), cloud_mask=initial.mask,
            residues=["GLY1_A", "GLY2_A", "GLY3_A"],
            residue_records=[dict(chain_index=0, residue_index=i) for i in range(3)],
            protein_residue_indices=[0, 1, 2])
        data.traj.__len__ = Mock(return_value=20)
        tree_builder = Mock(side_effect=ReachedTree)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["ci-mist", "-t", "trajectory.xtc", "-s", "topology.pdb",
                                                   "-o", str(Path(folder) / "run"), *argv]))
            stack.enter_context(patch.object(fit, "load_dataset", return_value=data))
            stack.enter_context(patch.object(fit, "vmap", return_value=Mock(return_value=initial)))
            stack.enter_context(patch.object(vmm, "update_r", side_effect=lambda theta, state: state))
            stack.enter_context(patch.object(fit.cst.MIST, "from_residue_states", tree_builder))
            stack.enter_context(patch.object(fit.logging, "basicConfig"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(ReachedTree):
                fit.main()
            tree_builder.assert_called_once()
        return tree_builder.call_args.args[0], initial

    def test_pipeline_preserves_default_and_honors_override(self):
        for batch in ([], ["--VMM_BATCH_SIZE", "2"]):
            with self.subTest(batch=batch):
                actual, initial = self.run_to_tree(batch)
                distances = dbscan.compute_distances(initial)
                expected = jax.vmap(dbscan.dbscan_eps_std)(
                    distances, initial.r.mean(axis=1), initial.r)
                explicit, _ = self.run_to_tree([*batch, "--min_mass", "0.01"])
                for field in actual._fields:
                    np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field), err_msg=field)
                    np.testing.assert_array_equal(getattr(explicit, field), getattr(expected, field), err_msg=field)
                np.testing.assert_array_equal(actual.n_states, [3, 3, 3])
                np.testing.assert_array_equal(actual.state_counts, [[18, 1, 1]] * 3)
                changed, _ = self.run_to_tree([*batch, "--min_mass", "0.1"])
                np.testing.assert_array_equal(changed.dbscan_min_probability_mass, [.1] * 3)
                np.testing.assert_array_equal(changed.n_states, [2, 2, 2])
                np.testing.assert_array_equal(changed.state_counts, [[18, 2, 0]] * 3)
                np.testing.assert_array_equal(changed.states_traj[:, 18], changed.states_traj[:, 19])
                self.assertTrue(np.all(changed.states_traj[:, 0] != changed.states_traj[:, 18]))


if __name__ == "__main__":
    unittest.main()
