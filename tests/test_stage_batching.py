"""Post-EM batching retains every residue, frame, component and state assignment."""
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist.ci import dbscan, vmm
from cimist.utils.chunked_vmap import vmap_chunked as cvmap


def synthetic_mixture(n_residues=5, n_frames=61, n_components=8, seed=20261021):
    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.uniform(-np.pi, np.pi, (n_residues, n_frames, 28)))
    mu = jnp.asarray(rng.uniform(-np.pi, np.pi, (n_residues, n_components, 28)))
    kappa = jnp.asarray(rng.uniform(10, 20, mu.shape))
    logw = jnp.asarray(rng.normal(0, 1, (n_residues, n_components)))
    mask = jnp.asarray(rng.random((n_residues, 28)) > 0.2)
    diagnostics = jnp.zeros((n_residues, 4))
    state = vmm.MixtureFitState(
        jnp.full(n_residues, n_components), mu, kappa, logw, mask,
        jnp.full((n_residues, n_frames, n_components), 1 / n_components),
        diagnostics, jnp.ones(n_residues, dtype=jnp.int64),
        jnp.ones(n_residues, dtype=bool), jnp.full(n_residues, 1e-2),
        diagnostics, diagnostics, diagnostics, diagnostics, diagnostics)
    return theta, state


def legacy_distances(state):
    cross_counts = jax.vmap(lambda r: jnp.sqrt(r).T @ jnp.sqrt(r))(state.r)
    normalize = lambda x: jnp.sqrt(jnp.diag(1 / jnp.diag(x))) @ x @ jnp.sqrt(jnp.diag(1 / jnp.diag(x)))
    return 1 - jax.vmap(normalize)(cross_counts)


class StageBatching(unittest.TestCase):
    def assert_numeric_equal(self, actual, expected):
        for a, e in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(a, e, rtol=1e-8, atol=1e-10)

    def test_final_refresh_and_fallback_keep_all_components(self):
        theta, state = synthetic_mixture()
        expected = jax.vmap(vmm.update_r)(theta, state)
        expected_fallback = jax.vmap(vmm.e_step)(theta, state.mu, state.kappa, state.logw, state.mask)
        for size in (None, 1, 2, 5, 8):
            with self.subTest(size=size):
                actual = cvmap(vmm.update_r, in_axes=(0, 0), chunk_size=size)(theta, state)
                self.assert_numeric_equal(actual, expected)
                fallback = cvmap(vmm.e_step, in_axes=(0, 0, 0, 0, 0), chunk_size=size)(
                    theta, state.mu, state.kappa, state.logw, state.mask)
                self.assert_numeric_equal(fallback, expected_fallback)
                np.testing.assert_allclose(actual.r.sum(axis=-1), 1, rtol=1e-8, atol=1e-10)

    def test_distances_and_dbscan_match_parent_with_uneven_tail(self):
        theta, state = synthetic_mixture()
        state = jax.vmap(vmm.update_r)(theta, state)
        distances = legacy_distances(state)
        self.assertTrue(np.isfinite(distances).all())
        weights = state.r.mean(axis=1)
        expected = jax.vmap(dbscan.dbscan_eps_std)(distances, weights, state.r)
        for size in (None, 1, 2, 5, 8):
            with self.subTest(size=size):
                actual_distances = dbscan.compute_distances(state, batch_size=size)
                self.assert_numeric_equal(actual_distances, distances)
                actual = cvmap(dbscan.dbscan_eps_std, in_axes=(0, 0, 0), chunk_size=size)(
                    actual_distances, weights, state.r)
                self.assert_numeric_equal(actual, expected)
                for field in ("states_traj", "state_counts", "n_states", "C_dbscan"):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


if __name__ == "__main__":
    unittest.main()
