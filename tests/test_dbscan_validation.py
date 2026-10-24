"""Undefined S2 overlaps must fail before producing hard states (float64)."""
from types import SimpleNamespace
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np

from cimist.ci import dbscan, vmm


class DbscanValidation(unittest.TestCase):
    def setUp(self):
        self.r = jnp.asarray([[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
                              [[0.7, 0.3], [0.6, 0.4], [0.4, 0.6], [0.3, 0.7]]])

    def test_rejects_invalid_responsibilities(self):
        invalid = (
            (self.r.at[0, :, 1].set(0), "zero or nonfinite total responsibility"),
            (self.r.at[0, 0, 0].set(jnp.nan), "must be finite"),
            (self.r.at[0, 0, 0].set(jnp.inf), "must be finite"),
            (self.r.at[0, 0, 0].set(-0.1), "must be nonnegative"),
            (self.r.at[0, 0, :].set(0), "Each observation"),
        )
        for r, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                dbscan.compute_distances(SimpleNamespace(r=r), batch_size=1)

    def test_k100_underflow_reproducer_fails(self):
        # Same bounded VMM parameters as the paper audit: 98 components have
        # zero realized responsibility, although their log weights are finite.
        theta = jnp.asarray([[0.]] * 20 + [[np.pi]] * 20)
        mu = jnp.asarray([[0.], [np.pi]] + [[np.pi / 2]] * 98)
        kappa = jnp.full((100, 1), 600.)
        logw = jnp.asarray([0., 0.] + [-500.] * 98)
        r = jax.jit(vmm.e_step)(theta, mu, kappa, logw, jnp.asarray([True]))
        np.testing.assert_array_equal(r.sum(axis=0), [20., 20.] + [0.] * 98)
        with self.assertRaisesRegex(ValueError, "S2 overlap is undefined"):
            dbscan.compute_distances(SimpleNamespace(r=r[None]))

    def test_k100_positive_masses_retain_two_known_states(self):
        # Zero entries are valid: all 100 components have positive mass, and
        # the two disjoint supports have normalized overlap zero (S2 D = 1).
        r = jnp.zeros((40, 100)).at[:20, :50].set(0.02).at[20:, 50:].set(0.02)
        expected_D = np.ones((100, 100))
        expected_D[:50, :50] = expected_D[50:, 50:] = 0
        for batch_size in (None, 1, 8):
            with self.subTest(batch_size=batch_size):
                D = dbscan.compute_distances(SimpleNamespace(r=r[None]), batch_size=batch_size)[0]
                np.testing.assert_allclose(D, expected_D, rtol=0, atol=1e-15)
                dbscan.validate_clustering_inputs(D, r.mean(axis=0), r)
                states = jax.jit(dbscan.dbscan_eps_std)(D, r.mean(axis=0), r)
                dbscan.validate_clustering_result(states)
                self.assertEqual(int(states.n_states), 2)
                np.testing.assert_array_equal(np.asarray(states.state_counts)[:2], [20, 20])
                np.testing.assert_allclose(states.S_mle, np.log(2), rtol=0, atol=1e-15)

    def test_valid_zero_entries_and_transforms_preserve_results(self):
        r = self.r.at[0, 0, :].set(jnp.asarray([1., 0.]))
        def legacy_distances(one):
            cross = jnp.sqrt(one).T @ jnp.sqrt(one)
            norm = jnp.sqrt(jnp.diag(1 / jnp.diag(cross)))
            return 1 - norm @ cross @ norm
        expected_D = jax.vmap(legacy_distances)(r)
        D = dbscan.compute_distances(SimpleNamespace(r=r), batch_size=1)
        np.testing.assert_array_equal(D, expected_D)
        weights = r.mean(axis=1)
        dbscan.validate_clustering_inputs(D, weights, r)
        direct = jax.vmap(dbscan.dbscan_eps_std)(D, weights, r)
        compiled = jax.jit(jax.vmap(dbscan.dbscan_eps_std))(D, weights, r)
        checked = jax.jit(checkify.checkify(jax.vmap(dbscan.dbscan_eps_std)))
        error, checked_result = checked(D, weights, r)
        error.throw()
        dbscan.validate_clustering_result(compiled)
        for actual in (compiled, checked_result):
            for a, e in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(direct)):
                np.testing.assert_allclose(a, e, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(actual.states_traj, direct.states_traj)
            np.testing.assert_array_equal(actual.state_counts, direct.state_counts)

    def test_invalid_distances_weights_and_epsilon_fail(self):
        D = dbscan.compute_distances(SimpleNamespace(r=self.r))
        weights = self.r.mean(axis=1)
        for bad_D, bad_weights, k, message in (
            (D.at[0, 0, 1].set(jnp.nan), weights, 1, "distances must be finite"),
            (D, weights.at[0, 0].set(jnp.nan), 1, "weights must be finite"),
            (D, weights.at[0, 0].set(-0.1), 1, "weights must be finite"),
            (D, weights, jnp.inf, "epsilon must be finite"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                dbscan.validate_clustering_inputs(bad_D, bad_weights, self.r, k=k)
        with self.assertRaisesRegex(ValueError, "distances must be finite"):
            dbscan.dbscan_eps_std(D[0].at[0, 0].set(jnp.nan), weights[0], self.r[0])
        with self.assertRaisesRegex(ValueError, "epsilon must be finite"):
            dbscan.matrix_dbscan(D[0], weights[0], self.r[0], jnp.nan, 0.01)

    def test_checkified_transformed_calls_report_invalid_inputs(self):
        D = dbscan.compute_distances(SimpleNamespace(r=self.r))
        weights = self.r.mean(axis=1)
        checked = jax.jit(checkify.checkify(jax.vmap(dbscan.dbscan_eps_std)))
        error, _ = checked(D.at[0, 0, 0].set(jnp.nan), weights, self.r)
        with self.assertRaisesRegex(checkify.JaxRuntimeError, "distances must be finite"):
            error.throw()
        checked_distances = jax.jit(checkify.checkify(
            lambda r: dbscan.compute_distances(SimpleNamespace(r=r))))
        error, _ = checked_distances(self.r.at[1, :, 1].set(0))
        with self.assertRaisesRegex(checkify.JaxRuntimeError, "S2 overlap is undefined"):
            error.throw()

    def test_invalid_clustering_results_do_not_reach_tree(self):
        D = dbscan.compute_distances(SimpleNamespace(r=self.r))
        states = jax.jit(jax.vmap(dbscan.dbscan_eps_std))(D, self.r.mean(axis=1), self.r)
        with self.assertRaisesRegex(ValueError, "nonfinite conformational"):
            dbscan.validate_clustering_result(states._replace(dbscan_S=jnp.full(2, jnp.nan)))
        with self.assertRaisesRegex(ValueError, "zero conformational"):
            dbscan.validate_clustering_result(states._replace(dbscan_p=jnp.zeros((2, 2))))


if __name__ == "__main__":
    unittest.main()
