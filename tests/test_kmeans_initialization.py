"""Frozen original initializer outputs; data seed 197, draw seeds 0/1/7.

Captured before the scan change from baseline 9f30db090f356131dbcf7103e9311f2b46f10151.
Original spherical_kmeans_plus_plus source SHA256:
3c2afdabe79d0799f4ae2f642347e3a2c3c142ff0bc04bba299a332f98b87b99.
NumPy 1.26.4, JAX/JAXlib 0.4.38, float64. Center indices below select the first
matching observation when duplicate observations have identical coordinates.
Parameter hashes independently freeze the original initialization key split,
concentrations, weights, masks, and modes; no expected values are regenerated.
"""
import hashlib
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist.ci.vmm import init_random_mixture_state, spherical_kmeans_plus_plus


CENTER_INDICES = {
    "random": {0: [7, 16, 11, 8, 6, 5, 10],
               1: [5, 16, 2, 12, 4, 9, 1],
               7: [7, 4, 16, 11, 10, 13, 6]},
    "one_component": {0: [7], 1: [12], 7: [5]},
    "duplicates": {0: [3, 0, 6] + [0] * 9,
                   1: [3, 6, 0] + [0] * 9,
                   7: [3, 0, 6] + [0] * 9},
    "constant": {seed: [0] * 9 for seed in (0, 1, 7)},
    "one_frame": {seed: [0] * 7 for seed in (0, 1, 7)},
    "two_frames": {0: [1, 0] + [0] * 5,
                   1: [0, 1] + [0] * 5,
                   7: [1, 0] + [0] * 5},
}
PARAMETER_HASHES = {
    0: {"mu": "fecfc98e05048fdc365b012424bc69e39012a0243e77f1a4f50096e32fb03759",
        "kappa": "d430d43b206b2edf47c08b191279392548f3ccde4d5037a021c7f58e016afe8d"},
    1: {"mu": "89c2bc9f8a4c831301f12ae9ca66994c9d872697a5538b0147296e742ef80c7a",
        "kappa": "465ef04f99c6e2f5c90f1c9ec249b2cf79bb3ed8054010a31ec999e562818e56"},
    7: {"mu": "377ac443d556c157aa58ab2862c08d3c0e8936052154074a22441e03b032dbce",
        "kappa": "ede7a60229eb6eafe18cf5272bd11f7dbf177a74283b49309c35d061bd745f58"},
}
COMMON_HASHES = {
    "logw": "4603ae6b66af3a25d0b9d12e53c2e61b33482b3ff4819807868bcb97dcec22d5",
    "mask": "fc4c7521bd768bb3951bfbf2e5f3214ebce38a0d55acecc133447a0facedc35a",
}


def observations():
    points = np.random.default_rng(197).uniform(-2.5, 2.5, (17, 28))
    points[:, ::5] = 0
    return points


class KmeansInitialization(unittest.TestCase):
    def test_frozen_center_selection_including_degenerate_samples(self):
        points = observations()
        constant = np.broadcast_to(np.linspace(.1, 2.8, 28), (5, 28)).copy()
        constant[:, ::5] = 0
        cases = {"random": (points, 7), "one_component": (points, 1),
                 "duplicates": (np.repeat(points[:3], 3, axis=0), 12),
                 "constant": (constant, 9), "one_frame": (points[:1], 7),
                 "two_frames": (points[:2], 7)}
        for name, (theta, components) in cases.items():
            for seed, indices in CENTER_INDICES[name].items():
                with self.subTest(case=name, seed=seed):
                    actual = spherical_kmeans_plus_plus(jnp.asarray(theta), components,
                                                        jax.random.PRNGKey(seed))
                    np.testing.assert_array_equal(actual, theta[indices])

    def test_vmap_preserves_original_center_selection(self):
        theta = jnp.asarray(observations())
        keys = jnp.stack([jax.random.PRNGKey(seed) for seed in (0, 1, 7)])
        actual = jax.jit(jax.vmap(lambda key: spherical_kmeans_plus_plus(theta, 7, key)))(keys)
        expected = np.stack([np.asarray(theta)[CENTER_INDICES["random"][seed]] for seed in (0, 1, 7)])
        np.testing.assert_array_equal(actual, expected)

    def test_frozen_masked_initial_parameters_and_key_split(self):
        theta = jnp.asarray(observations())
        mask = jnp.asarray(np.arange(28) % 5 != 0)
        for seed, expected in PARAMETER_HASHES.items():
            with self.subTest(seed=seed):
                state = init_random_mixture_state(theta, mask, jax.random.PRNGKey(seed),
                                                   n_components=7, maxiter=4)
                for name, digest in {**expected, **COMMON_HASHES}.items():
                    actual = hashlib.sha256(np.asarray(getattr(state, name)).tobytes()).hexdigest()
                    self.assertEqual(actual, digest, name)
                np.testing.assert_array_equal(np.asarray(state.kappa)[:, ~np.asarray(mask)], 0)


if __name__ == "__main__":
    unittest.main()
