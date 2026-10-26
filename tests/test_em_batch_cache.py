"""Portable native-setup regression checks for the isolated outer-JIT experiment."""
import ast
from functools import partial
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist.ci import vmm
from cimist.utils.chunked_vmap import vmap_chunked


def setup_advance(state, batch_size, solver="newton", jit=jax.jit,
                  cvmap=vmap_chunked):
    """Execute the actual imported fit setup without running the pipeline."""
    path = Path(vmm.__file__).resolve().parents[1] / "fit.py"
    main = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")

    def assignment(node, name):
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name)

    first = next(i for i, node in enumerate(main.body)
                 if assignment(node, "step_if_not_converged"))
    last = next(i for i, node in enumerate(main.body)
                if assignment(node, "num_converged"))
    code = compile(ast.Module(body=main.body[first:last], type_ignores=[]), str(path), "exec")
    namespace = dict(args=SimpleNamespace(VMM_BATCH_SIZE=batch_size, solver=solver),
                     mixture_state=state, partial=partial, jit=jit, cvmap=cvmap,
                     cst=SimpleNamespace(ci=SimpleNamespace(vmm=vmm)))
    exec(code, namespace)
    return namespace["_advance"]


def fixture():
    rng = np.random.default_rng(20261022)
    n, frames, k, dimensions = 7, 31, 100, 28
    theta = jnp.asarray(rng.normal(0., .25, (n, frames, dimensions)))
    history = jnp.full((n, 8), -123., dtype=jnp.float64)
    state = vmm.MixtureFitState(
        jnp.full(n, k), jnp.asarray(rng.normal(0., .25, (n, k, dimensions))),
        jnp.full((n, k, dimensions), 15., dtype=jnp.float64),
        jnp.full((n, k), -np.log(k)), jnp.ones((n, dimensions), dtype=bool),
        jnp.full((n, frames, k), 1 / k, dtype=jnp.float64), history,
        jnp.zeros(n, dtype=jnp.int64),
        jnp.asarray([True, True, True, True, False, False, False]),
        jnp.full(n, 1e-2), history, history, history, history, history)
    return theta, state


class OuterJitTests(unittest.TestCase):
    def test_native_finite_newton_reuses_compilation_and_preserves_every_field(self):
        theta, initial = fixture()
        candidate = setup_advance(initial, 3)
        original = vmap_chunked(jax.jit(partial(vmm.step_if_not_converged,
            gtol=1e-3, gmaxiter=500, solver="newton")), in_axes=(0, 0), chunk_size=3)
        actual = expected = initial
        jax.block_until_ready((theta, initial))
        messages, counts = [], []

        class CompileLog(logging.Handler):
            def emit(self, record):
                if record.name.startswith("jax"):
                    messages.append(record.getMessage())

        logger = logging.getLogger("jax")
        handler, previous_level = CompileLog(), logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            for update in range(4):
                expected = original(theta, expected)
                jax.block_until_ready(expected)
                start = len(messages)
                with jax.log_compiles(True):
                    actual = candidate(theta, actual)
                    jax.block_until_ready(actual)
                counts.append(sum(message.startswith("Compiling ")
                                  for message in messages[start:]))
                for field in initial._fields:
                    with self.subTest(update=update, field=field):
                        a = np.asarray(getattr(actual, field))
                        b = np.asarray(getattr(expected, field))
                        self.assertEqual(a.shape, b.shape)
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertEqual(a.tobytes(), b.tobytes())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        self.assertGreater(counts[0], 0, "No cold compile event observed")
        self.assertEqual(counts[1:], [0, 0, 0], "Native setup did not reuse its executable")
        print("Native finite-Newton compile counts:", counts)

    def test_other_routes_retain_the_original_wrapper(self):
        state = SimpleNamespace(converged=[False] * 7)
        for solver, batch, outer in (("newton", 3, True),
                ("newton", None, False), ("newton", 7, False),
                ("newton", 10, False), ("bfgs", 3, False),
                ("bfgs", None, False), ("bfgs", 7, False), ("bfgs", 10, False)):
            with self.subTest(solver=solver, batch=batch):
                jit_calls, cvmap_calls = [], []
                wrapper, compiled = object(), object()

                def fake_jit(function):
                    jit_calls.append(function)
                    return compiled if function is wrapper else function

                def fake_cvmap(function, **options):
                    cvmap_calls.append((function, options))
                    return wrapper

                result = setup_advance(state, batch, solver, jit=fake_jit, cvmap=fake_cvmap)
                self.assertEqual(len(jit_calls), 2 if outer else 1)
                self.assertEqual(len(cvmap_calls), 1)
                self.assertIs(cvmap_calls[0][0], jit_calls[0])
                self.assertEqual(cvmap_calls[0][1], dict(in_axes=(0, 0), chunk_size=batch))
                self.assertEqual(jit_calls[0].keywords,
                                 dict(gtol=1e-3, gmaxiter=500, solver=solver))
                self.assertIs(result, compiled if outer else wrapper)


if __name__ == "__main__":
    unittest.main()
