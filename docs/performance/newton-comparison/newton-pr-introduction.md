Hello,

This PR replaces the concentration-parameter optimizer inside ciMIST’s EM procedure with safeguarded Newton updates. It builds on the sparse-storage, batching, forest-repair and export improvements in the equivalent-performance PR.

The motivation is to solve the existing bounded M-step more directly and reliably. Once responsibilities and component means are fixed, each concentration parameter has an independent one-dimensional optimization problem. The original implementation solves those parameters together using BFGS; this PR uses their analytic structure instead.

The product von Mises mixture model, trajectory frames, mixture-component count, priors, and downstream entropy and uncertainty estimators remain unchanged. Solving the M-step differently changes the EM trajectory and fitted results, so this is not an output-equivalent change relative to BFGS.

## Algorithm changes

For each active component and angle, let W be the sum of its responsibilities and C the responsibility-weighted cosine sum about its updated mean. With responsibilities and means fixed, the concentration-dependent objective is

$$Q(\kappa)=C\kappa-W\log I_0(\kappa)+\mathrm{constant}.$$

Its stationary point satisfies

$$I_1(\kappa)/I_0(\kappa)=C/W.$$

The solver evaluates this equation independently for each component and angle, retaining the existing bounds of 10 ≤ κ ≤ 600. It checks for a constrained optimum at either bound, maintains a bracket for an interior root, accepts a Newton step only when the derivative is positive and the step stays inside that bracket, and otherwise bisects. Scaled Bessel functions avoid overflowing the individual Bessel functions.

## Why this is a legitimate M-step

The concentration-dependent objective separates into independent scalar terms; solving each term independently does not discard coupling in this conditional objective. For positive component weight, I1(κ)/I0(κ) is strictly increasing, so each scalar objective is concave. An interior root is its unique conditional maximum; if the root is outside the permitted interval, the constrained maximum is at a bound.

This targets the same bounded conditional objective without subsampling frames, pruning probabilities or approximating the mixture model. The outer EM procedure continues to update responsibilities, weights, means and concentrations in the existing order. A correctly solved conditional M-step does not establish a global mixture optimum: EM can still reach different local solutions.

Masked dimensions remain zero. Zero-weight components retain clipped previous concentrations. Invalid input or failed iterations retain previous concentrations rather than accepting an invalid update. Resultant overshoot within 64 machine epsilons is clipped and recorded as accepted roundoff status 3; larger overshoot is rejected.

The residual threshold is `max(3e-15, 8 eps)` and the bracket-width threshold is `max(1e-10, 8 eps abs(kappa))`, with an iteration cap of `min(maxiter, 100)`. These replace BFGS gradient/line-search stopping rules, so diagnostic counts are not interchangeable. `--solver bfgs` remains available with the original frame-based objective.

## Default settings and output

The CLI defaults to Newton, full-residue EM, pair batches of 128 and exact sparse probability storage. The equivalent-performance PR defaults to BFGS, EM batches of 8, pair batches of 128 and sparse storage. Dense arrays/pickles are opt-in with `--probability_storage dense`. Default sparse tree pickles use `cimist.io.load_sparse_tree(stream)`.

HDF5 retains its logically dense scientific schema and uses lossless gzip level 1 plus shuffle by default. **All performance measurements below used compression-disabled output**, corresponding to `--hdf5_compression none` on the current CLI. They exclude compression overhead; the default compression path is covered separately by exact round-trip and reader-compatibility tests.

## Comparison scope

The tables compare **mainline at `502c65f`**, **sparse BFGS / equivalent-performance PR**, and **Newton PR** on matching inputs, K=100 and seed 0. Wall time covers imports through raw scientific export; CPU time is cumulative process CPU consumption. Validation and later retention compression are excluded.

**Unmodified mainline ran out of memory on mGlu5.** Its mainline entries are therefore unavailable. A separately identified dense BFGS reference with EM16/pair128 memory batching completed and provides the mGlu5 reference fit. It must not be presented as an unmodified-mainline result.

These are single whole-configuration observations with node, CPU-affinity and threading differences. They describe measured resource use and do not isolate causal speedup from the Newton solver alone.
