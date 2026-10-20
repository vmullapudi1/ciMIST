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

## Wall time

| System (residues × frames) | Mainline (s) | Sparse BFGS / equivalent PR (s) | Newton PR (s) |
|---|---:|---:|---:|
| PDZ (101 × 10,500) | 1,197.897 | 483.604 | 201.825 |
| ERα (500 × 11,082) | 10,585.319 | 4,090.280 | 1,474.188 |
| mGlu5 (1,608 × 3,236) | OOM — no completed run | 11,741.805 | 7,055.840 |

## CPU time

| System (residues × frames) | Mainline (s) | Sparse BFGS / equivalent PR (s) | Newton PR (s) |
|---|---:|---:|---:|
| PDZ (101 × 10,500) | 25,215.830 | 18,148.814 | 612.919 |
| ERα (500 × 11,082) | 223,157.764 | 166,832.646 | 7,670.450 |
| mGlu5 (1,608 × 3,236) | OOM — no completed run | 307,984.806 | 14,676.622 |

## Peak RAM

| System (residues × frames) | Mainline (GiB) | Sparse BFGS / equivalent PR (GiB) | Newton PR (GiB) |
|---|---:|---:|---:|
| PDZ (101 × 10,500) | 142.745 | 26.666 | 10.469 |
| ERα (500 × 11,082) | 496.227 | 47.244 | 34.352 |
| mGlu5 (1,608 × 3,236) | OOM — no completed run | 57.390 | 38.882 |

## Additional runnable mGlu5 reference

| Configuration | Wall s | CPU s | Peak RAM GiB |
|---|---:|---:|---:|
| Dense BFGS + EM16/pair128 memory batching | 62,821.722 | 388,136.236 | 363.905 |

This modified reference completed; unmodified mainline did not. Its audited outputs match sparse BFGS / equivalent-performance.

## Observed Newton improvements

| System | Wall speedup vs mainline | Wall speedup vs sparse BFGS | Peak RAM reduction vs sparse BFGS | CPU-time reduction vs sparse BFGS |
|---|---:|---:|---:|---:|
| PDZ | 5.94× | 2.40× | 60.74% | 96.62% |
| ERα | 7.18× | 2.77× | 27.29% | 95.40% |
| mGlu5 | Unavailable — mainline OOM | 1.66× | 32.25% | 95.23% |

## Scientific results

| System | Mainline entropy ± SE (nats) | Sparse BFGS entropy ± SE | Newton entropy ± SE | Newton / sparse BFGS shared edges | MI Spearman |
|---|---:|---:|---:|---:|---:|
| PDZ | 86.938032 ± 1.059569 | 86.938032 ± 1.059569 | 87.044021 ± 0.855270 | 58/100 (58.00%) | 0.930147 |
| ERα | 418.471375 ± 1.999903 | 418.471375 ± 1.999903 | 433.315277 ± 1.766322 | 301/499 (60.32%) | 0.960746 |
| mGlu5 | OOM — no result | 1394.869247 ± 28.205934 | 1874.127190 ± 39.927457 | 698/1,607 (43.43%) | 0.852906 |

Mainline and equivalent-performance have identical retained tree edges, entropy, reported SE and MI arrays on PDZ and ERα. For mGlu5 this equality is against the batched dense-BFGS reference. Newton changes these outputs; higher fitted likelihood does not establish biological superiority. Shared-edge counts use undirected residue-name pairs; MI rank correlation uses identical pair ordering.

## EM solver diagnostics

| System | Mainline failed / recorded M-steps | Sparse BFGS failed / recorded M-steps | Newton failed / recorded M-steps |
|---|---:|---:|---:|
| PDZ | 426 / 1,261 | 426 / 1,261 | 0 / 1,343 |
| ERα | 2,250 / 6,216 | 2,250 / 6,216 | 0 / 6,941 |
| mGlu5 | OOM — no completed result | 9,103 / 20,614 | 0 / 23,117 |

The solvers have different stopping rules and diagnostics; these are their recorded failures, not a common residual test. The mGlu5 BFGS counts come from the batched dense reference, whose fit is preserved by equivalent-performance. Zero inner failures does not establish a global EM optimum.


## How the fitted results and trees change

On full PDZ, Newton retains 58 of the 100 BFGS/equivalent-performance tree edges. Its entropy changes from 86.938032 to 87.044021 nats, while reported standard error changes from 1.059569 to 0.855270 nats. The MI rank correlation is 0.930147. A small change in total entropy therefore coexists with substantial changes in selected edges.

On full ERα, the new run uses all 11,082 frames at commit `588f0c7`. Newton retains 301/499 BFGS/equivalent tree edges; MI rank correlation is 0.960746. Entropy changes from 418.471375 to 433.315277 nats, and reported SE from 1.999903 to 1.766322 nats. Input provenance matches the full equivalent-performance run. Execution validation checks shapes, finite outputs, outer convergence and absence of failed Newton M-steps; it is not a claim of BFGS equivalence. Fresh final-parameter likelihood comparisons have not been repeated for this new full ERα run.

On full mGlu5, Newton retains 698 of 1,607 edges from the batched dense-BFGS reference and equivalent-performance fit. Entropy changes from 1394.869247 to 1874.127190 nats and reported standard error from 28.205934 to 39.927457 nats. MI rank correlation is 0.852906. Unmodified mainline did not complete; these are comparisons with the runnable batched reference.

Fresh final-parameter likelihood diagnostics found higher residue likelihoods for all 101 PDZ residues and all 1,608 mGlu5 residues. Gains were 0.454855–14.375457 and 0.614079–11.917854 nats/frame/residue, respectively. Recorded failed inner M-steps changed from 426/1,261 to 0/1,343 on PDZ and from 9,103/20,614 to 0/23,117 on mGlu5. These diagnostics concern the accepted Newton fits preserved by the final composition's equivalence audits. They describe training likelihood, not predictive accuracy or biological superiority.

## Validation and figures

All 11 portable Newton tests passed, covering an independent SciPy Brent reference, bounds, masks, zero-weight handling, objective/gradient checks, ordered forest ties, sparse export, CLI defaults, sparse/dense decoded equality, lossless compression and reader compatibility.

The new comparison reads retained tree edges, entropy, uncertainty, MI arrays and EM diagnostic counts with matched residue ordering. It does not repeat the earlier complete scientific audit or read the large probability tensors.

The [comparison report](README.md) contains system-specific figures in PNG, SVG and PDF, plotted values in CSV, and per-edge membership tables. Adjacency diagrams show selected edges by residue index; they are not spatial contact maps.
