# Equivalent-output performance changes

Reference: `502c65f04c72fdec7b618a23c1136c66af9333d2`.
These changes preserve the paper's product von Mises model (supplement S1,
EM derivation on PDF pages 34–35), preprocessing, frames, K=100, priors,
algorithm order and uncertainty estimator. Forest repair implements the same
residue-deletion replicates (supplement page S6, lines 944–952).

## Measured complete pipelines

| Input (residues × frames) | Wall seconds, reference → candidate | CPU seconds | Peak RSS GiB |
|---|---:|---:|---:|
| PDZ 101 × 10,500 | 1197.897 → 483.604 | 25215.830 → 18148.814 | 142.745 → 26.666 |
| ER 500 × 11,082 | 10585.319 → 4090.280 | 223157.764 → 166832.646 | 496.227 → 47.244 |
| mGlu5 1608 × 3,236 | 62821.722 → 11741.805 | 388136.236 → 307984.806 | 363.905 → 57.390 |

Candidates use EM8/pair128/sparse, seed 0 and K=100. The mGlu5 reference
uses original dense BFGS with EM16/pair128: unbatched baseline ran out of memory.
These are single observations, not controlled solver speedups. Xeon Gold 6354
nodes differ; thread controls differ for ER and mGlu5. The interval covers
imports through raw export, excluding validation and retention compression.
Evidence receipts retain hardware, dependency and measurement provenance.

## Scientific acceptance

Prespecified float64 rtol=1e-8, atol=1e-10 was not relaxed. Audited paired
absolute and relative differences were zero for these completed comparisons:
mixture parameters, states, probabilities, mutual information, ordered trees,
entropy, uncertainty, both scientific pickles and fresh final-parameter
likelihoods/responsibilities. Discrete outputs were checked separately, with
component alignment. Matching nonfinite placeholders preserve baseline behavior.
The only PDB exception is its ten-byte creation date. Three expensive optional
method probes and original pre-pickle alias topology are not certified on large
cases. BFGS failures are retained: mGlu5 has 9,103 failed inner steps in 20,614
M-steps in both versions. Equivalence does not imply successful optimization.

## Reproduction and review

Use Python 3.10.19 and the recorded constraints (JAX/JAXlib 0.4.38,
NumPy 1.26.4, SciPy 1.15.3, NetworkX 2.8.8). From this checkout:

```sh
python -m pip install -c docs/performance/tested-constraints.txt -e .
JAX_ENABLE_X64=1 python -m unittest discover -s tests -v
mkdir -p results
ci-mist -t TRAJECTORY.xtc -s TOPOLOGY.pdb -o results/run --seed 0 \
  --VMM_BATCH_SIZE 8 --PAIR_BATCH_SIZE 128 --probability_storage sparse
```

Use identical trajectory/topology bytes, frame selection, seed and dependencies
for each comparison. Record input checksums, CPU model/affinity, thread controls,
wall time, process-tree CPU/RSS and convergence statuses. Evaluate likelihoods
from saved final parameters, not EM history. Compare all HDF5 and decoded pickle
fields, align labels, check discrete assignments/ordered edges separately, and
report absolute/relative errors with unchanged tolerances.

`source-hashes.json` binds every package source file to the measured composition.
`evidence/` contains original acceptance receipts, not the raw datasets/results;
absolute cluster paths in them are provenance and will not resolve on GitHub.
The complete campaign harness and raw evidence remain in the research workspace.
Public independent full-pipeline reproduction requires publishing that harness
and permitted input/result artifacts separately. These receipts are an audit
trail, not a substitute for downloadable raw evidence. The small portable tests
exercise algorithms and source identity; they do not rerun the large campaign.

## Output compatibility

Dense remains default. Sparse output uses `cimist.sparse-probabilities.v1`:
read trusted binary pickles with `cimist.io.load_sparse_tree(stream)` and use
`SparseProbability.dense_numpy()` when consumers need arrays. Ordinary bare
pickle consumers need adaptation. HDF5 scientific fields/schema are unchanged;
export uses bounded uncompressed blocks. Compression was not selected.
