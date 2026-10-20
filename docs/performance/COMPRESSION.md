# Lossless HDF5 compression default

The HDF5 save functions now default to gzip level 1 plus byte shuffle for
nonempty numeric arrays. Both dense and sparse graph probabilities, tree
statistics, coarse-grained states and saved mixture arrays use this policy.
Scalar datasets, empty arrays and strings remain unfiltered. Attributes,
dataset names, shapes and dtypes are unchanged. No scale-offset or precision
reduction is used. Scientific pickles are unaffected.

Compression is transparent to the existing HDF5 reader. Gzip is a standard
HDF5 filter; no extra codec plugin is needed. See the
[h5py lossless filter documentation](https://docs.h5py.org/en/stable/high/dataset.html#lossless-compression-filters).

## Controls

The CLI default is `--hdf5_compression gzip`. Select
`--hdf5_compression none` for uncompressed export. The Python functions
`save_tree_h5`, `save_coarse_graining_h5` and `save_VMM_h5` accept
`compression="gzip"` (default) or `compression=None`.

Chunks retain complete trailing dimensions (complete probability matrices)
where possible, target at most 1 MiB, and contain at most 128 rows. Oversized
rows are split across chunks. Sparse export keeps the existing 8 MiB/128-row
explicit-buffer target and aligns writes to chunk row boundaries. One row
larger than 8 MiB still requires a larger explicit buffer; HDF5 and codec
buffers are additional. Dense export retains its existing stack allocation.

## Validation scope

`tests/test_hdf5_compression.py` exercises byte-exact numeric round trips,
including signed zero, NaN, infinity, integers and booleans; unfiltered
scalar/empty/string datasets; mixed sparse/dense input; final partial blocks;
oversized rows; on-disk size reduction for a sparse fixture; CLI defaults and
opt-out; all three export groups; and unchanged-reader equivalence between
compressed and uncompressed output. Existing model/forest tests remain in the
suite. These are portable regression fixtures, not new full-protein fits.

## Benchmark configuration and provenance

The performance tables report runs with compression disabled, corresponding
to `--hdf5_compression none` on the current CLI. The acceptance receipts and
measured source inventory predate the gzip default; the tables do not include
its compression overhead. Compression can trade CPU and write/read time for
smaller files. Full PDZ/ER/mGlu5 compressed pipeline timings have not been run.

`benchmark-source-hashes.json` retains the original measured source hashes.
`source-hashes.json` identifies the current source; only `src/cimist/io.py`
and `src/cimist/fit.py` differ from that benchmark inventory. `fit.py` also
selects the measured memory configuration by default (sparse storage and
pair128; EM8 for equivalent, full-residue EM for Newton). This distinction
preserves the campaign's evidence without claiming it tested the new default.
The Newton branch's benchmark inventory refers to its own historical Newton
composition, not the equivalent-output parent's composition.

## Executed regression check

All 9 tests passed on 2026-10-20 (3.948 seconds reported by unittest),
in an isolated source copy on NucleusA089, CPUs 0–3, using the existing
benchmark environment (Python 3.10.19, h5py 3.16.0), with JAX float64 enabled
and OpenBLAS/OpenMP threads limited to one. No full-protein fit was rerun.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=1 PYTHONPATH=src \
  python -m unittest discover -s tests -v
```
