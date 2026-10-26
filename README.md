# ciMIST
### (conformational inference/maximum information spanning tree)

### Decoding protein dynamics with residue-wise conformational inference and tree-structured Potts models

<img src="https://github.com/justktln2/ciMIST/blob/main/images/GR_on_HSP90.png" width=600>

## About
ciMIST is a Python tool for inferring predictive models of conformational entropy from molecular dynamics simulations.
ciMIST infers conformations of single residues and models their global statistics using the maximum information spanning tree approach.
The output of ciMIST is a thermodynamic network model that makes predictions about conformational entropy at local and global scales with Bayesian uncertainty estimation.
[You can read about ciMIST in our preprint](https://www.biorxiv.org/content/10.1101/2025.05.28.656549v2). In the preprint, we show that ciMIST can

- predict global protein conformational entropies consistent with experiment without any fitting to experimental data
- predict local entropies consistent with experimentally-probed dynamics (NMR, HDX)
- identify allosteric hotspots consistent with mutagenesis
- provide thermodynamically quantifiable insight into mechanisms hidden in conformational entropy
- facilitate the visual interpretation of molecular dynamics trajectories

### How it works
1. The trajectory is transformed to internal coordinates with ciMIST's bundled, patched copy of [nerfax](https://github.com/PeptoneLtd/nerfax).
2. Residue configurational probability densities are estimated with von Mises mixture models.
3. Mixture components are clustered using a vectorized implementation of DBSCAN, producing residue conformations.
4. From these, residue conformational entropies and mutual informations are estimated.
5. The Chow-Liu (maximum mutual information spanning tree) algorithm is used for network inference.
6. Entropies are calculated from the network.

Most of this is implemented in JAX, but some of the tree handling is done in networkX.

## Installation
Clone this repository using `git clone https://github.com/justktln2/ciMIST.git .`
After downloading, navigate to the directory containing ciMIST and run the terminal command:
```
python -m pip install .
```

## Requirements (software)
Python requirements are listed in `pyproject.toml`.
MDTraj 1.10.3 or newer is required; 1.10.3 is the validated multichain baseline.

The bundled [nerfax source](src/cimist/utils/nerfax) includes changes for physical
bonded geometry, multichain identities, residue selection, reconstruction and
JAX compatibility. It is distributed under the [Apache 2.0 license](LICENSE).
The [nerfax change notice](src/cimist/utils/nerfax/NOTICE) records the upstream
reference and local modifications; modified source files also carry notices.
Use this bundled implementation when reproducing ciMIST's coordinate extraction.

## Requirements (hardware)
We have typically run ciMIST on CPUs with between 256GB and 512GB of RAM.

## Recommendations
ciMIST has given good quantitative results on between 5 and 10 microseconds of molecular dynamics data sampled at frequencies of once or twice per nanosecond for proteins up to about 300 amino acids long. However, we have found it to be a useful visual aid to the interpretation of trajectories of any length.

## Usage
ciMIST ships with a command line tool `ci-mist`. An analysis template illustrating basic aspects of the API is provided in the outputs of each run in the form of a Jupyter notebook.

This branch loads all protein chains as distinct residue instances and evaluates
every intra- and interchain residue pair. Bonded geometry respects actual
topology connectivity and periodic boxes. Chain IDs and residue identities are
retained in results and visualization. No chain pooling or symmetry assumption
is applied. Internal coordinates describe residue conformations; whole-chain
translation and rotation are not additional variables in this model.

The coordinate extractor supports canonical amino-acid templates. Unsupported
protein residues, including retained ASH names, raise an error identifying the
residue and chain before feature selection can omit them. EM must converge
within 100 updates; successful fits always refresh responsibilities from the
final parameters. Undefined component overlaps or nonfinite clustering results
stop analysis before tree inference.

```
usage: ci-mist [-h] -t TRAJECTORY -s TOPOLOGY -o OUTPUT_PREFIX [--seed SEED]
                  [--min_mass MIN_MASS] [--prior {percs,haldane,jeffreys,laplace}]
                  [--residue-selection RESIDUE_SELECTION]

options:
  -h, --help            show this help message and exit
  -t, --trajectory TRAJECTORY
                        The path to the trajectory file,
                        or to a directory that contains all trajectory files and nothing else.
                        Note that if a directory is supplied, all files in that directory must be valid molecular
                        dynamics trajectory files.
  -s, --topology TOPOLOGY
                        The path to the topology file.
  -o, --output_prefix OUTPUT_PREFIX
                        The prefix for the output directory.
  --residue-selection RESIDUE_SELECTION
                        MDTraj selection of protein residue centers; default: all.
  --seed SEED           The random number generator seed, default 0.
  --min_mass MIN_MASS   Minimum neighborhood probability mass for a DBSCAN core
                        component, in [0, 1]; default 0.01.
  --prior {percs,haldane,jeffreys,laplace}
                        Prior to use for residue entropy and pairwise mutual information estimation with the Dirichlet distribution.
                        Each prior corresponds to adding the same number of pseudocounts to each conformation.
                        Options are:
                            -'haldane' : 0 pseudocounts (DEFAULT)
                            -'percs' : 1/K pseudocounts, where K is the number of conformations
                            -'jeffreys' : 1/2 pseudocounts
                            -'laplace' : 1 pseudocount
                            
                        Note that of these options, only 'haldane' and 'percs' add the same total number of pseudocounts to each distribution.

```

An example demonstrating how to analyze outputs is given in [examples/CRIPT.ipynb](examples/CRIPT.ipynb).

### Memory settings and output format

Newton remains the default solver. For either solver, the defaults are 32
residues per conformational-inference batch and 128 pairs per probability batch,
matching the preceding Newton performance branch. `--VMM_BATCH_SIZE` limits
each EM, responsibility, component-distance and DBSCAN batch by residue;
`--PAIR_BATCH_SIZE` limits pair batches. Both retain every frame and selected
pair. Increase these values to use larger batches; a VMM batch at least as large
as the selected residue count processes all selected residues together. Smaller
batches reduce temporary memory, but do not bound total memory: the trajectory,
fitted models and candidate graph still need storage. Use `ci-mist --help` for
all options.

`RESULTS_ciMIST.h5` is a lossless, versioned sparse snapshot, compressed with
gzip by default. Load it with `cimist.io.load_h5_tree`; use
`cimist.io.H5ProbabilityReader` to read selected pair tables without loading the
whole model. `--probability_storage` controls the in-memory representation;
exports use the sparse format in either mode. Older HDF5 outputs are not
supported by the new reader. Re-run an analysis to create the new format.

### Selecting a marginal subsystem

Select residue centers with an MDTraj expression, for example:

```bash
ci-mist -t trajectory.xtc -s topology.pdb -o selected --seed 0 \
  --residue-selection '(chainid 0 and resSeq 10 to 30) or (chainid 1 and resSeq 40 to 60)'
```

`chainid` is the zero-based **original topology chain index**; `resSeq` is the
input residue number, and `resid` is the zero-based original topology residue
index. Ranges are inclusive. Selections retain canonical topology order;
exported node identities include chain ID, chain index, residue number and
original residue index. Empty selections fail; a singleton gives its marginal
entropy with no pair edges.

Excluded residues remain physically present and supply neighboring atoms needed
for valid angles and torsions. Their states are omitted from inference, so the
new tree estimates the selected residues' marginal entropy, averaged over the
excluded environment. Core-mediated correlations can remain; this does not
condition on the core or remove its physical influence. Every trajectory frame
and every selected intra- and interchain pair is retained. Initialization uses
each residue's original full-protein random key. Fewer centers reduce fitting
and pairwise work, but the full trajectory is still loaded for geometric context.

Python loaders `cimist.io.load_dataset`, `cimist.io.traj_to_dataset` and
`cimist.fit.load_dataset` accept the same `residue_selection` keyword, or an
iterable of original topology residue indices. Their `residue_records` describe
the selected rows; `protein_residue_indices` retains the full key ordering.

Reuse an existing full analysis without repeating conformational inference:

```python
from cimist.io import load_h5_tree, save_tree_h5

full = load_h5_tree("RESULTS_ciMIST.h5")
names = [name for name in full.resnames if 10 <= full.nodes[name]["resSeq"] <= 30]
subset = full.refit_to_subset(names)
save_tree_h5(subset, "selected_tree.h5")
```

Refitting builds a new tree from all selected empirical pairs, preserves the
sample count, prior and estimator, and recomputes uncertainty without changing
the parent model. Missing selected pairs or required pair fields cause an error;
they are not replaced with independent residues. `full.trim(names)` instead
returns a display graph containing the selected nodes and their connecting paths
through the full tree. Its
`full_model_entropy_contribution` attributes follow paper Eq. 3; summing selected
contributions gives their allocated share of full-system entropy, not a
subsystem marginal entropy. For a singleton, aggregate uncertainty is retained;
its per-residue deletion-contribution standard error is undefined (NaN).

To export that full-tree focus view, use
`full.to_pymol(structure, "focus_view", focus_nodes=names)`, where `structure`
is the original reference trajectory. PyMOL selections `focus_residues` and
`connector_residues` distinguish the two groups. `focus.json` records original
identities and explicitly labeled full-model Eq. 3 allocations. To visualize
the newly inferred subsystem, use `subset.to_pymol(structure, "subset_view")`.

`subset.information_matrix(diag=True)` shows empirical pairwise mutual
information, with marginal entropy on the diagonal. `kind="chow_liu"` instead
shows mutual information implied by the fitted tree. Both use empirical
probabilities; posterior estimates for tree edges remain available through
`tree_MIs()`. Passing `resis=names` selects a matrix view of the existing model
in canonical node order; it does not refit a subsystem tree.

For a partial candidate graph, missing empirical pairs appear as `NaN`, while
measured zero MI remains zero. A missing tree edge does not imply zero MI:
`kind="chow_liu"` calculates dependence along the connecting tree path, even
when the direct empirical pair is unavailable. For a disconnected tree (forest),
this view and `joint_distribution` treat the components as independent, extending
paper Eq. 1 to a product of component distributions. Cross-component tree-implied
MI is therefore zero; this is a model assumption, not measured independence.
`kind="hybrid"` places empirical values above and tree-implied values below the
diagonal. These views do not add edges or change the fitted model. Normal
trajectory inference still evaluates every selected pair and fits a connected
tree (or a singleton).

## Visualization of results in PyMOL
The program will create a directory of your choosing containing pre-generated scripts that will visualize trees on protein structure.
These visualizations have the option to be color-coded using the [cmocean](https://matplotlib.org/cmocean/) or [cmasher](https://github.com/1313e/CMasher) colormaps, which provide nice contrasts for protein structures.
In order to access these in PyMOL, you will need to add a line to your `.pymolrc` file that runs the script `pymol_palettes/pymol_palettes.py` included with ciMIST.

## mpnnMIST (experimental)
We have added the option to infer residue states using the ColabFold implementation of ProteinMPNN. This allows residue conformations to be identified with amino acids, in some sense. 

To use this option, you will need to [install my fork of ColabDesign](https://github.com/justktln2/ColabDesign).

This is experimental code whose predictive performance has not been validated.

```
usage: mpnn-mist [-h] [-t TRAJECTORY] [-s TOPOLOGY] [-o OUTPUT_PREFIX]
                 [--temperature TEMPERATURE] [--weights {original,soluble}]
                 [--dropout DROPOUT] [--temperature_mpnn TEMPERATURE_MPNN]
                 [--seed SEED] [--prior {haldane,laplace,jeffreys,percs}]
                 [--mpnn_batch_size MPNN_BATCH_SIZE]

ProteinMPNN-MIST.
    Run maximum information spanning tree on a molecular dynamics ensemble using ProteinMPNN inverse folding to determine residue states.
    WARNING: EXPERIMENTAL.

options:
  -h, --help            show this help message and exit
  -t, --trajectory TRAJECTORY
                        The path to the trajectory file,
                        or to a directory that contains all trajectory files and nothing else.
                        Note that if a directory is supplied, all files in that directory must be valid molecular
                        dynamics trajectory files.
  -s, --topology TOPOLOGY
                        The path to the topology file.
  -o, --output_prefix OUTPUT_PREFIX
                        The prefix for the output directory.
  --temperature TEMPERATURE
                        Temperature parameter for ProteinMPNN
  --weights {original,soluble}
                        ProteinMPNN weights to use.
  --dropout DROPOUT     'dropout' argument for ProteinMPNN
  --temperature_mpnn TEMPERATURE_MPNN
                        Sampling temperature for ProteinMPNN
  --seed SEED           Random seed.
  --prior {haldane,laplace,jeffreys,percs}
                        Prior to use for residue entropy and pairwise mutual information estimation with the Dirichlet distribution.
                        Each prior corresponds to adding the same number of pseudocounts to each conformation.
                        Options are:
                            -'haldane' : 0 pseudocounts (DEFAULT)
                            -'percs' : 1/K pseudocounts, where K is the number of conformations
                            -'jeffreys' : 1/2 pseudocounts
                            -'laplace' : 1 pseudocount
                            
                        Note that of these options, only 'haldane' and 'percs' add the same total number of pseudocounts to each distribution.
  --mpnn_batch_size MPNN_BATCH_SIZE
                        Batch size (in number of trajectory frames) to disatch to ProteinMPNN via jax.vmap. Should be set depending on available memory.
                            DEFAULT: 500.
```


### HDF5 compression

HDF5 output uses lossless gzip level 1 plus shuffle for nonempty numeric
arrays by default. Use `--hdf5_compression none` for uncompressed output;
Python save functions accept `compression=None`. Load the native sparse tree
format with `cimist.io.load_h5_tree("RESULTS_ciMIST.h5")`, or read selected
pairs with `cimist.io.H5ProbabilityReader`. This format replaces the older dense
HDF5 layout; older files are not supported by this reader.


### Fitting and memory options

The `ci-mist` CLI defaults to sparse probability storage, pair batches of 128,
and EM batches of 32 residues. Run normally to use these settings;
select dense in-memory arrays with `--probability_storage dense`.
`--VMM_BATCH_SIZE` also bounds the final responsibility update, component
distance calculation and DBSCAN by residue. It preserves all frames and
components; initialization retains its existing separate residue batches.
Both modes write native sparse HDF5 directly; the CLI no longer writes a
full-tree pickle. The copied analysis notebook loads the HDF5 tree directly.

`--solver bfgs` selects the original bounded concentration objective and optimizer.
Newton solves the same bounded M-step more tightly and can change fitted states
and scientific outputs. For either solver, convergence uses the latest two
post-update mean log-likelihoods. A run that has not converged within 100 EM
updates stops before clustering or scientific export.
