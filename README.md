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
1. The trajectory is transformed to internal coordinates with [nerfax](https://github.com/PeptoneLtd/nerfax).
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
usage: ci-mist [-h] [-t TRAJECTORY] [-s TOPOLOGY] [-o OUTPUT_PREFIX] [--seed SEED]
                  [--min_mass MIN_MASS] [--prior {percs,haldane,jeffreys,laplace}]

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
  --seed SEED           The random number generator seed, default 0.
  --min_mass MIN_MASS   Minimum probability for conformations, default 0.01.
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

The `ci-mist` CLI defaults to sparse probability storage, pair batches of 2048,
and EM batches covering all residues. Run normally to use these settings;
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
