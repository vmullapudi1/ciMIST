
"""
Author: Kaitlin
"""


import os
import argparse
import logging
from typing import List, NamedTuple
from functools import partial, wraps

def positive_batch_size(value):
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("batch size must be a positive integer")
    return size


def probability_mass(value):
    mass = float(value)
    if not 0 <= mass <= 1:
        raise argparse.ArgumentTypeError("probability mass must be finite and between 0 and 1")
    return mass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-t", "--trajectory", type=str,
    help=f"""The path to the trajectory file,
or to a directory that contains all trajectory files and nothing else.
Note that if a directory is supplied, all files in that directory must be valid molecular
dynamics trajectory files.""")
    parser.add_argument("-s", "--topology", type=str,
    help=f"""The path to the topology file.""")
    parser.add_argument("--residue-selection", default=None,
                        help="MDTraj selection of protein residue centers, e.g. 'chainid 0 and resSeq 10 to 30'. chainid and resid are zero-based original topology indices; resSeq is the input residue number. Default: all supported protein residues. Covalent context and every frame are retained.")
    
    parser.add_argument("-o", "--output_prefix", 
    help=f"""The prefix for the output directory.""")
    parser.add_argument("--seed", type=int, default=0,
    help="The random number generator seed, default 0.")
    parser.add_argument("--min_mass", type=probability_mass, default=0.01,
    help="Minimum neighborhood probability mass for a DBSCAN core component, in [0, 1]; default 0.01.")
    parser.add_argument("--prior",
                        type=str,
                        default="haldane",
                        choices={"percs", "haldane", "jeffreys", "laplace"},
                        help="""Prior to use for residue entropy and pairwise mutual information estimation with the Dirichlet distribution.
Each prior corresponds to adding the same number of pseudocounts to each conformation.
Options are:
    -'haldane' : 0 pseudocounts (DEFAULT)
    -'percs' : 1/K pseudocounts, where K is the number of conformations
    -'jeffreys' : 1/2 pseudocounts
    -'laplace' : 1 pseudocount
    
Note that of these options, only 'haldane' and 'percs' add the same total number of pseudocounts to each distribution."""
                        )
    parser.add_argument("--solver", choices=("newton", "bfgs"), default="newton",
                        help="Default: newton. Safeguarded scalar Newton solves the bounded M-step; bfgs uses the original clipped objective. Newton can change scientific outputs.")
    parser.add_argument("--VMM_BATCH_SIZE", type=positive_batch_size, default=None,
                        help="Maximum residues per EM, responsibility, component-distance and DBSCAN batch. Default: all residues, for either solver. Must be positive; smaller batches reduce temporary memory without dropping frames. Initialization is unchanged.")
    parser.add_argument("--PAIR_BATCH_SIZE", type=positive_batch_size, default=2048,
                        help="Maximum residue pairs evaluated together; default: 2048. A value at least the pair count uses the original full map. All pairs and frames are retained.")
    parser.add_argument("--probability_storage", choices=("dense", "sparse"), default="sparse", help="In-memory graph probabilities: sparse (default) reduces memory; dense uses arrays. Tree output is native sparse HDF5 in either mode.")
    parser.add_argument("--hdf5_compression", choices=("gzip", "none"), default="gzip",
                        help="HDF5 numeric arrays: lossless gzip level 1 + shuffle (default), or none.")
    return parser.parse_args(argv)

if __name__ == "__main__":
    # some setup so we don't have to wait for imports to see if the inputs are bad
    args = parse_args()
    

import time
t0 = time.time()

import jax
jax.config.update('jax_enable_x64', True)
import cimist as cst
from cimist.utils.chunked_vmap import vmap_chunked as cvmap
import dill as pkl
from jax import jit, vmap, Array
import jax.numpy as jnp
import mdtraj as md

dt = time.time() - t0
m, s = divmod(dt, 60)

class MDData(NamedTuple):
    traj: md.Trajectory
    angles: Array
    cloud_mask: Array
    residues: List
    residue_records: List = None
    protein_residue_indices: List = None

def time_and_message(message, func, printer=logging.info):
    @wraps(func)
    def wrapped(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        dt = time.time() - t0
        m, s = divmod(dt, 60)
        printer(message + f" in {int(m)} minutes and {round(s, 3)} seconds.")
        return result
    return wrapped

def combine_mixture_fit_batches(batches):
    return cst.ci.vmm.MixtureFitState(*[jnp.concatenate(x) for x in list(zip(*batches))])

def load_dataset(xyz, topology, stride=1, *, residue_selection=None) -> MDData:
    data = cst.io.load_dataset(xyz, topology, stride=stride,
                               residue_selection=residue_selection)
    return MDData(data.traj, data.angles, data.cloud_mask, data.residues,
                  data.residue_records, data.protein_residue_indices)


def make_initialization_keys(data, seed):
    """Retain each residue's original full-protein initialization key."""
    positions = {index: row for row, index in enumerate(data.protein_residue_indices)}
    rows = jnp.asarray([positions[record["residue_index"]] for record in data.residue_records])
    keys = jax.random.split(jax.random.PRNGKey(seed), len(data.protein_residue_indices))
    return keys[rows]


def _describe_residues(data, rows):
    return ", ".join(f"{data.residues[row]} (original topology residue index "
                     f"{data.residue_records[row]['residue_index']}, feature row {row})"
                     for row in rows)


def main():
    args = parse_args()


    import datetime
    today = datetime.datetime.today()
    # CONFIGURE LOGGING
    output_prefix = f"{args.output_prefix}_seed_{args.seed}" + os.sep
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    logfile = output_prefix + "logging.txt"
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w')
    logging.info("Began pipeline at " + str(today) + ".")
    logging.info("Solver: %s; EM batch size: %s; pair batch size: %s",
                 args.solver, args.VMM_BATCH_SIZE, args.PAIR_BATCH_SIZE)

    data = load_dataset(args.trajectory, args.topology, residue_selection=args.residue_selection)
    reference_structure = data.traj.slice(0, copy=False)
    logging.info("Loaded %d residues in %d chains and %d synchronized frames; "
                 "using physical bond/torsion references with terminal masks",
                 len(data.residues), len({r["chain_index"] for r in data.residue_records}),
                 len(data.traj))
    logging.info("Selected original topology residues %s from %d protein residues; "
                 "inferring their marginal network over all frames",
                 [record["residue_index"] for record in data.residue_records],
                 len(data.protein_residue_indices))

    output_prefix = f"{args.output_prefix}_seed_{args.seed}" + os.sep
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    
    # make batches for kmeans++
    init_keys = make_initialization_keys(data, args.seed)
    n_chunks = max(1, data.angles.shape[0] // 20)
    angles_batches = jnp.array_split(data.angles, n_chunks)
    mask_batches = jnp.array_split(data.cloud_mask, n_chunks)
    keys_batches = jnp.array_split(init_keys, n_chunks)
    init_states_func = vmap(cst.ci.vmm.init_random_mixture_state)

    # define path to folder and create it if it does not exist
    logging.info("Initializing mixture model fits")
    mixture_state = combine_mixture_fit_batches(
            [init_states_func(*args) for args in zip(angles_batches, mask_batches, keys_batches)]
    )
    
    logging.info("Beginning von Mises mixture models fits with expectation-maximization...")
    step_if_not_converged = jit(partial(cst.ci.vmm.step_if_not_converged,
                                      gtol=1e-3, gmaxiter=500, solver=args.solver))
    _advance = cvmap(step_if_not_converged, in_axes=(0, 0), chunk_size=args.VMM_BATCH_SIZE)
    num_residues = len(mixture_state.converged)
    if (args.solver == "newton" and args.VMM_BATCH_SIZE is not None
            and args.VMM_BATCH_SIZE < num_residues):
        _advance = jit(_advance)
    num_converged = jnp.sum(mixture_state.converged)
    for i in range(100):
        if num_converged == num_residues:
            break
        print(80*"-")
        advance = time_and_message(f"Completed EM iteration {i} for all residues", _advance)
        mixture_state = advance(data.angles, mixture_state)
        num_converged = jnp.sum(mixture_state.converged)
        if args.solver == "newton":
            last_step = (mixture_state.n_iter - 1)[:, None]
            successful = jnp.take_along_axis(mixture_state.successes, last_step, axis=1)[:, 0]
            if not bool(jnp.all(successful)):
                failed = jnp.where(~successful.astype(bool))[0].tolist()
                statuses = jnp.take_along_axis(mixture_state.statuses, last_step, axis=1)[:, 0]
                message = (f"Newton M-step failed for residues {_describe_residues(data, failed)}; "
                           f"statuses {statuses[jnp.asarray(failed)].tolist()}")
                logging.error(message)
                raise RuntimeError(message)
        
                
        logging.info(f"Reached convergence for {jnp.sum(mixture_state.converged)} out of {len(mixture_state.converged)} residues.")

    if num_converged != num_residues:
        unfinished = jnp.where(~mixture_state.converged)[0].tolist()
        message = (f"EM did not converge after 100 updates for residues {_describe_residues(data, unfinished)}; "
                   "no conformational states or tree were exported")
        logging.error(message)
        raise RuntimeError(message)

    # Refresh after the last M-step, including convergence on the final update.
    update_r = time_and_message("Carried out final update of mixture component responsibilities",
        cvmap(cst.ci.vmm.update_r, in_axes=(0, 0), chunk_size=args.VMM_BATCH_SIZE))
    mixture_state = update_r(data.angles, mixture_state)
    
    D = cst.ci.dbscan.compute_distances(mixture_state, batch_size=args.VMM_BATCH_SIZE)
    weights = mixture_state.r.mean(axis=1)
    cst.ci.dbscan.validate_clustering_inputs(D, weights, mixture_state.r)
    with open(output_prefix + "component_distance.pkl", "wb") as f:
        pkl.dump(D, f)

    compute_states = time_and_message("Determined residue conformations",
        cvmap(partial(cst.ci.dbscan.dbscan_eps_std, min_probability_mass=args.min_mass),
              in_axes=(0, 0, 0), chunk_size=args.VMM_BATCH_SIZE))
    states = compute_states(D, weights, mixture_state.r)
    cst.ci.dbscan.validate_clustering_result(states)
    build_tree = time_and_message("Calculated MIs and inferred maximimum information spanning tree", cst.MIST.from_residue_states)
    tree = build_tree(states, [str(r) for r in data.residues], prior=args.prior, uncertainty=True,
                      pair_batch_size=args.PAIR_BATCH_SIZE, probability_storage=args.probability_storage)
    tree.MI_graph.graph["coordinate_convention"] = "physical-bond-angles-v1"
    tree.T.graph["coordinate_convention"] = "physical-bond-angles-v1"
    selection = dict(semantics="marginal", expression=args.residue_selection,
                     residue_indices=[record["residue_index"] for record in data.residue_records],
                     protein_residue_indices=data.protein_residue_indices,
                     initialization="full-protein-split", seed=args.seed)
    for graph in (tree.MI_graph, tree.T):
        graph.graph["residue_selection"] = selection.copy()
    for record in data.residue_records:
        tree.MI_graph.nodes[record["label"]].update(record)
        tree.T.nodes[record["label"]].update(record)
    fname = output_prefix + "RESULTS_ciMIST.h5"

    cst.io.save_tree_h5(tree, fname,
        compression=None if args.hdf5_compression == "none" else args.hdf5_compression)

    import shutil
    ANALYSIS_PATH = cst.__path__[0] + os.sep + "templates/analysis_template.ipynb"
    shutil.copy(ANALYSIS_PATH, output_prefix + "analysis_template.ipynb")
    logging.info("copied analysis template to output directory.")
    
    

    cst.pymol.tree_cartoon(tree, reference_structure, output_prefix + "pymol" + os.sep)
    cst.io.save_coarse_graining_h5(states, fname,
        compression=None if args.hdf5_compression == "none" else args.hdf5_compression)
    cst.io.save_VMM_h5(mixture_state, fname,
        compression=None if args.hdf5_compression == "none" else args.hdf5_compression)
    

    return None

if __name__ == "__main__":
    main()
