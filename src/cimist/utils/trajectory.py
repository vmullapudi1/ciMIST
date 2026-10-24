"""
Functions for manipulating trajectories and converting between Cartesian and internal coordinates.
"""
from functools import partial
from typing import Callable, Optional, Tuple
from . import nerfax

from jax.tree_util import tree_map
from jax import Array, vmap, pmap
from jax.lax import scan
import jax.numpy as jnp
import mdtraj as md
import numpy as np

def traj_to_internal(traj: md.Trajectory, residue_indices=None,
                     frame_batch_size=64, residue_batch_size=128) -> dict:
    """Extract physical internal coordinates using the original topology.

    Selection chooses residue centers without removing their covalent context.
    Scaffold arrays retain their frame axes; feature_mask and residue_records
    are static. Atom-presence cloud_mask remains distinct from feature_mask.
    """
    references, feature_mask, records = nerfax.parser.protein_feature_references(
        traj.topology, residue_indices)
    bonds, angles = nerfax.parser.get_data_masks(
        traj.xyz, references, traj.unitcell_vectors, frame_batch_size, residue_batch_size)
    canonical = nerfax.parser.IUPACData.protein_letters_3to1
    seq = "".join(canonical[record["name"].title()] for record in records)
    point_ref, cloud_mask = nerfax.parser.get_point_ref_and_cloud_mask(seq)
    # Folding assumes one complete peptide chain in row order, without caps.
    n, ca, c = references[:, 0, :3].T
    previous_c = np.r_[-1, c[:-1]]
    previous_ca = np.r_[-1, ca[:-1]]
    next_n = np.r_[n[1:], -1]
    expected = np.stack((np.stack((n, ca, c, next_n), axis=-1),
                         np.stack((previous_ca, previous_c, n, ca), axis=-1),
                         np.stack((previous_c, n, ca, c), axis=-1)), axis=1)
    reconstructable = (len(records) >= 2
                       and len({record["chain_index"] for record in records}) == 1
                       and np.array_equal(references[:, :3], expected))
    return dict(angles_mask=angles, bond_mask=bonds,
                cloud_mask=np.broadcast_to(cloud_mask, (len(traj),) + cloud_mask.shape),
                point_ref_mask=np.broadcast_to(point_ref[:3], (len(traj),) + point_ref[:3].shape),
                feature_mask=feature_mask, residue_records=records,
                reconstruction_supported=reconstructable)

def internal_to_traj(internal_coordinates: dict,
                     reference_traj: md.Trajectory,
                     superpose: bool = True,
                     mean_bond_length: bool = True
                     ) -> md.Trajectory:
    records = internal_coordinates.get("residue_records")
    if records is not None and records != nerfax.parser.residue_records(reference_traj.topology):
        raise ValueError("Reconstruction requires a reference with the same complete residue mapping")
    _, _restrict, _to_list = nerfax.parser.get_scnet_loader_fns(reference_traj)
    cartesian = internal_to_cartesian_scnet(internal_coordinates,
                                            _to_list,
                                            mean_bond_length=mean_bond_length
                                            )
    traj = scnet_cartesian_to_traj(cartesian, reference_traj)
    # Folding chooses an arbitrary rigid pose, independent of the input cell.
    traj.unitcell_vectors = None
    if superpose:
        traj.superpose(_restrict(reference_traj))
    return traj

def restrict_to_scnet(traj: md.Trajectory):
    _, _restrict, _ = nerfax.parser.get_scnet_loader_fns(traj)
    return _restrict(traj)

def internal_to_cartesian_scnet(internal_coordinates: dict,
                     scnet_to_list: Callable,
                     mean_bond_length: Optional[bool] = True
                     ) -> Array:
    
    if not internal_coordinates.get("reconstruction_supported", True):
        raise ValueError("Internal-coordinate reconstruction requires one complete connected chain with at least two residues and no external covalent context")
    # Static feature/identity metadata must not be passed through vmap.
    cloud_mask = jnp.asarray(internal_coordinates["cloud_mask"][0])
    point_ref_mask = jnp.asarray(internal_coordinates["point_ref_mask"][0])
    angles_mask = vmap(convert_scnet_to_natural)(internal_coordinates["angles_mask"])
    bond_mask = jnp.asarray(internal_coordinates["bond_mask"])
    if mean_bond_length:
        bond_length = bond_mask.mean(axis=0)
        cartesian = vmap(lambda A: 
                    nerfax.plugin.protein_fold(cloud_mask, point_ref_mask, 
                                                  A,
                                                  bond_length
                                            )
            )(angles_mask)
    else:
        cartesian = vmap(lambda A, l: 
                    nerfax.plugin.protein_fold(cloud_mask, point_ref_mask,
                                                A, l
                                                )
                    )(angles_mask, bond_mask)
    return cartesian

def scnet_cartesian_to_traj(cartesian: Array,
                            reference_traj: md.Trajectory,
                            ) -> md.Trajectory:
    _, _restrict, _to_list = nerfax.parser.get_scnet_loader_fns(reference_traj)
    # coordinates, properly formatted for mdtraj
    xyz = vmap(_to_list)(cartesian)
    # only need the first frame for topology
    topology = _restrict(reference_traj.slice(0, copy=False)).topology
    traj = md.Trajectory(xyz, topology)
    if len(traj) == len(reference_traj):
        traj.time = reference_traj.time.copy()
        traj.unitcell_vectors = reference_traj.unitcell_vectors
    return traj

def flatten_masks(internal: dict) -> Tuple[Array, Array]:
    """Return residue/frame/features and the physical statistical mask."""
    values = internal["angles_mask"]
    # NumPy scaffolds share the feature-contiguous backing allocation.
    if isinstance(values, np.ndarray):
        angles = values.transpose(2, 0, 1, 3).reshape(values.shape[2], values.shape[0], 28)
    else:
        angles = jnp.concatenate((values[:, 0], values[:, 1]), axis=-1).swapaxes(0, 1)
    if "feature_mask" in internal:
        mask = internal["feature_mask"]
    else:
        mask = jnp.concatenate((internal["cloud_mask"][0], internal["cloud_mask"][0]), axis=-1)
    return jnp.asarray(angles), jnp.asarray(mask)

def traj_to_scnet(traj: md.Trajectory) -> Tuple[Array, str]:
    """
    convert mdtraj.Trajectory object to cartesian coordinates in sidechainnet
    format

    Parameters
    ----------
    traj : mdtraj.Trajectory
        DESCRIPTION.

    Returns
    -------
    coords : jax array of dimensions T x L x 14 x 3
        cartesian coordinates.
    seq : str
        the concatenated protein sequence; retain the topology for chain boundaries

    """
    _parse_coords, _, _ = nerfax.parser.get_scnet_loader_fns(traj)
    coords = vmap(_parse_coords)(traj.xyz)
    seq = nerfax.parser.protein_sequence(traj.topology)
    return coords, seq
    
#@partial(jit, static_argnames=("seq",))
def scnet_cartesian_to_internal(scnet_coords: Array, seq: str) -> dict:
    """Convert (T, R, 14, 3) Angstrom coordinates for one connected chain.

    Sequence alone cannot describe chain boundaries, covalent gaps, caps, or
    periodic imaging. Use traj_to_internal with the original topology for those.
    """
    internal = nerfax.parser.make_scaffolds(scnet_coords, seq)
    internal["angles_mask"] = vmap(convert_natural_to_scnet)(internal["angles_mask"])
    return internal

# -----------------------------------------------------------------------------
#adapted from 
# https://github.com/PeptoneLtd/nerfax/blob/8d04e1a919c7a86ad9372917b12de3991a117cbe/nerfax/plugin.py#L7-L30
roll_first_col_in_last_axis = lambda x, roll=1: jnp.concatenate([jnp.roll(x[...,:1], roll, axis=-2), x[...,1:]], axis=-1)

# Convert between scnet odd conventions and 'natural' (i.e. nerfax) ones.
def converter(input, roll):
    '''
    if a dict, it's assumed to have `angles_mask` key which is altered.
    if an array, assumed to be the `angles_mask` array in mp_nerf to alter
    '''
    def _converter(angles_mask):
        # Roll the first angle and dihedral round by 1 to match 'natural' syntax
        angles_mask = roll_first_col_in_last_axis(angles_mask, roll=roll)

        # Fix difference in how angle is specified
        angles, torsions = angles_mask
        # no conversion needed here because I don't care, but this is in nerfax code
        #angles = jnp.pi-angles # due to historical scnet reasons, the scnet angle is defined as pi-angle
        angles_mask = jnp.stack([angles, torsions])
        return angles_mask
    if isinstance(input, dict):
        # is scaffolds dict, where we fix angles mask
        return {**input, 'angles_mask': _converter(input['angles_mask'])}
    else:
        # Is angles_mask tensor of shape (2,L,14)
        return _converter(input)
    
convert_scnet_to_natural = partial(converter, roll=1)
convert_natural_to_scnet = partial(converter, roll=-1)

def chunked_pmap(f, chunksize, *, batch_size=None):
    ## Adapted from https://github.com/Joshuaalbert/jaxns/blob/master/jaxns/internals/maps.py#L101
    def _f(*args, batch_size=batch_size, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = scan(body, (), (args, kwargs))
            return result

        if chunksize > 1:
            if batch_size is None:
                batch_size = args[0].shape[0] if len(args) > 0 else None
            assert batch_size is not None, "Couldn't get batch_size, please provide explicitly"
            remainder = batch_size % chunksize
            extra = (chunksize - remainder) % chunksize
            args = tree_map(lambda arg: _pad_extra(arg, chunksize), args)
            kwargs = tree_map(lambda arg: _pad_extra(arg, chunksize), kwargs)
            result = pmap(queue)(*args, **kwargs)
            result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
            if extra > 0:
                result = tree_map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f


def _pad_extra(arg, chunksize):
    ## Adapted from https://github.com/Joshuaalbert/jaxns/blob/master/jaxns/internals/maps.py#L101
    N = arg.shape[0]
    remainder = N % chunksize
    #print(N, remainder, chunksize)
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = (chunksize - remainder) % chunksize
        arg = jnp.concatenate([arg] + [arg[0:1]] * extra, axis=0)
        N = N + extra
    else:
        extra = 0
    arg = jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:])
    return arg
