"""
Input and output operations for molecular dynamics trajectories and ciMIST objects.
"""

import mdtraj as md
import os
import dill as pkl
from itertools import islice
from functools import partial
from typing import Any, NamedTuple, Optional
from . import utils

from jax import Array
import cimist as cst
import numpy as np
import jax.numpy as jnp
import networkx as nx
import pandas as pd
import h5py
from .sparse_probability import SparseProbability

class DataSet(NamedTuple):
    traj: md.Trajectory
    angles: Array
    cloud_mask: Array

def load_dataset(trajfile: str, topfile: str, stride: int=1) -> DataSet:
    """Load protein angular features while preserving original atom identities.

    All trajectory atoms are retained as covalent context, including caps or
    other nonprotein atoms. Feature rows follow
    ``utils.nerfax.parser.residue_records(dataset.traj.topology)``. Frames, times,
    chain IDs and periodic boxes retain their input order and values.
    """

    traj = load_traj(trajfile, topfile, stride=stride, protein_only=False)
    return traj_to_dataset(traj)

def traj_to_dataset(traj: md.Trajectory):
    """Extract physical protein angles without modifying the supplied trajectory.

    Feature rows follow ``utils.nerfax.parser.residue_records(traj.topology)``;
    the returned trajectory also retains any nonprotein context atoms.
    """

    angles, cloud_mask = utils.flatten_masks(utils.traj_to_internal(traj))
    return DataSet(traj, jnp.asarray(angles), jnp.asarray(cloud_mask))

def load_traj(trajpath: str, top: str, stride: int = 1, protein_only=True) -> md.Trajectory:
    """
    Load and preprocess molecular dynamics trajectory for analysis.
    
    Handles both single trajectory files and directories containing multiple
    trajectory files. Automatically filters to protein atoms and preserves
    chain identities, bonded connectivity, times and periodic boxes.
    
    Parameters
    ----------
    trajpath : str
        Path to trajectory file or directory containing trajectory files.
        Supported formats: XTC, DCD, H5, PDB, etc. (MDTraj compatible)
    top : str
        Path to topology file defining molecular structure and connectivity
    stride : int, default=1
        Frame sampling interval - loads every nth frame to reduce memory usage
    protein_only : bool, default=True
        If True, extract only protein atoms and discard water/ions
        
    Returns
    -------
    md.Trajectory
        Preprocessed trajectory with:
        - Protein atoms only (if protein_only=True)
        - Original coordinates, times and periodic boxes
        - Reduced to every stride-th frame
        
    Raises
    ------
    IOError
        If trajpath does not exist as file or directory
    ValueError
        If protein filtering would omit an unsupported peptide residue
        
    Examples
    --------
    >>> # Load every 10th frame from single file
    >>> traj = load_traj("sim.xtc", "protein.pdb", stride=10)
    >>> 
    >>> # Load from directory of files
    >>> traj = load_traj("trajectory_dir/", "system.pdb")
    """

    if os.path.isfile(trajpath):
        traj = md.load(trajpath, top=top, stride=stride)
    elif os.path.isdir(trajpath):
        files = [trajpath + os.sep + f for f in sorted(os.listdir(trajpath))]
        if not files:
            raise ValueError(f"Trajectory directory is empty: {trajpath}")
        parts = [md.load(path, top=top, stride=stride) for path in files]
        original_topology = parts[0].topology
        chain_ids = [chain.chain_id for chain in original_topology.chains]
        if any(part.topology != original_topology
               or [chain.chain_id for chain in part.topology.chains] != chain_ids
               for part in parts[1:]):
            raise ValueError("Trajectory files must have identical ordered atom and chain identities")
        traj = md.join(parts)
        # MDTraj's join copies topology without retaining chain IDs.
        traj.topology = original_topology
        del parts
    else:
        raise IOError(f"No such file or path \'{trajpath}\'")
    if protein_only:
        # Validate before MDTraj's protein selection can silently remove ASH.
        utils.nerfax.parser.residue_records(traj.topology)
        indices = traj.topology.select('protein')
        if not len(indices):
            raise ValueError("Topology contains no protein atoms")
        if len(indices) != traj.n_atoms:
            original_residues = [residue for residue in traj.topology.residues
                                 if residue.is_protein]
            original_chains = list(dict.fromkeys(residue.chain for residue in original_residues))
            traj = traj.atom_slice(indices)
            # MDTraj 1.10 drops chain IDs and changes zero residue numbers on slicing.
            for chain, original in zip(traj.topology.chains, original_chains):
                chain.chain_id = original.chain_id
            for residue, original in zip(traj.topology.residues, original_residues):
                residue.resSeq = original.resSeq
    # Keep coordinates and periodic boxes in the same frame for bonded geometry.
    return traj

def to_pickle(obj: Any, path: str) -> None:
    """
    Serialize object using dill
    """
    with open(path, "wb") as f:
        pkl.dump(obj, f)


SPARSE_PICKLE_FORMAT = "cimist.sparse-probabilities.v1"


def dump_sparse_tree(tree, file):
    """Versioned opt-in representation; retains every tree field, no dense restore."""
    pkl.dump({"format": SPARSE_PICKLE_FORMAT, "tree": tree}, file)


def load_sparse_tree(file):
    """Read only trusted pickles. This returns sparse edge objects, not arrays."""
    payload = pkl.load(file)
    if not isinstance(payload, dict) or payload.get("format") != SPARSE_PICKLE_FORMAT:
        raise ValueError("Not a supported sparse tree pickle")
    return payload["tree"]


def _h5_compression_options(shape, dtype, compression):
    """Lossless filters with row-oriented chunks targeting at most 1 MiB."""
    if compression not in (None, "gzip"):
        raise ValueError("HDF5 compression must be 'gzip' or None")
    dtype = np.dtype(dtype)
    if compression is None or not shape or 0 in shape or dtype.kind not in "biufc":
        return {}
    row_bytes = int(np.prod(shape[1:], dtype=object)) * dtype.itemsize
    rows = max(1, min(shape[0], (1024**2) // max(1, row_bytes)))
    chunks = [rows, *shape[1:]]
    # Oversized individual rows must be split to keep codec buffers bounded.
    while int(np.prod(chunks, dtype=object)) * dtype.itemsize > 1024**2:
        axis = max(range(len(chunks)), key=chunks.__getitem__)
        chunks[axis] = max(1, (chunks[axis] + 1) // 2)
    return dict(compression="gzip", compression_opts=1, shuffle=True,
                chunks=tuple(chunks))


def _create_h5_dataset(group, name, data=None, *, shape=None, dtype=None,
                       compression="gzip"):
    """Write numeric arrays losslessly; leave scalars/empty/string data unfiltered."""
    options = {}
    if data is not None:
        array = np.asarray(data)
        shape = array.shape if shape is None else shape
        storage_dtype = array.dtype if dtype is None else np.dtype(dtype)
        options["data"] = data
    else:
        storage_dtype = np.dtype(dtype)
    if dtype is not None:
        options["dtype"] = dtype
    options.update(_h5_compression_options(shape, storage_dtype, compression))
    return group.create_dataset(name, shape=shape, **options)


SPARSE_HDF5_FORMAT = "cimist.sparse-hdf5.v1"
_H5_BLOCK_ROWS = 4096
_H5_BLOCK_BYTES = 8 * 1024**2


class ProbabilityRecord(NamedTuple):
    row: int
    edge: tuple
    axes: tuple
    probability: SparseProbability


def _write_value(parent, name, value, compression):
    """Save supported scientific state; reject executable or custom objects."""
    group = parent.create_group(name)
    if value is None:
        group.attrs["kind"] = "none"
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        group.attrs["kind"] = "dataframe" if isinstance(value, pd.DataFrame) else "series"
        if isinstance(value, pd.DataFrame) and len(set(value.dtypes)) > 1:
            raise TypeError("Mixed-dtype DataFrames are not supported")
        _write_value(group, "values", value.to_numpy(), compression)
        _write_index(group, "index", value.index, compression)
        if isinstance(value, pd.DataFrame):
            _write_index(group, "columns", value.columns, compression)
        else:
            _write_value(group, "name", value.name, compression)
    elif isinstance(value, dict):
        group.attrs["kind"] = "dict"
        for index, (key, item) in enumerate(value.items()):
            entry = group.create_group(str(index))
            _write_value(entry, "key", key, compression)
            _write_value(entry, "value", item, compression)
    elif isinstance(value, (tuple, list)):
        group.attrs["kind"] = "tuple" if isinstance(value, tuple) else "list"
        for index, item in enumerate(value):
            _write_value(group, str(index), item, compression)
    else:
        array = np.asarray(value)
        if array.dtype.kind in "US" or (array.dtype.kind == "O" and
                all(isinstance(item, str) for item in array.flat)):
            group.attrs["kind"] = "string" if array.ndim == 0 else "strings"
            group.attrs["numpy_dtype"] = array.dtype.str
            group.create_dataset("value", data=array.astype(object), dtype=h5py.string_dtype("utf-8"))
        elif array.dtype.kind in "biufc":
            group.attrs["kind"] = "scalar" if array.ndim == 0 else "array"
            _create_h5_dataset(group, "value", data=array, compression=compression)
        else:
            raise TypeError(f"Unsupported saved state {name!r}: {type(value).__name__}")
    return group


def _write_index(parent, name, index, compression):
    if isinstance(index, pd.MultiIndex):
        raise TypeError("MultiIndex scientific state is not supported")
    group = parent.create_group(name)
    group.attrs["range"] = isinstance(index, pd.RangeIndex)
    _write_value(group, "name", index.name, compression)
    if isinstance(index, pd.RangeIndex):
        group.attrs["bounds"] = (index.start, index.stop, index.step)
    else:
        _write_value(group, "values", index.to_numpy(), compression)


def _read_index(group):
    name = _read_value(group["name"])
    if group.attrs["range"]:
        return pd.RangeIndex(*group.attrs["bounds"], name=name)
    return pd.Index(_read_value(group["values"]), name=name)


def _read_value(group):
    kind = group.attrs["kind"]
    if kind == "none":
        return None
    if kind in ("scalar", "array"):
        return group["value"][()]
    if kind in ("string", "strings"):
        value = group["value"].asstr()[()]
        return value if kind == "string" else np.asarray(value, dtype=group.attrs["numpy_dtype"])
    if kind in ("list", "tuple"):
        value = [_read_value(group[str(i)]) for i in range(len(group))]
        return tuple(value) if kind == "tuple" else value
    if kind == "dict":
        return {_read_value(group[str(i)]["key"]): _read_value(group[str(i)]["value"])
                for i in range(len(group))}
    if kind == "dataframe":
        return pd.DataFrame(_read_value(group["values"]), index=_read_index(group["index"]),
                            columns=_read_index(group["columns"]))
    if kind == "series":
        return pd.Series(_read_value(group["values"]), index=_read_index(group["index"]),
                         name=_read_value(group["name"]))
    raise ValueError(f"Unsupported saved value kind {kind!r}")


def _probability(value):
    return value if isinstance(value, SparseProbability) else SparseProbability.encode(value)


def _validate_sparse(value):
    shape, dtype = tuple(value.shape), np.dtype(value.dtype)
    positions, values = np.asarray(value.positions), np.asarray(value.values)
    cells = int(np.prod(shape, dtype=object))
    if (len(shape) != 2 or np.asarray(shape).dtype.kind not in "iu"
            or any(size < 0 for size in shape) or dtype.kind != "f"
            or positions.ndim != 1 or values.ndim != 1 or len(positions) != len(values)
            or positions.dtype.kind not in "iu" or values.dtype != dtype
            or (len(positions) and (positions[0] < 0 or positions[-1] >= cells))
            or np.any(positions[1:] <= positions[:-1])
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError("Invalid sparse probability shape, dtype, positions or values")


def _write_probabilities(group, graph, compression="gzip", key="P"):
    """Stream sparse coordinates directly, without decoding or stacking matrices."""
    first = next((attrs[key] for _, _, attrs in graph.edges(data=True) if key in attrs), None)
    first = _probability(first) if first is not None else None
    shape = tuple(first.shape) if first is not None else (0, 0)
    dtype = np.dtype(first.dtype) if first is not None else np.dtype(float)
    cells = int(np.prod(shape, dtype=object))
    position_dtype = np.dtype(np.uint16 if cells <= 2**16 else np.uint32 if cells <= 2**32 else np.uint64)
    target = group.create_group(key)
    target.attrs["shape"], target.attrs["dtype"] = shape, dtype.str
    datasets = {}
    for name, storage_dtype in (("values", dtype), ("positions", position_dtype)):
        options = dict(chunks=(max(1, 1024**2 // storage_dtype.itemsize),), maxshape=(None,))
        if compression == "gzip":
            options.update(compression="gzip", compression_opts=1, shuffle=True)
        elif compression is not None:
            raise ValueError("HDF5 compression must be 'gzip' or None")
        datasets[name] = target.create_dataset(name, shape=(0,), dtype=storage_dtype, **options)
    count = graph.number_of_edges()
    offsets = _create_h5_dataset(target, "offsets", shape=(count + 1,), dtype=np.int64, compression=compression)
    present = _create_h5_dataset(target, "present", shape=(count,), dtype=bool, compression=compression)
    sparse_input = _create_h5_dataset(target, "sparse_input", shape=(count,), dtype=bool, compression=compression)
    offsets[0] = 0
    row = end = 0
    iterator = iter(graph.edges(data=True))
    while row < count:
        probabilities, lengths, flags, input_types, byte_count = [], [], [], [], 0
        while len(flags) < _H5_BLOCK_ROWS and byte_count < _H5_BLOCK_BYTES:
            try:
                _, _, attrs = next(iterator)
            except StopIteration:
                break
            value = _probability(attrs[key]) if key in attrs else None
            if value is not None:
                _validate_sparse(value)
                if tuple(value.shape) != shape or np.dtype(value.dtype) != dtype:
                    raise ValueError(f"Inconsistent {key} probability shape/dtype")
                probabilities.append(value)
                lengths.append(len(value.values))
                byte_count += value.values.nbytes + value.positions.nbytes
            else:
                lengths.append(0)
            flags.append(value is not None)
            input_types.append(isinstance(attrs.get(key), SparseProbability))
        new_end = end + sum(lengths)
        for name in ("values", "positions"):
            dataset = datasets[name]
            dataset.resize((new_end,))
            if new_end > end:
                dataset[end:new_end] = np.concatenate([getattr(value, name) for value in probabilities])
        offsets[row + 1:row + len(flags) + 1] = end + np.cumsum(lengths, dtype=np.int64)
        present[row:row + len(flags)] = flags
        sparse_input[row:row + len(flags)] = input_types
        row += len(flags)
        end = new_end


def _field_spec(value):
    array = np.asarray(value)
    if array.dtype.kind not in "biufcUS" and not (array.dtype.kind == "O" and
            all(isinstance(item, str) for item in array.flat)):
        raise TypeError(f"Unsupported graph field type {type(value).__name__}")
    return array.shape, array.dtype


def _write_fields(group, records, count, compression):
    """Fixed-shape fields preserve absent attributes separately from NaN values."""
    specs = {}
    for attrs in records():
        for key, value in attrs.items():
            if not isinstance(key, str) or "/" in key:
                raise TypeError("Graph field names must be strings without slashes")
            if key not in specs:
                specs[key] = _field_spec(value)
    targets = {}
    for key, (shape, dtype) in specs.items():
        field = group.create_group(key)
        field.attrs["numpy_dtype"] = dtype.str
        storage_dtype = h5py.string_dtype("utf-8") if dtype.kind in "OUS" else dtype
        values = _create_h5_dataset(field, "values", shape=(count, *shape), dtype=storage_dtype, compression=compression)
        present = _create_h5_dataset(field, "present", shape=(count,), dtype=bool, compression=compression)
        targets[key] = values, present
    iterator = iter(records())
    for start in range(0, count, _H5_BLOCK_ROWS):
        block = list(islice(iterator, _H5_BLOCK_ROWS))
        for key, (shape, dtype) in specs.items():
            strings = dtype.kind in "OUS"
            flags = np.array([key in attrs for attrs in block], dtype=bool)
            values = np.full((len(block), *shape), "", dtype=object) if strings else np.zeros((len(block), *shape), dtype=dtype)
            for index, attrs in enumerate(block):
                if key in attrs:
                    candidate = np.asarray(attrs[key])
                    if candidate.shape != shape or (candidate.dtype != dtype and not
                            (strings and candidate.dtype.kind in "OUS")):
                        raise ValueError(f"Inconsistent graph field {key!r} shape/dtype")
                    values[index] = candidate.item() if strings and candidate.ndim == 0 else candidate
            target, present = targets[key]
            target[start:start + len(block)] = values
            present[start:start + len(block)] = flags


def _read_fields(group, start, stop):
    records = [{} for _ in range(stop - start)]
    for key, field in group.items():
        values = field["values"]
        block = values.asstr()[start:stop] if h5py.check_string_dtype(values.dtype) else values[start:stop]
        flags = field["present"][start:stop]
        for index in np.flatnonzero(flags):
            records[index][key] = block[index]
    return records


def _write_graph(parent, name, graph, compression):
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("Only simple undirected ciMIST graphs are supported")
    group = parent.create_group(name)
    names = list(graph.nodes)
    if names and not (all(isinstance(node, (str, np.str_)) for node in names) or
                      all(isinstance(node, (int, np.integer)) and not isinstance(node, (bool, np.bool_)) for node in names)):
        raise TypeError("Graph node names must be uniformly strings or integers")
    saved_names = np.asarray(names)
    if names and saved_names.dtype.kind not in "iuUS":
        raise ValueError("Graph node names cannot be stored losslessly as strings or integers")
    indices = {node: index for index, node in enumerate(names)}
    _write_value(group, "node_names", saved_names, compression)
    _write_value(group, "metadata", graph.graph, compression)
    nodes = group.create_group("vertices")
    _write_fields(nodes, lambda: (attrs for _, attrs in graph.nodes(data=True)), len(names), compression)
    edges = group.create_group("edges")
    count = graph.number_of_edges()
    pairs = _create_h5_dataset(edges, "node_indices", shape=(count, 2), dtype=np.int64, compression=compression)
    axes = _create_h5_dataset(edges, "axes", shape=(count, 2), dtype=np.int64, compression=compression)
    row_counts = np.zeros(len(names), dtype=np.int64)
    iterator = iter(graph.edges(data=True))
    for start in range(0, count, _H5_BLOCK_ROWS):
        block = list(islice(iterator, _H5_BLOCK_ROWS))
        endpoints = np.asarray([(indices[u], indices[v]) for u, v, _ in block], dtype=np.int64)
        if np.any(endpoints[:, 0] >= endpoints[:, 1]):
            raise ValueError("Self edges or inconsistent undirected edge order")
        oriented = []
        for u, v, attrs in block:
            orientation = tuple(attrs.get("axes", ()))
            if len(orientation) != 2 or set(orientation) != {u, v}:
                raise ValueError(f"Invalid probability axes for edge {(u, v)!r}")
            if "P" not in attrs:
                raise ValueError(f"Missing probability P for edge {(u, v)!r}")
            oriented.append([indices[node] for node in orientation])
        pairs[start:start + len(block)] = endpoints
        axes[start:start + len(block)] = oriented
        np.add.at(row_counts, endpoints[:, 0], 1)
    _create_h5_dataset(edges, "node_offsets", data=np.r_[0, np.cumsum(row_counts)], compression=compression)
    # Replaying graph.edges does not preserve each node's neighbor order.
    neighbor_offsets = np.r_[0, np.cumsum([graph.degree[node] for node in names])]
    _create_h5_dataset(group, "neighbor_offsets", data=neighbor_offsets, compression=compression)
    neighbors = _create_h5_dataset(group, "neighbors", shape=(2 * count,), dtype=np.int64, compression=compression)
    for index, node in enumerate(names):
        neighbors[neighbor_offsets[index]:neighbor_offsets[index + 1]] = [indices[v] for v in graph[node]]
    probability_keys = {key for _, _, attrs in graph.edges(data=True) for key, value in attrs.items()
                        if key in ("P", "P_CL") or isinstance(value, SparseProbability)}
    for key in sorted(probability_keys):
        _write_probabilities(edges, graph, compression, key=key)
    edges.attrs["probability_fields"] = np.asarray(sorted(probability_keys), dtype=h5py.string_dtype("utf-8"))
    fields = edges.create_group("fields")
    _write_fields(fields, lambda: ({key: value for key, value in attrs.items()
                                   if key != "axes" and key not in probability_keys}
                                  for _, _, attrs in graph.edges(data=True)), count, compression)


def save_tree_h5(tree, path, mode="w", attrs_dict=None, compression="gzip"):
    """Save a lossless sparse snapshot with bounded buffers and gzip-1 chunks.

    Tree and candidate-graph state are separate, including adjacency order,
    optional fields, identities and jackknife outputs. Unsupported custom or
    executable state raises TypeError. No legacy dense HDF5 is written.
    """
    if compression not in (None, "gzip"):
        raise ValueError("HDF5 compression must be 'gzip' or None")
    with h5py.File(path, mode) as file:
        if any(name in file for name in ("tree", "MI_graph", "model", "statistics")):
            raise ValueError("The target file already contains snapshot groups")
        for key, value in (attrs_dict or {}).items():
            file.attrs[key] = value
        file.attrs["format"] = SPARSE_HDF5_FORMAT
        file.attrs["complete"] = False
        _write_graph(file, "tree", tree.T, compression)
        _write_graph(file, "MI_graph", tree.MI_graph, compression)
        model = file.create_group("model")
        for key, value in vars(tree).items():
            if key not in {"T", "MI_graph", "nodes", "edges"}:
                _write_value(model, key, value, compression)
        statistics = file.create_group("statistics")
        _write_value(statistics, "entropy", tree.entropy(), compression)
        _write_value(statistics, "sum_MI", tree.sum_MIs(), compression)
        file.attrs["complete"] = True


def save_coarse_graining_h5(states, path, mode="a", compression="gzip"):
    create_dataset = partial(_create_h5_dataset, compression=compression)
    with h5py.File(path, mode) as f:
        States = f.create_group("states")
        for field, value in zip(states._fields, states):
            create_dataset(States, field, data=np.array(value))
            
def save_VMM_h5(mixture, path, mode="a", compression="gzip"):
    create_dataset = partial(_create_h5_dataset, compression=compression)
    with h5py.File(path, mode) as f:
        try:
            Mixture = f.create_group("VMM")
        except ValueError:
            Mixture = f["VMM"]
        for field, value in zip(mixture._fields, mixture):
            if field != "r":
                create_dataset(Mixture, field, data=np.array(value))
    
class H5ProbabilityReader:
    """Context-managed selected reads without constructing a graph.

    read_rows/read_pairs return ProbabilityRecord objects in request order,
    including duplicate requests. Pair lookup is undirected; returned axes retain
    saved orientation. Decode an individual record with probability.dense_numpy().
    """
    def __init__(self, path, graph="MI_graph"):
        if graph not in ("tree", "MI_graph"):
            raise ValueError("graph must be 'tree' or 'MI_graph'")
        self.path, self.graph_name, self.file = path, graph, None

    def __enter__(self):
        if self.file is not None:
            raise RuntimeError("H5ProbabilityReader is already open")
        self.file = h5py.File(self.path, "r")
        try:
            _check_h5_format(self.file)
            self.group = self.file[self.graph_name]
            self.names = list(_read_value(self.group["node_names"]))
            self.indices = {node: index for index, node in enumerate(self.names)}
            if len(self.indices) != len(self.names):
                raise ValueError("Duplicate saved node names")
            self.edges = self.group["edges"]
            self.n_edges = len(self.edges["node_indices"])
            _validate_probability_layout(self.edges, self.n_edges)
            if (self.edges["node_indices"].shape != (self.n_edges, 2)
                    or self.edges["axes"].shape != (self.n_edges, 2)
                    or self.edges["node_indices"].dtype.kind not in "iu"
                    or self.edges["axes"].dtype.kind not in "iu"
                    or self.edges["node_offsets"].shape != (len(self.names) + 1,)):
                raise ValueError("Invalid saved edge index shapes or dtypes")
            row_offsets = self.edges["node_offsets"][:]
            if (row_offsets.dtype.kind not in "iu" or row_offsets[0] != 0
                    or row_offsets[-1] != self.n_edges or np.any(row_offsets[1:] < row_offsets[:-1])):
                raise ValueError("Invalid node row offsets")
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __exit__(self, *args):
        self.close()

    def _check_open(self):
        if self.file is None:
            raise RuntimeError("Use H5ProbabilityReader inside a with block")

    def _read_probabilities(self, key, start, stop):
        group = self.edges[key]
        offsets = group["offsets"][start:stop + 1]
        if (len(offsets) != stop - start + 1 or offsets.dtype.kind not in "iu"
                or offsets[0] < 0 or np.any(offsets[1:] < offsets[:-1])
                or offsets[-1] > len(group["values"]) or offsets[-1] > len(group["positions"])):
            raise ValueError("Invalid sparse probability offsets")
        first, last = int(offsets[0]), int(offsets[-1])
        positions, values = group["positions"][first:last], group["values"][first:last]
        shape, dtype = tuple(group.attrs["shape"]), np.dtype(group.attrs["dtype"])
        positions.flags.writeable = values.flags.writeable = False
        result = []
        for index, present in enumerate(group["present"][start:stop]):
            begin, end = int(offsets[index]) - first, int(offsets[index + 1]) - first
            if not present:
                if begin != end:
                    raise ValueError("Absent probability contains sparse values")
                result.append(None)
            else:
                value = SparseProbability(shape, dtype.str, positions[begin:end], values[begin:end])
                _validate_sparse(value)
                result.append(value)
        return result

    def _read_block(self, start, stop):
        endpoints, axes = self.edges["node_indices"][start:stop], self.edges["axes"][start:stop]
        if (np.any(endpoints < 0) or np.any(endpoints >= len(self.names))
                or np.any(endpoints[:, 0] >= endpoints[:, 1])
                or not np.array_equal(np.sort(axes, axis=1), endpoints)):
            raise ValueError("Invalid edge endpoints or probability axes")
        probabilities = self._read_probabilities("P", start, stop)
        if any(value is None for value in probabilities):
            raise ValueError("Missing probability P")
        return [ProbabilityRecord(start + i, tuple(self.names[j] for j in edge),
                tuple(self.names[j] for j in orientation), probability)
                for i, (edge, orientation, probability) in enumerate(zip(endpoints, axes, probabilities))]

    def read_rows(self, rows):
        self._check_open()
        if isinstance(rows, (int, np.integer)):
            rows = [rows]
        records = []
        for row in rows:
            if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
                raise TypeError("Probability rows must be integer indices")
            row = int(row)
            if not 0 <= row < self.n_edges:
                raise IndexError(f"Probability row {row} out of bounds")
            records.extend(self._read_block(row, row + 1))
        return records

    def read_pairs(self, pairs):
        self._check_open()
        rows = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("Each pair must contain two node names")
            try:
                u, v = sorted(self.indices[node] for node in pair)
            except KeyError as error:
                raise KeyError(f"Unknown node in pair {pair!r}") from error
            start, stop = map(int, self.edges["node_offsets"][u:u + 2])
            if not 0 <= start <= stop <= self.n_edges:
                raise ValueError("Invalid node row offsets")
            found = None
            for offset in range(start, stop, _H5_BLOCK_ROWS):
                block = self.edges["node_indices"][offset:min(stop, offset + _H5_BLOCK_ROWS)]
                matches = np.flatnonzero((block[:, 0] == u) & (block[:, 1] == v))
                if len(matches):
                    found = offset + int(matches[0])
                    break
            if found is None:
                raise KeyError(f"Pair {pair!r} is not present in the saved graph")
            rows.append(found)
        return self.read_rows(rows)


def _check_h5_format(file):
    if file.attrs.get("format") != SPARSE_HDF5_FORMAT:
        raise ValueError("Not a supported native sparse ciMIST HDF5 file")
    if not file.attrs.get("complete", False):
        raise ValueError("The ciMIST HDF5 snapshot was not completely written")


def _validate_probability_layout(edges, count):
    keys = list(edges.attrs["probability_fields"])
    if len(set(keys)) != len(keys) or (count and "P" not in keys):
        raise ValueError("Invalid probability field list")
    for key in keys:
        group = edges[key]
        shape = np.asarray(group.attrs["shape"])
        dtype = np.dtype(group.attrs["dtype"])
        values, positions, offsets = group["values"], group["positions"], group["offsets"]
        if (shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape < 0)
                or dtype.kind != "f" or values.dtype != dtype
                or values.ndim != 1 or positions.ndim != 1 or positions.dtype.kind not in "iu"
                or len(values) != len(positions) or offsets.shape != (count + 1,)
                or offsets.dtype.kind not in "iu" or offsets[0] != 0 or offsets[-1] != len(values)
                or group["present"].shape != (count,) or group["present"].dtype.kind != "b"
                or group["sparse_input"].shape != (count,) or group["sparse_input"].dtype.kind != "b"):
            raise ValueError(f"Invalid sparse probability layout for {key}")


def _validate_field_layout(group, count):
    for name, field in group.items():
        values, present = field["values"], field["present"]
        dtype = np.dtype(field.attrs["numpy_dtype"])
        if (values.ndim < 1 or values.shape[0] != count or present.shape != (count,)
                or present.dtype.kind != "b" or (dtype.kind in "biufc" and values.dtype != dtype)):
            raise ValueError(f"Invalid saved graph field layout for {name}")


def _load_graph(reader):
    group, edges = reader.group, reader.edges
    graph = nx.Graph()
    _validate_field_layout(group["vertices"], len(reader.names))
    _validate_field_layout(edges["fields"], reader.n_edges)
    graph.graph.update(_read_value(group["metadata"]))
    node_attrs = _read_fields(group["vertices"], 0, len(reader.names))
    graph.add_nodes_from(zip(reader.names, node_attrs))
    probability_fields = list(edges.attrs["probability_fields"])
    start = 0
    while start < reader.n_edges:
        stop = min(reader.n_edges, start + _H5_BLOCK_ROWS)
        # Bound temporary payloads even when individual sparse rows are large.
        for key in probability_fields:
            probability = edges[key]
            offsets = probability["offsets"][start:stop + 1]
            row_bytes = probability["values"].dtype.itemsize + probability["positions"].dtype.itemsize
            limit = offsets[0] + _H5_BLOCK_BYTES // max(1, row_bytes * len(probability_fields))
            stop = min(stop, start + max(1, int(np.searchsorted(offsets, limit, side="right")) - 1))
        fields = _read_fields(edges["fields"], start, stop)
        extra = {key: reader._read_probabilities(key, start, stop) for key in probability_fields if key != "P"}
        sparse_input = {key: edges[key]["sparse_input"][start:stop] for key in extra}
        for index, (record, attrs) in enumerate(zip(reader._read_block(start, stop), fields)):
            attrs.update(P=record.probability, axes=record.axes)
            for key, values in extra.items():
                if values[index] is not None:
                    attrs[key] = values[index] if sparse_input[key][index] else values[index].dense_numpy()
            graph.add_edge(*record.edge, **attrs)
        start = stop
    offsets = group["neighbor_offsets"][:]
    if (offsets.dtype.kind not in "iu" or len(offsets) != len(reader.names) + 1 or offsets[0] != 0
            or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(group["neighbors"])):
        raise ValueError("Invalid saved adjacency offsets")
    for index, node in enumerate(reader.names):
        neighbors = group["neighbors"][offsets[index]:offsets[index + 1]]
        if neighbors.dtype.kind not in "iu" or np.any(neighbors < 0) or np.any(neighbors >= len(reader.names)):
            raise ValueError("Invalid saved adjacency indices")
        order = [reader.names[i] for i in neighbors]
        if len(order) != len(graph[node]) or set(order) != set(graph[node]):
            raise ValueError("Invalid saved graph adjacency order")
        graph._adj[node] = {neighbor: graph._adj[node][neighbor] for neighbor in order}
    return graph


def load_h5_tree(path):
    """Restore the complete saved MIST snapshot without fitting or recomputing.

    Primary P probabilities remain SparseProbability values; optional cached
    P_CL fields retain their original dense/sparse representation. This loads
    the full candidate graph; use H5ProbabilityReader for a bounded selection
    instead. Only native sparse HDF5 files are supported.
    """
    with H5ProbabilityReader(path, "tree") as reader:
        tree_graph = _load_graph(reader)
        attributes = {key: _read_value(value) for key, value in reader.file["model"].items()}
    with H5ProbabilityReader(path, "MI_graph") as reader:
        mi_graph = _load_graph(reader)
    tree = cst.MIST.__new__(cst.MIST)
    tree.__dict__.update(attributes)
    tree.T, tree.MI_graph = tree_graph, mi_graph
    tree.nodes, tree.edges = tree.T.nodes, tree.T.edges
    return tree
