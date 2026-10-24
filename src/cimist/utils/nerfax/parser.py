import numpy as np
import mdtraj as md
from Bio.Data import IUPACData
from collections import Counter
import mp_nerf
import warnings
from jax import vmap, numpy as jnp, lax

AA = set(mp_nerf.kb_proteins.SC_BUILD_INFO.keys())-set('_')
aa_to_index = lambda seq: np.vectorize({v:k for k,v in enumerate(AA)}.__getitem__)(list(seq))
_CANONICAL_RESIDUES = {name.upper(): code for name, code in IUPACData.protein_letters_3to1.items()}
# These force-field names are not all recognized by MDTraj's is_protein.
_AMINO_ACID_VARIANTS = {"ASH", "GLH", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYM", "CYX", "LYN"}
_PEPTIDE_CAPS = {"ACE", "NME"}


def calculate_angle(c1, c2, c3):
    # Credit to https://github.com/EleutherAI/mp_nerf
    u1 = c2 - c1
    u2 = c3 - c2
    # dont use acos since norms involved. 
    # better use atan2 formula: atan2(cross, dot) from here: 
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html
    return jnp.arctan2( jnp.linalg.norm(jnp.cross(u1,u2)), 
                        jnp.dot(u1,u2) ) 

@vmap
def compute_dihedrals(xyz):
    # assert xyz.shape == (4,3) 
    d = xyz[1:]-xyz[:-1]
    index = jnp.arange(3)
    cross_pairs = jnp.stack([index[:-1], index[1:]])
    c = jnp.cross(*d[cross_pairs]) # c is the cross products of consecutive displacements
    p1 = jnp.einsum('i,i', d[0],c[1]) * jnp.linalg.norm(d[1])
    p2 = jnp.einsum('i,i', c[0],c[1])
    return jnp.arctan2(p1, p2)

@vmap
def decompose_quad(xyz):
    a,b,c, real_pos = xyz
    l = jnp.linalg.norm(real_pos-c)
    theta = calculate_angle(b,c,real_pos)
    chi = compute_dihedrals(jnp.array([[a,b,c,real_pos]]))[0]
    return l, theta, chi

def xyz_to_internal_coords(xyz):
    ''' Expects just the backbone'''
    dih_quads = jnp.stack([lax.dynamic_slice(xyz, (i,0), (xyz.shape[0]-3,3)) for i in jnp.arange(4)], axis=-2)
    l, theta, chi = decompose_quad(dih_quads)
    theta, chi, l = [a.reshape((-1,3)) for a in [theta,chi,l]]
    return l, theta, chi # lengths, angles, torsions

def _unsupported_residue(residue):
    chain_id = residue.chain.chain_id or f"chain{residue.chain.index}"
    return ValueError(f"Unsupported protein residue {residue.name}{residue.resSeq}_{chain_id} "
                      f"at topology index {residue.index} (chain index {residue.chain.index}, "
                      f"chain ID {residue.chain.chain_id!r}); canonical amino acids are required")


def residue_records(topology):
    """Describe protein residues, rejecting unsupported peptide chemistry.

    Validate the full topology before any center selection. ACE/NME may remain
    covalent context, but feature extraction must exclude them as centers.
    Names are those in the loaded topology; MDTraj may normalize PDB aliases.
    """
    residues = list(topology.residues)
    protein_indices = {r.index for r in residues
                       if r.is_protein or r.name in _AMINO_ACID_VARIANTS}
    backbone_candidates = {r.index for r in residues
                           if {"N", "CA", "C"}.issubset(a.name for a in r.atoms)}
    for first, second in topology.bonds:
        first_index, second_index = first.residue.index, second.residue.index
        if first_index == second_index or {first.name, second.name} != {"C", "N"}:
            continue
        # Unknown residue templates may lack internal bonds. A peptide link
        # distinguishes these candidates from unconnected ligands such as DHF.
        if first_index in backbone_candidates and (
                second_index in protein_indices or second_index in backbone_candidates):
            protein_indices.add(first_index)
        if second_index in backbone_candidates and (
                first_index in protein_indices or first_index in backbone_candidates):
            protein_indices.add(second_index)
    records = []
    for residue in residues:
        if residue.index not in protein_indices:
            continue
        if residue.name not in _CANONICAL_RESIDUES and residue.name not in _PEPTIDE_CAPS:
            raise _unsupported_residue(residue)
        chain_id = residue.chain.chain_id or ""
        chain_label = chain_id or f"chain{residue.chain.index}"
        ca_atoms = [a.index for a in residue.atoms if a.name == "CA"]
        records.append(dict(residue_index=residue.index,
                            chain_index=residue.chain.index, chain_id=chain_id,
                            resSeq=residue.resSeq, name=residue.name,
                            segment_id=residue.segment_id,
                            ca_atom_index=ca_atoms[0] if len(ca_atoms) == 1 else None,
                            label=f"{residue.name}{residue.resSeq}_{chain_label}"))
    counts = Counter(record["label"] for record in records)
    for record in records:
        if counts[record["label"]] > 1:
            record["label"] += f"[r{record['residue_index']}]"
    return records


def protein_feature_references(topology, residue_indices=None):
    """Return atom quartets, physical feature masks, and residue records.

    Selection chooses residue centers, while references use the full topology.
    Peptide neighbors come from C--N bonds, never residue numbers or array order.
    The returned quartets have shape (residues, 14, 4); -1 denotes an unavailable
    reference. A bond angle only needs the final three atoms of its quartet.
    """
    records = residue_records(topology)
    by_index = {record["residue_index"]: record for record in records}
    if residue_indices is not None:
        indices = list(residue_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("Residue selection contains duplicate topology indices")
        if any(index not in by_index for index in indices):
            raise ValueError("Residue selection must contain protein topology indices")
        records = [by_index[index] for index in indices]
    if not records:
        raise ValueError("No protein residues were selected")

    atom_maps = []
    duplicate_atoms = {}
    for residue in topology.residues:
        atoms = {}
        for atom in residue.atoms:
            if atom.name in atoms:
                duplicate_atoms[residue.index] = atom.name
            atoms[atom.name] = atom.index
        atom_maps.append(atoms)
    neighbors = [set() for _ in topology.atoms]
    for first, second in topology.bonds:
        neighbors[first.index].add(second.index)
        neighbors[second.index].add(first.index)
    references = np.full((len(records), 14, 4), -1, dtype=np.int32)
    for row, record in enumerate(records):
        residue = topology.residue(record["residue_index"])
        if residue.index in duplicate_atoms:
            raise ValueError(f"Duplicate atom name {duplicate_atoms[residue.index]!r} in residue index {residue.index}")
        if residue.name not in _CANONICAL_RESIDUES:
            raise _unsupported_residue(residue)
        code = _CANONICAL_RESIDUES[residue.name]
        names = ["N", "CA", "C", "O"] + mp_nerf.kb_proteins.SC_BUILD_INFO[code]["atom-names"]
        atoms = atom_maps[residue.index]
        missing = [name for name in names if name not in atoms]
        if missing:
            raise ValueError(f"Missing heavy atoms in {record['label']} at topology index {residue.index}: {', '.join(missing)}")
        n, ca, c = (atoms[name] for name in ("N", "CA", "C"))

        def peptide_neighbor(atom_index, name):
            matches = [index for index in neighbors[atom_index]
                       if topology.atom(index).name == name
                       and topology.atom(index).residue.index != residue.index]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous peptide connectivity for {record['label']} atom {topology.atom(atom_index).name}")
            return matches[0] if matches else -1

        previous_c = peptide_neighbor(n, "C")
        next_n = peptide_neighbor(c, "N")
        for context in (previous_c, next_n):
            if context >= 0 and topology.atom(context).residue.index in duplicate_atoms:
                raise ValueError(f"Duplicate atom names in covalent context for {record['label']}")
        previous_ca = (atom_maps[topology.atom(previous_c).residue.index].get("CA", -1)
                       if previous_c >= 0 else -1)
        references[row, :3] = ((n, ca, c, next_n),
                              (previous_ca, previous_c, n, ca),
                              (previous_c, n, ca, c))
        local = [atoms[name] for name in names]
        for slot in range(3, len(names)):
            template = mp_nerf.make_idx_mask(code)[slot - 3].astype(int)
            references[row, slot] = [local[index] for index in template] + [local[slot]]
        if "CB" in atoms:
            if previous_c >= 0:
                references[row, 4] = previous_c, n, ca, atoms["CB"]
            else:
                # Retain the physical legacy start-of-chain convention; O is
                # an in-residue reference when a singleton has no next N.
                references[row, 4] = next_n if next_n >= 0 else atoms["O"], c, ca, atoms["CB"]

        # Atom presence alone is insufficient when a topology lacks bonds.
        for quartet in references[row]:
            for first, second in zip(quartet[:-1], quartet[1:]):
                if first >= 0 and second >= 0 and second not in neighbors[first]:
                    raise ValueError(f"Missing topology bond {topology.atom(first).name}--{topology.atom(second).name} for {record['label']}")
    mask = np.concatenate((np.all(references[:, :, 1:] >= 0, axis=-1),
                           np.all(references >= 0, axis=-1)), axis=-1)
    return references, mask, records


def _minimum_image(vectors, boxes):
    inverse = np.linalg.inv(boxes)
    # The caller checks imaged covalent vectors against half the smallest face
    # spacing, which guarantees their images are unique even in triclinic boxes.
    if np.any(1.0 / np.linalg.norm(inverse, axis=1) <= 0.6):
        raise ValueError("Periodic cell face spacing must exceed 0.6 nm for covalent imaging")
    fractional = np.einsum("t...i,tij->t...j", vectors, inverse)
    return vectors - np.einsum("t...i,tij->t...j", np.rint(fractional), boxes)


def get_scnet_loader_fns(t):
    """Map canonical protein residue instances to SidechainNet rows (Angstrom)."""
    _, _, records = protein_feature_references(t.topology)
    rows = {record["residue_index"]: row for row, record in enumerate(records)}
    canonical = {name.upper(): code for name, code in IUPACData.protein_letters_3to1.items()}
    atom_indices, res_indices, slots = [], [], []
    for atom in t.topology.atoms:
        residue = atom.residue
        if residue.index not in rows:
            continue
        code = canonical[residue.name]
        names = ["N", "CA", "C", "O"] + mp_nerf.kb_proteins.SC_BUILD_INFO[code]["atom-names"]
        if atom.name in names:
            atom_indices.append(atom.index)
            res_indices.append(rows[residue.index])
            slots.append(names.index(atom.name))
    atom_indices, res_indices, slots = map(np.asarray, (atom_indices, res_indices, slots))

    def _parse_coords(coords):
        return jnp.zeros((len(records), 14, 3)).at[res_indices, slots].set(coords[atom_indices]) * 10

    def _restrict_to_scnet_atoms(traj):
        # MDTraj 1.10 copies lose chain IDs and change residue numbering zero.
        residues = [traj.topology.residue(record["residue_index"]) for record in records]
        chains = list(dict.fromkeys(residue.chain for residue in residues))
        restricted = traj.atom_slice(atom_indices)
        for chain, original in zip(restricted.topology.chains, chains):
            chain.chain_id = original.chain_id
        for residue, original in zip(restricted.topology.residues, residues):
            residue.resSeq = original.resSeq
            residue.segment_id = original.segment_id
        return restricted

    def _scnet_to_list(coords):
        if coords.shape != (len(records), 14, 3):
            raise ValueError("SidechainNet coordinates do not match reference residue mapping")
        return coords[res_indices, slots] / 10

    return _parse_coords, _restrict_to_scnet_atoms, _scnet_to_list


def protein_sequence(topology):
    records = residue_records(topology)
    for record in records:
        if record["name"] not in _CANONICAL_RESIDUES:
            raise _unsupported_residue(topology.residue(record["residue_index"]))
    return "".join(_CANONICAL_RESIDUES[record["name"]] for record in records)


def load_traj(path):
    return md.load(path)


def load_to_sc_coord_format(path, first_frame_only=True):
    from ..trajectory import traj_to_scnet
    coords, seq = traj_to_scnet(load_traj(path))
    return (coords[0] if first_frame_only else coords), seq


def get_point_ref_with_final(aa):
    ref_indices = mp_nerf.make_idx_mask(aa).astype(int)
    final_index = np.arange(11)+3
    mask = mp_nerf.make_cloud_mask(aa)[3:].astype(bool)
    final_index[~mask]=0
    return np.concatenate([ref_indices, final_index[...,None]], axis=-1)

def get_point_ref_and_cloud_mask(seq):
    # point ref here (4,L,11) differs from mp_nerf point_ref_mask (3,L,11) as it
    # includes the final index explicitly so is 4 long, not 3
    point_ref_per_aa = np.stack(list(map(get_point_ref_with_final,AA))).astype(int)
    point_ref = jnp.einsum('ijk->kij', point_ref_per_aa[aa_to_index(seq)])

    cloud_mask_per_aa = jnp.stack(list(map(mp_nerf.make_cloud_mask,AA))).astype(bool)
    cloud_mask = cloud_mask_per_aa[aa_to_index(seq)]
    return point_ref, cloud_mask

def get_data_masks(xyz, references, unitcell_vectors=None, frame_batch_size=64,
                   residue_batch_size=128):
    """Extract natural bond lengths (Angstrom) and SCNET angles from real atoms.

    xyz has shape (frames, atoms, 3), in nm. References have shape
    (residues, 14, 4), with -1 for unavailable atoms. Geometry temporaries are
    bounded by both batch sizes; missing features are zero, not ghost angles.
    """
    if frame_batch_size <= 0 or residue_batch_size <= 0:
        raise ValueError("Coordinate batch sizes must be positive")
    if len(xyz) == 0:
        raise ValueError("Trajectory contains no frames")
    mask = np.concatenate((np.all(references[:, :, 1:] >= 0, axis=-1),
                           np.all(references >= 0, axis=-1)), axis=-1)
    n_frames, n_residues = len(xyz), len(references)
    angles = np.zeros((n_frames, n_residues, 2, 14), dtype=np.float64).swapaxes(1, 2)
    bonds = np.zeros((n_frames, n_residues, 14), dtype=np.float64)
    boxes = unitcell_vectors
    longest_bond = 0.0
    first_long_bond = None
    for frame_start in range(0, n_frames, frame_batch_size):
        frame_end = min(frame_start + frame_batch_size, n_frames)
        frame_xyz = xyz[frame_start:frame_end]
        box = None if boxes is None else np.asarray(boxes[frame_start:frame_end], dtype=np.float64)
        if box is not None and (not np.all(np.isfinite(box)) or np.any(np.linalg.det(box) <= 0)):
            raise ValueError("Periodic unit cells must be finite and have positive volume")
        half_face_spacing = (None if box is None else
                             0.5 / np.linalg.norm(np.linalg.inv(box), axis=1).max(axis=1))
        for start in range(0, n_residues, residue_batch_size):
            end = min(start + residue_batch_size, n_residues)
            refs = references[start:end]
            coords = np.asarray(frame_xyz[:, np.maximum(refs, 0)], dtype=np.float64)
            coords = np.where(refs[None, ..., None] >= 0, coords, 0.0)
            if not np.all(np.isfinite(coords)):
                raise ValueError(f"Non-finite referenced coordinate in frames {frame_start}:{frame_end}")
            vectors = np.diff(coords, axis=-2)
            if box is not None:
                vectors = _minimum_image(vectors, box)
            required_bonds = (refs[:, :, :-1] >= 0) & (refs[:, :, 1:] >= 0)
            lengths = np.linalg.norm(vectors, axis=-1)
            invalid = required_bonds[None] & (lengths <= 1e-8)
            if np.any(invalid):
                frame, row, slot, bond = np.argwhere(invalid)[0]
                raise ValueError(f"Invalid covalent geometry for {start + row} in frame {frame_start + frame}: bond length {lengths[frame, row, slot, bond]:.6g} nm; check topology connectivity and periodic imaging")
            if half_face_spacing is not None:
                ambiguous = required_bonds[None] & (lengths >= half_face_spacing[:, None, None, None])
                if np.any(ambiguous):
                    frame, row, slot, bond = np.argwhere(ambiguous)[0]
                    raise ValueError(f"Periodic covalent geometry for residue row {start + row} in frame {frame_start + frame} exceeds the unique-image bound: {lengths[frame, row, slot, bond]:.6g} nm must be less than half the smallest box face spacing ({half_face_spacing[frame]:.6g} nm)")
            long_bonds = required_bonds[None] & (lengths > 0.3)
            if np.any(long_bonds):
                frame, row, _, _ = np.argwhere(long_bonds)[0]
                occurrence = (frame_start + int(frame), start + int(row))
                first_long_bond = (occurrence if first_long_bond is None
                                   else min(first_long_bond, occurrence))
                longest_bond = max(longest_bond, float(lengths[long_bonds].max()))
            cross = np.cross(vectors[..., :-1, :], vectors[..., 1:, :])
            angle = np.arctan2(np.linalg.norm(cross[..., 1, :], axis=-1),
                               np.sum(vectors[..., 1, :] * vectors[..., 2, :], axis=-1))
            p1 = np.sum(vectors[..., 0, :] * cross[..., 1, :], axis=-1) * lengths[..., 1]
            p2 = np.sum(cross[..., 0, :] * cross[..., 1, :], axis=-1)
            torsion = np.arctan2(p1, p2)
            valid_torsions = mask[start:end, 14:][None]
            collinear = np.any(np.linalg.norm(cross, axis=-1) < 1e-12, axis=-1)
            if np.any(valid_torsions & collinear):
                raise ValueError("Collinear atoms make a selected torsion undefined")
            features = np.concatenate((angle, torsion), axis=-1)
            features = np.where(mask[start:end][None], features, 0.0)
            angles[frame_start:frame_end, :, start:end] = features.reshape(
                frame_end - frame_start, end - start, 2, 14).swapaxes(1, 2)
            # Bond lengths retain the original natural (N, CA, C, ...) layout.
            local_bonds = np.where(required_bonds[None, :, :, 2], lengths[..., 2], 0.0)
            local_bonds[:, :, 0] = np.where(required_bonds[None, :, 2, 0], lengths[:, :, 2, 0], 0.0)
            bonds[frame_start:frame_end, start:end] = local_bonds * 10.0
    if first_long_bond is not None:
        warnings.warn(f"Covalent bond lengths above 0.3 nm: maximum {longest_bond:.6g} nm; first occurrence frame {first_long_bond[0]}, residue row {first_long_bond[1]}. Coordinates and frames are retained; check the input topology and trajectory.",
                      RuntimeWarning, stacklevel=2)
    return bonds, angles


def make_scaffolds(coords, seq):
    """Extract natural scaffolds for one connected chain of SidechainNet atoms.

    Coordinates are Angstrom, shaped (R, 14, 3) or (T, R, 14, 3).
    Sequence alone defines a connected chain. Use trajectory.traj_to_internal
    when chain boundaries, covalent gaps, caps, or periodic boxes matter.
    """
    coords = np.asarray(coords)
    single_frame = coords.ndim == 3
    if single_frame:
        coords = coords[None]
    if coords.shape[1:] != (len(seq), 14, 3):
        raise ValueError("SidechainNet coordinates must match sequence length and have 14 atom slots")
    topology = md.Topology()
    chain = topology.add_chain()
    for row, code in enumerate(seq):
        if code not in AA:
            raise ValueError(f"Unsupported amino acid {code!r}")
        residue = topology.add_residue(IUPACData.protein_letters_1to3[code].upper(), chain, resSeq=row + 1)
        names = ["N", "CA", "C", "O"] + mp_nerf.kb_proteins.SC_BUILD_INFO[code]["atom-names"]
        for slot in range(14):
            name = names[slot] if slot < len(names) else f"PAD{slot}"
            element = md.element.get_by_symbol(name[0]) if slot < len(names) else md.element.virtual
            topology.add_atom(name, element, residue)
    topology.create_standard_bonds()
    references, feature_mask, _ = protein_feature_references(topology)
    bonds, angles = get_data_masks(coords.reshape(len(coords), -1, 3) / 10, references)
    # Only angles' first column changes between the inherited two conventions.
    angles[..., 0] = np.roll(angles[..., 0], 1, axis=-1)
    point_ref, cloud_mask = get_point_ref_and_cloud_mask(seq)
    result = dict(angles_mask=angles, bond_mask=bonds,
                  cloud_mask=np.broadcast_to(cloud_mask, (len(coords),) + cloud_mask.shape),
                  point_ref_mask=np.broadcast_to(point_ref[:3], (len(coords),) + point_ref[:3].shape))
    result = {key: value[0] if single_frame else value for key, value in result.items()}
    result["feature_mask"] = feature_mask
    result["reconstruction_supported"] = len(seq) >= 2
    return result


def load_pdb(path, first_frame_only=True):
    from ..trajectory import traj_to_internal, convert_scnet_to_natural
    traj = load_traj(path)
    if first_frame_only:
        traj = traj.slice(0, copy=False)
    result = traj_to_internal(traj)
    result["angles_mask"] = vmap(convert_scnet_to_natural)(result["angles_mask"])
    if first_frame_only:
        for key in ("angles_mask", "bond_mask", "cloud_mask", "point_ref_mask"):
            result[key] = result[key][0]
    return result
