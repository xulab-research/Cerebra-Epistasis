"""WT-aligned atom37 labels; derived supervision is computed only for loss."""
from pathlib import Path
import numpy as np
import torch
from . import residue_constants as rc

FEATURE_FILENAME = 'embedding_ESMC_ESM3_Structure_for_Cerebra_Seq.pt'


def read_sequence(path):
    sequence = ''.join(line.strip() for line in Path(path).read_text().splitlines()
                       if not line.startswith('>')).upper()
    if not sequence or any(aa not in rc.restype_order for aa in sequence):
        raise ValueError(f'Expected one nonempty canonical protein sequence: {path}')
    return sequence


def _column(raw, key, default=None, size=None):
    value = raw.get(key)
    if value is None:
        if size is None:
            raise ValueError(f'Missing mmCIF field: {key}')
        return [default] * size
    return value if isinstance(value, list) else [value]


def structure_features(path, sequence, chain_id, model_index=0):
    """Align WT to full entity sequence, then map label_seq_id to atom37.

    Gaps are allowed; a substitution in the best-scoring alignment is rejected.
    Equal-score alignments use the first result, as in the original label maker.
    """
    from Bio.Align import PairwiseAligner
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    raw = MMCIF2Dict(str(path))
    auth = _column(raw, '_atom_site.auth_asym_id')
    size = len(auth)
    labels = _column(raw, '_atom_site.label_asym_id')
    entities = _column(raw, '_atom_site.label_entity_id')
    polymer_ids = set(_column(raw, '_entity_poly_seq.entity_id'))
    choices = {(labels[i], entities[i]) for i in range(size)
               if auth[i] == chain_id and entities[i] in polymer_ids}
    if len(choices) != 1:
        raise ValueError(f'Author chain {chain_id!r} must identify one polymer chain: {sorted(choices)}')
    label_chain, entity = next(iter(choices))
    component_ids = _column(raw, '_chem_comp.id', '', 0)
    parents = _column(raw, '_chem_comp.mon_nstd_parent_comp_id', '?', len(component_ids))
    parent_map = dict(zip(component_ids, parents))
    def residue_letter(name):
        if name in rc.restype_3to1:
            return rc.restype_3to1[name]
        parent = parent_map.get(name, 'MET' if name == 'MSE' else '')
        return rc.restype_3to1.get(parent, 'X')
    monomers = {}
    for entity_id, number, name in zip(_column(raw, '_entity_poly_seq.entity_id'),
                                     _column(raw, '_entity_poly_seq.num'),
                                     _column(raw, '_entity_poly_seq.mon_id')):
        if entity_id != entity:
            continue
        number = int(number)
        aa = residue_letter(name)
        if number in monomers and monomers[number] != aa:
            raise ValueError(f'Microheterogeneous sequence at entity residue {number}')
        monomers[number] = aa
    seq_ids = sorted(monomers)
    full_sequence = ''.join(monomers[i] for i in seq_ids)
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                             open_gap_score=-5, extend_gap_score=-0.5)
    alignment = aligner.align(sequence, full_sequence)[0]
    pairs = [(int(i), int(j)) for i, j in alignment.indices.T if i >= 0 and j >= 0]
    mismatches = [(i+1, sequence[i], full_sequence[j]) for i,j in pairs if sequence[i] != full_sequence[j]]
    if mismatches:
        raise ValueError(f'Residue substitutions are not allowed (WT position, WT, mmCIF): {mismatches[:12]}')
    source_to_wt = {seq_ids[j]: i for i,j in pairs}
    model_numbers = _column(raw, '_atom_site.pdbx_PDB_model_num', '1', size)
    models = list(dict.fromkeys(model_numbers))
    if not 0 <= model_index < len(models):
        raise ValueError(f'Model index {model_index} not available')
    seq_numbers = _column(raw, '_atom_site.label_seq_id')
    atom_names = _column(raw, '_atom_site.label_atom_id')
    comp_names = _column(raw, '_atom_site.label_comp_id')
    altlocs = _column(raw, '_atom_site.label_alt_id', '.', size)
    occupancies = _column(raw, '_atom_site.occupancy', '1', size)
    xs,ys,zs = [_column(raw, '_atom_site.Cartn_'+axis) for axis in ('x','y','z')]
    residues = {}
    for k in range(size):
        if labels[k] != label_chain or model_numbers[k] != models[model_index] or seq_numbers[k] in ('.','?'):
            continue
        wt_i = source_to_wt.get(int(seq_numbers[k]))
        if wt_i is None:
            continue
        if residue_letter(comp_names[k]) != sequence[wt_i]:
            raise ValueError(f'Coordinate residue differs from WT at position {wt_i+1}: {comp_names[k]}')
        name = atom_names[k]
        if name == 'SE' and comp_names[k] == 'MSE':
            name = 'SD'
        if name not in rc.atom_order:
            continue
        occupancy = float(occupancies[k]) if occupancies[k] not in ('.','?') else 0.
        xyz = np.asarray([xs[k],ys[k],zs[k]],dtype=np.float32)
        if occupancy <= 0 or not np.isfinite(xyz).all():
            continue
        alt = '' if altlocs[k] in ('.','?') else altlocs[k]
        residues.setdefault(wt_i, []).append((name, alt, occupancy, xyz))
    positions = np.zeros((len(sequence),37,3),dtype=np.float32)
    mask = np.zeros((len(sequence),37),dtype=bool)
    for wt_i, atoms in residues.items():
        scores = {}
        for name,alt,occupancy,xyz in atoms:
            if alt: scores[alt] = scores.get(alt,0.) + occupancy
        selected = min(scores,key=lambda key:(-scores[key],key)) if scores else ''
        best = {}
        for name,alt,occupancy,xyz in atoms:
            if alt not in ('',selected):continue
            if name not in best or occupancy > best[name][0]:best[name]=(occupancy,xyz)
        for name,(_,xyz) in best.items():
            positions[wt_i,rc.atom_order[name]]=xyz
            mask[wt_i,rc.atom_order[name]]=True
    frame = mask[:,[rc.atom_order[a] for a in ('N','CA','C')]].all(-1)
    if not frame.any():
        raise ValueError('No valid N/CA/C backbone frame available')
    return {'sequence':sequence,'all_atom_positions':torch.from_numpy(positions),
            'all_atom_mask':torch.from_numpy(mask)}


def load_structure_features(path, wt_sequence):
    feature = torch.load(path, map_location='cpu', weights_only=True)
    if feature['sequence'] != wt_sequence:
        raise ValueError(f'Structure sequence differs from wt.fasta: {path}')
    coordinate_keys = {"all_atom_positions", "all_atom_mask"}
    present = coordinate_keys.intersection(feature)
    if not present:
        return None
    if present != coordinate_keys:
        raise ValueError(f"Incomplete structural coordinates/mask: {path}")
    positions, mask = feature['all_atom_positions'], feature['all_atom_mask']
    length = len(wt_sequence)
    if positions.shape != (length, 37, 3) or mask.shape != (length, 37):
        raise ValueError(f'Invalid atom37 feature shapes: {path}')
    if not torch.isfinite(positions).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f'Invalid coordinates or mask: {path}')
    if not mask[:, [rc.atom_order[a] for a in ('N','CA','C')]].bool().all(-1).any():
        raise ValueError(f'No valid backbone frames: {path}')
    return {'sequence': wt_sequence, 'all_atom_positions': positions.float(),
            'all_atom_mask': mask.bool()}


def structure_loss_labels(feature):
    # comp_label currently constructs CPU tensors internally.
    from model.utils.Cerebra_Seq_utils import comp_label, sequence_to_hhblits_ids
    sequence = feature['sequence']
    positions = feature['all_atom_positions'].detach().cpu().float()
    mask = feature['all_atom_mask'].detach().cpu().bool()
    n, ca, c, cb = [rc.atom_order[a] for a in ('N', 'CA', 'C', 'CB')]
    xyz = positions[:, [ca, c, n, cb]].clone()
    b, d = positions[:, ca] - positions[:, n], positions[:, c] - positions[:, ca]
    pseudo_cb = (-0.58273431 * torch.cross(b, d, dim=-1)
                 + 0.56802827 * b - 0.54067466 * d + positions[:, ca])
    missing_cb = ~mask[:, cb]
    xyz[missing_cb, 3] = pseudo_cb[missing_cb]
    frame_mask = mask[:, [n, ca, c]].all(-1).float()
    xyz, cb_dist, _, quaternion, psi_phi = comp_label(xyz)
    return {'seq': sequence_to_hhblits_ids(sequence, torch.device('cpu')),
            'residue_index': torch.arange(len(sequence), dtype=torch.long),
            'xyz': xyz, 'CB_dist': cb_dist, 'quaternion': quaternion,
            'psi_phi': psi_phi, 'mask': frame_mask,
            'all_atoms_pos': positions, 'all_atoms_mask': mask.float()}
