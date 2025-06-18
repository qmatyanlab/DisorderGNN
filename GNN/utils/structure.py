import torch
import numpy as np

from pymatgen.core.periodic_table import Element

from torch_geometric.data import Data

def getPymatgenElementProperties(number):
    element = Element.from_Z(number)
    properties = []
    properties.append(element.number)
    properties.append(float(element.atomic_mass))
    properties.append(element.atomic_radius)
    properties.append(element.electron_affinity)
    properties.append(element.row)
    properties.append(element.group)
    # properties += list(element.data['Atomic orbitals'].values())[-6:-1]
    return properties

def calculateEdgeAttributes(dist, r_cutoff, dr):
    if dr == 0:
        return dist
    else:
        rgrid = np.arange(0, r_cutoff, dr)
        sigma = r_cutoff / 3
        attr = np.exp(-0.5 * (rgrid - dist)**2 /sigma**2) / np.sqrt(2 * np.pi) / sigma
        return attr

def structureToGraph(structure, E, optcond, elecond, atom_dict,
                     r_cutoff=10, dr=0.1, max_neighbors=20):
    edge_src, edge_dst, edge_shift = [], [], []
    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, r_cutoff)
        for neighbor in neighbors:
            edge_src.append(i)
            edge_dst.append(neighbor.index)
            edge_shift.append(neighbor.image)

    edge_src = torch.tensor(edge_src, dtype=torch.long)
    edge_dst = torch.tensor(edge_dst, dtype=torch.long)
    edge_shift = torch.tensor(edge_shift, dtype=torch.float)

    positions = torch.tensor(structure.cart_coords, dtype=torch.float)
    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(0)
    edge_vec = positions[edge_dst] - positions[edge_src] + torch.einsum('ni,nij->nj', edge_shift, lattice.expand(edge_shift.shape[0], -1, -1))
    edge_len = torch.norm(edge_vec, dim=1).numpy()
    edge_len = np.around(edge_len, decimals=2)

    symbols = [site.specie.Z for site in structure]
    node_z = torch.tensor([atom_dict[symbol] for symbol in symbols], dtype=torch.float)

    energy = torch.tensor([E], dtype=torch.float).unsqueeze(0)
    optcond = torch.tensor(optcond, dtype=torch.float).unsqueeze(0)
    elecond = torch.tensor(elecond, dtype=torch.float).unsqueeze(0)

    data = Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        x=node_z,
        z=node_z,
        edge_index=torch.stack([edge_src, edge_dst], dim=0),
        edge_shift=edge_shift,
        edge_vec=edge_vec,
        edge_len=edge_len,
        energy=energy,
        optcond=optcond,
        elecond=elecond,
    )
    return data

def normalize(unnorm_quantity, normalization_func, axis):
    if isinstance(axis, str):
        if axis not in ["configuration", "xgrid", "all"]:
            raise ValueError(f'The input {axis} must be within ["configuration", "xgrid", "all"].')
        axis_dict = {'configuration': 0, 'xgrid': 1, 'all': None}
        axis = axis_dict[axis]

    stats = {'normalization_func': normalization_func, 'axis': axis}
    if normalization_func == 'min_max':
        min_val = np.min(unnorm_quantity, axis=axis, keepdims=True)
        max_val = np.max(unnorm_quantity, axis=axis, keepdims=True)
        norm_quantity = 2 * (unnorm_quantity - min_val) / (max_val - min_val) - 1
        stats['min_val'], stats['max_val'] = min_val, max_val
    elif normalization_func == 'mean':
        mean_val = np.mean(unnorm_quantity, axis=axis, keepdims=True)
        norm_quantity = unnorm_quantity / mean_val
        stats['mean_val'] = mean_val
    elif normalization_func == 'median':
        median_val = np.median(np.max(unnorm_quantity, axis=1))
        norm_quantity = unnorm_quantity / median_val
        stats['median_val'] = median_val
    elif normalization_func == 'gaussian':
        mean_val = np.mean(unnorm_quantity, axis=axis, keepdims=True)
        std_val = np.std(unnorm_quantity, axis=axis, keepdims=True)
        norm_quantity = (unnorm_quantity - mean_val) / std_val
        stats['mean_val'], stats['std_val'] = mean_val, std_val
    elif normalization_func == 'log':
        norm_quantity = np.log(unnorm_quantity)

    return norm_quantity, stats

def denormalize(norm_quantity, stats):
    normalization_func = stats['normalization_func']
    # axis = stats['axis']

    if normalization_func == 'min_max':
        min_val = stats['min_val']
        max_val = stats['max_val']
        unnorm_quantity = (norm_quantity + 1) * (max_val - min_val) / 2 + min_val
    elif normalization_func == 'mean':
        mean_val = stats['mean_val']
        unnorm_quantity = norm_quantity * mean_val
    elif normalization_func == 'median':
        median_val = stats['median_val']
        unnorm_quantity = norm_quantity * median_val
    elif normalization_func == 'gaussian':
        mean_val = stats['mean_val']
        std_val = stats['std_val']
        unnorm_quantity = norm_quantity * std_val + mean_val
    elif normalization_func == 'log':
        unnorm_quantity = np.exp(norm_quantity)
    else:
        raise ValueError(f'Unknown normalization function: {normalization_func}')

    return unnorm_quantity
