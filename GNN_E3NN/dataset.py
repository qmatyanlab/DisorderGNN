import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import InMemoryDataset

from pymatgen.core import Structure

from settings import atomic_embedding_CGCNN

from utils import (
    structureToGraph, getPymatgenElementProperties, normalize,
    connectJobStore, query
)

class Dataset(InMemoryDataset):
    def __init__(self, root, atomic_feature_type, normalization_func, axis, interpolate_elecond=False,
                 dr=0.1, transform=None, pre_transform=None):
        if not os.path.exists(root):
            os.makedirs(root)

        self.dr = dr
        self.atomic_feature_type = atomic_feature_type
        self.normalization_func = normalization_func
        self.axis = axis
        self.interpolate_elecond = interpolate_elecond

        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.metadata = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self): pass

    def process(self):
        store = connectJobStore(jobstore_config)
        atom_dict = getAtomicFeatures(atomic_feature_type=self.atomic_feature_type)

        conductivity_entries = query({'name': 'conductivityWannier90'}, store=store)
        E_list, structure_list = [], []
        optcond_list, elecond_list = [], []
        for entry in conductivity_entries:
            omega, optcond_xx, optcond_yy = entry['output']['optical']['omega'], entry['output']['optical']['xx']['real'], entry['output']['optical']['yy']['real']
            temperature, elecond_xx, elecond_yy = entry['output']['electrical']['temperature'], entry['output']['electrical']['xx'], entry['output']['electrical']['yy']
            optcond = (np.array(optcond_xx) + np.array(optcond_yy)) / 2
            elecond = (np.array(elecond_xx) + np.array(elecond_yy)) / 2

            scf_entry = query({'name': 'static', 'metadata.index': int(entry['metadata']['index'])}, store=store)
            E = float(scf_entry['output']['output']['energy'])
            structure = Structure.from_dict(scf_entry['output']['output']['structure'])

            structure_list.append(structure)
            E_list.append(E)
            optcond_list.append(optcond)
            elecond_list.append(elecond)

        E_list, E_stats = normalize(np.array(E_list), normalization_func='min_max', axis='all')
        optcond_list, optcond_stats = normalize(np.array(optcond_list), self.normalization_func, self.axis)
        elecond_list, elecond_stats = normalize(np.array(elecond_list), self.normalization_func, self.axis)

        data_list = []
        for E, structure, optcond, elecond in tqdm(zip(E_list, structure_list, optcond_list, elecond_list), total=len(E_list)):
            data = structureToGraph(structure=structure, E=E, optcond=optcond, elecond=elecond,
                                    atom_dict=atom_dict, r_cutoff=6, dr=self.dr)
            data_list.append(data)

        self.metadata = {
            'energy': {'stats': E_stats},
            'optcond': {'xgrid': omega, 'stats': optcond_stats},
            'elecond': {'xgrid': temperature, 'stats': elecond_stats}
        }

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, self.metadata), self.processed_paths[0])

def getAtomicFeatures(atomic_feature_type):
    if 'number' in atomic_feature_type:
        atom_dict = {k: [k] for k in range(1, 100)}
    elif 'pymatgen' in atomic_feature_type:
        atom_dict = {k: getPymatgenElementProperties(k) for k in range(1, 100)}
    elif 'CGCNN' in atomic_feature_type:
        atom_dict = {int(key): value for key, value in atomic_embedding_CGCNN.items()}
    return atom_dict

def loadDataset(root, atomic_feature_type, normalization_func, axis, dr=0.1):
    dataset = Dataset(root, atomic_feature_type, normalization_func, axis, dr)
    dataset = dataset.shuffle()
    return dataset