import copy
import io
import os
import logging
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def getSequentialInput(dataset, targets):
    seq_input = {}
    data = dataset[0]
    for tgt in targets:
        if not hasattr(data, tgt):
            raise ValueError(f'The dataset does not have the target {tgt}.')
        if 'xgrid' in dataset.metadata[tgt]:
            seq_input[tgt] = dataset.metadata[tgt]['xgrid']
    return seq_input

def numpy(tensor: torch.Tensor | list[torch.Tensor]):
    def numpyTensor(t: torch.Tensor):
        return t.data.cpu().numpy() if t.device.type != 'cpu' else t.data.numpy()

    if isinstance(tensor, list):
        return [numpyTensor(t) for t in tensor]
    return numpyTensor(tensor)

def loadTrainedModel(modelname, device=torch.device('cpu')):
    filename = f'./save/GNN_model/{modelname}.pkl'
    if not os.path.exists(filename):
        raise FileNotFoundError(f'The model {modelname} does not exist under the directory ./save/GNN_model.')

    with open(filename, 'rb') as f:
        if device == torch.device('cuda'):
            model = pkl.load(f)
        elif device == torch.device('cpu'):
            model = CPU_Unpickler(f).load()
    return model.to(device)

def loadDatasetMetadata(dataset_metadata):
    filename = f'./save/dataset_metadata/{dataset_metadata}.pkl'
    if not os.path.exists(filename):
        raise FileNotFoundError(f'The dataset metadata {dataset_metadata}'
                                f' does not exist under the directory ./save/dataset_metadata.')
    with open(filename, 'rb') as f:
        metadata = pkl.load(f)
        return metadata