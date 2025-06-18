import copy
import io
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
        return super().find_class(module, name)

def get_logger(name):
    logger = logging.getLogger(name)
    filename = name + '.log'
    fh = logging.FileHandler(filename, mode='w')
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger

def getSequentialInput(dataset, targets):
    seq_input = {}
    data = dataset[0]
    for tgt in targets:
        if not hasattr(data, tgt):
            raise ValueError(f'The dataset does not have the target {tgt}.')
        if 'xgrid' in dataset.metadata[tgt]:
            seq_input[tgt] = dataset.metadata[tgt]['xgrid']
    return seq_input

def splitTrainValTest(dataset, batchsz, train_ratio=0.8, val_ratio=0.1):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    train_loader = DataLoader(dataset[: train_size], batch_size=batchsz)
    val_loader = DataLoader(dataset[train_size : train_size + val_size], batch_size=batchsz)
    test_loader = DataLoader(dataset[train_size + val_size :], batch_size=1)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def denormalizeLoss(normalized_loss, min, max):
    return normalized_loss * (max - min) / 2

def numpy(tensor: torch.Tensor | list[torch.Tensor]):
    def numpyTensor(t: torch.Tensor):
        return t.data.cpu().numpy() if t.device.type != 'cpu' else t.data.numpy()

    if isinstance(tensor, list):
        return np.array([numpyTensor(t) for t in tensor])
    return numpyTensor(tensor)

def stackingTensorsToNumpy(tensor_list: list[torch.Tensor]):
    tensor_list = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensor_list]
    return numpy(torch.cat(tensor_list, dim=0))

def saveGNNResults(name, **kwargs):
    with open(f'./save/{name}.pkl', 'wb') as f:
        pkl.dump(kwargs, f)

# def saveTrainedModel(name, model):
#     # with open(f'./save/{name}_model.pkl', 'wb') as f:
#     #     pkl.dump(model, f)
#     torch.sav
#
# def loadTrainedModel(name, model):
#     model.load
#     # with open(f'./save/{name}_model.pkl', 'rb') as f:
#     #     return pkl.load(f)

def gatherFlatGrad(grad, params):
    results = []
    for p, param in zip(grad, params):
        if p is not None:
            results.append(p.contiguous().view(-1))
        else:
            results.append(torch.zeros_like(param).view(-1))
    return torch.cat(results)
#
# def gatherFlatGrad(grad):
#     return torch.cat([p.contiguous().view(-1) for p in grad if not p is None])
