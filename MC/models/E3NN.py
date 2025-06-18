import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from typing import Dict, Union

import e3nn
from e3nn import o3

from models.Network import Network, visualize_layers, scatterAdd, scatterMean

class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):
        # override the `reduce_output` keyword to instead perform an averge over atom contributions
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True

        super().__init__(**kwargs)

        # embed the one-hot encoding
        self.em_z = nn.Linear(in_dim, em_dim)  # Linear layer for atom type
        self.em_x = nn.Linear(in_dim, em_dim)  # Linear layer for atom type

    def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))
        output = super().forward(data)
        # RELU issue, from e3nn discussion, removing because it might break the symmetry
        # output = torch.relu(output)

        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            output = scatterMean(output, data.batch, dim=0)  # take mean over atoms per example
            # output = torch_scatter.scatter_add(output, data.batch, dim=0)  # take mean over atoms per example
            # output, _ = torch_scatter.scatter_max(output, data.batch, dim=0)  # max over atoms per examples
        return output

if __name__ == '__main__':
    out_dim1 = 1
    out_dim2 = 251  # about 200 points
    out_dim3 = 91
    em_dim = 64

    model = PeriodicNetwork(
        in_dim=118,
        em_dim=em_dim,
        irreps_in=str(em_dim) + "x0e",
        irreps_out=str(out_dim1) + "x0e+" + str(out_dim2) + "x0e+" + str(out_dim3) + "x0e",
        irreps_node_attr=str(em_dim) + "x0e",
        layers=2,
        mul=32,
        lmax=2,
        max_radius=6,
        num_neighbors=63,
        reduce_output=True,
    )
    print(sum(param.numel() for param in model.parameters()))
    print(sum(param.numel() for param in model.parameters() if param.requires_grad))

    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} requires_grad={param.requires_grad}")
