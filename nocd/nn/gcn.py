import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix

__all__ = [
    'GCN',
]


class GCN(nn.Module):
    """Graph convolution network using PyG's GCNConv.

    References:
        "Semi-supervised learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int64)
        self.layers = nn.ModuleList([GCNConv(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(GCNConv(layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = nn.ModuleList([
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ])
        else:
            self.batch_norm = None

    @staticmethod
    def build_edge_index(adj: sp.spmatrix, device: torch.device = None):
        """Convert scipy sparse adjacency matrix to PyG edge_index.

        Adds self-loops as part of GCN normalization.
        """
        adj = adj.tolil()
        adj.setdiag(1)
        adj = adj.tocsr()
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        if device is not None:
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, edge_index, edge_weight)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
