import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import from_scipy_sparse_matrix

__all__ = [
    'GAT',
]


class GAT(nn.Module):
    """Graph Attention Network for overlapping community detection.

    Uses multi-head attention to learn adaptive neighbor weighting,
    allowing nodes at community boundaries to selectively attend to
    same-community neighbors.

    References:
        "Graph Attention Networks", Velickovic et al., ICLR 2018

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list of int
        Hidden layer sizes (per head).
    output_dim : int
        Number of output communities.
    heads : int, default=4
        Number of attention heads per layer (output layer always uses 1).
    dropout : float, default=0.5
        Dropout rate on features.
    attn_dropout : float, default=0.3
        Dropout rate on attention coefficients.
    batch_norm : bool, default=False
        Whether to use batch normalization after hidden layers.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, heads=4,
                 dropout=0.5, attn_dropout=0.3, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        self.heads = heads

        layer_dims = [int(d) for d in hidden_dims] + [int(output_dim)]

        # First hidden layer: multi-head, concat
        self.layers = nn.ModuleList([
            GATConv(int(input_dim), layer_dims[0], heads=heads,
                    dropout=attn_dropout, concat=True)
        ])

        # Additional hidden layers (if any)
        for idx in range(len(layer_dims) - 2):
            self.layers.append(
                GATConv(layer_dims[idx] * heads, layer_dims[idx + 1],
                        heads=heads, dropout=attn_dropout, concat=True)
            )

        # Output layer: single head, no concat
        if len(layer_dims) > 1:
            self.layers.append(
                GATConv(layer_dims[-2] * heads, layer_dims[-1],
                        heads=1, dropout=attn_dropout, concat=False)
            )

        if batch_norm:
            self.batch_norm = nn.ModuleList([
                nn.BatchNorm1d(dim * heads, affine=False, track_running_stats=False)
                for dim in hidden_dims
            ])
        else:
            self.batch_norm = None

    @staticmethod
    def build_edge_index(adj: sp.spmatrix, device: torch.device = None):
        """Convert scipy sparse adjacency matrix to PyG edge_index.

        Adds self-loops for self-attention.
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
        # GAT ignores edge_weight — attention replaces it
        for idx, layer in enumerate(self.layers):
            if self.dropout != 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            if idx != len(self.layers) - 1:
                x = F.elu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
