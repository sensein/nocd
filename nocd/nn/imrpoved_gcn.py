import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import from_scipy_sparse_matrix, add_remaining_self_loops
from torch_geometric.utils import degree

__all__ = [
    'ImprovedGCN',
    'ImpGraphConvolution',
]


class ImpGraphConvolution(MessagePassing):
    """Graph convolution layer with separate self and neighbor weight matrices.

    Computes: adj_norm @ (x @ W_neighbor) + x @ W_self + bias
    Using PyG's message-passing framework with symmetric normalization.
    """
    def __init__(self, in_features, out_features):
        super().__init__(aggr='add')
        self.in_features = in_features
        self.out_features = out_features
        self.weight_own = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_nbr = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_own, gain=2.0)
        nn.init.xavier_uniform_(self.weight_nbr, gain=2.0)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        # Neighbor aggregation: D^{-1/2} A D^{-1/2} (x @ W_nbr)
        x_nbr = x @ self.weight_nbr
        out = self.propagate(edge_index, x=x_nbr, edge_weight=edge_weight)
        # Self transformation
        out = out + x @ self.weight_own + self.bias
        return out

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j


class ImprovedGCN(nn.Module):
    """An improved GCN architecture for overlapping community detection.

    Uses two weight matrices for self-propagation and aggregation,
    Tanh activation, and optional layer normalization.

    Inspired by https://arxiv.org/abs/1906.12192
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, layer_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int64)
        self.layers = nn.ModuleList([ImpGraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(ImpGraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if layer_norm:
            self.layer_norm = nn.ModuleList([
                nn.LayerNorm([dim], elementwise_affine=False) for dim in hidden_dims
            ])
        else:
            self.layer_norm = None

    @staticmethod
    def build_edge_index(adj: sp.spmatrix, device: torch.device = None):
        """Convert scipy sparse adjacency to PyG edge_index with symmetric normalization.

        Unlike standard GCN, ImprovedGCN does NOT add self-loops (self-connection
        is handled by the separate weight_own matrix).
        """
        adj = adj.tolil()
        adj.setdiag(0)
        adj = adj.tocsr()

        # Compute D^{-1/2} A D^{-1/2} manually
        deg = np.ravel(adj.sum(1))
        deg_sqrt_inv = np.zeros_like(deg)
        nonzero = deg > 0
        deg_sqrt_inv[nonzero] = 1.0 / np.sqrt(deg[nonzero])
        adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])

        edge_index, edge_weight = from_scipy_sparse_matrix(adj_norm)
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
                x = torch.tanh(x)
                if self.layer_norm is not None:
                    x = self.layer_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
