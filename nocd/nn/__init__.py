from .decoder import *
from .gat import *
from .gcn import *
from .imrpoved_gcn import *

import torch
import torch.nn.functional as F


def build_gnn(model_type, input_dim, hidden_dims, output_dim, **kwargs):
    """Construct a GNN encoder.

    Parameters
    ----------
    model_type : str
        'gcn', 'improved', or 'gat'.
    input_dim : int
        Input feature dimension.
    hidden_dims : list of int
        Hidden layer sizes.
    output_dim : int
        Number of output communities.
    **kwargs :
        Additional keyword arguments (dropout, batch_norm, layer_norm,
        heads, attn_dropout).

    Returns
    -------
    gnn : nn.Module
    """
    if model_type == 'improved':
        return ImprovedGCN(
            input_dim, hidden_dims, output_dim,
            dropout=kwargs.get('dropout', 0.5),
            layer_norm=kwargs.get('layer_norm', False),
        )
    elif model_type == 'gat':
        return GAT(
            input_dim, hidden_dims, output_dim,
            heads=kwargs.get('heads', 4),
            dropout=kwargs.get('dropout', 0.5),
            attn_dropout=kwargs.get('attn_dropout', 0.3),
            batch_norm=kwargs.get('batch_norm', False),
        )
    else:
        return GCN(
            input_dim, hidden_dims, output_dim,
            dropout=kwargs.get('dropout', 0.5),
            batch_norm=kwargs.get('batch_norm', False),
        )


def build_edge_index(model_type, adj, device=None):
    """Build PyG edge_index and edge_weight for the given model type.

    Parameters
    ----------
    model_type : str
        'gcn', 'improved', or 'gat'.
    adj : scipy.sparse matrix
        Adjacency matrix.
    device : torch.device or None

    Returns
    -------
    edge_index : torch.LongTensor
    edge_weight : torch.FloatTensor or None
    """
    if model_type == 'improved':
        return ImprovedGCN.build_edge_index(adj, device=device)
    elif model_type == 'gat':
        return GAT.build_edge_index(adj, device=device)
    else:
        return GCN.build_edge_index(adj, device=device)


def load_checkpoint(path, device=None):
    """Load a NOCD model checkpoint.

    Returns
    -------
    gnn : nn.Module
        Model with weights loaded, in eval mode, on the specified device.
    checkpoint : dict
        Full checkpoint dictionary (for metadata access).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    gnn = build_gnn(
        model_type=checkpoint['model_type'],
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        output_dim=checkpoint['output_dim'],
        dropout=checkpoint['dropout'],
        layer_norm=checkpoint.get('layer_norm', False),
        batch_norm=checkpoint.get('batch_norm', False),
    )
    gnn.load_state_dict(checkpoint['model_state_dict'])
    if device is not None:
        gnn = gnn.to(device)
    gnn.eval()
    return gnn, checkpoint


def infer(gnn, x, edge_index, edge_weight=None):
    """Run forward pass and return community membership as numpy array.

    Parameters
    ----------
    gnn : nn.Module
        NOCD GNN encoder (should be in eval mode).
    x : torch.Tensor
        Node feature matrix.
    edge_index : torch.LongTensor
    edge_weight : torch.FloatTensor or None

    Returns
    -------
    Z : np.ndarray, shape [N, K]
        Soft community membership scores (ReLU-activated).
    """
    gnn.eval()
    with torch.no_grad():
        Z = F.relu(gnn(x, edge_index, edge_weight))
    return Z.cpu().numpy()
