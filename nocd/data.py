import warnings

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize


def _load_sparse_csr(loader, prefix):
    """Load a scipy CSR matrix from npz keys with a given prefix."""
    return sp.csr_matrix(
        (loader[f'{prefix}.data'], loader[f'{prefix}.indices'], loader[f'{prefix}.indptr']),
        shape=loader[f'{prefix}.shape'],
    )


def _remove_self_loops(A):
    A = A.tolil()
    A.setdiag(0)
    return A.tocsr()


def load_dataset(file_name):
    """Load a graph from a Numpy binary file (NOCD format with labels).

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'Z' : The community labels as np.ndarray
            * Further dictionaries mapping node, class and attribute IDs
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = _load_sparse_csr(loader, 'adj_matrix')

        X = None
        if 'attr_matrix.data' in loader:
            X = _load_sparse_csr(loader, 'attr_matrix')

        Z = _load_sparse_csr(loader, 'labels')

        A = _remove_self_loops(A)

        if sp.issparse(Z):
            Z = Z.toarray().astype(np.float32)

        graph = {'A': A, 'X': X, 'Z': Z}

        for key in ('node_names', 'attr_names', 'class_names'):
            val = loader.get(key)
            if val is not None:
                graph[key] = val.tolist()

        return graph


def load_graph(path):
    """Load a graph from .npz (scipy sparse) or .csv (edge list).

    Supports NOCD-format .npz, raw CSR .npz, and CSV edge lists.

    Parameters
    ----------
    path : str
        Path to graph file.

    Returns
    -------
    A : sp.csr_matrix
        Adjacency matrix (self-loops removed).
    X : sp.csr_matrix or None
        Node attribute matrix if available.
    Z_gt : np.ndarray or None
        Ground truth community labels if available.
    """
    X = None
    Z_gt = None

    if path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        if 'adj_matrix.data' in data:
            A = _load_sparse_csr(data, 'adj_matrix')
            if 'attr_matrix.data' in data:
                X = _load_sparse_csr(data, 'attr_matrix')
            if 'labels.data' in data:
                Z_gt = _load_sparse_csr(data, 'labels').toarray().astype(np.float32)
        elif 'data' in data and 'indices' in data and 'indptr' in data:
            A = sp.csr_matrix((data['data'], data['indices'], data['indptr']),
                              shape=data['shape'])
        else:
            raise ValueError(f"Unrecognized .npz format. Keys: {list(data.keys())}")
    elif path.endswith('.csv'):
        edges = np.loadtxt(path, delimiter=',', skiprows=1)
        if edges.shape[1] == 2:
            rows, cols = edges[:, 0].astype(int), edges[:, 1].astype(int)
            weights = np.ones(len(rows))
        else:
            rows, cols, weights = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2]
        n = max(rows.max(), cols.max()) + 1
        A = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
        A = A.maximum(A.T)
    else:
        raise ValueError(f"Unsupported format: {path}. Use .npz or .csv")

    A = _remove_self_loops(A)
    return A, X, Z_gt


def load_features(path):
    """Load a node feature matrix from a .npz file.

    Returns
    -------
    X : sp.csr_matrix
    """
    data = _np_load(path)
    if 'attr_matrix.data' in data:
        return _load_sparse_csr(data, 'attr_matrix')
    if 'X' in data:
        X = data['X']
        return sp.csr_matrix(X) if not sp.issparse(X) else X
    raise ValueError(f"Cannot load features from {path}")


def prepare_features(A, X=None, feature_type='X', device=None):
    """Normalize features and convert to a dense tensor.

    Parameters
    ----------
    A : sp.spmatrix
        Adjacency matrix.
    X : sp.spmatrix or None
        Node attribute matrix.
    feature_type : str
        One of 'X' (attributes), 'A' (adjacency), 'AX' (both).
    device : torch.device or None

    Returns
    -------
    x_dense : torch.Tensor
        Dense feature tensor on the specified device.
    """
    if X is not None and feature_type in ('X', 'AX'):
        x_norm = normalize(X)
        if feature_type == 'AX':
            x_norm = sp.hstack([x_norm, normalize(A)])
    else:
        if feature_type == 'X' and X is None:
            warnings.warn("Model expects node features (X) but none available. Using adjacency.")
        x_norm = normalize(A)

    if sp.issparse(x_norm):
        x_dense = torch.FloatTensor(x_norm.toarray())
    else:
        x_dense = torch.FloatTensor(x_norm)

    if device is not None:
        x_dense = x_dense.to(device)
    return x_dense
