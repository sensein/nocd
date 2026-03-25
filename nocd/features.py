"""Scikit-learn compatible feature transformers for graph-structural features."""

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Compute fixed-dimensional structural features from a graph adjacency matrix.

    Produces topology-derived node features that are independent of any
    domain-specific attributes, enabling models trained on one graph to
    generalize to graphs from different domains.

    Features computed per node:
        - degree (normalized by max degree)
        - log(degree + 1)
        - clustering coefficient
        - square clustering coefficient
        - average neighbor degree (normalized)
        - PageRank
        - HITS hub and authority scores
        - core number (normalized)

    Optionally concatenates with an existing feature matrix X.

    Parameters
    ----------
    include_x : bool, default=False
        If True and X is provided to transform(), concatenate structural
        features with the (normalized) X matrix.
    pagerank_alpha : float, default=0.85
        Damping factor for PageRank.
    pagerank_max_iter : int, default=100
        Maximum iterations for PageRank.
    normalize_output : bool, default=True
        Whether to L2-normalize each feature column.

    Attributes
    ----------
    n_features_ : int
        Number of structural features produced (excluding X).
    """

    _N_STRUCTURAL = 9  # number of structural feature columns

    def __init__(self, include_x=False, pagerank_alpha=0.85,
                 pagerank_max_iter=100, normalize_output=True):
        self.include_x = include_x
        self.pagerank_alpha = pagerank_alpha
        self.pagerank_max_iter = pagerank_max_iter
        self.normalize_output = normalize_output

    def fit(self, A, X=None, y=None):
        """Fit is a no-op (stateless transformer). Returns self."""
        self.n_features_ = self._N_STRUCTURAL
        return self

    def transform(self, A, X=None):
        """Compute structural features for the given adjacency matrix.

        Parameters
        ----------
        A : scipy.sparse matrix, shape (n_nodes, n_nodes)
            Adjacency matrix (symmetric, no self-loops).
        X : array-like or scipy.sparse matrix or None, shape (n_nodes, n_features)
            Optional node attribute matrix to concatenate.

        Returns
        -------
        features : np.ndarray, shape (n_nodes, n_structural [+ n_features])
            Structural features, optionally concatenated with normalized X.
        """
        A = sp.csr_matrix(A)
        n = A.shape[0]

        feats = []

        # 1. Degree features
        degree = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
        max_deg = degree.max() if degree.max() > 0 else 1.0
        feats.append(degree / max_deg)
        feats.append(np.log1p(degree))

        # 2. Clustering coefficient
        feats.append(_clustering_coefficient(A))

        # 3. Square clustering coefficient
        feats.append(_square_clustering(A))

        # 4. Average neighbor degree (normalized)
        avg_nbr_deg = _avg_neighbor_degree(A, degree)
        feats.append(avg_nbr_deg / max_deg)

        # 5. PageRank
        feats.append(_pagerank(A, alpha=self.pagerank_alpha,
                               max_iter=self.pagerank_max_iter))

        # 6. HITS hub and authority scores
        hub, auth = _hits(A)
        feats.append(hub)
        feats.append(auth)

        # 7. Core number (k-core decomposition, normalized)
        core = _core_number(A)
        max_core = core.max() if core.max() > 0 else 1.0
        feats.append(core / max_core)

        result = np.column_stack(feats).astype(np.float32)

        if self.normalize_output:
            # Per-column normalization to [0, 1]
            col_max = result.max(axis=0)
            col_max[col_max == 0] = 1.0
            result = result / col_max

        if self.include_x and X is not None:
            if sp.issparse(X):
                X_dense = normalize(X).toarray()
            else:
                X_dense = normalize(X)
            result = np.hstack([result, X_dense])

        return result

    def fit_transform(self, A, X=None, y=None):
        return self.fit(A, X, y).transform(A, X)


class SpectralFeatures(BaseEstimator, TransformerMixin):
    """Compute spectral embedding features from the graph Laplacian.

    Uses the smallest non-trivial eigenvectors of the normalized Laplacian
    as positional encodings. These capture global graph structure and have
    fixed dimensionality regardless of graph size.

    Parameters
    ----------
    n_components : int, default=16
        Number of eigenvectors to use (excluding the trivial constant one).
    include_x : bool, default=False
        If True and X is provided to transform(), concatenate spectral
        features with the (normalized) X matrix.
    normalize_output : bool, default=True
        Whether to L2-normalize each eigenvector column.

    Attributes
    ----------
    n_features_ : int
        Number of spectral features produced (excluding X).
    """

    def __init__(self, n_components=16, include_x=False, normalize_output=True):
        self.n_components = n_components
        self.include_x = include_x
        self.normalize_output = normalize_output

    def fit(self, A, X=None, y=None):
        """Fit is a no-op (stateless transformer). Returns self."""
        self.n_features_ = self.n_components
        return self

    def transform(self, A, X=None):
        """Compute spectral features for the given adjacency matrix.

        Parameters
        ----------
        A : scipy.sparse matrix, shape (n_nodes, n_nodes)
            Adjacency matrix (symmetric, no self-loops).
        X : array-like or scipy.sparse matrix or None, shape (n_nodes, n_features)
            Optional node attribute matrix to concatenate.

        Returns
        -------
        features : np.ndarray, shape (n_nodes, n_components [+ n_features])
            Spectral embedding features, optionally concatenated with normalized X.
        """
        from scipy.sparse.linalg import eigsh

        A = sp.csr_matrix(A, dtype=np.float64)
        n = A.shape[0]
        k = min(self.n_components + 1, n - 1)

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        degree = np.asarray(A.sum(axis=1)).ravel()
        deg_sqrt_inv = np.zeros_like(degree)
        nonzero = degree > 0
        deg_sqrt_inv[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        D_inv_sqrt = sp.diags(deg_sqrt_inv)
        L_norm = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        # Compute smallest eigenvectors (skip the trivial zero eigenvalue)
        eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='SM')

        # Sort by eigenvalue and drop the first (constant) eigenvector
        order = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, order]
        result = eigenvectors[:, 1:self.n_components + 1].astype(np.float32)

        # Pad if graph is too small
        if result.shape[1] < self.n_components:
            pad = np.zeros((n, self.n_components - result.shape[1]), dtype=np.float32)
            result = np.hstack([result, pad])

        if self.normalize_output:
            norms = np.linalg.norm(result, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            result = result / norms

        if self.include_x and X is not None:
            if sp.issparse(X):
                X_dense = normalize(X).toarray()
            else:
                X_dense = normalize(X)
            result = np.hstack([result, X_dense])

        return result

    def fit_transform(self, A, X=None, y=None):
        return self.fit(A, X, y).transform(A, X)


# --- Helper functions for structural features ---

def _clustering_coefficient(A):
    """Compute local clustering coefficient for each node."""
    A = sp.csr_matrix(A, dtype=np.float64)
    A_bool = (A > 0).astype(np.float64)
    n = A.shape[0]
    degree = np.asarray(A_bool.sum(axis=1)).ravel()
    # Number of triangles through each node: diag(A^3) / 2
    A2 = A_bool @ A_bool
    triangles = np.asarray(A2.multiply(A_bool).sum(axis=1)).ravel() / 2.0
    # Possible triangles: d*(d-1)/2
    denom = degree * (degree - 1) / 2.0
    cc = np.zeros(n)
    mask = denom > 0
    cc[mask] = triangles[mask] / denom[mask]
    return cc


def _square_clustering(A):
    """Compute square clustering coefficient (fraction of possible squares)."""
    A = sp.csr_matrix(A, dtype=np.float64)
    A_bool = (A > 0).astype(np.float64)
    n = A.shape[0]
    degree = np.asarray(A_bool.sum(axis=1)).ravel()
    # Squares through node i: sum of (A^2)_{ij}^2 for j != i, minus triangles
    A2 = A_bool @ A_bool
    # Number of paths of length 2 through each pair
    # Square clustering for node i = sum_{j in N(i)} (|N(i) ∩ N(j)| - 1) / (d_j - 1)
    # Simplified: use (A^2 elementwise squared sum - degree) as proxy
    sq = np.zeros(n)
    A2_diag = np.asarray(A2.diagonal())
    denom = degree * (degree - 1)
    mask = denom > 0
    # Approximate: fraction of 2-hop paths that close into squares
    sq[mask] = (A2_diag[mask] - degree[mask]) / denom[mask]
    return np.clip(sq, 0, 1)


def _avg_neighbor_degree(A, degree):
    """Average degree of neighbors for each node."""
    A_bool = (A > 0).astype(np.float64)
    nbr_deg_sum = np.asarray(A_bool @ degree.reshape(-1, 1)).ravel()
    d = np.asarray(A_bool.sum(axis=1)).ravel()
    avg = np.zeros_like(d)
    mask = d > 0
    avg[mask] = nbr_deg_sum[mask] / d[mask]
    return avg


def _pagerank(A, alpha=0.85, max_iter=100, tol=1e-6):
    """Power iteration PageRank."""
    A = sp.csr_matrix(A, dtype=np.float64)
    n = A.shape[0]
    out_degree = np.asarray(A.sum(axis=1)).ravel()
    out_degree[out_degree == 0] = 1.0
    # Transition matrix M = D^{-1} A
    M = sp.diags(1.0 / out_degree) @ A
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = (1 - alpha) / n + alpha * (M.T @ pr)
        if np.abs(pr_new - pr).sum() < tol:
            break
        pr = pr_new
    return pr / pr.max() if pr.max() > 0 else pr


def _hits(A, max_iter=100, tol=1e-6):
    """HITS algorithm returning hub and authority scores."""
    A = sp.csr_matrix(A, dtype=np.float64)
    n = A.shape[0]
    auth = np.ones(n) / n
    hub = np.ones(n) / n
    for _ in range(max_iter):
        auth_new = A.T @ hub
        norm = np.linalg.norm(auth_new)
        auth_new = auth_new / norm if norm > 0 else auth_new
        hub_new = A @ auth_new
        norm = np.linalg.norm(hub_new)
        hub_new = hub_new / norm if norm > 0 else hub_new
        if np.abs(auth_new - auth).sum() + np.abs(hub_new - hub).sum() < tol:
            break
        auth = auth_new
        hub = hub_new
    return hub, auth


def _core_number(A):
    """K-core decomposition returning core number for each node."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    degree = np.asarray(A.sum(axis=1)).ravel().astype(int)
    core = degree.copy()
    # Build adjacency lists
    remaining = np.ones(n, dtype=bool)
    sorted_nodes = np.argsort(degree)

    for node in sorted_nodes:
        if not remaining[node]:
            continue
        k = core[node]
        # Find neighbors
        row = A.getrow(node)
        neighbors = row.indices
        for nbr in neighbors:
            if remaining[nbr] and core[nbr] > k:
                core[nbr] = max(core[nbr] - 1, k)
        remaining[node] = False

    return core.astype(np.float64)
