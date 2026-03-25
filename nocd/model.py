"""Scikit-learn compatible NOCD estimator."""

import os
import warnings
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator

from nocd import data as data_mod
from nocd import metrics, sampler, utils
from nocd.nn import (
    BerpoDecoder, build_edge_index, build_gnn, infer,
)
from nocd.train import ModelSaver, NoImprovementStopping


class NOCD(BaseEstimator):
    """Neural Overlapping Community Detection.

    Scikit-learn compatible estimator for overlapping community detection
    using Graph Neural Networks.

    Parameters
    ----------
    num_communities : int
        Number of communities to detect.
    model_type : str, default='improved'
        GNN variant: 'improved' or 'gcn'.
    hidden_dims : list of int, default=(128,)
        Hidden layer sizes.
    feature_type : str, default='X'
        Input features: 'X' (attributes), 'A' (adjacency), 'AX' (both).
    dropout : float, default=0.5
        Dropout rate.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-2
        L2 regularization strength.
    max_epochs : int, default=500
        Maximum training epochs.
    patience : int, default=10
        Early stopping patience (in display_step units).
    display_step : int, default=25
        Epochs between validation checks.
    balance_loss : bool, default=True
        Balance edge/non-edge loss contributions.
    stochastic_loss : bool, default=True
        Use mini-batch edge sampling for loss.
    batch_size : int, default=20000
        Edge batch size for stochastic training.
    batch_norm : bool, default=False
        Use batch normalization (GCN only).
    layer_norm : bool, default=False
        Use layer normalization (ImprovedGCN only).
    threshold : float, default=0.5
        Binarization threshold for community membership.
    device : str or None, default=None
        Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.

    Attributes
    ----------
    gnn_ : nn.Module
        Trained GNN encoder.
    Z_ : np.ndarray, shape (n_nodes, num_communities)
        Soft community membership scores from the last fit.
    labels_ : np.ndarray, shape (n_nodes, num_communities)
        Binary community assignments from the last fit.
    n_features_in_ : int
        Number of input features.
    """

    def __init__(
        self,
        num_communities,
        model_type='improved',
        hidden_dims=(128,),
        feature_type='X',
        dropout=0.5,
        lr=1e-3,
        weight_decay=1e-2,
        max_epochs=500,
        patience=10,
        display_step=25,
        balance_loss=True,
        stochastic_loss=True,
        batch_size=20000,
        batch_norm=False,
        layer_norm=False,
        threshold=0.5,
        device=None,
    ):
        self.num_communities = num_communities
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.feature_type = feature_type
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.display_step = display_step
        self.balance_loss = balance_loss
        self.stochastic_loss = stochastic_loss
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.threshold = threshold
        self.device = device

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return utils.get_device()

    def fit(self, A, X=None, y=None, verbose=True):
        """Fit the NOCD model.

        Parameters
        ----------
        A : scipy.sparse matrix, shape (n_nodes, n_nodes)
            Adjacency matrix.
        X : scipy.sparse matrix or np.ndarray or None, shape (n_nodes, n_features)
            Node attribute matrix. If None, adjacency is used as features.
        y : np.ndarray or None, shape (n_nodes, num_communities)
            Ground truth community labels (used only for NMI logging, not for training).
        verbose : bool, default=True
            Print training progress.

        Returns
        -------
        self
        """
        device = self._get_device()
        if verbose:
            print(f"Using device: {device}")

        N = A.shape[0]
        hidden_dims = list(self.hidden_dims)

        # Prepare features
        x_dense = data_mod.prepare_features(A, X, self.feature_type, device=device)
        self.n_features_in_ = x_dense.shape[1]

        # Build model
        gnn = build_gnn(
            self.model_type, self.n_features_in_, hidden_dims, self.num_communities,
            dropout=self.dropout, layer_norm=self.layer_norm, batch_norm=self.batch_norm,
        ).to(device)
        edge_index, edge_weight = build_edge_index(self.model_type, A, device=device)

        decoder = BerpoDecoder(N, A.nnz, balance_loss=self.balance_loss)
        opt = torch.optim.Adam(gnn.parameters(), lr=self.lr)

        # Edge sampler
        edge_sampler = sampler.get_edge_sampler(A, self.batch_size, self.batch_size, num_workers=2)

        # Early stopping
        val_loss = np.inf
        early_stopping = NoImprovementStopping(lambda: val_loss, patience=self.patience)
        model_saver = ModelSaver(gnn)

        for epoch, batch in enumerate(edge_sampler):
            if epoch > self.max_epochs:
                break

            if epoch % self.display_step == 0:
                with torch.no_grad():
                    gnn.eval()
                    Z = F.relu(gnn(x_dense, edge_index, edge_weight))
                    val_loss = decoder.loss_full(Z, A).item()

                    if verbose:
                        msg = f'Epoch {epoch:4d}, loss = {val_loss:.4f}'
                        if y is not None:
                            Z_np = Z.cpu().numpy()
                            nmi = metrics.overlapping_nmi(Z_np > self.threshold, y)
                            msg += f', nmi = {nmi:.4f}'
                        print(msg)

                    early_stopping.next_step()
                    if early_stopping.should_save():
                        model_saver.save()
                    if early_stopping.should_stop():
                        if verbose:
                            print(f'Early stopping at epoch {epoch}')
                        break

            # Training step
            gnn.train()
            opt.zero_grad()
            Z = F.relu(gnn(x_dense, edge_index, edge_weight))
            ones_idx, zeros_idx = batch
            ones_idx = ones_idx.to(device)
            zeros_idx = zeros_idx.to(device)

            if self.stochastic_loss:
                loss = decoder.loss_batch(Z, ones_idx, zeros_idx)
            else:
                loss = decoder.loss_full(Z, A)
            loss += utils.l2_reg_loss(gnn, scale=self.weight_decay)
            loss.backward()
            opt.step()

        # Restore best model
        model_saver.restore()
        self.gnn_ = gnn

        # Store predictions on training graph
        self.Z_ = infer(gnn, x_dense, edge_index, edge_weight)
        self.labels_ = (self.Z_ > self.threshold).astype(np.float32)

        if verbose:
            if y is not None:
                nmi = metrics.overlapping_nmi(self.labels_, y)
                print(f"\nTraining complete. Best NMI: {nmi:.4f}")
            else:
                print("\nTraining complete.")

        return self

    def predict(self, A, X=None):
        """Predict community memberships for a graph.

        Parameters
        ----------
        A : scipy.sparse matrix, shape (n_nodes, n_nodes)
            Adjacency matrix.
        X : scipy.sparse matrix or np.ndarray or None
            Node attribute matrix.

        Returns
        -------
        Z_binary : np.ndarray, shape (n_nodes, num_communities)
            Binary community assignment matrix.
        """
        Z = self.predict_proba(A, X)
        return (Z > self.threshold).astype(np.float32)

    def predict_proba(self, A, X=None):
        """Predict soft community membership scores.

        Parameters
        ----------
        A : scipy.sparse matrix
            Adjacency matrix.
        X : scipy.sparse matrix or np.ndarray or None
            Node attribute matrix.

        Returns
        -------
        Z : np.ndarray, shape (n_nodes, num_communities)
            Soft community membership scores.
        """
        self._check_is_fitted()
        device = self._get_device()
        x_dense = data_mod.prepare_features(A, X, self.feature_type, device=device)
        edge_index, edge_weight = build_edge_index(self.model_type, A, device=device)
        self.gnn_ = self.gnn_.to(device)
        return infer(self.gnn_, x_dense, edge_index, edge_weight)

    def score(self, A, X=None, y=None):
        """Compute overlapping NMI if ground truth is provided, else negative loss.

        Parameters
        ----------
        A : scipy.sparse matrix
            Adjacency matrix.
        X : scipy.sparse matrix or None
            Node attribute matrix.
        y : np.ndarray or None
            Ground truth community labels.

        Returns
        -------
        score : float
        """
        Z = self.predict_proba(A, X)
        if y is not None:
            return float(metrics.overlapping_nmi((Z > self.threshold).astype(np.float32), y))
        # Return negative loss as unsupervised score (higher is better)
        unsup = metrics.evaluate_unsupervised((Z > self.threshold).astype(np.float32), A)
        return float(unsup['coverage'])

    def save(self, path):
        """Save model checkpoint to disk.

        Parameters
        ----------
        path : str
            Output file path.
        """
        self._check_is_fitted()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.gnn_.cpu().state_dict(),
            'model_type': self.model_type,
            'input_dim': int(self.n_features_in_),
            'hidden_dims': [int(d) for d in self.hidden_dims],
            'output_dim': int(self.num_communities),
            'dropout': float(self.dropout),
            'batch_norm': bool(self.batch_norm),
            'layer_norm': bool(self.layer_norm),
            'balance_loss': bool(self.balance_loss),
            'features': self.feature_type,
            'threshold': float(self.threshold),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device=None):
        """Load a trained NOCD model from a checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint file.
        device : str or None
            Device to load onto.

        Returns
        -------
        model : NOCD
            Fitted NOCD estimator ready for prediction.
        """
        dev = torch.device(device) if device else utils.get_device()
        checkpoint = torch.load(path, map_location=dev, weights_only=False)

        model = cls(
            num_communities=checkpoint['output_dim'],
            model_type=checkpoint['model_type'],
            hidden_dims=tuple(checkpoint['hidden_dims']),
            feature_type=checkpoint.get('features', 'A'),
            dropout=checkpoint['dropout'],
            batch_norm=checkpoint.get('batch_norm', False),
            layer_norm=checkpoint.get('layer_norm', False),
            balance_loss=checkpoint.get('balance_loss', True),
            threshold=checkpoint.get('threshold', 0.5),
            device=str(dev),
        )

        gnn = build_gnn(
            model.model_type, checkpoint['input_dim'], list(model.hidden_dims),
            model.num_communities,
            dropout=model.dropout, layer_norm=model.layer_norm, batch_norm=model.batch_norm,
        )
        gnn.load_state_dict(checkpoint['model_state_dict'])
        gnn = gnn.to(dev)
        gnn.eval()
        model.gnn_ = gnn
        model.n_features_in_ = checkpoint['input_dim']

        return model

    def _check_is_fitted(self):
        if not hasattr(self, 'gnn_'):
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
