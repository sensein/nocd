"""CLI entry points for training, prediction, and visualization."""

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nocd import data as data_mod
from nocd import metrics, utils
from nocd.model import NOCD


def train_main():
    parser = argparse.ArgumentParser(description='Train NOCD model')
    parser.add_argument('--dataset', type=str, default='data/mag_cs.npz',
                        help='Path to dataset .npz file')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'improved'],
                        help='GNN model type')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-norm', action='store_true', default=True)
    parser.add_argument('--layer-norm', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--display-step', type=int, default=25)
    parser.add_argument('--balance-loss', action='store_true', default=True)
    parser.add_argument('--stochastic-loss', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=20000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--features', type=str, default='X',
                        choices=['X', 'A', 'AX', 'structural', 'spectral'],
                        help='Feature type: X (attributes), A (adjacency), AX (both), '
                             'structural (topology), spectral (Laplacian eigenvectors)')
    parser.add_argument('--n-components', type=int, default=16,
                        help='Number of spectral components (only for --features spectral)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='checkpoints/nocd_model.pt',
                        help='Output checkpoint path')
    args = parser.parse_args()

    # Load dataset
    loader = data_mod.load_dataset(args.dataset)
    A, X, Z_gt = loader['A'], loader['X'], loader['Z']
    N, K = Z_gt.shape
    print(f"Dataset: {args.dataset}")
    print(f"  Nodes: {N}, Communities: {K}, Edges: {A.nnz}")

    # Create and fit model
    model = NOCD(
        num_communities=K,
        model_type=args.model,
        hidden_dims=tuple(args.hidden_dims),
        feature_type=args.features,
        n_components=args.n_components,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        display_step=args.display_step,
        balance_loss=args.balance_loss,
        stochastic_loss=args.stochastic_loss,
        batch_size=args.batch_size,
        batch_norm=args.batch_norm,
        layer_norm=args.layer_norm,
        threshold=args.threshold,
    )
    model.fit(A, X, y=Z_gt)
    model.save(args.output)
    print(f"Model saved to {args.output}")


def predict_main():
    parser = argparse.ArgumentParser(description='Predict overlapping communities with trained NOCD model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--graph', type=str, required=True,
                        help='Path to graph (.npz scipy sparse or .csv edge list)')
    parser.add_argument('--features', type=str, default=None,
                        help='Path to external node features .npz (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary community assignment')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npz file with soft and binary predictions')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Output .json file with community lists')
    args = parser.parse_args()

    model = NOCD.load(args.checkpoint)
    model.threshold = args.threshold
    print(f"Model: {model.model_type}, Communities: {model.num_communities}")

    # Load graph
    A, X_from_graph, _ = data_mod.load_graph(args.graph)
    N = A.shape[0]
    print(f"Graph: {N} nodes, {A.nnz} edges")

    X = data_mod.load_features(args.features) if args.features else X_from_graph

    # Predict
    Z_np = model.predict_proba(A, X)
    Z_binary = model.predict(A, X)

    # Summary
    community_sizes = Z_binary.sum(axis=0).astype(int)
    nodes_in_any = (Z_binary.sum(axis=1) > 0).sum()
    multi_membership = (Z_binary.sum(axis=1) > 1).sum()
    print(f"\nResults (threshold={args.threshold}):")
    print(f"  Nodes assigned to at least 1 community: {nodes_in_any}/{N}")
    print(f"  Nodes in multiple communities: {multi_membership}")
    for i, size in enumerate(community_sizes):
        print(f"  Community {i}: {size} nodes")

    unsup = metrics.evaluate_unsupervised(Z_binary, A)
    print(f"\nUnsupervised metrics:")
    for k, v in unsup.items():
        print(f"  {k}: {v:.4f}")

    if args.output:
        np.savez(args.output, Z_soft=Z_np, Z_binary=Z_binary, threshold=args.threshold)
        print(f"\nPredictions saved to {args.output}")

    if args.output_json:
        communities = utils.coms_matrix_to_list(Z_binary)
        result = {
            'num_communities': len(communities),
            'communities': {i: [int(n) for n in c] for i, c in enumerate(communities)},
            'metrics': {k: float(v) for k, v in unsup.items()},
        }
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Community list saved to {args.output_json}")


def visualize_main():
    parser = argparse.ArgumentParser(description='Visualize community detection results')
    parser.add_argument('--graph', type=str, required=True, help='Path to graph .npz file')
    parser.add_argument('--predictions', type=str, default=None,
                        help='Path to predictions .npz (from nocd-predict)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (alternative to --predictions)')
    parser.add_argument('--ground-truth', action='store_true',
                        help='Also plot ground truth communities (if available in graph .npz)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--markersize', type=float, default=0.05)
    parser.add_argument('--output', type=str, default='communities.png',
                        help='Output image path')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    if args.predictions is None and args.checkpoint is None:
        parser.error("Provide either --predictions or --checkpoint")

    A, X, Z_gt = data_mod.load_graph(args.graph)

    if args.predictions is not None:
        pred_data = np.load(args.predictions, allow_pickle=True)
        Z_pred = pred_data['Z_soft']
    else:
        model = NOCD.load(args.checkpoint)
        Z_pred = model.predict_proba(A, X)

    show_gt = args.ground_truth and Z_gt is not None
    ncols = 2 if show_gt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 10))

    if ncols == 1:
        axes = [axes]

    _plot_communities(A, Z_pred, 'Predicted Communities', axes[0],
                      threshold=args.threshold, markersize=args.markersize)

    if show_gt:
        _plot_communities(A, Z_gt, 'Ground Truth Communities', axes[1],
                          threshold=args.threshold, markersize=args.markersize)

    plt.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved to {args.output}")
    plt.close(fig)


def _plot_communities(A, Z, title, ax, threshold=0.5, markersize=0.05):
    """Plot reordered adjacency matrix for a community assignment."""
    z = np.argmax(Z, axis=1)
    o = np.argsort(z)
    utils.plot_sparse_clustered_adjacency(A, Z.shape[1], z, o, ax=ax, markersize=markersize)
    ax.set_title(title, fontsize=14)
