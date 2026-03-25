# Overlapping Community Detection with Graph Neural Networks

PyTorch + PyG implementation of the **Neural Overlapping Community Detection** method from
["Overlapping Community Detection with Graph Neural Networks"](http://www.kdd.in.tum.de/nocd).

Modernized from the [original repository](https://github.com/shchur/overlapping-community-detection)
to use PyTorch 2.x, PyTorch Geometric (PyG), and device-agnostic training (CUDA / MPS / CPU).

## Setup

Requires [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv venv --python 3.12
uv pip install -e .
```

This installs `torch`, `torch-geometric`, `numpy`, `scipy`, `scikit-learn`, and `matplotlib`.

## Training

```bash
uv run nocd-train --dataset data/mag_cs.npz
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `data/mag_cs.npz` | Path to dataset `.npz` file |
| `--model` | `gcn` | GNN variant: `gcn` (with batch norm) or `improved` |
| `--hidden-dims` | `128` | Hidden layer sizes (space-separated) |
| `--features` | `X` | Input features: `X` (attributes), `A` (adjacency), `AX` (both) |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-2` | L2 regularization strength |
| `--dropout` | `0.5` | Dropout rate |
| `--max-epochs` | `500` | Maximum training epochs |
| `--patience` | `10` | Early stopping patience |
| `--batch-size` | `20000` | Edge batch size for stochastic training |
| `--balance-loss` | `True` | Balance edge/non-edge loss contributions |
| `--output` | `checkpoints/nocd_model.pt` | Checkpoint save path |

The training script auto-detects CUDA, MPS (Apple Silicon), or CPU.

## Prediction

Detect overlapping communities on a new similarity graph using a trained model:

```bash
uv run nocd-predict \
    --checkpoint checkpoints/nocd_model.pt \
    --graph my_graph.npz \
    --output predictions.npz \
    --output-json communities.json
```

Supported graph input formats:
- **`.npz`** -- scipy sparse adjacency matrix (NOCD format with optional features, or raw CSR arrays with keys `data`, `indices`, `indptr`, `shape`)
- **`.csv`** -- edge list with columns `node_i, node_j[, weight]` (header row expected)

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `0.5` | Binarization threshold for community membership |
| `--features` | (none) | External node features `.npz` (optional) |
| `--output` | (none) | Save soft + binary predictions as `.npz` |
| `--output-json` | (none) | Save community lists as `.json` |

## Visualization

Generate a reordered adjacency matrix plot showing community structure:

```bash
# From saved predictions
uv run nocd-visualize \
    --graph data/mag_cs.npz \
    --predictions predictions.npz \
    --ground-truth \
    --output communities.png

# Directly from a checkpoint
uv run nocd-visualize \
    --graph data/mag_cs.npz \
    --checkpoint checkpoints/nocd_model.pt
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--ground-truth` | off | Show ground truth side-by-side (if labels in `.npz`) |
| `--threshold` | `0.5` | Community membership threshold |
| `--markersize` | `0.05` | Dot size in the spy plot |
| `--dpi` | `150` | Output image resolution |
| `--output` | `communities.png` | Output image path |

## Python API

The `NOCD` class provides a scikit-learn compatible interface:

```python
from nocd import NOCD
from nocd.data import load_dataset

# Load data
graph = load_dataset('data/mag_cs.npz')
A, X, Z_gt = graph['A'], graph['X'], graph['Z']
N, K = Z_gt.shape

# Train
model = NOCD(num_communities=K, model_type='gcn', hidden_dims=(128,), batch_norm=True)
model.fit(A, X, y=Z_gt)
model.save('checkpoints/nocd-gcn-X.pt')

# Predict (same or new graph)
Z_binary = model.predict(A, X)      # binary community assignments
Z_soft = model.predict_proba(A, X)  # soft membership scores

# Load a saved model for prediction only
model = NOCD.load('checkpoints/nocd-gcn-X.pt')
Z_binary = model.predict(A, X)

# Evaluate
from nocd.metrics import overlapping_nmi, evaluate_unsupervised
nmi = overlapping_nmi(Z_binary, Z_gt)
metrics = evaluate_unsupervised(Z_binary, A)
```

## Datasets

Included in `data/`:

- **Facebook Ego Networks** (`data/facebook_ego/`)
- **Microsoft Academic Graph**: `mag_cs.npz`, `mag_chem.npz`, `mag_eng.npz`, `mag_med.npz`

## Pretrained Checkpoints

Six pretrained checkpoints are provided in `checkpoints/`, covering all
combinations of model type and feature type. Each checkpoint has an
accompanying [ML Croissant](https://mlcommons.org/working-groups/data/croissant/)
JSON-LD metadata file.

Naming scheme: `nocd-{model}-{features}[-k{n_components}].pt`

| Checkpoint | Model | Features | NMI | Coverage | Conductance |
|---|---|---|---|---|---|
| `nocd-gcn-X.pt` | GCN + BatchNorm | Node attributes (X) | **0.48** | 0.93 | 0.22 |
| `nocd-gcn-spectral-k32.pt` | GCN + BatchNorm | Spectral (k=32) | 0.09 | 0.32 | 0.38 |
| `nocd-gcn-structural.pt` | GCN + BatchNorm | Structural (9-dim) | 0.00 | 0.85 | 0.35 |
| `nocd-improved-X.pt` | ImprovedGCN | Node attributes (X) | 0.27 | 0.81 | 0.26 |
| `nocd-improved-spectral-k32.pt` | ImprovedGCN | Spectral (k=32) | 0.00 | 0.20 | 0.65 |
| `nocd-improved-structural.pt` | ImprovedGCN | Structural (9-dim) | 0.00 | 0.89 | 0.06 |

All trained on `data/mag_cs.npz` (21,957 nodes, 193,500 edges, 18 communities).
Device: MPS (Apple Silicon), Python 3.14, PyTorch 2.11, PyG 2.7.

**Key findings:**
- GCN + BatchNorm with node attributes achieves the best NMI (~0.48), matching the original paper
- Structural and spectral features find real community structure (high coverage) but the
  communities don't align with ground-truth labels (low NMI) — this is expected since
  topology-only features lack the domain-specific signal in the keyword attributes
- The `structural` and `spectral` feature types enable cross-graph transfer (fixed input dim)
- Coverage and conductance show the models are finding meaningful graph partitions
  even when NMI is low

To regenerate all checkpoints: `uv run python scripts/train_all_checkpoints.py`

## Cite

Please cite the original paper if you use this code or the datasets:

```
@article{
    shchur2019overlapping,
    title={Overlapping Community Detection with Graph Neural Networks},
    author={Oleksandr Shchur and Stephan G\"{u}nnemann},
    journal={Deep Learning on Graphs Workshop, KDD},
    year={2019},
}
```
