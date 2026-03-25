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
| `--model` | `improved` | GNN variant: `improved` or `gcn` |
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
model = NOCD(num_communities=K, model_type='improved', hidden_dims=(128,))
model.fit(A, X, y=Z_gt)
model.save('checkpoints/nocd_model.pt')

# Predict (same or new graph)
Z_binary = model.predict(A, X)      # binary community assignments
Z_soft = model.predict_proba(A, X)  # soft membership scores

# Load a saved model for prediction only
model = NOCD.load('checkpoints/nocd_model.pt')
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

## Pretrained Checkpoint

A pretrained checkpoint (`checkpoints/nocd_model.pt`) is included, trained on `data/mag_cs.npz` with:

```
nocd-train --dataset data/mag_cs.npz --model improved --hidden-dims 128 \
    --features X --dropout 0.5 --lr 1e-3 --weight-decay 1e-2 \
    --max-epochs 500 --patience 10 --balance-loss --stochastic-loss \
    --batch-size 20000
```

Device: MPS (Apple Silicon), Python 3.14, PyTorch 2.11, PyG 2.7. Best NMI: ~0.32.

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
