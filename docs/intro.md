# NOCD: Neural Overlapping Community Detection

This book demonstrates the modernized NOCD package for overlapping community
detection using Graph Neural Networks.

NOCD implements the method from
["Overlapping Community Detection with Graph Neural Networks"](http://www.kdd.in.tum.de/nocd)
(Shchur & Gunnemann, KDD 2019), updated to use PyTorch 2.x and PyTorch Geometric.

## Contents

- **Original Demo**: Training and evaluating the GCN-based community detector
  on the Microsoft Academic Graph dataset (reproducing the original notebook).
- **Feature Transformers Demo**: Using `StructuralFeatures` and `SpectralFeatures`
  for domain-agnostic community detection that can generalize across different
  types of graphs.

## Installation

```bash
uv venv --python 3.14
uv pip install -e .
```

## Links

- [GitHub Repository](https://github.com/sensein/nocd)
- [Original Paper](http://www.kdd.in.tum.de/nocd)
