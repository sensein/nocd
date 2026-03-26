#!/usr/bin/env python
"""Train transferable checkpoints and generate Croissant metadata.

Ships only domain-agnostic checkpoints (structural/spectral features)
trained on the largest dataset (mag_med). X-feature checkpoints are
trained for comparison but not committed to the repo.
"""

import hashlib
import json
import os
from datetime import date

from nocd import NOCD
from nocd.data import load_dataset
from nocd.metrics import overlapping_nmi, evaluate_unsupervised

OUTDIR = "checkpoints"

DATASETS = {
    "mag_cs": "data/mag_cs.npz",
    "mag_chem": "data/mag_chem.npz",
    "mag_eng": "data/mag_eng.npz",
    "mag_med": "data/mag_med.npz",
}

# Domain-agnostic features trained on the largest dataset
AGNOSTIC_DATASET = "mag_med"

# Shipped checkpoints: domain-agnostic features on largest dataset
SHIPPED_CONFIGS = [
    {"model_type": "gcn", "feature_type": "structural", "hidden_dims": (64, 32), "batch_norm": True},
    {"model_type": "gcn", "feature_type": "spectral", "hidden_dims": (64, 32), "batch_norm": True, "n_components": 32},
]

# Comparison-only: X features on each dataset (not shipped)
COMPARISON_CONFIG = {"model_type": "gcn", "feature_type": "X", "hidden_dims": (128,), "batch_norm": True}


def checkpoint_name(model_type, feature_type, dataset_key, n_components=None):
    name = f"nocd-{model_type}-{feature_type}"
    if feature_type == "spectral" and n_components:
        name += f"-k{n_components}"
    name += f"-{dataset_key}"
    return name


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_croissant(name, path, cfg, dataset_key, nmi, unsup, n_nodes, n_edges, n_communities):
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
        },
        "@type": "sc:Dataset",
        "dct:conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": name,
        "description": (
            f"Pretrained NOCD checkpoint ({cfg['model_type']} model, "
            f"{cfg['feature_type']} features) for overlapping community detection. "
            f"Trained on Microsoft Academic Graph ({dataset_key})."
        ),
        "license": "https://opensource.org/licenses/MIT",
        "url": f"https://github.com/sensein/nocd/tree/main/checkpoints/{name}.pt",
        "creator": {
            "@type": "Organization",
            "name": "sensein",
            "url": "https://github.com/sensein",
        },
        "datePublished": str(date.today()),
        "version": "0.2.0",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": f"{name}.pt",
                "name": f"{name}.pt",
                "contentUrl": f"{name}.pt",
                "encodingFormat": "application/x-pytorch",
                "sha256": sha256_file(path),
            }
        ],
        "cr:trainedOn": {
            "@type": "sc:Dataset",
            "name": dataset_key,
            "description": f"Microsoft Academic Graph - {dataset_key} co-authorship network",
            "sc:numberOfNodes": n_nodes,
            "sc:numberOfEdges": n_edges,
            "sc:numberOfCommunities": n_communities,
        },
        "cr:modelArchitecture": cfg["model_type"],
        "cr:featureType": cfg["feature_type"],
        "cr:hiddenDims": list(cfg["hidden_dims"]),
        "cr:batchNorm": cfg.get("batch_norm", False),
        "cr:layerNorm": cfg.get("layer_norm", False),
        "cr:dropout": 0.5,
        "cr:nComponents": cfg.get("n_components", None),
        "cr:metrics": {
            "overlappingNMI": round(float(nmi), 4),
            "coverage": round(float(unsup["coverage"]), 4),
            "conductance": round(float(unsup["conductance"]), 4),
            "density": round(float(unsup["density"]), 4),
        },
    }


def train_one(cfg, dataset_key, dataset_path, save=True):
    graph = load_dataset(dataset_path)
    A, X, Z_gt = graph["A"], graph["X"], graph["Z"]
    N, K = Z_gt.shape

    name = checkpoint_name(
        cfg["model_type"], cfg["feature_type"], dataset_key,
        n_components=cfg.get("n_components"),
    )
    pt_path = os.path.join(OUTDIR, f"{name}.pt")
    json_path = os.path.join(OUTDIR, f"{name}.json")

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Dataset: {dataset_path} ({N} nodes, {A.nnz} edges, {K} communities)")
    print(f"{'='*60}")

    model = NOCD(
        num_communities=K,
        max_epochs=500,
        patience=10,
        display_step=50,
        balance_loss=True,
        stochastic_loss=True,
        batch_size=20000,
        dropout=0.5,
        lr=1e-3,
        weight_decay=1e-2,
        **cfg,
    )
    model.fit(A, X, y=Z_gt)

    Z_pred = model.predict(A, X if cfg["feature_type"] == "X" else None)
    nmi = overlapping_nmi(Z_pred, Z_gt)
    unsup = evaluate_unsupervised(Z_pred, A)
    print(f"Final NMI: {nmi:.4f}, Coverage: {unsup['coverage']:.4f}")

    if save:
        model.save(pt_path)
        croissant = make_croissant(name, pt_path, cfg, dataset_key, nmi, unsup, N, A.nnz, K)
        with open(json_path, "w") as f:
            json.dump(croissant, f, indent=2)
        print(f"Saved: {pt_path}")

    return name, nmi, unsup


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Clean old checkpoints
    for f in os.listdir(OUTDIR):
        if f.startswith("nocd-") and (f.endswith(".pt") or f.endswith(".json")):
            os.remove(os.path.join(OUTDIR, f))

    results = []

    # 1. Shipped checkpoints: domain-agnostic on largest dataset
    print("\n" + "=" * 60)
    print("SHIPPED CHECKPOINTS (domain-agnostic, transferable)")
    print("=" * 60)
    for cfg in SHIPPED_CONFIGS:
        name, nmi, unsup = train_one(cfg, AGNOSTIC_DATASET, DATASETS[AGNOSTIC_DATASET], save=True)
        results.append((name, nmi, unsup, True))

    # 2. Comparison: X features on each dataset (not shipped)
    print("\n" + "=" * 60)
    print("COMPARISON CHECKPOINTS (X features, not shipped)")
    print("=" * 60)
    for ds_key, ds_path in DATASETS.items():
        name, nmi, unsup = train_one(COMPARISON_CONFIG, ds_key, ds_path, save=False)
        results.append((name, nmi, unsup, False))

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Checkpoint':<45} {'Ship':>5} {'NMI':>8} {'Cov':>8} {'Cond':>8}")
    print(f"{'-'*80}")
    for name, nmi, unsup, shipped in results:
        tag = "  Y" if shipped else "  -"
        print(f"{name:<45} {tag:>5} {nmi:>8.4f} {unsup['coverage']:>8.4f} {unsup['conductance']:>8.4f}")


if __name__ == "__main__":
    main()
