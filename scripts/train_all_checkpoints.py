#!/usr/bin/env python
"""Train all checkpoint variants and generate Croissant metadata."""

import hashlib
import json
import os
from datetime import date

from nocd import NOCD
from nocd.data import load_dataset
from nocd.metrics import overlapping_nmi

DATASET = "data/mag_cs.npz"
OUTDIR = "checkpoints"

CONFIGS = [
    {"model_type": "gcn", "feature_type": "X", "hidden_dims": (128,), "batch_norm": True},
    {"model_type": "gcn", "feature_type": "structural", "hidden_dims": (64, 32), "batch_norm": True},
    {"model_type": "gcn", "feature_type": "spectral", "hidden_dims": (64, 32), "batch_norm": True, "n_components": 32},
    {"model_type": "improved", "feature_type": "X", "hidden_dims": (128,), "layer_norm": False},
    {"model_type": "improved", "feature_type": "structural", "hidden_dims": (64, 32), "layer_norm": False},
    {"model_type": "improved", "feature_type": "spectral", "hidden_dims": (64, 32), "layer_norm": False, "n_components": 32},
]


def checkpoint_name(cfg):
    name = f"nocd-{cfg['model_type']}-{cfg['feature_type']}"
    if cfg["feature_type"] == "spectral":
        name += f"-k{cfg.get('n_components', 16)}"
    return name


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_croissant(name, path, cfg, nmi, n_nodes, n_edges, n_communities):
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
            f"Trained on Microsoft Academic Graph (Computer Science)."
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
            "name": "mag_cs",
            "description": "Microsoft Academic Graph - Computer Science co-authorship network",
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
        "cr:overlappingNMI": round(float(nmi), 4),
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    graph = load_dataset(DATASET)
    A, X, Z_gt = graph["A"], graph["X"], graph["Z"]
    N, K = Z_gt.shape

    for cfg in CONFIGS:
        name = checkpoint_name(cfg)
        pt_path = os.path.join(OUTDIR, f"{name}.pt")
        json_path = os.path.join(OUTDIR, f"{name}.json")

        print(f"\n{'='*60}")
        print(f"Training: {name}")
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

        Z_pred = model.predict(A, X)
        nmi = overlapping_nmi(Z_pred, Z_gt)
        print(f"Final NMI: {nmi:.4f}")

        model.save(pt_path)
        print(f"Saved: {pt_path}")

        croissant = make_croissant(name, pt_path, cfg, nmi, N, A.nnz, K)
        with open(json_path, "w") as f:
            json.dump(croissant, f, indent=2)
        print(f"Saved: {json_path}")

    # Remove old generic checkpoint
    old = os.path.join(OUTDIR, "nocd_model.pt")
    if os.path.exists(old):
        os.remove(old)
        print(f"\nRemoved old checkpoint: {old}")


if __name__ == "__main__":
    main()
