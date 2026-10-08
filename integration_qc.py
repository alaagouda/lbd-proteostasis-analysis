"""
iLISI and average silhouette width before and after Harmony, as a
check that batch correction is mixing samples without erasing cell
type structure. Subsamples for speed since silhouette scoring is
slow on the full dataset.

Usage:
    python integration_qc.py --input data/nido_analysis_ready.h5ad \
        --outdir results/nido_qc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

from common import resolve_columns

N_SUBSAMPLE = 10_000
N_SILHOUETTE = 3_000
N_NEIGHBORS = 30


def compute_lisi(X, labels, n_neighbors=N_NEIGHBORS):
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nn.kneighbors(X)
    indices = indices[:, 1:]

    scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        neighbour_labels = labels[indices[i]]
        _, counts = np.unique(neighbour_labels, return_counts=True)
        simpson = np.sum((counts / counts.sum()) ** 2)
        scores[i] = 1 / simpson if simpson > 0 else 1
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    ct_col, _, sample_col = resolve_columns(adata)

    if "X_pca" not in adata.obsm or "X_pca_harmony" not in adata.obsm:
        raise SystemExit("need both X_pca and X_pca_harmony in adata.obsm to compare before/after")

    rng = np.random.RandomState(42)
    idx = rng.choice(adata.n_obs, min(N_SUBSAMPLE, adata.n_obs), replace=False)
    sub = adata[idx]

    batch_labels = LabelEncoder().fit_transform(sub.obs[sample_col].values)
    ct_labels = LabelEncoder().fit_transform(sub.obs[ct_col].astype(str).values)

    metrics = {}
    for label, embed_key in [("before", "X_pca"), ("after", "X_pca_harmony")]:
        embed = sub.obsm[embed_key][:, :30]
        metrics[f"iLISI_{label}"] = compute_lisi(embed, batch_labels).mean()
        metrics[f"cLISI_{label}"] = compute_lisi(embed, ct_labels).mean()

        sil_idx = rng.choice(len(sub), min(N_SILHOUETTE, len(sub)), replace=False)
        metrics[f"ASW_batch_{label}"] = silhouette_score(embed[sil_idx], batch_labels[sil_idx])
        metrics[f"ASW_celltype_{label}"] = silhouette_score(embed[sil_idx], ct_labels[sil_idx])

    print(pd.Series(metrics).to_string())
    pd.DataFrame([metrics]).to_csv(outdir / "integration_qc_metrics.csv", index=False)


if __name__ == "__main__":
    main()
