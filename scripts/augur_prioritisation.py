"""
Ranks cell types by how well a classifier can tell disease from
control nuclei using their transcriptome, following the Augur logic
(cross-validated AUROC per cell type). Downsamples large cell types to
500 nuclei per group so runtime stays reasonable and no single
population dominates just by having more cells.

Usage:
    python augur_prioritisation.py --input data/nido_analysis_ready.h5ad \
        --cohort Nido --outdir results/nido_augur
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import resolve_columns, standardise_diagnosis, ensure_lognorm

MIN_CELLS_PER_GROUP = 50
SUBSAMPLE_PER_GROUP = 500


def run_augur(adata, ct_col):
    rng = np.random.RandomState(42)
    results = []

    for ct in adata.obs[ct_col].value_counts().index:
        if str(ct).lower() in ("unassigned", "unknown", "nan"):
            continue
        sub = adata[adata.obs[ct_col] == ct]
        disease_idx = np.where((sub.obs["Dx"] == "Disease").values)[0]
        control_idx = np.where((sub.obs["Dx"] == "Control").values)[0]
        if len(disease_idx) < MIN_CELLS_PER_GROUP or len(control_idx) < MIN_CELLS_PER_GROUP:
            print(f"  {ct}: too few cells ({len(disease_idx)} disease / {len(control_idx)} control), skipping")
            continue

        keep = np.concatenate([
            rng.choice(disease_idx, min(SUBSAMPLE_PER_GROUP, len(disease_idx)), replace=False),
            rng.choice(control_idx, min(SUBSAMPLE_PER_GROUP, len(control_idx)), replace=False),
        ])
        ct_sub = sub[keep]
        X = ct_sub.X.toarray() if sps.issparse(ct_sub.X) else np.asarray(ct_sub.X)
        y = (ct_sub.obs["Dx"] == "Disease").astype(int).values

        aucs = []
        for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:, 1]))

        results.append({"Cell_Type": str(ct), "AUC": np.mean(aucs),
                         "n_disease": len(disease_idx), "n_control": len(control_idx)})
        print(f"  {ct}: AUC={np.mean(aucs):.3f}")

    return pd.DataFrame(results).sort_values("AUC", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    ensure_lognorm(adata)
    ct_col, dx_col, _ = resolve_columns(adata)
    adata = standardise_diagnosis(adata, dx_col)

    result = run_augur(adata, ct_col)
    result.to_csv(outdir / f"{args.cohort}_augur.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
