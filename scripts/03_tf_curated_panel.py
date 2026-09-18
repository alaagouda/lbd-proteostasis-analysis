"""
Transcription factor activity using a curated panel of literature-
established direct target genes per factor, compared between disease
and control donors within each cell type. This is the primary method
used for the transcription-factor results reported in the manuscript;
decoupleR (tf_activity.py) is used separately as an orthogonal check
against the broader, database-derived CollecTRI regulon set.

Activity is scored per donor as the mean z-score across a factor's
target genes, computed at the donor-level pseudobulk (CPM-normalised
mean per donor), then compared between disease and control groups with
a Mann-Whitney U test. The donor, not the nucleus, is the unit of
replication: `resolve_columns()` returns the donor/sample column
specifically so activity can be aggregated to one value per donor
before any group comparison, consistent with module_scoring.py and
pseudobulk_de.py elsewhere in this repository.

Usage:
    python 03_tf_curated_panel.py --input data/GSE253462_annotated.h5ad \
        --cohort Ma --outdir results/ma_tf_activity
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from scipy.stats import mannwhitneyu

from common import resolve_columns, standardise_diagnosis

REGULONS = {
    "ATF4":   ["DDIT3", "ATF3", "ASNS", "PSAT1", "PHGDH", "SLC7A11", "HERPUD1"],
    "TP53":   ["CDKN1A", "MDM2", "BBC3", "PMAIP1", "BAX", "GADD45A", "SESN2"],
    "NFE2L2": ["HMOX1", "NQO1", "GSTM1", "TXNRD1", "PRDX1", "TXN", "SRXN1"],
    "XBP1":   ["HSPA5", "DNAJB9", "EDEM1", "PDIA4", "SEC61A1", "DERL1"],
    "ATF6":   ["HSPA5", "DDIT3", "HERPUD1", "PDIA4", "DNAJB11"],
    "HSF1":   ["HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "CRYAB",
               "HSP90AA1", "HSPA6", "HSPD1", "HSPE1", "DNAJB6",
               "HSPA2", "HSPB8", "HSPA8", "DNAJB2", "HSPH1", "BAG3", "STIP1"],
    "STAT3":  ["BCL2", "MYC", "CCND1", "MCL1", "VEGFA", "HIF1A", "TWIST1"],
    "NFKB1":  ["TNF", "IL6", "IL1B", "CXCL10", "CCL2", "PTGS2",
               "ICAM1", "VCAM1", "BCL2", "BIRC3"],
}


def build_donor_pseudobulk(adata, sample_col, dx_col, target_sum=1e6):
    """CPM-normalised mean expression per donor. Densifies after the
    groupby step: pandas keeps a Sparse dtype through groupby().mean()
    in current versions, and .std() cannot be computed on a Sparse
    column. Densifying here is safe -- the result is one row per donor,
    not per cell."""
    X = adata.X
    if sps.issparse(X):
        counts_per_cell = np.asarray(X.sum(axis=1)).flatten()
        counts_per_cell[counts_per_cell == 0] = 1
        scale = target_sum / counts_per_cell
        X_cpm = (sps.diags(scale) @ X).tocsr()
        df = pd.DataFrame.sparse.from_spmatrix(X_cpm, index=adata.obs_names, columns=adata.var_names)
    else:
        counts_per_cell = X.sum(axis=1)
        counts_per_cell[counts_per_cell == 0] = 1
        scale = target_sum / counts_per_cell
        df = pd.DataFrame(X * scale[:, None], index=adata.obs_names, columns=adata.var_names)

    df["sample"] = adata.obs[sample_col].values
    df["diagnosis"] = adata.obs[dx_col].values
    pb = df.groupby("sample").mean(numeric_only=True)
    pb = pb.sparse.to_dense() if hasattr(pb, "sparse") else pb.astype(float)
    pb["Diagnosis"] = df.groupby("sample")["diagnosis"].first()
    return pb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to the cohort's h5ad file")
    ap.add_argument("--cohort", required=True, help="Cohort label, e.g. Ma or Nido")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    # NOTE: deliberately not calling ensure_lognorm() here -- this script's
    # own CPM normalisation in build_donor_pseudobulk() requires genuinely
    # raw counts as input; log-transforming first would corrupt that step.
    _, dx_col, sample_col = resolve_columns(adata)
    if sample_col is None:
        raise ValueError("No donor/sample column resolved -- cannot aggregate to donor "
                          "level, and testing at the cell level would be pseudoreplication.")
    adata = standardise_diagnosis(adata, dx_col)

    pb = build_donor_pseudobulk(adata, sample_col=sample_col, dx_col="Dx")
    pd_mask = pb["Diagnosis"].astype(str).str.lower().str.contains("pd|case|lbd", na=False)

    rows = []
    for tf, targets in REGULONS.items():
        avail = [g for g in targets if g in pb.columns]
        if len(avail) < 3:
            continue
        z = (pb[avail] - pb[avail].mean()) / (pb[avail].std() + 1e-8)
        activity = z.mean(axis=1)
        pd_act = activity[pd_mask].values
        ctrl_act = activity[~pd_mask].values
        if len(pd_act) < 3 or len(ctrl_act) < 3:
            continue
        _, p = mannwhitneyu(pd_act, ctrl_act, alternative="two-sided")
        rows.append({
            "Cohort": args.cohort, "TF": tf, "n_targets": len(avail),
            "n_disease_donors": len(pd_act), "n_control_donors": len(ctrl_act),
            "mean_disease": pd_act.mean(), "mean_control": ctrl_act.mean(),
            "mean_diff": pd_act.mean() - ctrl_act.mean(), "p_value": p,
        })

    out = pd.DataFrame(rows).sort_values("p_value")
    out_path = outdir / f"{args.cohort}_tf_curated_panel.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
