"""
Pseudobulk differential expression, one cell type at a time.

Sums raw counts per donor within each cell type, then runs PyDESeq2
on the resulting donor-level count matrix. This is the same script
for all three cohorts - pass --cohort to pick the diagnosis/sample
column mapping.

Usage:
    python pseudobulk_de.py --input data/GSE253462_annotated.h5ad \
        --cohort Ma --outdir results/ma_de
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps

from common import resolve_columns, standardise_diagnosis


def build_pseudobulk(adata, cell_type_col, sample_col, min_total_count=10):
    counts_by_celltype = {}
    for ct in sorted(adata.obs[cell_type_col].dropna().unique()):
        sub = adata[adata.obs[cell_type_col] == ct]
        donors = sorted(sub.obs[sample_col].unique())
        mat = np.zeros((sub.shape[1], len(donors)))
        for j, donor in enumerate(donors):
            X = sub[sub.obs[sample_col] == donor].X
            mat[:, j] = np.asarray(X.sum(axis=0)).flatten() if sps.issparse(X) else X.sum(axis=0)
        df = pd.DataFrame(mat, index=sub.var_names, columns=donors).round().astype(int)
        df = df[df.sum(axis=1) >= min_total_count]
        if df.shape[1] < 4:
            print(f"  {ct}: only {df.shape[1]} donors, skipping")
            continue
        counts_by_celltype[ct] = df
    return counts_by_celltype


def run_deseq2(counts, dx_by_donor, disease_label="Disease", control_label="Control"):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    donors = counts.columns.tolist()
    meta = pd.DataFrame({"condition": [dx_by_donor[d] for d in donors]}, index=donors)
    if meta["condition"].nunique() < 2:
        return None

    dds = DeseqDataSet(counts=counts.T, metadata=meta, design="~condition", refit_cooks=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", disease_label, control_label])
    stats.summary()
    res = stats.results_df.reset_index().rename(columns={"index": "gene"})
    res["Z_score"] = res["stat"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"  {adata.n_obs:,} cells")

    ct_col, dx_col, sample_col = resolve_columns(adata)
    if not all([ct_col, dx_col, sample_col]):
        raise SystemExit("could not resolve one of cell_type/diagnosis/sample columns, check adata.obs manually")

    adata = standardise_diagnosis(adata, dx_col)
    dx_by_donor = adata.obs.drop_duplicates(sample_col).set_index(sample_col)["Dx"].to_dict()

    print("building pseudobulk per cell type")
    pseudobulk = build_pseudobulk(adata, ct_col, sample_col)

    heat_shock_rows = []
    from gene_modules import PROTEOSTASIS

    for ct, counts in pseudobulk.items():
        print(f"running DESeq2 for {ct}")
        res = run_deseq2(counts, dx_by_donor)
        if res is None:
            continue
        res["Cell_Type"] = ct
        res["Cohort"] = args.cohort
        n_sig = (res["padj"] < 0.05).sum()
        print(f"  {n_sig} genes FDR<0.05")

        clean_name = str(ct).replace(" ", "_").replace("/", "_")
        res.to_csv(outdir / f"{args.cohort}_DE_{clean_name}.csv", index=False)

        heat_shock_rows.append(res[res["gene"].isin(PROTEOSTASIS)])

    if heat_shock_rows:
        combined = pd.concat(heat_shock_rows, ignore_index=True)
        combined.to_csv(outdir / f"{args.cohort}_proteostasis_by_celltype.csv", index=False)
        print(f"saved combined proteostasis DE table ({len(combined)} rows)")


if __name__ == "__main__":
    main()
