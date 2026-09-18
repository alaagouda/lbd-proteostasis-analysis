"""
Spearman correlation between hypoxia and proteostasis module scores,
restricted to endothelial nuclei and aggregated to donor level.

Usage:
    python endothelial_coupling.py --input data/nido_analysis_ready.h5ad \
        --cohort Nido --outdir results/nido_endothelial
"""

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

from common import resolve_columns, standardise_diagnosis, ensure_lognorm
from gene_modules import PROTEOSTASIS, HYPOXIA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    ct_col, dx_col, sample_col = resolve_columns(adata)
    adata = standardise_diagnosis(adata, dx_col)

    endo_mask = adata.obs[ct_col].astype(str).str.contains("endo", case=False, na=False)
    print(f"endothelial nuclei: {endo_mask.sum():,}")
    if endo_mask.sum() < 100:
        raise SystemExit("too few endothelial nuclei to run this reliably, check cell type labels")

    endo = adata[endo_mask].copy()
    ensure_lognorm(endo)

    hyp_genes = [g for g in HYPOXIA if g in endo.var_names]
    prot_genes = [g for g in PROTEOSTASIS if g in endo.var_names]
    sc.tl.score_genes(endo, hyp_genes, score_name="Hypoxia_score", ctrl_size=50)
    sc.tl.score_genes(endo, prot_genes, score_name="Proteostasis_score", ctrl_size=50)

    agg = (endo.obs.groupby(sample_col, observed=True)
           .agg({"Hypoxia_score": "mean", "Proteostasis_score": "mean", "Dx": "first"})
           .reset_index())

    rho, p = spearmanr(agg["Hypoxia_score"], agg["Proteostasis_score"])
    print(f"rho={rho:.4f}  p={p:.4f}  n={len(agg)} donors")

    agg.to_csv(outdir / f"{args.cohort}_endothelial_pseudobulk.csv", index=False)
    pd.DataFrame([{"Cohort": args.cohort, "rho": rho, "p": p, "n": len(agg)}]).to_csv(
        outdir / f"{args.cohort}_endothelial_correlation.csv", index=False)


if __name__ == "__main__":
    main()
