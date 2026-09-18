"""
Tests GO:0006986 (response to unfolded protein) as a pre-ranked
enrichment against a cohort-level ranked gene list. This was
pre-specified as a primary outcome and needs to be run the same way
for all three cohorts - it is not the same gene set as the MSigDB
Hallmark "Unfolded Protein Response" collection used in
hallmarks_gsea.py, and the two shouldn't be reported interchangeably.

Ranking is by Cohen's d per gene at donor pseudobulk level, averaged
across cell types.

Usage:
    python go_enrichment.py --input data/nido_analysis_ready.h5ad \
        --cohort Nido --outdir results/nido_go
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
import gseapy as gp

from common import resolve_columns, standardise_diagnosis, ensure_lognorm, cohens_d


def build_ranked_list(adata, sample_col):
    X = adata.X.toarray() if sps.issparse(adata.X) else np.asarray(adata.X)
    df = pd.DataFrame(X, columns=adata.var_names)
    df[sample_col] = adata.obs[sample_col].values
    df["Dx"] = adata.obs["Dx"].values
    pseudobulk = df.groupby(sample_col).mean(numeric_only=True)
    dx_by_sample = df.groupby(sample_col)["Dx"].first()

    disease_samples = dx_by_sample[dx_by_sample == "Disease"].index
    control_samples = dx_by_sample[dx_by_sample == "Control"].index

    scores = {}
    for gene in pseudobulk.columns:
        d = cohens_d(pseudobulk.loc[disease_samples, gene].values,
                      pseudobulk.loc[control_samples, gene].values)
        if not np.isnan(d):
            scores[gene] = d
    return pd.Series(scores).sort_values(ascending=False)


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
    _, dx_col, sample_col = resolve_columns(adata)
    adata = standardise_diagnosis(adata, dx_col)

    print("building ranked gene list")
    ranked = build_ranked_list(adata, sample_col)
    print(f"  {len(ranked)} genes ranked")

    print("fetching GO Biological Process gene sets")
    go_bp = gp.get_library(name="GO_Biological_Process_2023", organism="human")
    term = next((t for t in go_bp if "0006986" in t or "response to unfolded protein" in t.lower()), None)
    if term is None:
        raise SystemExit("GO:0006986 not found in the downloaded library, check gseapy's GO version")

    print(f"testing term: {term} ({len(go_bp[term])} genes)")
    rnk = ranked.reset_index()
    rnk.columns = ["gene_name", "score"]
    pre_res = gp.prerank(rnk=rnk, gene_sets={term: go_bp[term]},
                          min_size=3, max_size=1000, permutation_num=1000, seed=42, outdir=None)

    if len(pre_res.res2d) == 0:
        raise SystemExit("not enough overlap between the term and this cohort's ranked list to test")

    row = pre_res.res2d.iloc[0]
    result = pd.DataFrame([{
        "Cohort": args.cohort, "Term": term,
        "NES": float(row["NES"]), "FDR": float(row["FDR q-val"]),
    }])
    result.to_csv(outdir / f"{args.cohort}_GO0006986.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
