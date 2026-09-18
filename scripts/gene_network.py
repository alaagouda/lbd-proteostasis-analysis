"""
Pairwise Spearman correlations among the 14 proteostasis genes at
donor pseudobulk level, tiered by significance, then cross-checked
against STRING and BioGRID for physical interaction evidence.

Usage:
    python gene_network.py --input data/GSE253462_annotated.h5ad \
        --cohort Ma --outdir results/ma_network
"""

import argparse
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
import requests
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from common import resolve_columns, ensure_lognorm
from gene_modules import PROTEOSTASIS

STRING_CONFIDENCE = 400
BIOGRID_API_KEY = os.environ.get("BIOGRID_API_KEY", "")


def pseudobulk_expression(adata, genes, sample_col):
    present = [g for g in genes if g in adata.var_names]
    means = {}
    for gene in present:
        idx = adata.var_names.get_loc(gene)
        expr = adata.X[:, idx]
        expr = np.asarray(expr.todense()).flatten() if sps.issparse(expr) else np.asarray(expr).flatten()
        s = pd.Series(expr, index=adata.obs[sample_col].values)
        means[gene] = s.groupby(level=0).mean()
    return pd.DataFrame(means).dropna(), present


def correlation_network(expr_df, genes):
    rows = []
    for g1, g2 in combinations(genes, 2):
        rho, p = spearmanr(expr_df[g1], expr_df[g2])
        rows.append({"Gene1": g1, "Gene2": g2, "rho": rho, "pvalue": p})
    df = pd.DataFrame(rows)
    df["padj"] = multipletests(df["pvalue"], method="fdr_bh")[1]

    df["Tier"] = "not significant"
    df.loc[df["padj"] < 0.25, "Tier"] = "Tier 3 (FDR<0.25)"
    df.loc[(df["padj"] < 0.05) & (df["rho"].abs() > 0.5), "Tier"] = "Tier 2 (FDR<0.05, |r|>0.5)"
    df.loc[(df["padj"] < 0.01) & (df["rho"].abs() > 0.7), "Tier"] = "Tier 1 (FDR<0.01, |r|>0.7)"
    return df


def fetch_string(genes):
    resp = requests.get(
        "https://string-db.org/api/tsv/network",
        params={
            "identifiers": "\r".join(genes),
            "species": 9606,
            "required_score": STRING_CONFIDENCE,
            "network_type": "functional",
            "caller_identity": "lbd_proteostasis_pipeline",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  STRING request failed ({resp.status_code})")
        return pd.DataFrame(columns=["Gene1", "Gene2", "Combined_score"])
    rows = []
    for line in resp.text.strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 6:
            rows.append({"Gene1": parts[2], "Gene2": parts[3], "Combined_score": float(parts[5])})
    return pd.DataFrame(rows)


def fetch_biogrid(genes):
    if not BIOGRID_API_KEY:
        print("  no BIOGRID_API_KEY set, skipping BioGRID cross-check")
        return set()
    resp = requests.get(
        "https://webservice.thebiogrid.org/interactions/",
        params={
            "searchNames": "true",
            "geneList": "|".join(genes),
            "taxId": 9606,
            "format": "tab2",
            "accessKey": BIOGRID_API_KEY,
            "interSpeciesExcluded": "true",
            "selfInteractionsExcluded": "true",
            "includeInteractors": "false",
        },
        timeout=30,
    )
    genes_upper = {g.upper() for g in genes}
    pairs = set()
    for line in resp.text.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        g1, g2 = parts[7].upper(), parts[8].upper()
        if g1 in genes_upper and g2 in genes_upper and g1 != g2:
            pairs.add(tuple(sorted([g1, g2])))
    return pairs


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
    _, _, sample_col = resolve_columns(adata)

    expr_df, present = pseudobulk_expression(adata, PROTEOSTASIS, sample_col)
    print(f"{len(present)}/{len(PROTEOSTASIS)} genes present, {expr_df.shape[0]} donors")

    corr = correlation_network(expr_df, present)
    corr.to_csv(outdir / f"{args.cohort}_correlation_network.csv", index=False)
    print(corr["Tier"].value_counts().to_string())

    tier1_genes = set(corr.loc[corr["Tier"].str.startswith("Tier 1"), ["Gene1", "Gene2"]].values.flatten())
    string_df = fetch_string(present)
    string_df.to_csv(outdir / f"{args.cohort}_string_network.csv", index=False)
    print(f"STRING interactions retrieved: {len(string_df)}")

    if len(string_df):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(present)
        for _, row in string_df.iterrows():
            if row["Gene1"] in present and row["Gene2"] in present:
                G.add_edge(row["Gene1"], row["Gene2"], weight=row["Combined_score"])
        hubs = pd.DataFrame({
            "Gene": list(G.nodes()),
            "Degree_centrality": [nx.degree_centrality(G)[g] for g in G.nodes()],
        }).sort_values("Degree_centrality", ascending=False)
        hubs.to_csv(outdir / f"{args.cohort}_hub_genes.csv", index=False)
        print(hubs.head(5).to_string(index=False))

    biogrid_pairs = fetch_biogrid(present)
    if biogrid_pairs:
        tier1_pairs = {tuple(sorted([r["Gene1"], r["Gene2"]]))
                       for _, r in corr[corr["Tier"].str.startswith("Tier 1")].iterrows()}
        confirmed = tier1_pairs & biogrid_pairs
        print(f"Tier 1 pairs confirmed by BioGRID: {len(confirmed)}/{len(tier1_pairs)}")


if __name__ == "__main__":
    main()
