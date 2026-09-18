"""
Preranked MSigDB Hallmarks GSEA at cohort level. Ranking is by mean
DESeq2 Wald statistic across cell types - point this at the *_DE_*.csv
files pseudobulk_de.py writes per cell type.

Usage:
    python hallmarks_gsea.py --de-dir results/ma_de --cohort Ma \
        --outdir results/ma_gsea
"""

import argparse
from pathlib import Path

import pandas as pd
import gseapy as gp


def load_ranked_list(de_dir, cohort):
    de_files = sorted(Path(de_dir).glob(f"{cohort}_DE_*.csv"))
    if not de_files:
        raise SystemExit(f"no {cohort}_DE_*.csv files found in {de_dir}, run pseudobulk_de.py first")

    per_gene_stats = {}
    for f in de_files:
        df = pd.read_csv(f)
        for _, row in df.dropna(subset=["stat"]).iterrows():
            per_gene_stats.setdefault(row["gene"], []).append(row["stat"])

    ranked = {g: sum(v) / len(v) for g, v in per_gene_stats.items()}
    return pd.Series(ranked).sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--de-dir", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fdr-thresholds", nargs="+", type=float, default=[0.25, 0.05])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ranked = load_ranked_list(args.de_dir, args.cohort)
    print(f"ranked list: {len(ranked)} genes averaged across cell types")

    rnk = ranked.reset_index()
    rnk.columns = ["gene_name", "score"]

    pre_res = gp.prerank(rnk=rnk, gene_sets="MSigDB_Hallmark_2020",
                          min_size=5, max_size=1000, permutation_num=1000, seed=42, outdir=None)
    res = pre_res.res2d.sort_values("NES", ascending=False)
    res.to_csv(outdir / f"{args.cohort}_hallmarks_gsea.csv", index=False)

    for threshold in args.fdr_thresholds:
        n_sig = (res["FDR q-val"].astype(float) < threshold).sum()
        print(f"pathways at FDR<{threshold}: {n_sig}/{len(res)}")

    watch = ["Unfolded Protein Response", "TNF-alpha Signaling via NF-kB", "Inflammatory Response"]
    for term in watch:
        match = res[res["Term"].str.contains(term, case=False, na=False)]
        if len(match):
            r = match.iloc[0]
            print(f"  {term}: NES={float(r['NES']):.3f}  FDR={float(r['FDR q-val']):.4f}")


if __name__ == "__main__":
    main()
