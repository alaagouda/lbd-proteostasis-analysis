"""
Combines per-cohort Z-scores for the seven pre-specified heat shock
genes into a single weighted estimate (Stouffer's method, weighted by
sqrt(n)). Expects the *_proteostasis_by_celltype.csv files produced by
pseudobulk_de.py for each cohort - averages Z across cell types within
a cohort first, then combines across cohorts.

Usage:
    python stouffer_meta_analysis.py \
        --jin results/jin_de/Jin_proteostasis_by_celltype.csv \
        --ma results/ma_de/Ma_proteostasis_by_celltype.csv \
        --nido results/nido_de/Nido_proteostasis_by_celltype.csv \
        --outdir results/meta_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from gene_modules import HEAT_SHOCK_PRIMARY

COHORT_N = {"Jin": 6, "Ma": 17, "Nido": 33}


def load_cohort_zscores(path, cohort_name):
    if path is None:
        return None
    df = pd.read_csv(path)
    per_gene_z = df.groupby("gene")["Z_score"].mean()
    return per_gene_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jin")
    ap.add_argument("--ma")
    ap.add_argument("--nido")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cohort_z = {
        "Jin": load_cohort_zscores(args.jin, "Jin"),
        "Ma": load_cohort_zscores(args.ma, "Ma"),
        "Nido": load_cohort_zscores(args.nido, "Nido"),
    }
    cohort_z = {k: v for k, v in cohort_z.items() if v is not None}
    print(f"cohorts loaded: {list(cohort_z.keys())}")

    rows = []
    for gene in HEAT_SHOCK_PRIMARY:
        row = {"gene": gene}
        z_vals, weights = [], []
        for cohort, series in cohort_z.items():
            z = series.get(gene, np.nan)
            row[f"{cohort}_Z"] = z
            if not np.isnan(z):
                z_vals.append(z)
                weights.append(np.sqrt(COHORT_N[cohort]))

        if len(z_vals) >= 2:
            w = np.array(weights)
            combined_z = np.sum(np.array(z_vals) * w) / np.sqrt(np.sum(w ** 2))
            combined_p = 2 * (1 - norm.cdf(abs(combined_z)))
            same_direction = len(set(np.sign(z_vals))) == 1
        else:
            combined_z, combined_p, same_direction = np.nan, np.nan, False

        row["combined_Z"] = combined_z
        row["combined_p"] = combined_p
        row["n_cohorts"] = len(z_vals)
        row["same_direction"] = same_direction
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "stouffer_results.csv", index=False)

    print("\ngene         Z_combined      p        n_cohorts")
    for _, r in out.iterrows():
        print(f"{r['gene']:<12} {r['combined_Z']:>8.3f}   {r['combined_p']:.4f}    {r['n_cohorts']}")


if __name__ == "__main__":
    main()
