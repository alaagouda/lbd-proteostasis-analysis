"""
Module scores and effect sizes for all eight gene modules, across
whichever cohort is passed in.

Cohen's d is computed at the donor level (mean score per sample,
then compare group means), not from per-cell scores directly: the
donor, not the nucleus, is the unit of replication in a case/control
snRNA-seq design, and testing at the cell level would treat thousands
of non-independent nuclei as independent observations. The groupby
step below is essential to this, not optional.

Usage:
    python module_scoring.py --input data/nido_analysis_ready.h5ad \
        --cohort Nido --outdir results/nido_modules
"""

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
from scipy import stats

from common import resolve_columns, standardise_diagnosis, ensure_lognorm, cohens_d, sample_level_scores
from gene_modules import MODULES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    print(f"{args.cohort}: {adata.n_obs:,} cells")
    ensure_lognorm(adata)

    _, dx_col, sample_col = resolve_columns(adata)
    adata = standardise_diagnosis(adata, dx_col)

    rows = []
    for module_name, genes in MODULES.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 3:
            print(f"  {module_name}: only {len(present)} genes present, skipping")
            continue

        score_col = f"score_{module_name}"
        sc.tl.score_genes(adata, present, score_name=score_col, ctrl_size=50)
        agg = sample_level_scores(adata, score_col, sample_col)

        disease = agg.loc[agg["Dx"] == "Disease", score_col].values
        control = agg.loc[agg["Dx"] == "Control", score_col].values
        if len(disease) < 2 or len(control) < 2:
            print(f"  {module_name}: not enough donors per group, skipping")
            continue

        d = cohens_d(disease, control)
        u, p = stats.mannwhitneyu(disease, control, alternative="two-sided")

        rows.append({
            "Cohort": args.cohort, "Module": module_name, "n_genes": len(present),
            "n_disease": len(disease), "n_control": len(control),
            "disease_mean": disease.mean(), "control_mean": control.mean(),
            "Cohens_d": d, "p_value": p,
        })
        print(f"  {module_name}: d={d:.3f}  p={p:.4f}  (n_genes={len(present)})")

    out = pd.DataFrame(rows).sort_values("Cohens_d", ascending=False)
    out.to_csv(outdir / f"{args.cohort}_module_effect_sizes.csv", index=False)
    print(f"saved {outdir / f'{args.cohort}_module_effect_sizes.csv'}")


if __name__ == "__main__":
    main()
