"""
Transcription factor regulon activity using decoupleR's univariate
linear model against the CollecTRI regulon database, compared between
disease and control donors within each cell type. This provides an
independent, orthogonal check against the curated target-gene panel
method used elsewhere in this project: CollecTRI's literature-curated
networks include substantially more targets per transcription factor
than a hand-picked panel, and are not specific to any one hypothesis.

Activity is aggregated to one mean value per donor before testing:
the donor, not the nucleus, is the unit of replication in a
case/control snRNA-seq design, and `resolve_columns()` returns the
donor/sample column specifically so this aggregation can happen before
any group comparison, consistent with module_scoring.py and
pseudobulk_de.py elsewhere in this repository.

Usage:
    python tf_activity.py --input data/GSE253462_annotated.h5ad \
        --cohort Ma --outdir results/ma_tf_activity
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from common import resolve_columns, standardise_diagnosis, ensure_lognorm

TFS_OF_INTEREST = ["ATF4", "HSF1", "NFKB1", "TP53", "ATF6", "XBP1", "NFE2L2", "STAT3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--by-celltype", action="store_true",
                     help="also report activity split by cell type, not just cohort-wide")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import decoupler as dc

    adata = sc.read_h5ad(args.input)
    ensure_lognorm(adata)
    ct_col, dx_col, sample_col = resolve_columns(adata)
    if sample_col is None:
        raise ValueError("No donor/sample column resolved -- cannot aggregate to donor "
                          "level, and testing at the cell level would be pseudoreplication. "
                          "Fix the column-matching in common.py before proceeding.")
    adata = standardise_diagnosis(adata, dx_col)

    print("fetching CollecTRI regulons")
    net = dc.get_collectri(organism="human", split_complexes=False)

    print("running ULM")
    dc.run_ulm(mat=adata, net=net, source="source", target="target", weight="weight", use_raw=False)
    acts = dc.get_acts(adata, obsm_key="ulm_estimate")

    groups = [None] if not args.by_celltype else sorted(adata.obs[ct_col].dropna().unique())
    rows = []

    for group in groups:
        mask = np.ones(adata.n_obs, dtype=bool) if group is None else (adata.obs[ct_col] == group).values
        sub = acts[mask]
        donors = sub.obs[sample_col].values
        dx = sub.obs["Dx"].values
        for tf in TFS_OF_INTEREST:
            if tf not in sub.var_names:
                continue
            cell_vals = np.asarray(sub[:, tf].X).flatten()
            # Aggregate to one mean value per donor BEFORE testing -- the donor,
            # not the cell, is the independent observation.
            df_cells = pd.DataFrame({"donor": donors, "dx": dx, "val": cell_vals})
            donor_means = df_cells.groupby(["donor", "dx"])["val"].mean().reset_index()
            disease = donor_means.loc[donor_means["dx"] == "Disease", "val"].values
            control = donor_means.loc[donor_means["dx"] == "Control", "val"].values
            if len(disease) < 3 or len(control) < 3:
                continue
            _, p = stats.mannwhitneyu(disease, control, alternative="two-sided")
            rows.append({
                "Cohort": args.cohort, "Cell_Type": group or "all",
                "TF": tf, "n_disease_donors": len(disease), "n_control_donors": len(control),
                "disease_mean": disease.mean(), "control_mean": control.mean(),
                "diff": disease.mean() - control.mean(), "p_value": p,
            })

    out = pd.DataFrame(rows)
    suffix = "_by_celltype" if args.by_celltype else ""
    out.to_csv(outdir / f"{args.cohort}_tf_activity{suffix}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
