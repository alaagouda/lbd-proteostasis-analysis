"""
Usage:
    python cell_cell_communication.py --input data/GSE253462_annotated.h5ad \
        --cohort Ma --outdir results/ma_liana
"""

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from common import resolve_columns, ensure_lognorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-perms", type=int, default=100)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import liana as li

    adata = sc.read_h5ad(args.input)
    ensure_lognorm(adata)
    ct_col, _, sample_col = resolve_columns(adata)

    print(f"running LIANA rank_aggregate by sample ({adata.obs[sample_col].nunique()} samples)")
    li.method.rank_aggregate.by_sample(
        adata, groupby=ct_col, sample_key=sample_col,
        expr_prop=0.1, min_cells=10, resource_name="consensus",
        n_perms=args.n_perms, use_raw=False, verbose=True,
    )

    res = adata.uns["liana_res"]
    if hasattr(res.index, "names"):
        res = res.reset_index()
    res.to_csv(outdir / f"{args.cohort}_liana_full.csv", index=False)
    print(f"total interactions: {len(res):,}")

    rank_col = next((c for c in res.columns if "magnitude_rank" in c or "rank" in c.lower()), None)
    ligand_col = next((c for c in res.columns if "ligand" in c.lower()), None)
    receptor_col = next((c for c in res.columns if "receptor" in c.lower()), None)

    if rank_col is None or ligand_col is None or receptor_col is None:
        print("could not find rank/ligand/receptor columns, inspect res.columns manually")
        return

    sig = res[res[rank_col] < 0.05]
    nlgn1 = sig[sig[ligand_col].str.contains("NLGN1", case=False, na=False) |
                sig[receptor_col].str.contains("NRXN", case=False, na=False)]
    tnf = sig[(sig[ligand_col] == "TNF") | sig[receptor_col].str.contains("TNFR", case=False, na=False)]

    total = max(len(sig), 1)
    summary = pd.DataFrame([{
        "Cohort": args.cohort, "total_significant": len(sig),
        "NLGN1_pairs": len(nlgn1), "NLGN1_pct": len(nlgn1) / total * 100,
        "TNF_pairs": len(tnf), "TNF_pct": len(tnf) / total * 100,
    }])
    summary.to_csv(outdir / f"{args.cohort}_liana_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
