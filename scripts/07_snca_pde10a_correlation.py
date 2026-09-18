#!/usr/bin/env python3
"""
Cell-type-resolved SNCA/PDE10A correlation, computed on raw (not
log-normalised) expression for this specific comparison, since PDE10A
and SNCA are both moderate-to-high-expression genes for which Spearman
rank correlation on raw counts and on log-normalised values give
materially different results depending on cohort composition and
sequencing depth. This is a deliberate, cohort-appropriate choice for
this correlation, distinct from the log-normalised scoring used
elsewhere in this project's module analyses.

Usage:
    python 07_snca_pde10a_correlation.py \\
        --nido path/to/nido.h5ad --ma path/to/ma.h5ad --jin path/to/jin.h5ad \\
        --outdir results/snca_pde10a
"""
import argparse
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

COHORT_COLS = {
    'Nido_Prefrontal': ('cell_type', 'Diagnosis'),
    'Ma_Cingulate':    ('cell_type', 'condition'),
    'Jin_Temporal':    ('broad_type', 'condition'),
}


def get_expr(adata, gene):
    if gene not in adata.var_names:
        return None
    idx = list(adata.var_names).index(gene)
    X = adata.X
    if sp.issparse(X):
        return np.array(X[:, idx].todense()).flatten()
    return X[:, idx].flatten()


def compute_correlations(adata, cohort, ct_col, min_cells=50):
    print(f"\n{'='*60}  {cohort}")
    print(f"  X dtype={adata.X.dtype}, X max={adata.X.max():.2f} "
          f"({'RAW COUNTS' if adata.X.max() > 20 else 'already log-normalised'})")

    # Exactly as in the original: no normalization call here.
    if 'Proteostasis_score' in adata.obs.columns:
        adata.obs['prot_score'] = adata.obs['Proteostasis_score']
    else:
        avail = [g for g in PROTEOSTASIS_14 if g in adata.var_names]
        sc.tl.score_genes(adata, avail, score_name='prot_score', use_raw=False)

    missing = [g for g in ['SNCA', 'PDE10A'] if g not in adata.var_names]
    if missing:
        print(f"  WARNING missing genes: {missing}")

    results = []
    for ct in sorted(adata.obs[ct_col].unique()):
        if str(ct).lower() in ('unassigned', 'unknown', 'nan'):
            continue
        sub = adata[adata.obs[ct_col] == ct]
        if sub.n_obs < min_cells:
            continue

        snca = get_expr(sub, 'SNCA')
        pde = get_expr(sub, 'PDE10A')
        prot = sub.obs['prot_score'].values
        if snca is None or pde is None:
            continue

        rho_sp, p_sp = spearmanr(snca, pde)
        rho_spr, p_spr = spearmanr(snca, prot)
        rho_pp, p_pp = spearmanr(pde, prot)

        results.append({
            'cohort': cohort, 'cell_type': str(ct), 'n_cells': sub.n_obs,
            'SNCA_x_PDE10A_rho': rho_sp, 'SNCA_x_PDE10A_p': p_sp,
            'SNCA_x_Proteostasis_rho': rho_spr, 'SNCA_x_Proteostasis_p': p_spr,
            'PDE10A_x_Proteostasis_rho': rho_pp, 'PDE10A_x_Proteostasis_p': p_pp,
        })
        print(f"  {str(ct):35s} n={sub.n_obs:6d}  "
              f"SNCA x PDE10A={rho_sp:+.3f} (p={p_sp:.4f})")

    df = pd.DataFrame(results)
    if len(df) > 1:
        _, df['SNCA_x_PDE10A_padj'], _, _ = multipletests(df['SNCA_x_PDE10A_p'], method='fdr_bh')
        _, df['SNCA_x_Proteostasis_padj'], _, _ = multipletests(df['SNCA_x_Proteostasis_p'], method='fdr_bh')
        _, df['PDE10A_x_Proteostasis_padj'], _, _ = multipletests(df['PDE10A_x_Proteostasis_p'], method='fdr_bh')
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nido", type=Path, required=True, help="Path to the Nido cohort h5ad")
    parser.add_argument("--ma", type=Path, required=True, help="Path to the Ma cohort h5ad")
    parser.add_argument("--jin", type=Path, required=True, help="Path to the Jin cohort h5ad")
    parser.add_argument("--outdir", type=Path, default=Path("results/comment4_5_repro"))
    args = parser.parse_args()
    OUTDIR = args.outdir
    OUTDIR.mkdir(parents=True, exist_ok=True)

    FILES = {
        "Nido_Prefrontal": args.nido,
        "Ma_Cingulate": args.ma,
        "Jin_Temporal": args.jin,
    }

    dfs = {}
    for cohort, path in FILES.items():
        if not path.exists():
            print(f"  {path} not found, skipping {cohort}")
            continue
        adata = sc.read_h5ad(path)
        ct_col, cond_col = COHORT_COLS[cohort]
        dfs[cohort] = compute_correlations(adata, cohort, ct_col)
        del adata
        import gc; gc.collect()

    df_all = pd.concat(list(dfs.values()), ignore_index=True)
    out_csv = OUTDIR / 'snca_pde10a_correlations_all_cohorts.csv'
    df_all.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("KEY NUMBERS - Ma_Cingulate, unnormalised (raw-count) method:")
    print("=" * 70)
    ma_rows = df_all[df_all["cohort"] == "Ma_Cingulate"]
    for _, row in ma_rows.iterrows():
        print(f"  {row['cell_type']:35s}  rho={row['SNCA_x_PDE10A_rho']:+.3f}  "
              f"p={row['SNCA_x_PDE10A_p']:.4f}")

    print(f"\nResults saved to {out_csv}.")
