"""
Shared helpers for the LBD proteostasis pipeline.

Column names differ slightly between the three cohorts (Jin, Ma, Nido)
depending on which annotation pass produced each h5ad, so most of this
is defensive lookups rather than hardcoded strings. If a lookup fails,
check adata.obs.columns directly before assuming the data is wrong.
"""

import numpy as np
import scipy.sparse as sps
import scanpy as sc


CELL_TYPE_CANDIDATES = ["broad_type", "cell_type_ct", "cell_type", "majority_voting", "celltype"]
DX_CANDIDATES = ["diagnosis", "condition", "disease", "group"]
SAMPLE_CANDIDATES = ["sample_id", "sample", "donor", "subject"]

DISEASE_KEYWORDS = ["LBD", "PD", "PARK", "CASE", "DISEASE", "AFFECTED"]
CONTROL_KEYWORDS = ["CTRL", "CONTROL", "HEALTHY", "NORMAL", "UNAFFECTED"]


def find_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for cl, orig in cols_lower.items():
            if cand.lower() in cl:
                return orig
    return None


def resolve_columns(adata):
    """Best-guess cell type / diagnosis / sample columns for a cohort.
    Always print these and eyeball them before trusting a run - the
    same h5ad has shown up with different column names across annotation
    passes more than once in this project."""
    ct = find_column(adata.obs.columns, CELL_TYPE_CANDIDATES)
    dx = find_column(adata.obs.columns, DX_CANDIDATES)
    sample = find_column(adata.obs.columns, SAMPLE_CANDIDATES)
    print(f"  columns resolved: cell_type={ct!r}  diagnosis={dx!r}  sample={sample!r}")
    return ct, dx, sample


def standardise_diagnosis(adata, dx_col):
    """Map whatever label scheme a cohort uses onto Disease / Control."""
    mapping = {}
    for val in adata.obs[dx_col].unique():
        v = str(val).upper()
        if any(k in v for k in DISEASE_KEYWORDS):
            mapping[val] = "Disease"
        elif any(k in v for k in CONTROL_KEYWORDS):
            mapping[val] = "Control"
        else:
            mapping[val] = val
    adata.obs["Dx"] = adata.obs[dx_col].map(mapping)
    unmapped = adata.obs["Dx"][~adata.obs["Dx"].isin(["Disease", "Control"])].unique()
    if len(unmapped):
        print(f"  warning: unmapped diagnosis values {list(unmapped)} - check DISEASE/CONTROL keywords")
    return adata


def ensure_lognorm(adata):
    """Normalise in place if X still looks like raw counts."""
    X = adata.X
    xmax = X.data.max() if sps.issparse(X) else float(np.asarray(X).max())
    if xmax <= 20:
        return
    print(f"  normalising (X max was {xmax:.0f})")
    if sps.issparse(X):
        row_sums = np.asarray(X.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        adata.X = sps.diags(1e4 / row_sums) @ X
    else:
        row_sums = np.asarray(X).sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adata.X = (np.asarray(X) / row_sums) * 1e4
    sc.pp.log1p(adata)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    pooled_sd = np.sqrt(pooled_var)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0


def sample_level_scores(adata, score_col, sample_col):
    """Aggregate a per-cell score to one value per donor before running
    any statistics on it. Computing Cohen's d straight from per-cell
    scores badly inflates n and gives a wrong effect size - donors, not
    cells, are the unit of replication here."""
    agg = (adata.obs.groupby(sample_col, observed=True)
           .agg({score_col: "mean", "Dx": "first"})
           .reset_index())
    return agg
