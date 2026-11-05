#!/usr/bin/env python3
"""
stouffer_from_counts.py : recompute the seven-gene Stouffer meta-analysis from DONOR-LEVEL pseudobulk counts.

    python stouffer_from_counts.py                  # reads data_snrna/*.h5ad (the three cohort objects), writes results/stouffer_from_counts/

WHY THIS EXISTS
  The per-cohort Z values behind Figure 5A, Figure 8A, 'CRYAB Z = 5.502' and 'all seven genes above 1.96' came from a saved table whose provenance is unknown: in 17 of its 19 rows the Z cannot belong to the p-value stored beside it
  (for example CRYAB in Jin: Z = 9.02 beside p = 0.0057). This script rebuilds them from the raw counts, the way the Methods describe: counts summed per donor, genes with fewer than 10 counts dropped, PyDESeq2 with
  disease versus control, and the Wald statistic of each gene as that cohort's Z. Combined Z = sum(sqrt(donors_c) * Z_c) / sqrt(sum donors_c) over the cohorts in which the gene is detected.

OUTPUT (results/stouffer_from_counts/)
  Meta_Analysis_Results_RECOMPUTED.csv   same columns as Meta_Analysis_Results_DONOR_LEVEL.csv, so the figure scripts can read it (set FIG_META_FILE)
  stouffer_comparison.csv                recomputed against the combined Z-scores in the manuscript, gene by gene
EXIT STATUS  0 if every recomputed combined Z equals the manuscript value to within 0.01. 1 if not: the numbers in the manuscript (Results, Fig. 5A, Fig. 8A, Abstract) must then be updated from the recomputed table.
"""

import os, runpy, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
GENES = ["CRYAB", "HSPB1", "DNAJB1", "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1"]
LOCKED = {
    "CRYAB": 3.436,
    "HSPB1": 3.371,
    "DNAJB1": 3.055,
    "HSPA1A": 5.260,
    "HSPA1B": 4.037,
    "HSP90AA1": 1.398,
    "HSP90AB1": 0.814,
}  # combined Z in the manuscript
COHORTS = ("Jin", "Nido", "Ma")
MIN_COUNTS = 10
TOL = 0.01
OUT = Path(os.environ.get("LBD_STOUFFER_OUT", "results/stouffer_from_counts"))


def pseudobulk(raw, donors, genes):
    """Raw counts summed per donor: (donors x genes) integer DataFrame. raw is nuclei x genes (sparse or dense)."""
    raw = raw.tocsr() if sps.issparse(raw) else sps.csr_matrix(raw)
    codes, uniq = pd.factorize(pd.Series(donors).astype(str))
    G = sps.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes))
    )
    return pd.DataFrame(
        np.asarray((G @ raw).todense()).round().astype(int), index=list(uniq), columns=list(genes)
    )


def deseq_wald(counts, condition):
    """PyDESeq2, disease versus control. Returns a DataFrame indexed by gene with stat (Wald Z), log2FoldChange and pvalue."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as e:
        raise SystemExit(
            "PyDESeq2 is not installed. Run:  pip install pydeseq2   (or: uv pip install pydeseq2), restart the kernel, and run again."
        ) from e
    counts = counts.loc[:, counts.sum(axis=0) >= MIN_COUNTS]
    meta = pd.DataFrame(
        {
            "condition": pd.Categorical(
                condition.loc[counts.index].values, categories=["Control", "PD"]
            )
        },
        index=counts.index,
    )
    try:
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~condition", quiet=True)
    except TypeError:
        dds = DeseqDataSet(counts=counts, metadata=meta, design_factors="condition", quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", "PD", "Control"], quiet=True)
    st.summary()
    return st.results_df


def combine(per_cohort, n_donors, genes=GENES):
    """per_cohort: {cohort: DataFrame(stat, log2FoldChange, pvalue)}. Returns the table in the stored schema."""
    rows = []
    for g in genes:
        row = {"Gene": g}
        z, w = [], []
        for c in COHORTS:
            r = per_cohort.get(c)
            ok = r is not None and g in r.index and pd.notna(r.loc[g, "stat"])
            row[f"{c}_Z"] = float(r.loc[g, "stat"]) if ok else np.nan
            row[f"{c}_log2FC"] = float(r.loc[g, "log2FoldChange"]) if ok else np.nan
            row[f"{c}_p"] = float(r.loc[g, "pvalue"]) if ok else np.nan
            if ok:
                z.append(row[f"{c}_Z"])
                w.append(np.sqrt(n_donors[c]))
        zc = (
            float(np.sum(np.array(z) * np.array(w)) / np.sqrt(np.sum(np.array(w) ** 2)))
            if z
            else np.nan
        )
        row.update(
            Combined_Z=zc,
            Combined_p=float(2 * stats.norm.sf(abs(zc))) if z else np.nan,
            N_cohorts=len(z),
            Same_direction=bool(len(z) and (np.all(np.array(z) > 0) or np.all(np.array(z) < 0))),
        )
        rows.append(row)
    cols = (
        ["Gene"]
        + [f"{c}_{k}" for c in COHORTS for k in ("Z", "log2FC", "p")]
        + ["Combined_Z", "Combined_p", "N_cohorts", "Same_direction"]
    )
    return pd.DataFrame(rows)[cols]


def compare(table, locked=LOCKED):
    t = table.set_index("Gene")
    rows = []
    for g, want in locked.items():
        got = float(t.loc[g, "Combined_Z"]) if g in t.index else np.nan
        rows.append(
            dict(
                gene=g,
                manuscript=want,
                recomputed=round(got, 3),
                difference=round(got - want, 3) if np.isfinite(got) else np.nan,
                agrees=bool(np.isfinite(got) and abs(got - want) <= TOL),
                above_1_96=bool(np.isfinite(got) and abs(got) > 1.96),
            )
        )
    return pd.DataFrame(rows)


def run_cohorts(data):
    """data: {cohort: (counts DataFrame donors x genes, condition Series indexed by donor)}. Returns (table, per_cohort, n_donors)."""
    per, n = {}, {}
    for c, (counts, cond) in data.items():
        per[c] = deseq_wald(counts, cond)
        n[c] = len(counts)
        print(
            f"  {c}: {len(counts)} donors ({int((cond.loc[counts.index] == 'PD').sum())} disease, {int((cond.loc[counts.index] == 'Control').sum())} control), {per[c].shape[0]:,} genes tested"
        )
    return combine(per, n), per, n


def load_real():
    import anndata

    ns = runpy.run_path(str(HERE / "make_figure4_cohens_d.py"), run_name="not_main")
    data = {}
    for c in COHORTS:
        path = ns["resolve_input_path"](ns["DATA_DIR"] / ns["FILES"][c], f"{c} cohort", False)
        print(f"reading {c}: {path}")
        adata = anndata.read_h5ad(path)
        diag, donor, dx = ns["resolve_diag_and_donor"](adata, c)
        raw, src = ns["find_raw_counts"](adata)
        print(
            f"  {c}: {adata.n_obs:,} nuclei | diagnosis '{diag}' | donor '{donor}' | raw counts from {src}"
        )
        counts = pseudobulk(raw, adata.obs[donor].astype(str).values, adata.var_names)
        cond = (
            pd.Series(np.asarray(dx), index=adata.obs[donor].astype(str).values)
            .groupby(level=0)
            .first()
        )
        data[c] = (counts, cond.replace({"Control": "Control", "PD": "PD"}))
    return data


def main(argv=None):
    import importlib.metadata as _im

    try:
        print(f"PyDESeq2 version used: {_im.version('pydeseq2')}  (record it in the Methods)")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    table, per, n = run_cohorts(load_real())
    cmp_ = compare(table)
    table.to_csv(OUT / "Meta_Analysis_Results_RECOMPUTED.csv", index=False)
    cmp_.to_csv(OUT / "stouffer_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nCOMBINED Z: manuscript against the recomputation from donor-level counts")
    print(cmp_.to_string(index=False))
    print("\nPER-COHORT Z (donor level):")
    print(table[["Gene", "Jin_Z", "Ma_Z", "Nido_Z", "Combined_Z"]].round(3).to_string(index=False))
    if bool(cmp_.agrees.all()):
        print("\nREPRODUCED: every combined Z agrees with the manuscript to within 0.01.")
        return 0
    print(
        "\nNOT REPRODUCED. The Stouffer numbers in the manuscript do not come from donor-level counts.\n"
        f"  {int((~cmp_.agrees).sum())} of {len(cmp_)} genes differ; {int((~cmp_.above_1_96).sum())} would not exceed Z = 1.96.\n"
        f"  Use {OUT / 'Meta_Analysis_Results_RECOMPUTED.csv'} (set FIG_META_FILE to it to redraw Fig. 5A and 8A) and update the Results, the Abstract and the Discussion from it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
