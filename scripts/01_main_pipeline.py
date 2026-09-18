"""
Gouda et al. 2026 - revision analysis, checkpointed and resumable.

Runs the same six analyses used for the original submission (pseudobulk
DE, Hallmarks GSEA, TNF/TNFR pathway check, Augur cell-type ranking,
per-cell-type UPR enrichment, per-cell-type TF activity) against the
three cohort h5ad files, saving every intermediate result to a local
checkpoint directory as it finishes. If the script is interrupted, a
rerun skips anything already checkpointed rather than starting over -
this matters mainly for TF activity, since fitting hundreds of regulon
models per cell type across all three cohorts is slow enough that
losing partial progress to a crash or timeout is expensive.

TF activity in particular runs per cell type rather than on a whole
cohort's matrix in one call, since fitting several hundred regulon
models against ~280,000 cells at once is a large enough memory spike to
be worth avoiding on a modest machine.

Usage:
    python 01_main_pipeline.py \\
        --jin path/to/jin.h5ad --ma path/to/ma.h5ad --nido path/to/nido.h5ad \\
        --checkpoint-dir results/revision_2026
"""

import gc
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROTEOSTASIS_GENES = [
    "HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "DNAJB4", "HSP90AA1", "HSP90AB1",
    "DNAJA1", "DNAJA4", "CHORDC1", "STIP1", "CRYAB", "BAG3", "HSPH1",
]
TNF_TNFR_GENES = ["TNF", "TNFRSF1A", "TNFRSF1B"]
CELL_TYPE_CANDIDATES = ["broad_type", "cell_type_ct", "cell_type", "celltype", "annotation"]
DX_CANDIDATES = ["diagnosis", "condition", "disease", "dx", "group"]
SAMPLE_CANDIDATES = ["sample_id", "sample", "donor", "subject", "biobank_id"]

OUT_DIR = None  # set from --checkpoint-dir in __main__


# =====================================================================
# CHECKPOINTING - one function to check/load/save any unit of work
# =====================================================================
def checkpoint_load(filename):
    """Returns a DataFrame if this exact output was already computed and
    saved to the checkpoint directory, otherwise None."""
    local_path = OUT_DIR / filename
    if local_path.exists():
        return pd.read_csv(local_path)
    return None


def checkpoint_save(df, filename):
    local_path = OUT_DIR / filename
    df.to_csv(local_path, index=False)
    return local_path


def find_col(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for cl, orig in cols_lower.items():
            if cand.lower() in cl:
                return orig
    return None


def validate_labels(adata, cfg, cohort_name):
    """Checks disease_label/control_label actually exist in the data
    before any DESeq2 time is spent. Cohorts have been annotated with
    different capitalization conventions before (Ma uses lowercase
    'control', Jin uses 'Control') - this catches that mismatch in
    under a second instead of after DESeq2 has already fit several
    cell types."""
    actual = set(adata.obs[cfg["dx_col"]].dropna().unique())
    expected = {cfg["disease_label"], cfg["control_label"]}
    missing = expected - actual
    if missing:
        raise ValueError(
            f"{cohort_name}: COHORT_CONFIG expects {expected} in column "
            f"'{cfg['dx_col']}', but the data actually contains {actual}. "
            f"Fix disease_label/control_label in COHORT_CONFIG to match "
            f"exactly (check capitalization) before running anything."
        )
    print(f"  {cohort_name}: label check passed - {expected} both found in '{cfg['dx_col']}'")


def diagnose(name, path):
    print(f"\n{'='*70}\nDIAGNOSING: {name}  ({path})\n{'='*70}")
    if not path.exists():
        print("  FILE NOT FOUND")
        return None
    adata = sc.read_h5ad(path)
    print(f"  shape: {adata.shape[0]} nuclei x {adata.shape[1]} genes")
    ct_col = find_col(adata.obs.columns, CELL_TYPE_CANDIDATES)
    dx_col = find_col(adata.obs.columns, DX_CANDIDATES)
    samp_col = find_col(adata.obs.columns, SAMPLE_CANDIDATES)
    print(f"  auto-detected -> cell_type_col={ct_col!r}, dx_col={dx_col!r}, sample_col={samp_col!r}")
    return adata


# =====================================================================
# 1. PSEUDOBULK - checkpointed per cell type
# =====================================================================
def pseudobulk_one_celltype(adata, ct, cell_type_col, sample_col, layer=None):
    mask_ct = (adata.obs[cell_type_col] == ct).values
    sub = adata[mask_ct]
    donors = sorted(sub.obs[sample_col].unique())
    counts = np.zeros((sub.shape[1], len(donors)), dtype=np.float64)
    for j, d in enumerate(donors):
        m = (sub.obs[sample_col] == d).values
        Xd = sub.layers[layer][m] if layer else sub.X[m]
        summed = np.asarray(Xd.sum(axis=0)).flatten() if hasattr(Xd, "sum") else Xd.sum(axis=0)
        counts[:, j] = summed
    counts_df = pd.DataFrame(counts, index=sub.var_names, columns=donors).round().astype(int)
    return counts_df[counts_df.sum(axis=1) >= 10]


def run_pydeseq2_one_celltype(counts_df, donor_dx_map, disease_label, control_label):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    donors = counts_df.columns.tolist()
    dx = [donor_dx_map[d] for d in donors]
    if len(set(dx)) < 2:
        return None
    clinical = pd.DataFrame({"condition": dx}, index=donors)
    dds = DeseqDataSet(counts=counts_df.T, metadata=clinical, design="~condition", refit_cooks=True)
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=["condition", disease_label, control_label])
    stat_res.summary()
    res = stat_res.results_df.reset_index().rename(columns={"index": "gene"})
    res["Z_score"] = res["stat"]
    return res


def get_de_for_cohort(adata, cfg, cohort_name):
    """Checkpointed per cell type. Returns {cell_type: DE dataframe},
    pulling each cell type from Drive if it's already there instead of
    re-running DESeq2 on it."""
    ct_col, sample_col, dx_col = cfg["cell_type_col"], cfg["sample_col"], cfg["dx_col"]
    donor_dx_map = adata.obs.drop_duplicates(sample_col).set_index(sample_col)[dx_col].to_dict()
    cell_types = sorted(adata.obs[ct_col].dropna().unique())

    de_tables = {}
    for ct in cell_types:
        clean_ct = str(ct).replace(" ", "_").replace(".", "").replace("/", "_")
        fname = f"{cohort_name}_DE_{clean_ct}.csv"

        cached = checkpoint_load(fname)
        if cached is not None:
            de_tables[ct] = cached
            print(f"    {ct}: loaded from checkpoint ({(cached['padj'] < 0.05).sum()} genes FDR<0.05)")
            continue

        print(f"    {ct}: running DESeq2...")
        counts_df = pseudobulk_one_celltype(adata, ct, ct_col, sample_col)
        if counts_df.shape[1] < 4:
            print(f"    {ct}: too few donors, skipping")
            continue
        res = run_pydeseq2_one_celltype(counts_df, donor_dx_map, cfg["disease_label"], cfg["control_label"])
        if res is None:
            continue
        res["Cell_Type"] = ct
        res["Cohort"] = cohort_name
        checkpoint_save(res, fname)
        de_tables[ct] = res
        print(f"    {ct}: DE complete + checkpointed, {(res['padj'] < 0.05).sum()} genes FDR<0.05")
    return de_tables


# =====================================================================
# 2. GSEA HALLMARKS - checkpointed at cohort level
# =====================================================================
def build_cohort_ranked_list(de_tables):
    all_stats = pd.concat([df.set_index("gene")["stat"] for df in de_tables.values()], axis=1)
    all_stats.columns = list(de_tables.keys())
    return all_stats.mean(axis=1, skipna=True).dropna().sort_values(ascending=False)


def run_gsea_hallmarks(ranked_series, cohort_name):
    fname = f"{cohort_name}_Hallmarks_GSEA_results.csv"
    cached = checkpoint_load(fname)
    if cached is not None:
        print(f"    Hallmarks GSEA loaded from checkpoint")
        return cached

    import gseapy as gp
    rnk = ranked_series.reset_index()
    rnk.columns = ["gene_name", "score"]
    pre_res = gp.prerank(rnk=rnk, gene_sets="MSigDB_Hallmark_2020",
                          min_size=5, max_size=1000, permutation_num=1000, seed=42, outdir=None)
    res_df = pre_res.res2d.copy()
    res_df["FDR q-val"] = res_df["FDR q-val"].astype(float)
    res_df = res_df.sort_values("NES", ascending=False)

    n_fdr25 = (res_df["FDR q-val"] < 0.25).sum()
    n_fdr05 = (res_df["FDR q-val"] < 0.05).sum()
    print(f"    {cohort_name}: pathways at FDR<0.25: {n_fdr25}/{len(res_df)}, at FDR<0.05: {n_fdr05}/{len(res_df)}")

    checkpoint_save(res_df, fname)
    return res_df


# =====================================================================
# 3. TNF/TNFR - cheap, no separate checkpoint needed, derives from DE tables
# =====================================================================
def check_tnf_tnfr_genes(de_tables, cohort_name):
    rows = []
    for ct, df in de_tables.items():
        sub = df[df["gene"].isin(TNF_TNFR_GENES)]
        for _, r in sub.iterrows():
            rows.append({"Cell_Type": ct, "gene": r["gene"],
                         "log2FoldChange": r["log2FoldChange"], "padj": r["padj"]})
    return pd.DataFrame(rows)


# =====================================================================
# 4. AUGUR - checkpointed at cohort level (raw counts, per pertpy's docs)
# =====================================================================
def run_augur(adata, cfg, cohort_name):
    fname = f"{cohort_name}_augur_results.csv"
    cached = checkpoint_load(fname)
    if cached is not None:
        print(f"    Augur loaded from checkpoint")
        return cached

    import pertpy as pt
    if "counts" not in adata.layers:
        raise ValueError(f"{cohort_name}: no 'counts' layer, cannot run Augur on raw data")
    adata = adata.copy()
    adata.X = adata.layers["counts"]

    ag = pt.tl.Augur("random_forest_classifier")
    loaded = ag.load(adata, cell_type_col=cfg["cell_type_col"], label_col=cfg["dx_col"],
                      condition_label=cfg["control_label"], treatment_label=cfg["disease_label"])
    _, results = ag.predict(loaded, subsample_size=20, n_threads=4, select_variance_features=True)

    summary = results["summary_metrics"]
    auc_series = summary.loc["mean_augur_score"] if "mean_augur_score" in summary.index else summary["mean_augur_score"]
    auc_df = auc_series.reset_index()
    auc_df.columns = ["Cell_Type", "AUC"]
    auc_df["Cohort"] = cohort_name
    auc_df = auc_df.sort_values("AUC", ascending=False)

    checkpoint_save(auc_df, fname)
    return auc_df


# =====================================================================
# 5. UPR PER CELL TYPE - checkpointed per cell type
# =====================================================================
def run_upr_by_celltype(de_tables, cohort_name):
    import gseapy as gp
    rows = []
    for ct, df in de_tables.items():
        clean_ct = str(ct).replace(" ", "_").replace(".", "").replace("/", "_")
        fname = f"{cohort_name}_UPR_{clean_ct}.csv"

        cached = checkpoint_load(fname)
        if cached is not None:
            rows.append(cached.iloc[0].to_dict())
            print(f"    {ct}: UPR loaded from checkpoint")
            continue

        rnk = df[["gene", "stat"]].dropna().sort_values("stat", ascending=False)
        rnk.columns = ["gene_name", "score"]
        try:
            pre_res = gp.prerank(rnk=rnk, gene_sets="MSigDB_Hallmark_2020",
                                  min_size=5, max_size=1000, permutation_num=1000, seed=42, outdir=None)
            res_df = pre_res.res2d
            upr_row = res_df[res_df["Term"].str.contains("Unfolded Protein Response", case=False, na=False)]
            if upr_row.empty:
                continue
            r = upr_row.iloc[0]
            row = {"Cell_Type": ct, "NES": float(r["NES"]), "FDR": float(r["FDR q-val"]),
                   "Significant_FDR05": float(r["FDR q-val"]) < 0.05}
            checkpoint_save(pd.DataFrame([row]), fname)
            rows.append(row)
            print(f"    {ct}: UPR NES={row['NES']:.3f}, FDR={row['FDR']:.4f} - checkpointed")
        except Exception as e:
            print(f"    {ct}: GSEA failed ({e})")
    return pd.DataFrame(rows)


# =====================================================================
# 6. TF ACTIVITY - now PER CELL TYPE, checkpointed per cell type.
#    This is the fix for the crash: each call to decoupler now runs on
#    one cell type's nuclei, not all 277,831 at once.
# =====================================================================
def run_tf_activity_by_celltype(adata, cfg, cohort_name,
                                  tfs_of_interest=("ATF4", "HSF1", "NFKB1")):
    import decoupler as dc

    ct_col, dx_col = cfg["cell_type_col"], cfg["dx_col"]
    cell_types = sorted(adata.obs[ct_col].dropna().unique())

    all_rows = []
    for ct in cell_types:
        clean_ct = str(ct).replace(" ", "_").replace(".", "").replace("/", "_")
        fname = f"{cohort_name}_TF_{clean_ct}.csv"

        cached = checkpoint_load(fname)
        if cached is not None:
            all_rows.append(cached)
            print(f"    {ct}: TF activity loaded from checkpoint")
            continue

        print(f"    {ct}: running decoupler ULM on this cell type only "
              f"({(adata.obs[ct_col] == ct).sum()} nuclei)...")
        sub = adata[(adata.obs[ct_col] == ct).values].copy()

        net = dc.op.collectri(organism="human", remove_complexes=False)
        dc.mt.ulm(data=sub, net=net, verbose=False)
        acts = dc.pp.get_obsm(adata=sub, key="score_ulm")

        rows = []
        for tf in tfs_of_interest:
            if tf not in acts.var_names:
                continue
            tf_vals = np.asarray(acts[:, tf].X).flatten()
            dx = sub.obs[dx_col].values
            disease_vals = tf_vals[dx == cfg["disease_label"]]
            control_vals = tf_vals[dx == cfg["control_label"]]
            if len(disease_vals) < 3 or len(control_vals) < 3:
                continue
            d_mean, c_mean = disease_vals.mean(), control_vals.mean()
            try:
                _, p = stats.mannwhitneyu(disease_vals, control_vals, alternative="two-sided")
            except ValueError:
                p = np.nan
            rows.append({"Cohort": cohort_name, "Cell_Type": ct, "TF": tf,
                         "disease_mean": d_mean, "control_mean": c_mean,
                         "diff": d_mean - c_mean, "p_value": p})

        ct_result = pd.DataFrame(rows)
        if len(ct_result):
            checkpoint_save(ct_result, fname)
            all_rows.append(ct_result)
            print(f"    {ct}: checkpointed")

        del sub, acts
        gc.collect()

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# =====================================================================
# 7. SEQUENCING DEPTH - cheap, checkpointed at cohort level
# =====================================================================
def report_sequencing_depth(adata, cohort_name):
    fname = f"{cohort_name}_seq_depth.csv"
    cached = checkpoint_load(fname)
    if cached is not None:
        return cached.iloc[0].to_dict()

    umi_col = find_col(adata.obs.columns, ["total_counts", "n_counts", "nCount"])
    gene_col = find_col(adata.obs.columns, ["n_genes_by_counts", "n_genes", "nFeature"])
    if umi_col is None or gene_col is None:
        sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
        umi_col, gene_col = "total_counts", "n_genes_by_counts"

    summary = {
        "Cohort": cohort_name,
        "median_UMIs_per_nucleus": adata.obs[umi_col].median(),
        "median_genes_per_nucleus": adata.obs[gene_col].median(),
        "n_nuclei": adata.n_obs,
    }
    checkpoint_save(pd.DataFrame([summary]), fname)
    return summary


# =====================================================================
# MAIN - safe to re-run after any interruption; anything already
# checkpointed is skipped rather than recomputed.
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jin", type=Path, required=True, help="Path to the Jin cohort h5ad")
    parser.add_argument("--ma", type=Path, required=True, help="Path to the Ma cohort h5ad")
    parser.add_argument("--nido", type=Path, required=True, help="Path to the Nido cohort h5ad")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/revision_2026"),
                         help="Where intermediate results are saved and resumed from")
    args = parser.parse_args()

    OUT_DIR = args.checkpoint_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FILES = {"Jin": args.jin, "Ma": args.ma, "Nido": args.nido}

    COHORT_CONFIG = {
        "Jin":  dict(cell_type_col="broad_type", dx_col="condition", sample_col="sample",
                     disease_label="LBD", control_label="Control"),
        "Ma":   dict(cell_type_col="cell_type", dx_col="condition", sample_col="sample",
                     disease_label="PD", control_label="control"),   # lowercase - matches the actual data
        "Nido": dict(cell_type_col="cell_type", dx_col="Diagnosis", sample_col="sample_id",
                     disease_label="PD", control_label="Control"),
    }

    all_seq_depth = []
    for cohort in ["Nido", "Jin", "Ma"]:
        if not FILES[cohort].exists():
            print(f"\nSKIPPING {cohort} - h5ad not found at {FILES[cohort]}")
            continue
        cfg = COHORT_CONFIG[cohort]

        print(f"\n{'#'*70}\n#  {cohort}\n{'#'*70}")
        adata = sc.read_h5ad(FILES[cohort])
        print(f"  loaded {adata.n_obs:,} nuclei")
        validate_labels(adata, cfg, cohort)

        all_seq_depth.append(report_sequencing_depth(adata, cohort))

        print("\n  [DE per cell type - checkpointed]")
        de_tables = get_de_for_cohort(adata, cfg, cohort)

        print("\n  [Hallmarks GSEA]")
        ranked = build_cohort_ranked_list(de_tables)
        run_gsea_hallmarks(ranked, cohort)

        print("\n  [TNF/TNFR check]")
        tnf_result = check_tnf_tnfr_genes(de_tables, cohort)
        tnf_result.to_csv(OUT_DIR / f"{cohort}_TNF_TNFR.csv", index=False)

        print("\n  [UPR per cell type - checkpointed]")
        run_upr_by_celltype(de_tables, cohort)

        print("\n  [Augur]")
        run_augur(adata, cfg, cohort)

        print("\n  [TF activity per cell type - checkpointed, this is the slow step]")
        run_tf_activity_by_celltype(adata, cfg, cohort)

        del adata
        gc.collect()

    seq_df = pd.DataFrame(all_seq_depth)
    checkpoint_save(seq_df, "all_cohorts_seq_depth.csv")
    print(f"\n\nAll done. Checkpoints saved under {OUT_DIR} - safe to re-run this script anytime.")
