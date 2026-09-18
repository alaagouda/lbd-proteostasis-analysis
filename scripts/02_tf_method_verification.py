"""
DOMAIN 1 - Independent investigation: does a genuine bulk-level decoupleR/
CollecTRI run reproduce the published Fig. 5B/C numbers, or does it reveal
they came from a different (custom z-score) method entirely?

This does NOT depend on any other domain in this investigation. Run it
standalone.

Root cause being tested: Fig. 5 legend claims "decoupleR v1.8.0 using the
CollecTRI database." ma_tf_activity.py (Ma's confirmed source script) uses
a hand-rolled REGULONS dict and z-scores target genes directly - not
decoupleR at all. Nido's source script has never been confirmed. This
script runs the REAL decoupleR method at bulk cohort level (not per cell
type) for both cohorts and reports whether it lands near the published
values or somewhere else entirely.

Expected output: a table with three columns per TF per cohort -
published value, this script's decoupleR bulk value, and the custom
z-score value from ma_tf_activity.py's method (recomputed here for
comparison) - so you can see directly which method the published numbers
actually match.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from scipy import stats
from pathlib import Path
import argparse

CUSTOM_REGULONS = {
    "ATF4":   ["DDIT3", "ATF3", "ASNS", "PSAT1", "PHGDH", "SLC7A11", "HERPUD1"],
    "TP53":   ["CDKN1A", "MDM2", "BBC3", "PMAIP1", "BAX", "GADD45A", "SESN2"],
    "NFE2L2": ["HMOX1", "NQO1", "GSTM1", "TXNRD1", "PRDX1", "TXN", "SRXN1"],
    "XBP1":   ["HSPA5", "DNAJB9", "EDEM1", "PDIA4", "SEC61A1", "DERL1"],
    "ATF6":   ["HSPA5", "DDIT3", "HERPUD1", "PDIA4", "DNAJB11"],
    "HSF1":   ["HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "CRYAB", "HSP90AA1", "HSPA6",
               "HSPD1", "HSPE1", "DNAJB6", "HSPA2", "HSPB8", "HSPA8", "DNAJB2",
               "HSPH1", "BAG3", "STIP1"],
    "STAT3":  ["BCL2", "MYC", "CCND1", "MCL1", "VEGFA", "HIF1A", "TWIST1"],
    "NFKB1":  ["TNF", "IL6", "IL1B", "CXCL10", "CCL2", "PTGS2", "ICAM1", "VCAM1", "BCL2", "BIRC3"],
}


def ensure_lognorm(adata):
    X = adata.X
    xmax = X.data.max() if sps.issparse(X) else float(np.asarray(X).max())
    if xmax > 20:
        if sps.issparse(X):
            rs = np.asarray(X.sum(axis=1)).flatten(); rs[rs == 0] = 1
            adata.X = sps.diags(1e4 / rs) @ X
        else:
            rs = np.asarray(X).sum(axis=1, keepdims=True); rs[rs == 0] = 1
            adata.X = (np.asarray(X) / rs) * 1e4
        sc.pp.log1p(adata)


def pseudobulk_by_donor(adata, sample_col):
    X = adata.X.toarray() if sps.issparse(adata.X) else np.asarray(adata.X)
    df = pd.DataFrame(X, columns=adata.var_names)
    df[sample_col] = adata.obs[sample_col].values
    return df.groupby(sample_col).mean(numeric_only=True)


def custom_zscore_activity(pb, dx_by_sample, disease_label, control_label):
    """Reproduces ma_tf_activity.py's exact method for direct comparison."""
    dis_samples = dx_by_sample[dx_by_sample == disease_label].index
    ctrl_samples = dx_by_sample[dx_by_sample == control_label].index
    rows = []
    for tf, targets in CUSTOM_REGULONS.items():
        avail = [g for g in targets if g in pb.columns]
        if len(avail) < 3:
            continue
        z = (pb[avail] - pb[avail].mean()) / (pb[avail].std() + 1e-8)
        activity = z.mean(axis=1)
        dis_act = activity.loc[dis_samples].values
        ctrl_act = activity.loc[ctrl_samples].values
        _, p = stats.mannwhitneyu(dis_act, ctrl_act, alternative="two-sided")
        rows.append({"TF": tf, "custom_zscore_mean_diff": dis_act.mean() - ctrl_act.mean(),
                     "custom_zscore_p": p})
    return pd.DataFrame(rows).set_index("TF")


PUBLISHED = {
    "Nido": {"ATF4": (0.576, 0.019), "HSF1": (0.288, 0.246), "NFKB1": (-0.080, 0.604), "TP53": (0.460, 0.007)},
    "Ma":   {"ATF4": (0.859, 0.059), "HSF1": (1.066, 0.008), "NFKB1": (-0.135, 0.673), "TP53": (None, 0.423)},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ma", type=Path, required=True, help="Path to the Ma cohort h5ad")
    parser.add_argument("--nido", type=Path, required=True, help="Path to the Nido cohort h5ad")
    parser.add_argument("--outdir", type=Path, default=Path("results/domain1_tf_bulk"))
    args = parser.parse_args()
    OUT_DIR = args.outdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    COHORT_FILES = {"Ma": args.ma, "Nido": args.nido}
    COHORT_CONFIG = {
        "Ma":   dict(dx_col="condition", sample_col="sample", disease_label="PD", control_label="control"),
        "Nido": dict(dx_col="Diagnosis", sample_col="sample_id", disease_label="PD", control_label="Control"),
    }
    all_results = []
    for cohort, path in COHORT_FILES.items():
        print(f"\n{'='*70}\n{cohort}\n{'='*70}")
        adata = sc.read_h5ad(path)
        ensure_lognorm(adata)
        cfg = COHORT_CONFIG[cohort]

        pb = pseudobulk_by_donor(adata, cfg["sample_col"])
        dx_by_sample = adata.obs.drop_duplicates(cfg["sample_col"]).set_index(cfg["sample_col"])[cfg["dx_col"]]

        # --- reproduce the custom method for direct comparison ---
        custom = custom_zscore_activity(pb, dx_by_sample, cfg["disease_label"], cfg["control_label"])
        print("\nCustom z-score method (ma_tf_activity.py's approach), recomputed here:")
        print(custom)

        # --- run genuine decoupleR bulk ULM ---
        import decoupler as dc
        net = dc.op.collectri(organism="human", remove_complexes=False)

        pb_adata = sc.AnnData(pb.values, obs=pd.DataFrame(index=pb.index), var=pd.DataFrame(index=pb.columns))
        dc.mt.ulm(data=pb_adata, net=net, verbose=True)
        acts = dc.pp.get_obsm(adata=pb_adata, key="score_ulm")

        rows = []
        for tf in CUSTOM_REGULONS:
            if tf not in acts.var_names:
                continue
            vals = pd.Series(np.asarray(acts[:, tf].X).flatten(), index=pb.index)
            dis_vals = vals.loc[dx_by_sample[dx_by_sample == cfg["disease_label"]].index.intersection(vals.index)]
            ctrl_vals = vals.loc[dx_by_sample[dx_by_sample == cfg["control_label"]].index.intersection(vals.index)]
            _, p = stats.mannwhitneyu(dis_vals, ctrl_vals, alternative="two-sided")
            rows.append({"TF": tf, "decoupler_bulk_mean_diff": dis_vals.mean() - ctrl_vals.mean(),
                         "decoupler_bulk_p": p})
        dec_df = pd.DataFrame(rows).set_index("TF")

        combined = custom.join(dec_df, how="outer")
        combined["published_mean"] = [PUBLISHED[cohort].get(tf, (None, None))[0] for tf in combined.index]
        combined["published_p"] = [PUBLISHED[cohort].get(tf, (None, None))[1] for tf in combined.index]
        combined["Cohort"] = cohort

        print(f"\n{cohort} - published vs custom z-score vs genuine decoupleR bulk:")
        print(combined)
        combined.to_csv(OUT_DIR / f"{cohort}_TF_method_comparison.csv")
        all_results.append(combined)

        del adata
        import gc; gc.collect()

    full = pd.concat(all_results)
    full.to_csv(OUT_DIR / "ALL_TF_method_comparison.csv")
    print("\n\nDone. Check which column (custom_zscore vs decoupler_bulk) the published_mean")
    print("column actually matches, per TF per cohort - that tells you definitively which")
    print("method produced the numbers currently in Fig. 5B/C.")
