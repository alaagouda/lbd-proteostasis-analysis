"""
DOMAIN 1b - targeted follow-up. Tests a single new hypothesis: the
published "mean activity" values are a ONE-SAMPLE statistic (mean TF
activity across donors, tested against zero), not a two-group
disease-vs-control difference. Domain 1 tested only the two-group
version and matched neither candidate method to the published numbers.

Three candidate statistics tested per TF, per cohort:
  (a) one-sample: all donors' activity vs 0
  (b) one-sample: disease-group donors only, activity vs 0
  (c) one-sample: control-group donors only, activity vs 0
...each computed under BOTH the custom z-score method and genuine
decoupleR bulk, so 6 candidate numbers per TF per cohort in total.

Takes the same two cohort h5ad files as 02_tf_method_verification.py.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from scipy import stats
from pathlib import Path
import argparse

CUSTOM_REGULONS = {
    "ATF4":  ["DDIT3", "ATF3", "ASNS", "PSAT1", "PHGDH", "SLC7A11", "HERPUD1"],
    "HSF1":  ["HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "CRYAB", "HSP90AA1", "HSPA6",
              "HSPD1", "HSPE1", "DNAJB6", "HSPA2", "HSPB8", "HSPA8", "DNAJB2",
              "HSPH1", "BAG3", "STIP1"],
    "NFKB1": ["TNF", "IL6", "IL1B", "CXCL10", "CCL2", "PTGS2", "ICAM1", "VCAM1", "BCL2", "BIRC3"],
    "TP53":  ["CDKN1A", "MDM2", "BBC3", "PMAIP1", "BAX", "GADD45A", "SESN2"],
}
PUBLISHED = {
    "Nido": {"ATF4": (0.576, 0.019), "HSF1": (0.288, 0.246), "NFKB1": (-0.080, 0.604), "TP53": (0.460, 0.007)},
    "Ma":   {"ATF4": (0.859, 0.059), "HSF1": (1.066, 0.008), "NFKB1": (-0.135, 0.673), "TP53": (None, 0.423)},
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


def one_sample_test(vals, label):
    if len(vals) < 3:
        return None
    try:
        _, p = stats.wilcoxon(vals)
    except ValueError:
        _, p = stats.ttest_1samp(vals, 0)
    return {"subset": label, "mean": float(np.mean(vals)), "n": len(vals), "p_vs_zero": p}


def custom_activity_series(pb, targets):
    avail = [g for g in targets if g in pb.columns]
    if len(avail) < 3:
        return None
    z = (pb[avail] - pb[avail].mean()) / (pb[avail].std() + 1e-8)
    return z.mean(axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ma", type=Path, required=True, help="Path to the Ma cohort h5ad")
    parser.add_argument("--nido", type=Path, required=True, help="Path to the Nido cohort h5ad")
    parser.add_argument("--outdir", type=Path, default=Path("results/domain1b_onesample"))
    args = parser.parse_args()
    OUT_DIR = args.outdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    COHORT_FILES = {"Ma": args.ma, "Nido": args.nido}
    COHORT_CONFIG = {
        "Ma":   dict(dx_col="condition", sample_col="sample", disease_label="PD", control_label="control"),
        "Nido": dict(dx_col="Diagnosis", sample_col="sample_id", disease_label="PD", control_label="Control"),
    }
    all_rows = []
    for cohort, path in COHORT_FILES.items():
        print(f"\n{'='*70}\n{cohort}\n{'='*70}")
        adata = sc.read_h5ad(path)
        ensure_lognorm(adata)
        cfg = COHORT_CONFIG[cohort]

        pb = pseudobulk_by_donor(adata, cfg["sample_col"])
        dx_by_sample = adata.obs.drop_duplicates(cfg["sample_col"]).set_index(cfg["sample_col"])[cfg["dx_col"]]
        dis_samples = dx_by_sample[dx_by_sample == cfg["disease_label"]].index.intersection(pb.index)
        ctrl_samples = dx_by_sample[dx_by_sample == cfg["control_label"]].index.intersection(pb.index)

        # --- genuine decoupleR bulk activity per donor ---
        import decoupler as dc
        net = dc.op.collectri(organism="human", remove_complexes=False)
        pb_adata = sc.AnnData(pb.values, obs=pd.DataFrame(index=pb.index), var=pd.DataFrame(index=pb.columns))
        dc.mt.ulm(data=pb_adata, net=net, verbose=False)
        acts = dc.pp.get_obsm(adata=pb_adata, key="score_ulm")

        for tf, targets in CUSTOM_REGULONS.items():
            custom_series = custom_activity_series(pb, targets)
            dec_series = pd.Series(np.asarray(acts[:, tf].X).flatten(), index=pb.index) if tf in acts.var_names else None

            for method_name, series in [("custom_zscore", custom_series), ("decoupler_bulk", dec_series)]:
                if series is None:
                    continue
                for label, idx in [("all_donors", pb.index), ("disease_only", dis_samples), ("control_only", ctrl_samples)]:
                    vals = series.loc[series.index.intersection(idx)].dropna().values
                    result = one_sample_test(vals, label)
                    if result:
                        result.update({"Cohort": cohort, "TF": tf, "method": method_name})
                        all_rows.append(result)

        pub = PUBLISHED[cohort]
        for tf, (mean, p) in pub.items():
            print(f"  published {tf}: mean={mean}, p={p}")

        del adata
        import gc; gc.collect()

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_DIR / "one_sample_candidates.csv", index=False)

    print("\n\nFull candidate table (compare each row's mean/p_vs_zero against the")
    print("published values printed above, per cohort):")
    pd.set_option("display.max_rows", None)
    print(results.sort_values(["Cohort", "TF", "method", "subset"]).to_string(index=False))
