"""
Tests the specific hypothesis proposed for why Augur's cell-type ranking
changed between the old (log-normalised) and corrected (raw-counts) runs:
that raw counts let a classifier partly detect sequencing-depth differences
between disease and control cells, and that this confound is uneven across
cell types -- larger in some, smaller in others -- which would explain a
RESHUFFLE in ranking, not just a uniform shift in every AUROC.

For each cell type, in both Jin and Ma, this computes the mean total UMI
count per cell in the disease group vs the control group, and reports the
gap. If cell types with the biggest AUROC change also show the biggest
depth gap between groups, that's direct evidence for the hypothesis. If
there's no such relationship, the depth-confound explanation is not
supported by this data and the ranking shift needs a different explanation.

NOTE ON THE "OLD" AUROC VALUES BELOW: these were read by eye off the bar
lengths in the previously uploaded figure image, not extracted from exact
numbers -- treat them as approximate (+/- 0.01-0.02), good enough to judge
which cell types shifted the most, not for precise comparison.
"""
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from scipy import stats
from pathlib import Path

COHORT_CONFIG = {
    "Jin": dict(cell_type_col="broad_type", dx_col="condition", disease_label="LBD", control_label="Control"),
    "Ma":  dict(cell_type_col="cell_type", dx_col="condition", disease_label="PD", control_label="control"),
}

# Approximate, read-by-eye from the previously uploaded (old, log-normalised) figure
OLD_AUROC_APPROX = {
    "Jin": {"Oligodendrocyte": 0.72, "Astrocyte": 0.65, "Excit. Neuron": 0.61,
            "OPC": 0.59, "Inhib. Neuron": 0.58},  # Microglia/Endothelial not shown in old figure
    "Ma": {"Microglia": 0.81, "Oligodendrocytes": 0.74, "Endothelial": 0.70,
           "Astrocytes": 0.68, "Pericytes": 0.64, "Excitatory_Neurons": 0.63,
           "OPCs": 0.61, "Inhibitory_Neurons": 0.60},  # Unassigned not shown
}
# Confirmed, corrected (raw counts) -- already applied to the manuscript
NEW_AUROC = {
    "Jin": {"Microglia": 0.692721, "Inhib. Neuron": 0.645522, "Astrocyte": 0.644433,
            "Excit. Neuron": 0.642517, "OPC": 0.627766, "Oligodendrocyte": 0.621304,
            "Endothelial": 0.573968},
    "Ma": {"Oligodendrocytes": 0.847540, "Endothelial": 0.783946, "Pericytes": 0.607721,
           "Unassigned": 0.606995, "Microglia": 0.581531, "Excitatory_Neurons": 0.568980,
           "Inhibitory_Neurons": 0.568662, "Astrocytes": 0.556916, "OPCs": 0.555726},
}


def get_depth(adata):
    """Total counts per cell from the raw counts layer if present, else X."""
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sps.issparse(X):
        return np.asarray(X.sum(axis=1)).flatten()
    return np.asarray(X).sum(axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jin", type=Path, required=True, help="Path to the Jin cohort h5ad")
    parser.add_argument("--ma", type=Path, required=True, help="Path to the Ma cohort h5ad")
    parser.add_argument("--outdir", type=Path, default=Path("results/depth_confound_check"))
    args = parser.parse_args()
    OUT_DIR = args.outdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FILES = {"Jin": args.jin, "Ma": args.ma}

    all_rows = []
    for cohort, path in FILES.items():
        adata = sc.read_h5ad(path)
        cfg = COHORT_CONFIG[cohort]
        depth = get_depth(adata)
        adata.obs["_depth"] = depth

        print(f"\n{'='*70}  {cohort}")
        print(f"{'Cell type':<25} {'Depth: disease':>16} {'Depth: control':>16} "
              f"{'Depth gap %':>12} {'Old AUROC':>10} {'New AUROC':>10} {'AUROC chg':>10}")

        for ct in sorted(adata.obs[cfg["cell_type_col"]].unique()):
            sub = adata.obs[adata.obs[cfg["cell_type_col"]] == ct]
            dis = sub.loc[sub[cfg["dx_col"]] == cfg["disease_label"], "_depth"]
            ctl = sub.loc[sub[cfg["dx_col"]] == cfg["control_label"], "_depth"]
            if len(dis) < 20 or len(ctl) < 20:
                continue
            dis_mean, ctl_mean = dis.mean(), ctl.mean()
            gap_pct = 100 * (dis_mean - ctl_mean) / ctl_mean

            old_auc = OLD_AUROC_APPROX.get(cohort, {}).get(str(ct))
            new_auc = NEW_AUROC.get(cohort, {}).get(str(ct))
            auc_chg = (new_auc - old_auc) if (old_auc is not None and new_auc is not None) else None

            old_str = f"{old_auc:.3f}" if old_auc is not None else "n/a"
            new_str = f"{new_auc:.3f}" if new_auc is not None else "n/a"
            chg_str = f"{auc_chg:+.3f}" if auc_chg is not None else "n/a"

            print(f"{str(ct):<25} {dis_mean:>16,.0f} {ctl_mean:>16,.0f} "
                  f"{gap_pct:>11.1f}% {old_str:>10} {new_str:>10} {chg_str:>10}")

            all_rows.append({"cohort": cohort, "cell_type": str(ct),
                              "depth_gap_pct": gap_pct, "auroc_change": auc_chg})

        del adata
        import gc; gc.collect()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "depth_confound_check.csv", index=False)

    testable = df.dropna(subset=["auroc_change"])
    if len(testable) >= 4:
        rho, p = stats.spearmanr(testable["depth_gap_pct"].abs(), testable["auroc_change"].abs())
        print(f"\n{'='*70}")
        print("HYPOTHESIS TEST: does |sequencing-depth gap| correlate with")
        print("|AUROC change| across cell types (pooling both cohorts)?")
        print(f"{'='*70}")
        print(f"  Spearman rho = {rho:.3f}, p = {p:.4f}, n = {len(testable)} cell types")
        if p < 0.10 and rho > 0.3:
            print("  -> SUPPORTS the depth-confound hypothesis: cell types with a bigger")
            print("     depth gap between disease/control tend to show a bigger AUROC change.")
        else:
            print("  -> Does NOT clearly support the depth-confound hypothesis on this data.")
            print("     Small n (few cell types) means this test has limited power either way --")
            print("     read the printed table above by eye as well, not just this one number.")
    else:
        print("\nToo few cell types with both old and new AUROC values to test formally --")
        print("read the printed table above by eye instead.")
