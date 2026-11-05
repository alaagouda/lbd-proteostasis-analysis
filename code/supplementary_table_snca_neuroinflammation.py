"""
Builds the supplementary table for the SNCA-Neuroinflammation donor-level analysis from the CSV files
the three verification scripts save. Nothing is typed in by hand: every number is read from those files.

Run, in this order, the three scripts on the real Ma file first (each writes one CSV into
results/snca_neuroinflammation_verification/):
    snca_neuroinflammation_primary.py   -> Ma_analysisC_recomputed_donor_level.csv  (PRIMARY)
    snca_neuroinflammation_donor_level.py -> Ma_analysisA_cpm_donor_level.csv          (sensitivity)
    snca_neuroinflammation_scoregenes.py  -> Ma_analysisB_scoregenes_donor_level.csv   (sensitivity)
Then run this script. It refuses to run if any of the three files is missing, so the table can never be
built from a mixture of old and new runs.
"""

from pathlib import Path
import sys
import pandas as pd

RESULTS = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/snca_neuroinflammation_verification")
)
FILES = [
    ("Primary: SNCA expression + score_genes module", "Ma_analysisC_recomputed_donor_level.csv"),
    ("Sensitivity A: per-nucleus CPM for both measures", "Ma_analysisA_cpm_donor_level.csv"),
    ("Sensitivity B: score_genes for both measures", "Ma_analysisB_scoregenes_donor_level.csv"),
]
ORIGINAL = {
    "Pericytes": "rho = -0.585, padj < 0.001 (reported as strongest)",
    "Excitatory_Neurons": "rho = -0.492, padj < 0.001",
    "Microglia": "rho = +0.119, padj < 0.001",
}

missing = [fn for _, fn in FILES if not (RESULTS / fn).exists()]
if missing:
    raise FileNotFoundError(
        f"Missing in {RESULTS.resolve()}: {missing}. Run all three verification scripts on the real Ma "
        f"file first, so that every analysis comes from the same final scripts."
    )

tables = []
for label, fn in FILES:
    d = pd.read_csv(RESULTS / fn)
    need = {"cell_type", "n_donors", "rho", "perm_p", "padj_BH"}
    if not need <= set(d.columns):
        raise ValueError(f"{fn} lacks columns {need - set(d.columns)}")
    d = d.copy()
    d["rank"] = d["rho"].abs().rank(ascending=False, method="min").astype(int)
    d = d.set_index("cell_type")[["n_donors", "rho", "perm_p", "padj_BH", "rank"]]
    d.columns = pd.MultiIndex.from_product(
        [[label], ["n donors", "Spearman rho", "permutation p", "BH-adjusted p", "rank by |rho|"]]
    )
    tables.append(d)

wide = pd.concat(tables, axis=1)
primary = tables[0].iloc[:, 1]  # primary rho column
wide = wide.loc[primary.abs().sort_values(ascending=False).index]
wide.insert(0, ("", "Original manuscript claim"), [ORIGINAL.get(c, "") for c in wide.index])
wide.index.name = "Cell type"

for c in wide.columns:
    if c[1] in ("Spearman rho",):
        wide[c] = wide[c].map(lambda v: f"{v:+.3f}")
    elif c[1] in ("permutation p", "BH-adjusted p"):
        wide[c] = wide[c].map(lambda v: f"{v:.4f}" if v >= 0.0001 else "<0.0001")
    elif c[1] in ("n donors", "rank by |rho|"):
        wide[c] = wide[c].astype(int)

out_csv = RESULTS / "Supplementary_Table_SNCA_Neuroinflammation_donor_level.csv"
wide.to_csv(out_csv)
try:
    wide.to_excel(RESULTS / "Supplementary_Table_SNCA_Neuroinflammation_donor_level.xlsx")
except Exception as e:
    print("xlsx not written (", e, ") - the CSV is complete.")

print(wide.to_string())
print("\nSummary:")
for label, fn in FILES:
    d = pd.read_csv(RESULTS / fn)
    sig = d.loc[d["padj_BH"] < 0.05, "cell_type"].tolist()
    p = (
        d.assign(r=d["rho"].abs().rank(ascending=False, method="min"))
        .query("cell_type == 'Pericytes'")
        .iloc[0]
    )
    print(
        f"  {label}: BH padj<0.05 -> {sig or 'none'}; pericytes rho={p['rho']:+.3f}, padj={p['padj_BH']:.3f}, rank {int(p['r'])}/{len(d)}"
    )
print(f"\nWritten to {out_csv}")
