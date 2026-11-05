"""
Verifies the SNCA-neuroinflammation per-cell-type correlation claims,
currently reported in the manuscript's Results text without any stated
donor-level aggregation -- the same computational pattern that was
confirmed to cause pseudoreplication-inflated significance for the
original PDE10A/SNCA-vs-Proteostasis analysis (see
step_D_pde10a_snca_donor_level_v3.py and the surrounding verification
work).

Specifically checks the three claims currently in the manuscript:
  - Prefrontal (Nido): near-zero correlations across all cell types
  - Cingulate (Ma) excitatory neurons: rho = -0.492, padj < 0.001
  - Cingulate (Ma) microglia: rho = +0.119, padj < 0.001
  - Cingulate (Ma) pericytes: rho = -0.585, padj < 0.001, described as
    "the strongest association in the dataset"

This script recomputes all of these at the DONOR level (one SNCA value
and one Neuroinflammation module score per donor, per cell type -- not
one value per nucleus) using the same per-nucleus-CPM-then-donor-mean
method confirmed for the rest of this manuscript's module scores, and
the same permutation-based significance testing (exact enumeration for
small n) used for the corrected PDE10A/SNCA analysis.

Before trusting this script's output over the manuscript's current
claim, read the printed donor-count column: if any cell type's donor
count is small (as pericytes likely will be, given a rare population
across only 17 Ma donors), treat that result with the same caution
the rest of this manuscript applies to small-n findings, and check
whether the ORIGINAL per-nucleus computation (which this script does
NOT attempt to reproduce, since reproducing a flawed method isn't the
goal here) is actually what generated the manuscript's current numbers.
"""

import os
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats

# Neuroinflammation module: 13 genes, exactly as in Supplementary Table 2 and in the
# pipeline's own gene_modules.py. NOTE: module_genes_Neuroinflammation.csv (15 genes)
# does NOT match either of those and is not used here; likewise
# module_genes_Proteostasis.csv (16 genes) differs from the paper's 14-gene module.
# preflight_checks() cross-checks this list against Supplementary Table 2 when that
# file is available locally.
NEUROINFLAMMATION_GENES = [
    "TNF",
    "IL1B",
    "IL6",
    "NFKB1",
    "CCL2",
    "CXCL10",
    "CD68",
    "AIF1",
    "TREM2",
    "C1QA",
    "C1QB",
    "C1QC",
    "TYROBP",
]
SUPP_TABLE2_XLSX = Path(
    "./Supplementary_Table2__module_gene_lists.xlsx"
)  # optional local cross-check

# Donor composition exactly as stored in GSE253462_annotated.h5ad (from a real run:
# {'PD': 9, 'control': 8}). Hard check, not a warning: a verification run on the
# wrong file or a changed annotation should stop, not print and carry on.
EXPECTED_DONOR_DIAGNOSIS = {"PD": 9, "control": 8}

# FIX: matches the Drive-access pattern already proven working elsewhere
# in this project (final_network_provenance_verification_v2.py) -- this
# script previously assumed a static local path, which would fail in the
# marimo environment exactly the way the earlier STRING/BioGRID scripts
# did before this same fix was applied to them.
H5AD_PATH = Path(
    "./data_snrna/GSE253462_annotated.h5ad"
)  # confirmed real filename, from the Drive error message's own file listing
CELL_TYPE_COL = "cell_type"
DONOR_COL = "sample"
DIAGNOSIS_COL = "condition"
COUNT_LAYER = "counts"

DRIVE_FOLDER_ID = os.environ.get("LBD_DRIVE_FOLDER_ID", "")
DRIVE_TOKEN_PATH = Path("drive_token.json")
# Default True here, unlike the final network audit script: this is a
# diagnostic check being run to decide whether a manuscript passage needs
# correcting, not a locked final verification run where ambiguity about
# which exact file version was used would be a problem in itself.
ALLOW_DRIVE_FETCH = bool(DRIVE_FOLDER_ID)  # R2: off unless LBD_DRIVE_FOLDER_ID is set


def _restore_drive_token_if_missing():
    if DRIVE_TOKEN_PATH.exists():
        return
    token_json = os.environ.get("DRIVE_TOKEN_JSON")
    if not token_json:
        raise FileNotFoundError("drive_token.json missing and no DRIVE_TOKEN_JSON env var.")
    DRIVE_TOKEN_PATH.write_text(token_json)


def _get_drive_service():
    _restore_drive_token_if_missing()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(str(DRIVE_TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        DRIVE_TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def _list_all_drive_files(service, folder_id):
    idx, pt = {}, None
    while True:
        r = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, size)",
                pageSize=100,
                pageToken=pt,
            )
            .execute()
        )
        for f in r.get("files", []):
            idx[f["name"]] = f
        pt = r.get("nextPageToken")
        if not pt:
            break
    return idx


def _download_drive_file(service, meta, local_path):
    from googleapiclient.http import MediaIoBaseDownload
    import io

    if local_path.exists() and local_path.stat().st_size == int(meta.get("size", 0)):
        return local_path
    request = service.files().get_media(fileId=meta["id"])
    fh = io.FileIO(local_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()
    return local_path


def resolve_input_path(path: Path, label: str, allow_drive_fetch: bool) -> Path:
    """Identical logic to the proven version in
    final_network_provenance_verification_v2.py: use the local file if
    present, otherwise fetch a same-named file from this project's Drive
    folder when explicitly allowed, and fail with a clear, actionable
    error -- including the full list of available Drive filenames --
    rather than a bare FileNotFoundError, if neither is available."""
    if path.exists():
        return path
    if not allow_drive_fetch:
        raise FileNotFoundError(
            f"{label} not found locally at {path}. This script does not fetch "
            f"inputs automatically -- provide the exact local file, or set "
            f"ALLOW_DRIVE_FETCH = True to permit retrieving a same-named file "
            f"from this project's Drive folder."
        )
    print(
        f"{label} not found locally at {path}, ALLOW_DRIVE_FETCH set: "
        f"trying Google Drive by filename '{path.name}'..."
    )
    service = _get_drive_service()
    remote_index = _list_all_drive_files(service, DRIVE_FOLDER_ID)
    if path.name not in remote_index:
        raise FileNotFoundError(
            f"{label} not found locally at {path}, and no file named '{path.name}' "
            f"exists in the Drive folder. Available files: {sorted(remote_index.keys())}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _download_drive_file(service, remote_index[path.name], path)
    return path


def per_nucleus_cpm_mean(adata, genes, nucleus_mask):
    """Confirmed Figure-4 method: CPM per nucleus, then mean across genes.

    FIX: no longer densifies the full nucleus-by-gene matrix. CPM's
    denominator (each nucleus's TOTAL count) genuinely needs every gene
    summed, but scipy sparse matrices support .sum(axis=1) natively
    without densifying -- only the handful of target gene columns are
    pulled out and densified afterward, a tiny slice rather than the
    whole subset-by-all-genes matrix this previously built."""
    raw = adata.layers[COUNT_LAYER][nucleus_mask]
    if hasattr(raw, "toarray"):
        totals = np.asarray(raw.sum(axis=1)).ravel()  # sparse sum, stays sparse until here
    else:
        totals = raw.sum(axis=1)
    totals = totals.astype(float)
    totals[totals == 0] = 1  # avoid division by zero on empty nuclei

    assert_genes_present(adata, genes, "CPM scoring")
    gene_idx = [adata.var_names.get_loc(g) for g in genes]
    gene_counts = raw[:, gene_idx]
    if hasattr(gene_counts, "toarray"):
        gene_counts = gene_counts.toarray()  # only this small slice is ever densified
    cpm = gene_counts / totals[:, None] * 1_000_000
    return cpm.mean(axis=1)


def donor_level_scores(adata, genes, cell_type, min_donors=4):
    """Returns a DataFrame of one row per donor: SNCA score, module score,
    diagnosis -- restricted to the given cell type. Donors with zero nuclei
    of this cell type are correctly excluded, not silently zero-filled."""
    mask_celltype = adata.obs[CELL_TYPE_COL] == cell_type
    donors = adata.obs.loc[mask_celltype, DONOR_COL].unique()
    rows = []
    for donor in donors:
        mask = mask_celltype & (adata.obs[DONOR_COL] == donor)
        n_nuclei = mask.sum()
        if n_nuclei == 0:
            continue
        snca_vals = per_nucleus_cpm_mean(adata, ["SNCA"], mask.values)
        module_vals = per_nucleus_cpm_mean(adata, genes, mask.values)
        # FIX: explicitly asserted rather than silently taking the first
        # value -- a verification script should catch a donor-diagnosis
        # data integrity problem, not quietly paper over it the same way
        # the issue being investigated here was originally missed.
        donor_diagnoses = adata.obs.loc[mask, DIAGNOSIS_COL].unique()
        assert len(donor_diagnoses) == 1, (
            f"Donor {donor} has {len(donor_diagnoses)} different diagnosis values "
            f"({list(donor_diagnoses)}) across its {cell_type} nuclei -- this should "
            f"never happen and indicates a data integrity problem that needs "
            f"investigating before trusting any result here."
        )
        diagnosis = donor_diagnoses[0]
        rows.append(
            {
                "donor": donor,
                "n_nuclei": n_nuclei,
                "snca": snca_vals.mean(),
                "module": module_vals.mean(),
                "diagnosis": diagnosis,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) < min_donors:
        print(
            f"  WARNING: only {len(df)} donors have any {cell_type} nuclei -- "
            f"too few for meaningful correlation; treat any result here as "
            f"unreliable regardless of the p-value it produces."
        )
    return df


def exact_or_monte_carlo_permutation_p(x, y, n_perm=10000, seed=0):
    """Same exact-enumeration-for-small-n logic used for Jin's PDE10A/SNCA
    verification: enumerates all possible permutations when that count is
    smaller than the Monte Carlo sample size, to avoid reporting a smaller
    p-value than is mathematically possible for the given n."""
    n = len(x)
    observed_rho, _ = stats.spearmanr(x, y)
    import math

    total_perms = math.factorial(n)
    if total_perms <= n_perm:
        null_rhos = []
        for perm in permutations(y):
            rho, _ = stats.spearmanr(x, perm)
            null_rhos.append(rho)
        null_rhos = np.array(null_rhos)
    else:
        rng = np.random.default_rng(seed)
        null_rhos = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    p = (np.sum(np.abs(null_rhos) >= np.abs(observed_rho)) + 1) / (len(null_rhos) + 1)
    return observed_rho, p


def assert_genes_present(adata, genes, label):
    """score_genes and a naive `if g in var_names` filter both DROP missing genes
    silently, which would turn a 15-gene module into a smaller one without any
    error. Refuse to run instead."""
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise KeyError(
            f"{label} genes missing from adata.var_names: {missing}. "
            f"Do not run until the gene identifiers are reconciled. If a gene is genuinely absent "
            f"from this file, note that the original module_scoring.py silently scored only the genes "
            f"present, so check n_genes in the original module-effect-size output before deciding "
            f"how to proceed."
        )


def preflight_checks(adata, resolved_path, module_genes):
    """Everything that can be checked cheaply is checked here, immediately after
    loading and BEFORE any code touches adata.layers[COUNT_LAYER], so a wrong
    name produces the diagnostic message below rather than a bare KeyError."""
    missing_obs_cols = [
        c for c in (CELL_TYPE_COL, DONOR_COL, DIAGNOSIS_COL) if c not in adata.obs.columns
    ]
    if missing_obs_cols:
        raise KeyError(
            f"Expected obs column(s) {missing_obs_cols} not found in {resolved_path}. "
            f"Actual obs columns: {sorted(adata.obs.columns.tolist())}. "
            f"Update CELL_TYPE_COL / DONOR_COL / DIAGNOSIS_COL to match before rerunning."
        )
    if COUNT_LAYER not in adata.layers:
        raise KeyError(
            f"Expected layer '{COUNT_LAYER}' not found in {resolved_path}. "
            f"Actual layers: {sorted(str(k) for k in adata.layers.keys() if k is not None)}. "
            f"Update COUNT_LAYER to match before rerunning."
        )
    assert_genes_present(adata, ["SNCA"], "SNCA")
    assert_genes_present(
        adata, module_genes, f"Neuroinflammation ({len(module_genes)}-gene module)"
    )

    if SUPP_TABLE2_XLSX.exists():
        import pandas as _pd

        _t = _pd.read_excel(SUPP_TABLE2_XLSX, header=None)
        _row = _t[_t.iloc[:, 0].astype(str).str.strip() == "Neuroinflammation"]
        if _row.empty:
            raise ValueError(f"No 'Neuroinflammation' row found in {SUPP_TABLE2_XLSX}.")
        published = {g.strip() for g in str(_row.iloc[0, 2]).split(",")}
        if published != set(module_genes):
            raise ValueError(
                f"Gene list in this script differs from {SUPP_TABLE2_XLSX.name}. "
                f"Only in script: {sorted(set(module_genes) - published)}; "
                f"only in Supplementary Table 2: {sorted(published - set(module_genes))}."
            )
        print(
            f"Gene list matches {SUPP_TABLE2_XLSX.name} (Neuroinflammation row) exactly "
            f"({len(published)} genes)."
        )
    else:
        print(
            f"NOTE: {SUPP_TABLE2_XLSX.name} not found locally, so the script's gene list was "
            f"not cross-checked against Supplementary Table 2. Place it next to the script to enable it."
        )

    donor_diag = (
        adata.obs[[DONOR_COL, DIAGNOSIS_COL]]
        .drop_duplicates()[DIAGNOSIS_COL]
        .value_counts()
        .to_dict()
    )
    if donor_diag != EXPECTED_DONOR_DIAGNOSIS:
        raise ValueError(
            f"Unexpected donor diagnosis composition: {donor_diag}. "
            f"Expected {EXPECTED_DONOR_DIAGNOSIS}. Wrong file, changed annotation, or a "
            f"donor carrying two diagnosis labels."
        )
    print(
        f"Confirmed schema, count layer, all {len(module_genes)} module genes + SNCA present, "
        f"donor composition {donor_diag}."
    )
    print(f"Cell types present: {sorted(adata.obs[CELL_TYPE_COL].unique().tolist())}")


if __name__ == "__main__":
    import scanpy as sc

    resolved_path = resolve_input_path(H5AD_PATH, "Ma annotated cohort", ALLOW_DRIVE_FETCH)

    print("Loading Ma cohort...")
    adata = sc.read_h5ad(resolved_path)
    preflight_checks(adata, resolved_path, NEUROINFLAMMATION_GENES)

    print(
        "\nANALYSIS A (sensitivity): SNCA and the Neuroinflammation module both scored by per-nucleus CPM, then "
        "averaged per donor. The original module scoring (module_scoring.py) used score_genes, so Analysis C is "
        "the faithful reconstruction; A and B show how much the answer depends on the scoring method."
    )

    # FIX: every cell type actually present is now tested, not just the
    # three the manuscript happens to name -- "strongest in the dataset"
    # is a claim about ALL cell types, and three out of nine cannot
    # confirm or refute it.
    all_cell_types = sorted(adata.obs[CELL_TYPE_COL].unique().tolist())
    manuscript_claims = {
        "Excitatory_Neurons": (-0.492, "p<0.001"),
        "Microglia": (0.119, "p<0.001"),
        "Pericytes": (-0.585, "p<0.001 (claimed strongest in dataset)"),
    }

    results = []
    for ct in all_cell_types:
        df = donor_level_scores(adata, NEUROINFLAMMATION_GENES, ct)
        if len(df) < 4:
            results.append({"cell_type": ct, "n_donors": len(df), "rho": np.nan, "perm_p": np.nan})
            continue
        rho, p = exact_or_monte_carlo_permutation_p(df["snca"].values, df["module"].values)
        results.append({"cell_type": ct, "n_donors": len(df), "rho": rho, "perm_p": p})

    results_df = pd.DataFrame(results)

    # FIX: BH-FDR correction across all tested cell types, giving a true
    # padj comparable to the manuscript's claimed "padj<0.001" -- a raw
    # permutation p alone cannot be compared to an adjusted p-value.
    def benjamini_hochberg(pvals):
        """Manual BH-FDR implementation, avoiding a dependency on
        scipy.stats.false_discovery_control (only added in scipy 1.11,
        June 2023) given this project's repeated environment-version
        mismatches tonight (pertpy, pandas) -- this is simple enough to
        implement directly and verify independently rather than assume
        a specific scipy version is available."""
        pvals = np.asarray(pvals)
        n = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order] * n / (np.arange(n) + 1)
        # enforce monotonicity (BH step-up procedure)
        adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        result = np.empty(n)
        result[order] = adjusted
        return result

    valid = results_df["perm_p"].notna()
    results_df["padj_BH"] = np.nan
    if valid.sum() > 0:
        results_df.loc[valid, "padj_BH"] = benjamini_hochberg(
            results_df.loc[valid, "perm_p"].values
        )

    results_df["abs_rho"] = results_df["rho"].abs()
    results_df = results_df.sort_values("abs_rho", ascending=False, na_position="last")

    print(
        f"\n{'Cell type':<20}{'n donors':<10}{'rho':<9}{'perm p':<10}{'BH padj':<10}{'manuscript claim'}"
    )
    for _, row in results_df.iterrows():
        claim = manuscript_claims.get(row["cell_type"], "(not specifically named in manuscript)")
        if pd.isna(row["rho"]):
            print(
                f"{row['cell_type']:<20}{int(row['n_donors']):<10}{'--':<9}{'--':<10}{'--':<10}{claim}"
            )
        else:
            print(
                f"{row['cell_type']:<20}{int(row['n_donors']):<10}{row['rho']:<9.3f}"
                f"{row['perm_p']:<10.4f}{row['padj_BH']:<10.4f}{claim}"
            )

    strongest = (
        results_df.loc[results_df["abs_rho"].idxmax()]
        if results_df["abs_rho"].notna().any()
        else None
    )
    if strongest is not None:
        print(
            f"\nStrongest |rho| across ALL tested cell types, donor-level: "
            f"{strongest['cell_type']} (rho={strongest['rho']:.3f}, "
            f"padj={strongest['padj_BH']:.4f})"
        )
        if strongest["cell_type"] != "Pericytes":
            print(
                "This does NOT match the manuscript's claim that pericytes are "
                "the strongest association in the dataset."
            )

    print("\nIf any donor-level rho above differs substantially in magnitude or")
    print("direction from the manuscript's per-nucleus claim, or if any BH-")
    print("adjusted p-value here is far less extreme than the manuscript's")
    print("'padj < 0.001', that confirms the same pseudoreplication pattern")
    print("already found and fixed elsewhere in this manuscript.")

    from pathlib import Path as _P

    _out = _P("results/snca_neuroinflammation_verification")
    _out.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(_out / "Ma_analysisA_cpm_donor_level.csv", index=False)
    print(f"Results written to {_out.resolve()}")
