"""
ANALYSIS B of two: donor-level SNCA-neuroinflammation sensitivity check
using scanpy's score_genes, the method the manuscript's Methods section
states was used for the eight original expression modules (including
Neuroinflammation), rather than the CPM-based method confirmed
specifically for the Proteostasis module (Figure 4) and used in
ANALYSIS A (snca_neuroinflammation_donor_level.py).

Why this second analysis exists, and isn't redundant with Analysis A:
CPM and score_genes are confirmed, on this project's own data, to NOT
always agree -- the same Proteostasis module gave Cohen's d = 0.357
under CPM versus d = 0.878 under score_genes for the identical donors
and genes. Given that confirmed, substantial gap for Proteostasis,
Analysis A's result (no significant donor-level SNCA-neuroinflammation
association in any of 9 Ma cell types) should not be trusted as
reflecting the manuscript's actual original method until checked here
too. If Analysis A and B agree, that is strong, method-independent
evidence. If they disagree, the disagreement itself is informative and
worth reporting rather than picking whichever result is more
convenient.

Everything else (donor-level inference, all 9 cell types, BH-FDR
correction, exact/Monte Carlo permutation, donor-diagnosis integrity
checks, Drive access) is identical to Analysis A -- only the per-
nucleus scoring function differs.

IMPORTANT, specific to score_genes: it requires log-normalized
expression in .X (not raw counts), and scanpy's control-gene matching
operates on the full dataset's expression distribution, not a subset
-- so unlike CPM (computed per donor-and-cell-type subset), score_genes
is run ONCE on the whole dataset before any subsetting, exactly
matching how scanpy's own documentation describes its intended use.
scanpy's score_genes also draws its control gene set with a random
seed; this is set explicitly below for reproducibility, but a
different seed could give a slightly different control set and
therefore a slightly different score -- this is an inherent property
of the method itself, not a bug in this script.
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


LOGNORM_MAX = 20  # same threshold common.ensure_lognorm uses to decide .X is already log-normalised
STORED_MODULE_SCORE_COL = "Neuroinflammation_score"  # name used for stored module scores in the pipeline's analysis-ready files
RESULTS_DIR = Path("results/snca_neuroinflammation_verification")


def looks_like_integer_counts(X, max_values=200000):
    """True if (almost) all stored values are whole numbers. Raw counts are integers;
    log-normalised expression essentially never is. common.ensure_lognorm relies on
    'max <= 20' alone, which misreads raw counts whose maximum happens to be small."""
    import scipy.sparse as _sp

    vals = X.data if _sp.issparse(X) else np.asarray(X).ravel()
    vals = np.asarray(vals)
    if vals.size == 0:
        return False
    if vals.size > max_values:
        vals = vals[np.random.default_rng(0).choice(vals.size, size=max_values, replace=False)]
    nz = vals[vals != 0]
    if nz.size == 0:
        return False
    return float(np.mean(np.abs(nz - np.round(nz)) < 1e-6)) > 0.99


def prepare_expression(adata, snca_gene="SNCA"):
    """Mirror common.ensure_lognorm: keep the stored .X when it already looks log-normalised
    (max <= LOGNORM_MAX); otherwise normalise from the counts layer (1e4 + log1p). On the
    'stored' route, print how the stored SNCA values compare with a plain 1e4 + log1p of the
    counts, so any difference in the original normalisation is visible rather than assumed."""
    import scanpy as sc

    xmax = float(adata.X.max())
    if adata.raw is not None:
        print(
            f"adata.raw is PRESENT (max {float(adata.raw.X.max()):.2f}). score_genes is called with "
            f"use_raw=False below, so it reads the prepared .X and not .raw."
        )
    else:
        print("adata.raw is not present.")
    integer_like = looks_like_integer_counts(adata.X)
    if xmax <= LOGNORM_MAX and not integer_like:
        print(
            f".X max = {xmax:.2f} (<= {LOGNORM_MAX}) and values are not whole numbers: treating stored .X as "
            f"log-normalised and using it unchanged, as common.ensure_lognorm does."
        )
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(adata.n_obs, size=min(3000, adata.n_obs), replace=False))
        gi = adata.var_names.get_loc(snca_gene)
        cnt = adata.layers[COUNT_LAYER][idx]
        tot = np.asarray(cnt.sum(axis=1)).ravel().astype(float)
        tot[tot == 0] = 1
        c = cnt[:, gi]
        c = c.toarray().ravel() if hasattr(c, "toarray") else np.asarray(c).ravel()
        recomputed = np.log1p(c / tot * 1e4)
        st = adata.X[idx][:, gi]
        st = st.toarray().ravel() if hasattr(st, "toarray") else np.asarray(st).ravel()
        diff = float(np.max(np.abs(recomputed - st)))
        r = (
            float(np.corrcoef(recomputed, st)[0, 1])
            if np.std(st) > 0 and np.std(recomputed) > 0
            else float("nan")
        )
        print(
            f"  Stored SNCA vs plain log1p(1e4*count/total) on {len(idx)} random nuclei: "
            f"max |difference| = {diff:.4g}, Pearson r = {r:.4f}."
        )
        if diff > 1e-3:
            print(
                "  NOTE: the stored .X was normalised differently from plain 1e4 + log1p. "
                "The stored values are used, as in the original pipeline."
            )
        return adata, "stored"
    why = "values are whole numbers" if integer_like else f"max {xmax:.1f} > {LOGNORM_MAX}"
    print(
        f".X looks like raw counts ({why}). Normalising from the '{COUNT_LAYER}' layer "
        f"(target_sum=1e4, log1p), equivalent to ensure_lognorm on raw counts."
    )
    adata.X = adata.layers[COUNT_LAYER].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata, "recomputed"


def compute_score_genes_once(adata, snca_gene, module_genes, seed=0):
    """Runs scanpy's score_genes ONCE on the full dataset for both SNCA
    and the Neuroinflammation module, matching scanpy's actual intended
    usage (control genes matched against the whole dataset's expression
    distribution, not a subset). Requires log-normalized .X; this is
    checked, not assumed, since this project has already found files
    where .X was unexpectedly still raw counts."""
    import scanpy as sc

    max_val = adata.X.max()
    if max_val > 50:  # a crude but effective check: genuine log-normalized
        # single-cell data is essentially never this large
        raise ValueError(
            f"adata.X has a maximum value of {max_val:.1f}, which is far too "
            f"large to plausibly be log-normalized expression. score_genes "
            f"requires log-normalized data -- this looks like raw counts are "
            f"still in .X. Run sc.pp.normalize_total() and sc.pp.log1p() on "
            f"a copy before calling this function, and confirm the result "
            f"looks like real log-normalized data before trusting its output."
        )

    assert_genes_present(adata, module_genes, "score_genes module")
    sc.tl.score_genes(
        adata,
        gene_list=[snca_gene],
        score_name="_snca_score",
        ctrl_size=50,
        random_state=seed,
        use_raw=False,
    )
    sc.tl.score_genes(
        adata,
        gene_list=module_genes,
        score_name="_module_score",
        ctrl_size=50,
        random_state=seed,
        use_raw=False,
    )
    return adata


def donor_level_scores(adata, genes, cell_type, min_donors=4):
    """Returns a DataFrame of one row per donor: SNCA score, module score,
    diagnosis -- restricted to the given cell type. Donors with zero nuclei
    of this cell type are correctly excluded, not silently zero-filled.

    Uses the pre-computed per-nucleus _snca_score / _module_score columns
    from compute_score_genes_once(), averaged within each donor -- unlike
    the CPM version, score_genes cannot be meaningfully recomputed on a
    small subset, since its control-gene matching is a whole-dataset
    operation."""
    mask_celltype = adata.obs[CELL_TYPE_COL] == cell_type
    donors = adata.obs.loc[mask_celltype, DONOR_COL].unique()
    rows = []
    for donor in donors:
        mask = mask_celltype & (adata.obs[DONOR_COL] == donor)
        n_nuclei = mask.sum()
        if n_nuclei == 0:
            continue
        snca_vals = adata.obs.loc[mask, "_snca_score"].values
        module_vals = adata.obs.loc[mask, "_module_score"].values
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

    # score_genes needs log-normalized expression in .X. Rather than assume
    # .X is already in the right state (this project has already found
    # files where it wasn't), normalize from the confirmed raw-count layer
    # explicitly, on a copy, so the raw counts layer itself is untouched.
    print("Preparing log-normalized expression for score_genes...")
    adata, route = prepare_expression(adata, "SNCA")

    adata = compute_score_genes_once(adata, "SNCA", NEUROINFLAMMATION_GENES, seed=0)
    print("score_genes computed for SNCA and the Neuroinflammation module.")

    print(
        "\nThis is ANALYSIS B (score_genes-based). Compare these results directly "
        "against snca_neuroinflammation_donor_level.py's ANALYSIS A "
        "(CPM-based) output for the same cell types. If both analyses agree "
        "(e.g. both show no significant association surviving BH correction, "
        "and similar rho signs), that is strong, method-independent evidence. "
        "If they disagree, report the disagreement itself rather than only "
        "the more convenient of the two results."
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "Ma_analysisB_scoregenes_donor_level.csv", index=False)
    print(f"Results written to {RESULTS_DIR.resolve()}")
