"""
STEP B: Three-cohort (Jin + Ma + Nido) gradient analysis, using the
CONFIRMED method from Step A -- per-nucleus CPM, mean within donor --
for Proteostasis, Neuroinflammation, and Complement.

WHY THIS RERUN MATTERS: an earlier three-cohort analysis used
scanpy.tl.score_genes and found a near-null Proteostasis result across
cohorts. That method is now known to NOT reproduce the manuscript's
published Proteostasis effect sizes (confirmed diff up to 0.31 for Ma).
This rerun uses the method independently confirmed tonight to reproduce
published values to within 0.0005 across all three cohorts -- the
earlier near-null result may simply have been measuring the wrong
thing, and should not be treated as final until re-tested this way.

TERMINOLOGY: per-nucleus CPM, donor-mean of nucleus scores. NOT
classical pseudobulk (ratio-of-sums). NOT scanpy score_genes.

DESIGN:
  - All three modules scored with the IDENTICAL method and framework --
    changing only the gene list between modules.
  - Per-cohort diagnosis labels are explicit config (Jin: lbd/control;
    Ma/Nido: pd/control) -- NOT a shared hardcoded string, which was a
    real, confirmed bug source earlier tonight.
  - Jin (n=6, 3v3) reported descriptively throughout, consistent with
    how this project has treated Jin's small sample size everywhere else.
  - Within-cohort disease-vs-control effect sizes (Cohen's d, 95% CI,
    Mann-Whitney p) are the PRIMARY comparison across cohorts. Absolute
    cross-cohort medians are reported as secondary/descriptive context:
    per-nucleus CPM is more directly comparable across cohorts than
    scanpy score_genes (no cohort-wide reference-gene-pool dependency),
    but sequencing depth/protocol differences between cohorts still
    mean this is not a fully controlled cross-cohort comparison.
  - Neuroinflammation is scored both with its full 13-gene list and a
    C1q-excluded 10-gene sensitivity variant, for consistency with how
    this module has been treated elsewhere in this project.
  - FDR correction applied within each cohort's disease-vs-control
    family separately (matching this project's established discipline);
    Jin is reported descriptively and excluded from FDR correction.

Self-contained -- no cross-file dependency.
"""

import io, os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

DRIVE_FOLDER_ID = os.environ.get("LBD_DRIVE_FOLDER_ID", "")
TOKEN_PATH = Path("drive_token.json")
DATA_DIR = Path("./data_snrna")
OUT_DIR = Path("./results/step_B_three_cohort_gradient")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "Jin": dict(
        file="GSE235914_LBD_annotated_FIXED.h5ad",
        donor_col="sample",
        dx_col="condition",
        disease_labels={"lbd", "dlb"},
        control_labels={"control", "ctrl"},
        descriptive_only=True,
        count_layer="counts",
    ),
    "Ma": dict(
        file="GSE253462_annotated.h5ad",
        donor_col="sample",
        dx_col="condition",
        disease_labels={"pd"},
        control_labels={"control", "ctrl"},
        descriptive_only=False,
        count_layer="counts",
    ),
    "Nido": dict(
        file="nido_analysis_ready.h5ad",
        donor_col="sample_id",
        dx_col="Diagnosis",
        disease_labels={"pd"},
        control_labels={"control", "ctrl"},
        descriptive_only=False,
        count_layer="counts",
    ),
}
COHORT_ORDER = [
    "Jin",
    "Ma",
    "Nido",
]  # matches the manuscript's temporal -> cingulate -> prefrontal claim

PROTEOSTASIS_GENES = [
    "HSPA1A",
    "HSPA1B",
    "HSPB1",
    "DNAJB1",
    "DNAJB4",
    "HSP90AA1",
    "HSP90AB1",
    "DNAJA1",
    "DNAJA4",
    "CHORDC1",
    "STIP1",
    "CRYAB",
    "BAG3",
    "HSPH1",
]
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
NEUROINFLAMMATION_C1Q_EXCLUDED = [
    g for g in NEUROINFLAMMATION_GENES if g not in ("C1QA", "C1QB", "C1QC")
]
COMPLEMENT_GENES = ["C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CFH", "CFB", "C5AR1", "CR1"]

PRIMARY_MODULES = {
    "Proteostasis": PROTEOSTASIS_GENES,
    "Neuroinflammation": NEUROINFLAMMATION_GENES,
    "Complement": COMPLEMENT_GENES,
}
SENSITIVITY_MODULES = {
    "Neuroinflammation_C1qExcluded": NEUROINFLAMMATION_C1Q_EXCLUDED,
}
MODULES = {**PRIMARY_MODULES, **SENSITIVITY_MODULES}


# ---------------------------------------------------------------------
# DRIVE ACCESS
# ---------------------------------------------------------------------
def restore_token_if_missing():
    if TOKEN_PATH.exists():
        return
    token_json = os.environ.get("DRIVE_TOKEN_JSON")
    if not token_json:
        raise FileNotFoundError("drive_token.json missing.")
    TOKEN_PATH.write_text(token_json)


def get_drive_service():
    if not DRIVE_FOLDER_ID:
        return None  # R2b: Drive disabled (set LBD_DRIVE_FOLDER_ID to enable)
    restore_token_if_missing()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def list_all_drive_files(service, folder_id):
    if service is None:  # R2b: Drive disabled, so every file is taken from the local data directory

        class _LocalIndex(dict):
            def __contains__(self, key):
                return True

            def __missing__(self, key):
                return {}

        return _LocalIndex()
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


def download_file(service, meta, local_path):
    if service is None:  # R2b: Drive disabled
        if local_path.exists():
            return local_path
        raise FileNotFoundError(
            f"{local_path} not found. Put the file there, or set LBD_DRIVE_FOLDER_ID (and provide a Drive token) to download it."
        )
    from googleapiclient.http import MediaIoBaseDownload

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


# ---------------------------------------------------------------------
# CORE LOGIC (confirmed method from Step A)
# ---------------------------------------------------------------------
def integer_like_fraction(X):
    vals = np.asarray(X.data if sp.issparse(X) else X, dtype=float).ravel()
    return 1.0 if vals.size == 0 else float(np.mean(np.isclose(vals, np.rint(vals), atol=1e-8)))


def select_raw_counts(adata, cfg):
    """FIX: explicit, required layer per cohort rather than a heuristic
    search -- being integer-like doesn't prove a matrix is the correct
    biological raw-count source, only that it looks numerically raw.
    Auditable and deterministic: fails loudly if the expected layer is
    absent or not sufficiently integer-like, rather than silently
    falling back to a different layer."""
    layer = cfg["count_layer"]
    if layer not in adata.layers:
        raise ValueError(
            f"Expected raw-count layer '{layer}' not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    X = adata.layers[layer]
    if integer_like_fraction(X) < 0.999999:
        raise ValueError(
            f"layers['{layer}'] is not sufficiently integer-like " f"to be treated as raw counts."
        )
    return X, f"layers['{layer}']"


def resolve_groups(adata, cfg):
    donor = adata.obs[cfg["donor_col"]].astype(str)
    raw_dx = adata.obs[cfg["dx_col"]].astype(str).str.strip().str.lower()

    def map_label(x):
        if x in cfg["disease_labels"]:
            return "Disease"
        if x in cfg["control_labels"]:
            return "Control"
        raise ValueError(f"Unrecognised diagnosis label '{x}'. Observed: {sorted(raw_dx.unique())}")

    group = raw_dx.map(map_label)
    mixed = pd.DataFrame({"donor": donor, "group": group}).groupby("donor")["group"].nunique()
    mixed = mixed[mixed > 1]
    if len(mixed):
        raise ValueError(f"Donors with mixed diagnosis labels: {mixed.index.tolist()}")
    return (
        pd.DataFrame({"donor": donor, "group": group})
        .drop_duplicates()
        .sort_values("donor")
        .reset_index(drop=True)
    )


def gene_indices(adata, genes, module_name):
    """FIX: missing genes are now FATAL. Silently dropping genes risked
    different cohorts effectively being scored with different biological
    modules if gene coverage differed between them -- exactly the kind
    of confound a cross-cohort comparison cannot tolerate."""
    names = pd.Index(adata.var_names.astype(str))
    missing = [g for g in genes if g not in names]
    if missing:
        raise ValueError(
            f"{module_name}: missing required gene(s): {missing}. "
            f"All cohorts must use the identical gene set for a valid comparison."
        )
    duplicated = names[names.duplicated()].unique().tolist()
    dup_targets = [g for g in genes if g in duplicated]
    if dup_targets:
        raise ValueError(f"{module_name}: duplicated target genes: {dup_targets}")
    return [names.get_loc(g) for g in genes], genes


def per_nucleus_cpm_donor_mean(adata, X, genes, donor_col, module_name):
    idx_genes, present_genes = gene_indices(adata, genes, module_name)
    donor = adata.obs[donor_col].astype(str)

    nucleus_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    nucleus_total[nucleus_total <= 0] = np.nan

    target = X[:, idx_genes]
    if sp.issparse(target):
        target = target.toarray()

    nucleus_cpm = target / nucleus_total[:, None] * 1e6
    nucleus_score = np.nanmean(nucleus_cpm, axis=1)

    # FIX: zero-library nuclei become NaN and are excluded from the donor
    # mean by pandas' default skipna behavior, but a naive "size" count
    # would still count them -- overstating how many nuclei actually
    # contributed. Report both explicitly.
    valid = np.isfinite(nucleus_score)
    tmp = pd.DataFrame({"donor": donor.values, "nucleus_score": nucleus_score, "valid": valid})
    return (
        tmp.groupby("donor", observed=True)
        .agg(
            donor_score=("nucleus_score", "mean"),
            n_nuclei=("valid", "sum"),
            n_nuclei_total=("valid", "size"),
        )
        .reset_index()
    )


def cohens_d(disease, control):
    disease, control = np.asarray(disease, dtype=float), np.asarray(control, dtype=float)
    n1, n2 = len(disease), len(control)
    pooled_var = ((n1 - 1) * np.var(disease, ddof=1) + (n2 - 1) * np.var(control, ddof=1)) / (
        n1 + n2 - 2
    )
    pooled_sd = np.sqrt(pooled_var)
    return np.nan if pooled_sd == 0 else (np.mean(disease) - np.mean(control)) / pooled_sd


def cohens_d_parametric_ci(d, n1, n2, confidence=0.95):
    df = n1 + n2 - 2
    se = np.sqrt((n1 + n2) / (n1 * n2) + (d**2) / (2 * df))
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df)
    return d - t_crit * se, d + t_crit * se


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("STEP B: THREE-COHORT GRADIENT, CONFIRMED METHOD (per-nucleus CPM,")
    print("donor-mean) -- Proteostasis, Neuroinflammation, Complement")
    print("=" * 78)

    service = get_drive_service()
    remote_index = list_all_drive_files(service, DRIVE_FOLDER_ID)

    cohort_donor_scores = {}  # cohort -> {module -> donor_df}
    cohort_meta = {}

    for cohort, cfg in COHORTS.items():
        print(f"\n{'='*78}\n{cohort}: loading and scoring\n{'='*78}")
        local_path = DATA_DIR / cfg["file"]
        download_file(service, remote_index[cfg["file"]], local_path)
        adata = sc.read_h5ad(local_path)
        print(f"  Cells: {adata.n_obs:,}")

        X, raw_source = select_raw_counts(adata, cfg)
        print(f"  Raw source: {raw_source}")

        meta = resolve_groups(adata, cfg)
        cohort_meta[cohort] = meta
        print(
            f"  Donors: {len(meta)} ({(meta.group=='Disease').sum()} disease, "
            f"{(meta.group=='Control').sum()} control)"
        )

        module_scores = {}
        for module_name, genes in MODULES.items():
            score_df = per_nucleus_cpm_donor_mean(adata, X, genes, cfg["donor_col"], module_name)
            module_scores[module_name] = score_df.merge(
                meta, on="donor", how="left", validate="one_to_one"
            )
        cohort_donor_scores[cohort] = module_scores

    # --- Analysis 1: descriptive cross-cohort medians ---
    print(
        f"\n{'='*78}\nANALYSIS 1 (PURELY DESCRIPTIVE): cross-cohort donor-level medians\n"
        f"Per-nucleus CPM is more directly comparable across cohorts than scanpy\n"
        f"score_genes (no cohort-wide reference-gene-pool dependency), but this is\n"
        f"still not a fully controlled cross-cohort comparison -- cohort, region,\n"
        f"disease composition, and processing protocol are all confounded (three\n"
        f"separate studies). No inferential test is reported here: a Kruskal-Wallis\n"
        f"p-value across cohorts this confounded cannot be interpreted as evidence\n"
        f"of a regional effect, so none is computed. Only the ordering itself is\n"
        f"reported, as pure description. Primary inferential evidence is Analysis 2.\n{'='*78}"
    )
    gradient_rows = []
    for module_name in MODULES:
        medians = {
            c: cohort_donor_scores[c][module_name]["donor_score"].median() for c in COHORT_ORDER
        }
        matches_ordering = medians["Jin"] > medians["Ma"] > medians["Nido"]
        print(
            f"  {module_name}: Jin={medians['Jin']:.4f}, Ma={medians['Ma']:.4f}, "
            f"Nido={medians['Nido']:.4f} -- {'MATCHES' if matches_ordering else 'does NOT match'} "
            f"temporal>cingulate>prefrontal ordering (descriptive only)"
        )
        gradient_rows.append(
            {
                "module": module_name,
                "median_Jin": medians["Jin"],
                "median_Ma": medians["Ma"],
                "median_Nido": medians["Nido"],
                "matches_ordering": matches_ordering,
            }
        )
    gradient_df = pd.DataFrame(gradient_rows)

    # --- Analysis 2: within-cohort disease-vs-control (PRIMARY) ---
    print(
        f"\n{'='*78}\nANALYSIS 2 (PRIMARY): within-cohort disease-vs-control, confirmed method\n"
        f"Jin (n=6, 3v3) reported descriptively -- too small for confident inference.\n{'='*78}"
    )
    disease_rows = []
    for cohort in COHORT_ORDER:
        cfg = COHORTS[cohort]
        for module_name in MODULES:
            df = cohort_donor_scores[cohort][module_name]
            disease = df.loc[df.group == "Disease", "donor_score"].to_numpy()
            control = df.loc[df.group == "Control", "donor_score"].to_numpy()
            n1, n2 = len(disease), len(control)
            if n1 < 2 or n2 < 2:
                continue

            d = cohens_d(disease, control)
            ci_lo, ci_hi = cohens_d_parametric_ci(d, n1, n2)
            _, p = stats.mannwhitneyu(disease, control, alternative="two-sided")

            print(
                f"  [{cohort}] {module_name}: n_disease={n1}, n_control={n2}, d={d:.4f}, "
                f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}], p={p:.4f}"
                f"{'  [DESCRIPTIVE ONLY]' if cfg['descriptive_only'] else ''}"
            )

            disease_rows.append(
                {
                    "cohort": cohort,
                    "module": module_name,
                    "n_disease": n1,
                    "n_control": n2,
                    "cohens_d": d,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "p_value": p,
                    "descriptive_only": cfg["descriptive_only"],
                }
            )

    disease_df = pd.DataFrame(disease_rows)
    # FIX: separate primary (Proteostasis, Neuroinflammation, Complement) from
    # sensitivity (Neuroinflammation_C1qExcluded) FDR families, per cohort --
    # a sensitivity variant should not silently dilute or be diluted by the
    # primary family's correction.
    powered_mask = ~disease_df["descriptive_only"]
    is_primary = disease_df["module"].isin(PRIMARY_MODULES.keys())
    disease_df["fdr_p_value"] = np.nan
    disease_df["fdr_significant"] = False
    for cohort in COHORT_ORDER:
        for family_name, family_mask_col in [("primary", is_primary), ("sensitivity", ~is_primary)]:
            mask = powered_mask & (disease_df["cohort"] == cohort) & family_mask_col
            if mask.sum() > 0:
                reject2, fdr_p2, _, _ = multipletests(
                    disease_df.loc[mask, "p_value"], method="fdr_bh"
                )
                disease_df.loc[mask, "fdr_p_value"] = fdr_p2
                disease_df.loc[mask, "fdr_significant"] = reject2
                print(f"  [{cohort}/{family_name}] {mask.sum()} test(s) in this FDR family")

    # --- Save ---
    print(f"\n{'='*78}\nSAVING RESULTS\n{'='*78}")
    gradient_path = OUT_DIR / "step_B_gradient_results.csv"
    disease_path = OUT_DIR / "step_B_disease_vs_control_results.csv"
    gradient_df.to_csv(gradient_path, index=False)
    disease_df.to_csv(disease_path, index=False)

    print("\nGradient (descriptive) results:")
    print(gradient_df.to_string(index=False))
    print("\nDisease-vs-control (PRIMARY) results:")
    print(disease_df.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")
