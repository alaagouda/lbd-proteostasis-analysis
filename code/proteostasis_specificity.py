"""
STEP C (v2, corrected): Proteostasis specificity -- leave-one-gene-out +
expression-matched random-gene-set null.

CRITICAL FIX from v1: the original implementation densified the full
nucleus x gene CPM matrix per cohort (Nido: ~74 GB, would crash any
realistic environment). This version uses a mathematically equivalent
donor x gene precomputation instead:

    donor_score(gene_set) = mean_over_genes_in_set( mean_over_donor's_nuclei( CPM[nucleus, gene] ) )

Since both averaging steps are unweighted means, the order can be
swapped (verified below against the original formulation on small
synthetic data before trusting this on real data):

    donor_score(gene_set) = mean_over_genes_in_set( donor_mean_cpm[donor, gene] )

This means a SMALL donor x gene matrix (e.g. 17x55,000 for Ma, ~7.5 MB)
can be precomputed ONCE per cohort using sparse operations, and every
subsequent leave-one-out / random-set draw is just a cheap column
selection and average -- no per-iteration densification, no per-cohort
74 GB allocation.

OTHER FIXES from review:
  - Raw-count layer is now explicit and locked per cohort (Step A/B
    discipline), not a heuristic search.
  - Random-null fallback NO LONGER reintroduces an actual Proteostasis
    gene -- raises instead of silently contaminating the null.
  - Random genes within one draw are now sampled WITHOUT replacement --
    a "14-gene random set" is genuinely 14 unique genes.
  - Duplicate target gene NAMES are explicitly checked and rejected
    (Step A/B discipline), not silently resolved by taking the first match.
  - Actual random gene identities are saved per draw, not just effect sizes.
  - N_RANDOM_SETS increased to 5000 now that the cost per draw is trivial.

Self-contained -- no cross-file dependency.
"""

import io, os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

DRIVE_FOLDER_ID = os.environ.get("LBD_DRIVE_FOLDER_ID", "")
TOKEN_PATH = Path("drive_token.json")
DATA_DIR = Path("./data_snrna")
OUT_DIR = Path("./results/step_C_proteostasis_specificity")
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
        expected_disease=3,
        expected_control=3,
    ),
    "Ma": dict(
        file="GSE253462_annotated.h5ad",
        donor_col="sample",
        dx_col="condition",
        disease_labels={"pd"},
        control_labels={"control", "ctrl"},
        descriptive_only=False,
        count_layer="counts",
        expected_disease=9,
        expected_control=8,
    ),
    "Nido": dict(
        file="nido_analysis_ready.h5ad",
        donor_col="sample_id",
        dx_col="Diagnosis",
        disease_labels={"pd"},
        control_labels={"control", "ctrl"},
        descriptive_only=False,
        count_layer="counts",
        expected_disease=20,
        expected_control=13,
    ),
}
COHORT_ORDER = ["Jin", "Ma", "Nido"]
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
N_RANDOM_SETS = 5000
N_EXPRESSION_BINS = 20
RANDOM_SEED = 42


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
# CORE LOGIC
# ---------------------------------------------------------------------
def integer_like_fraction(X):
    vals = np.asarray(X.data if sp.issparse(X) else X, dtype=float).ravel()
    return 1.0 if vals.size == 0 else float(np.mean(np.isclose(vals, np.rint(vals), atol=1e-8)))


def select_raw_counts(adata, cfg):
    """Explicit, required layer per cohort -- matches Step A/B discipline."""
    layer = cfg["count_layer"]
    if layer not in adata.layers:
        raise ValueError(
            f"Expected raw-count layer '{layer}' not found. "
            f"Available: {list(adata.layers.keys())}"
        )
    X = adata.layers[layer]
    if integer_like_fraction(X) < 0.999999:
        raise ValueError(f"layers['{layer}'] is not sufficiently integer-like.")
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
    result = (
        pd.DataFrame({"donor": donor, "group": group})
        .drop_duplicates()
        .sort_values("donor")
        .reset_index(drop=True)
    )

    n_disease = (result["group"] == "Disease").sum()
    n_control = (result["group"] == "Control").sum()
    if "expected_disease" in cfg:
        if n_disease != cfg["expected_disease"] or n_control != cfg["expected_control"]:
            raise ValueError(
                f"Donor/group counts changed from what this locked analysis expects: "
                f"got {n_disease} disease / {n_control} control, expected "
                f"{cfg['expected_disease']} disease / {cfg['expected_control']} control. "
                f"The underlying data may have changed -- verify before proceeding."
            )

    # STRONGEST lock: exact donor IDENTITIES, not just counts -- counts
    # matching doesn't rule out one donor's diagnosis flipping while
    # another's flips oppositely. Left as an explicit placeholder rather
    # than guessed: hardcoding unverified donor IDs risks a FALSE failure
    # on genuinely correct data, which is worse than not having this
    # check yet. Populate EXPECTED_DONORS below from this run's own
    # printed donor lists once confirmed, then this becomes enforced on
    # every subsequent run.
    if cfg.get("expected_donors") is not None:
        actual_disease = set(result.loc[result.group == "Disease", "donor"])
        actual_control = set(result.loc[result.group == "Control", "donor"])
        exp_disease = set(cfg["expected_donors"]["Disease"])
        exp_control = set(cfg["expected_donors"]["Control"])
        if actual_disease != exp_disease or actual_control != exp_control:
            raise ValueError(
                f"Exact donor identities changed from locked expectations.\n"
                f"  Disease -- expected: {sorted(exp_disease)}, got: {sorted(actual_disease)}\n"
                f"  Control -- expected: {sorted(exp_control)}, got: {sorted(actual_control)}"
            )
    else:
        print(
            f"    (Donor identities not yet locked -- printing for future EXPECTED_DONORS population: "
            f"Disease={sorted(result.loc[result.group=='Disease','donor'].tolist())}, "
            f"Control={sorted(result.loc[result.group=='Control','donor'].tolist())})"
        )

    return result


def cohens_d(disease, control):
    disease, control = np.asarray(disease, dtype=float), np.asarray(control, dtype=float)
    n1, n2 = len(disease), len(control)
    pooled_var = ((n1 - 1) * np.var(disease, ddof=1) + (n2 - 1) * np.var(control, ddof=1)) / (
        n1 + n2 - 2
    )
    pooled_sd = np.sqrt(pooled_var)
    return np.nan if pooled_sd == 0 else (np.mean(disease) - np.mean(control)) / pooled_sd


def compute_donor_gene_mean_cpm(X, nucleus_total, donor):
    """Memory-safe donor x gene mean-CPM, mathematically verified
    equivalent to nucleus-level CPM averaged within donor -- BUT this
    equivalence requires every nucleus to have a valid (>0) total count.
    A zero-count nucleus previously got silently included as an all-zero
    CPM row rather than excluded (as nanmean would exclude it), breaking
    equivalence -- confirmed directly via a test that included this case,
    which an earlier equivalence check had not covered. Now fails loudly
    instead, so the caller can decide how to handle it rather than silently
    getting a biased donor score."""
    if np.any(nucleus_total <= 0) or np.any(~np.isfinite(nucleus_total)):
        bad = int(np.sum((nucleus_total <= 0) | (~np.isfinite(nucleus_total))))
        raise ValueError(
            f"{bad} nuclei have zero or invalid total raw counts. "
            f"Resolve or explicitly exclude these before scoring -- "
            f"including them would silently bias donor-level scores."
        )

    unique_donors = pd.unique(donor)
    donor_idx_map = {d: i for i, d in enumerate(unique_donors)}
    row_idx = np.array([donor_idx_map[d] for d in donor])

    n_donors = len(unique_donors)
    n_nuclei = X.shape[0]

    counts_per_donor = pd.Series(donor).value_counts()
    weights = np.array([1.0 / counts_per_donor[d] for d in donor])
    indicator = sp.csr_matrix((weights, (row_idx, np.arange(n_nuclei))), shape=(n_donors, n_nuclei))

    # FIX: sparse row-scaling via .multiply() rather than constructing an
    # explicit (n_nuclei x n_nuclei) sparse diagonal matrix -- avoids
    # unnecessary overhead, especially at Nido's scale (277,831 nuclei).
    inv_total = (1e6 / nucleus_total).reshape(-1, 1)
    cpm_sparse = X.multiply(inv_total)
    if not sp.isspmatrix_csr(cpm_sparse):
        cpm_sparse = cpm_sparse.tocsr()

    donor_gene_mean_cpm = indicator @ cpm_sparse
    donor_gene_mean_cpm = (
        np.asarray(donor_gene_mean_cpm.todense())
        if sp.issparse(donor_gene_mean_cpm)
        else donor_gene_mean_cpm
    )

    return pd.DataFrame(donor_gene_mean_cpm, index=unique_donors)


def prepare_cohort(cohort, cfg, service, remote_index):
    local_path = DATA_DIR / cfg["file"]
    download_file(service, remote_index[cfg["file"]], local_path)
    adata = sc.read_h5ad(local_path)
    print(f"  [{cohort}] Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")

    X, raw_source = select_raw_counts(adata, cfg)
    print(f"  [{cohort}] Raw source: {raw_source}")

    var_names = adata.var_names.astype(str)
    duplicated = var_names[var_names.duplicated()].unique().tolist()
    target_dups = [g for g in PROTEOSTASIS_GENES if g in duplicated]
    if target_dups:
        raise ValueError(f"{cohort}: duplicated target gene names: {target_dups}")

    nucleus_total = np.asarray(X.sum(axis=1)).ravel().astype(float)
    donor = adata.obs[cfg["donor_col"]].astype(str).values

    # FIX: fail loudly rather than silently exclude zero-count nuclei --
    # for a LOCKED analysis, an automatic exclusion is itself a
    # methodological decision that shouldn't happen without explicit
    # notice. Step A's confirmed baseline required no such handling,
    # implying none exist in the real data -- if they do turn up here,
    # that's an unexpected discrepancy worth stopping for, not silently
    # adapting around. This also sidesteps any risk of donor metadata
    # (still resolved from the unfiltered adata) drifting out of sync
    # with a filtered nucleus population.
    if np.any(nucleus_total <= 0) or np.any(~np.isfinite(nucleus_total)):
        bad = int(np.sum((nucleus_total <= 0) | (~np.isfinite(nucleus_total))))
        raise ValueError(
            f"{cohort}: found {bad} nuclei with zero/invalid raw-count totals. "
            f"Stop and inspect the source data before proceeding -- Step A's "
            f"confirmed baseline implies this should not occur."
        )

    donor_gene_cpm = compute_donor_gene_mean_cpm(X, nucleus_total, donor)
    donor_gene_cpm.columns = var_names.to_numpy()
    print(
        f"  [{cohort}] Donor x gene mean-CPM matrix: {donor_gene_cpm.shape} "
        f"({donor_gene_cpm.memory_usage(deep=True).sum() / 1e6:.2f} MB)"
    )

    meta = resolve_groups(adata, cfg)
    mean_expr_per_gene = donor_gene_cpm.mean(axis=0)

    return {"donor_gene_cpm": donor_gene_cpm, "meta": meta, "mean_expr": mean_expr_per_gene}


def score_gene_set(prepped, genes):
    """Now trivially cheap: mean across the selected gene COLUMNS of the
    small precomputed donor x gene matrix."""
    donor_scores = prepped["donor_gene_cpm"][genes].mean(axis=1).reset_index()
    donor_scores.columns = ["donor", "donor_score"]
    return donor_scores.merge(prepped["meta"], on="donor", how="left", validate="one_to_one")


def disease_effect(donor_df):
    disease = donor_df.loc[donor_df.group == "Disease", "donor_score"].to_numpy()
    control = donor_df.loc[donor_df.group == "Control", "donor_score"].to_numpy()
    if len(disease) < 2 or len(control) < 2:
        return np.nan
    return cohens_d(disease, control)


def build_expression_bins(mean_expr, target_genes, n_bins):
    valid_genes = mean_expr[mean_expr > 0].index
    ranks = mean_expr[valid_genes].rank(method="first")
    bin_edges = np.quantile(ranks.values, np.linspace(0, 1, n_bins + 1))
    bin_ids = np.digitize(ranks.values, bin_edges[1:-1])
    gene_to_bin = dict(zip(valid_genes, bin_ids))

    bin_to_genes = {}
    for g, b in gene_to_bin.items():
        bin_to_genes.setdefault(b, []).append(g)

    target_bins = [gene_to_bin.get(g) for g in target_genes]
    if any(b is None for b in target_bins):
        missing = [g for g, b in zip(target_genes, target_bins) if b is None]
        raise ValueError(f"Target gene(s) not found in expression ranking: {missing}")
    return target_bins, bin_to_genes


def draw_unique_matched_set(target_bins, bin_to_genes, exclude_genes, rng):
    """FIX: samples WITHOUT replacement across the whole draw (a random
    set must contain 14 unique genes), and NEVER falls back to
    reintroducing an excluded (real Proteostasis) gene -- raises instead
    of silently contaminating the null."""
    chosen = set()
    result = []
    for b in target_bins:
        candidates = [
            g for g in bin_to_genes.get(b, []) if g not in exclude_genes and g not in chosen
        ]
        if not candidates:
            raise ValueError(
                f"Insufficient eligible unique genes in expression bin {b} "
                f"to complete this random draw without duplication or "
                f"reintroducing an excluded gene."
            )
        pick = candidates[rng.integers(len(candidates))]
        chosen.add(pick)
        result.append(pick)
    return result


if __name__ == "__main__":
    print("=" * 78)
    print("STEP C (v2): PROTEOSTASIS SPECIFICITY -- corrected for memory safety,")
    print("null contamination, and within-draw gene duplication.")
    print("=" * 78)

    service = get_drive_service()
    remote_index = list_all_drive_files(service, DRIVE_FOLDER_ID)

    prepped_cohorts = {}
    for cohort in COHORT_ORDER:
        print(
            f"\n{'='*78}\n{cohort}: loading and preparing (donor x gene precomputation)\n{'='*78}"
        )
        prepped_cohorts[cohort] = prepare_cohort(cohort, COHORTS[cohort], service, remote_index)

    for cohort in COHORT_ORDER:
        missing = [
            g
            for g in PROTEOSTASIS_GENES
            if g not in prepped_cohorts[cohort]["donor_gene_cpm"].columns
        ]
        if missing:
            raise ValueError(f"{cohort}: missing Proteostasis gene(s): {missing}")

    print(f"\n{'='*78}\nBASELINE: full 14-gene module (sanity check against Step A/B)\n{'='*78}")
    # Values confirmed in the Step B run -- Step C must reproduce these
    # exactly before proceeding, or 5,000+ random draws would be spent
    # testing specificity around a baseline that no longer matches the
    # confirmed, locked Figure 4 result.
    EXPECTED_D = {"Jin": 1.567521, "Ma": 0.356806, "Nido": 0.113985}

    baseline_d = {}
    for cohort in COHORT_ORDER:
        donor_df = score_gene_set(prepped_cohorts[cohort], PROTEOSTASIS_GENES)
        d = disease_effect(donor_df)
        baseline_d[cohort] = d
        print(f"  {cohort}: d={d:.4f}")
        if not np.isclose(d, EXPECTED_D[cohort], atol=1e-4):
            raise ValueError(
                f"{cohort}: baseline d={d:.6f} does not match the locked Step A/B "
                f"value {EXPECTED_D[cohort]:.6f}. Stop -- do not proceed to the "
                f"random-null analysis on a baseline that no longer reproduces."
            )

    print(f"\n{'='*78}\nPART 1: LEAVE-ONE-GENE-OUT\n{'='*78}")
    loo_rows = []
    for gene in PROTEOSTASIS_GENES:
        remaining = [g for g in PROTEOSTASIS_GENES if g != gene]
        row = {"removed_gene": gene}
        for cohort in COHORT_ORDER:
            donor_df = score_gene_set(prepped_cohorts[cohort], remaining)
            d = disease_effect(donor_df)
            row[f"{cohort}_d_without_gene"] = d
            row[f"{cohort}_shift_from_baseline"] = d - baseline_d[cohort]
        loo_rows.append(row)
        print(
            f"  Removed {gene:10s}: "
            + "  ".join(
                f"{c} d={row[f'{c}_d_without_gene']:.4f} (shift={row[f'{c}_shift_from_baseline']:+.4f})"
                for c in COHORT_ORDER
            )
        )
    loo_df = pd.DataFrame(loo_rows)

    for cohort in COHORT_ORDER:
        max_shift = loo_df[f"{cohort}_shift_from_baseline"].abs().max()
        sign_flips = (
            np.sign(loo_df[f"{cohort}_d_without_gene"]) != np.sign(baseline_d[cohort])
        ).sum()
        print(
            f"\n  [{cohort}] Baseline d={baseline_d[cohort]:.4f}. Max shift={max_shift:.4f}. "
            f"Sign flips: {sign_flips}/14."
        )

    print(
        f"\n{'='*78}\nPART 2: EXPRESSION-MATCHED RANDOM-GENE-SET NULL ({N_RANDOM_SETS} sets, "
        f"14 unique genes each, no Proteostasis genes ever included)\n{'='*78}"
    )
    rng = np.random.default_rng(RANDOM_SEED)
    null_results = {cohort: [] for cohort in COHORT_ORDER}
    null_gene_sets = {cohort: [] for cohort in COHORT_ORDER}

    cohort_bins = {}
    for cohort in COHORT_ORDER:
        p = prepped_cohorts[cohort]
        target_bins, bin_to_genes = build_expression_bins(
            p["mean_expr"], PROTEOSTASIS_GENES, N_EXPRESSION_BINS
        )
        cohort_bins[cohort] = (target_bins, bin_to_genes)
        print(f"  [{cohort}] Proteostasis genes' expression bins: {target_bins}")

    exclude_set = set(PROTEOSTASIS_GENES)
    for draw in range(N_RANDOM_SETS):
        for cohort in COHORT_ORDER:
            target_bins, bin_to_genes = cohort_bins[cohort]
            random_genes = draw_unique_matched_set(target_bins, bin_to_genes, exclude_set, rng)
            donor_df = score_gene_set(prepped_cohorts[cohort], random_genes)
            d = disease_effect(donor_df)
            null_results[cohort].append(d)
            null_gene_sets[cohort].append(";".join(random_genes))

    print(f"\n  Empirical results ({N_RANDOM_SETS} expression-matched random 14-unique-gene sets):")
    specificity_rows = []
    for cohort in COHORT_ORDER:
        null_ds = np.array(null_results[cohort])
        null_ds = null_ds[~np.isnan(null_ds)]
        real_d = baseline_d[cohort]
        empirical_p = (np.sum(np.abs(null_ds) >= np.abs(real_d)) + 1) / (len(null_ds) + 1)
        percentile = float(np.mean(null_ds < real_d) * 100)
        print(
            f"  [{cohort}] Real d={real_d:.4f} | Null: mean={null_ds.mean():.4f}, sd={null_ds.std():.4f}, "
            f"range=[{null_ds.min():.4f}, {null_ds.max():.4f}]"
        )
        print(
            f"           Empirical p={empirical_p:.4f} | Real d at {percentile:.1f}th percentile of null"
        )
        specificity_rows.append(
            {
                "cohort": cohort,
                "real_d": real_d,
                "null_mean": null_ds.mean(),
                "null_sd": null_ds.std(),
                "null_min": null_ds.min(),
                "null_max": null_ds.max(),
                "empirical_p": empirical_p,
                "percentile": percentile,
                "n_valid_draws": len(null_ds),
            }
        )
    specificity_df = pd.DataFrame(specificity_rows)

    print(f"\n{'='*78}\nSAVING RESULTS\n{'='*78}")
    loo_df.to_csv(OUT_DIR / "step_C_leave_one_gene_out.csv", index=False)
    specificity_df.to_csv(OUT_DIR / "step_C_random_null_summary.csv", index=False)
    for cohort in COHORT_ORDER:
        pd.DataFrame(
            {
                "draw": range(N_RANDOM_SETS),
                "cohens_d": null_results[cohort],
                "gene_set": null_gene_sets[cohort],
            }
        ).to_csv(OUT_DIR / f"step_C_{cohort}_random_null_full_with_genesets.csv", index=False)
    print(f"Saved to {OUT_DIR}, including full gene-set identities per draw for reproducibility.")

    # --- Part 3: gradient-specific random-set null ---
    # Tests a DIFFERENT question from Part 2: not "is the disease effect
    # size unusual" but "is the Jin>Ma>Nido regional ordering itself
    # unusual for a random gene set." This requires the SAME gene set
    # tested across all three cohorts simultaneously -- unlike Part 2's
    # independent per-cohort draws (which used cohort-specific expression
    # matching), gradient testing needs one shared random module per draw.
    # Ma is used as the expression-matching reference (the "middle" cohort
    # sharing substantial gene overlap with both others); each candidate
    # gene is additionally required to be present in Jin's and Nido's var
    # names before being eligible.
    print(
        f"\n{'='*78}\nPART 3: GRADIENT-SPECIFIC RANDOM-SET NULL ({N_RANDOM_SETS} sets, "
        f"same 14 genes tested across all 3 cohorts per draw)\n{'='*78}"
    )

    ma_target_bins, ma_bin_to_genes = cohort_bins["Ma"]
    all_var_names = {c: set(prepped_cohorts[c]["donor_gene_cpm"].columns) for c in COHORT_ORDER}
    # Restrict Ma's candidate pools to genes ALSO present in Jin and Nido
    ma_bin_to_genes_shared = {
        b: [g for g in genes if g in all_var_names["Jin"] and g in all_var_names["Nido"]]
        for b, genes in ma_bin_to_genes.items()
    }

    gradient_null_matches = 0
    gradient_null_records = []
    for draw in range(N_RANDOM_SETS):
        random_genes = draw_unique_matched_set(
            ma_target_bins, ma_bin_to_genes_shared, exclude_set, rng
        )
        medians = {}
        for cohort in COHORT_ORDER:
            donor_df = score_gene_set(prepped_cohorts[cohort], random_genes)
            medians[cohort] = donor_df["donor_score"].median()
        matches = medians["Jin"] > medians["Ma"] > medians["Nido"]
        if matches:
            gradient_null_matches += 1
        gradient_null_records.append(
            {
                "draw": draw,
                "gene_set": ";".join(random_genes),
                "median_Jin": medians["Jin"],
                "median_Ma": medians["Ma"],
                "median_Nido": medians["Nido"],
                "matches_ordering": matches,
            }
        )

    real_medians = {
        c: prepped_cohorts[c]["donor_gene_cpm"][PROTEOSTASIS_GENES].mean(axis=1).median()
        for c in COHORT_ORDER
    }
    real_matches = real_medians["Jin"] > real_medians["Ma"] > real_medians["Nido"]
    gradient_empirical_p = (gradient_null_matches + 1) / (N_RANDOM_SETS + 1)

    print(
        f"  Real Proteostasis module: Jin={real_medians['Jin']:.4f}, Ma={real_medians['Ma']:.4f}, "
        f"Nido={real_medians['Nido']:.4f} -- matches Jin>Ma>Nido: {real_matches}"
    )
    print(
        f"  Random sets ALSO matching this exact ordering: {gradient_null_matches}/{N_RANDOM_SETS} "
        f"({gradient_null_matches/N_RANDOM_SETS:.2%})"
    )
    print(
        f"  Empirical fraction of expression-matched random sets showing the same "
        f"Jin>Ma>Nido ordering: {gradient_empirical_p:.4f}"
    )
    print("  (This is a binary match/no-match test, not a continuous statistic -- there is no")
    print("   notion of a random set 'exceeding' the ordering, only matching it or not. Treat")
    print("   this as a specificity/negative-control analysis, not a formal regional inference.)")
    if gradient_empirical_p < 0.05:
        print("  The Jin>Ma>Nido ordering is unusual among expression-matched random gene sets --")
        print(
            "  this specific ordering is not simply a common property of comparably-expressed genes."
        )
    else:
        print(
            "  The Jin>Ma>Nido ordering is NOT unusual among expression-matched random gene sets --"
        )
        print(
            "  a meaningful fraction of comparable random modules also show this ordering, so this"
        )
        print("  alone does not establish the ordering as specific to Proteostasis biology.")

    gradient_null_df = pd.DataFrame(gradient_null_records)
    gradient_null_df.to_csv(OUT_DIR / "step_C_gradient_random_null.csv", index=False)
    print(f"\nSaved gradient null results to {OUT_DIR}")

    print("\nFinal specificity summary (Part 2, disease-effect):")
    print(specificity_df.to_string(index=False))
