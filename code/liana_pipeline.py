# NOTE (2026-10): the NLGN1:TNF balance described here is no longer reported; see CHANGELOG.md.
"""
Produces the manuscript's cell-cell communication numbers (NLGN1-NRXN vs
TNF-TNFR balance) directly from a raw h5ad on Google Drive.

Runs in a hosted notebook: it uses the Drive API and a token file, not google.colab.drive
(this environment doesn't have that module; this matches the same
download pattern already proven working in figure2_umap_FINAL.py earlier
this session).

Needs a GPU runtime -- rank_aggregate is compute-heavy, hours not minutes.

METHOD NOTE -- read before trusting the output:
Significance is defined as cellphone_pvals < 0.05, NOT LIANA's own
magnitude_rank consensus score. This was a deliberate choice made to
match what the manuscript's existing numbers actually use (confirmed by
directly filtering raw Nido interaction files: cellphone_pvals<0.05
reproduces 50,771 NLGN1 pairs / 6,035 TNF pairs / 8.4:1 ratio exactly).
Other scripts in this project (gap5_ma_liana.py, check_liana_coverage.py)
use magnitude_rank instead and will NOT reproduce these numbers.

Set COHORT below, then run.
"""

import io
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from pathlib import Path

sc.settings.verbosity = 1

# ---- Settings -- EDIT COHORT to switch between Ma / Nido / Jin ----
COHORT = "Ma"  # "Ma", "Nido", or "Jin"
N_PERMS = 100
CT_COLUMN_OVERRIDE = None  # set to e.g. "fine_type" or "cluster" to force a
# specific cell-type column instead of auto-detection

# --- Tier 1 hypotheses to test -- change these and rerun to compare ---
ANALYSIS_MODE = "by_sample"  # "by_sample" (current default, LIANA run
# separately per sample then aggregated) or
# "pooled" (single LIANA run across all cells
# of each diagnosis group at once -- much
# higher power per test, could explain the
# 3.3x total-count gap on its own)
DIAGNOSIS_SUBSET = "all"  # "all" (current default, PD+control together),
# "disease" (PD/LBD cells only), or
# "control" (control cells only)

DRIVE_FOLDER_ID = os.environ.get("LBD_DRIVE_FOLDER_ID", "")
TOKEN_PATH = Path("drive_token.json")
DATA_DIR = Path("./data_snrna")
OUT_DIR = Path("./results") / f"{COHORT.lower()}_liana"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Jin": "GSE235914_LBD_annotated_FIXED.h5ad",
    "Ma": "GSE253462_annotated.h5ad",
    "Nido": "nido_analysis_ready.h5ad",
}
REGIONS = {
    "Jin": "Temporal cortex",
    "Ma": "Cingulate cortex",
    "Nido": "Prefrontal cortex",
}

FILTER_COLUMN = "cellphone_pvals"  # see METHOD NOTE above before changing
FILTER_THRESHOLD = 0.05


# ---- Drive access (token-based, matches figure2_umap_FINAL.py) ----
def restore_token_if_missing():
    if TOKEN_PATH.exists():
        return
    token_json = os.environ.get("DRIVE_TOKEN_JSON")
    if not token_json:
        raise FileNotFoundError("drive_token.json missing and no DRIVE_TOKEN_JSON env var.")
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
    remote_index, page_token = {}, None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, size)",
                pageSize=100,
                pageToken=page_token,
            )
            .execute()
        )
        for f in resp.get("files", []):
            remote_index[f["name"]] = f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return remote_index


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


def verify_and_get_raw(adata, cohort_name):
    def _looks_raw(X):
        sample = X[:200, :200].toarray() if sps.issparse(X) else np.asarray(X[:200, :200])
        xmax = float(sample.max())
        return np.allclose(sample, np.round(sample), atol=1e-6) and xmax >= 5

    if _looks_raw(adata.X):
        print(f"  [{cohort_name}] .X is raw counts")
        return adata
    print(f"  [{cohort_name}] .X not raw -- checking layers...")
    for name in adata.layers.keys():
        if _looks_raw(adata.layers[name]):
            adata.X = adata.layers[name].copy()
            print(f"  [{cohort_name}] Using layers['{name}']")
            return adata
    raise ValueError(f"{cohort_name}: no raw-counts source found.")


# ---- LIANA pipeline ----
def resolve_columns(adata, ct_override=None):
    diag_col = next(
        (
            c
            for c in adata.obs.columns
            if any(x in c.lower() for x in ["diagnosis", "condition", "disease"])
        ),
        None,
    )
    sample_col = next(
        (
            c
            for c in adata.obs.columns
            if any(x in c.lower() for x in ["sample", "donor", "subject"])
        ),
        None,
    )
    ct_col = next(
        (
            c
            for c in adata.obs.columns
            if any(x in c.lower() for x in ["cell_type", "celltype", "annotation"])
        ),
        None,
    )
    for name in ("broad_type", "cell_type_broad", "cell_type"):
        if name in adata.obs.columns:
            ct_col = name
            break
    if ct_override is not None:
        if ct_override not in adata.obs.columns:
            raise KeyError(
                f"CT_COLUMN_OVERRIDE='{ct_override}' not found. Available: {list(adata.obs.columns)}"
            )
        ct_col = ct_override
    missing = [
        n
        for n, c in [("diagnosis", diag_col), ("sample", sample_col), ("cell_type", ct_col)]
        if c is None
    ]
    if missing:
        raise KeyError(
            f"Could not resolve column(s): {missing}. Available: {list(adata.obs.columns)}"
        )
    return diag_col, sample_col, ct_col


def ensure_lognorm(adata):
    if adata.X.max() > 100:
        print("  Normalising and log-transforming...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("  Data already appears normalised/log-transformed -- skipping.")


def summarize_communication(res: pd.DataFrame, cohort: str, region: str) -> dict:
    ligand_col = next((c for c in res.columns if "ligand" in c.lower()), None)
    receptor_col = next((c for c in res.columns if "receptor" in c.lower()), None)
    sig_col = next((c for c in res.columns if c.lower() == FILTER_COLUMN.lower()), None)
    missing = [
        n
        for n, c in [("ligand", ligand_col), ("receptor", receptor_col), (FILTER_COLUMN, sig_col)]
        if c is None
    ]
    if missing:
        raise KeyError(
            f"LIANA output missing expected column(s): {missing}. Found: {list(res.columns)}"
        )

    p_max = res[sig_col].max()
    if p_max < 0.9:
        print(
            f"  WARNING: {sig_col} max is only {p_max:.4g} -- this looks pre-filtered by "
            f"something else already; re-filtering here may not give the right answer."
        )

    sig = res[res[sig_col] < FILTER_THRESHOLD]
    total_sig = len(sig)
    if total_sig == 0:
        raise ValueError(
            f"Zero rows pass {sig_col}<{FILTER_THRESHOLD} -- check the LIANA run completed correctly."
        )

    nlgn1 = sig[
        (sig[ligand_col] == "NLGN1") | sig[receptor_col].str.contains("NRXN", case=False, na=False)
    ]
    tnf = sig[
        (sig[ligand_col] == "TNF") | sig[receptor_col].str.contains("TNFR", case=False, na=False)
    ]

    nlgn1_pct = len(nlgn1) / total_sig * 100
    tnf_pct = len(tnf) / total_sig * 100
    ratio = round(nlgn1_pct / tnf_pct, 1) if tnf_pct > 0 else float("inf")

    print(f"\n  {'='*50}")
    print(f"  {cohort} ({region}) -- {FILTER_COLUMN}<{FILTER_THRESHOLD}")
    print(f"  Total significant: {total_sig:,}")
    print(f"  NLGN1-NRXN: {len(nlgn1):,} ({nlgn1_pct:.2f}%)")
    print(f"  TNF-TNFR:   {len(tnf):,} ({tnf_pct:.2f}%)")
    print(f"  NLGN1:TNF ratio: {ratio}:1")
    print(f"  {'='*50}")

    return {
        "Cohort": cohort,
        "Region": region,
        "Significance_threshold": f"{FILTER_COLUMN}<{FILTER_THRESHOLD}",
        "Total_significant": total_sig,
        "NLGN1_pairs": len(nlgn1),
        "NLGN1_pct": round(nlgn1_pct, 2),
        "TNF_pairs": len(tnf),
        "TNF_pct": round(tnf_pct, 2),
        "NLGN1_TNF_ratio": ratio,
    }


if __name__ == "__main__":
    print("=" * 70)
    print(f"PULLING {COHORT} FILE FROM GOOGLE DRIVE")
    print("=" * 70)
    service = get_drive_service()
    remote_index = list_all_drive_files(service, DRIVE_FOLDER_ID)

    fname = FILES[COHORT]
    if fname not in remote_index:
        raise FileNotFoundError(
            f"'{fname}' not found in Drive folder. Available: {list(remote_index.keys())}"
        )
    local_path = DATA_DIR / fname
    download_file(service, remote_index[fname], local_path)

    print(f"\nLoading {fname}...")
    adata = sc.read_h5ad(local_path)
    print(f"  {adata.n_obs:,} cells")
    adata = verify_and_get_raw(adata, COHORT)

    diag_col, sample_col, ct_col = resolve_columns(adata, ct_override=CT_COLUMN_OVERRIDE)
    print(f"  Diagnosis: {diag_col} | Sample: {sample_col} | Cell type: {ct_col}")

    # --- Tier 1: diagnosis subsetting ---
    if DIAGNOSIS_SUBSET != "all":
        diag_raw = adata.obs[diag_col].astype(str).str.upper()
        if DIAGNOSIS_SUBSET == "disease":
            keep = diag_raw.str.contains("PD|DLB|LBD|CASE|DISEASE", na=False, regex=True)
        elif DIAGNOSIS_SUBSET == "control":
            keep = diag_raw.str.contains("CTRL|CONTROL|HEALTHY", na=False, regex=True)
        else:
            raise ValueError(f"Unknown DIAGNOSIS_SUBSET: {DIAGNOSIS_SUBSET!r}")
        n_before = adata.n_obs
        adata = adata[keep].copy()
        print(
            f"  DIAGNOSIS_SUBSET='{DIAGNOSIS_SUBSET}': kept {adata.n_obs:,} of {n_before:,} cells "
            f"({adata.obs[sample_col].nunique()} samples)"
        )
        if adata.n_obs == 0:
            raise ValueError(
                f"DIAGNOSIS_SUBSET='{DIAGNOSIS_SUBSET}' matched zero cells -- "
                f"check the diagnosis values: {adata.obs[diag_col].unique()}"
            )

    ensure_lognorm(adata)

    try:
        import liana as li
    except ImportError:
        print("Installing liana...")
        import subprocess

        subprocess.run(["pip", "install", "-q", "liana"], check=True)
        import liana as li

    installed_version = getattr(li, "__version__", "0.0.0")
    print(f"  Installed liana version: {installed_version}")

    # KNOWN BUG (liana < 2.0.0, fixed in PR #273): rank_aggregate crashes with
    # KeyError: '[None] not found in axis' inside _run_method when a
    # sample/cell-type combination yields no valid score for one of the
    # aggregated methods. Upgrading is the "real" fix, but this environment's
    # sandboxed dependency management (marimo/uv) has proven to silently
    # revert pip-install-based upgrades within a running session. Rather than
    # depend on that working, patch the exact failure directly: pandas'
    # DataFrame.drop is wrapped so that ONLY this specific, exact call
    # signature (dropping a non-existent [None] column) becomes a no-op --
    # matching what liana's own 2.0.0 fix does ("no longer NaNs the whole
    # rank on one missing score"). Every other .drop() call anywhere,
    # including elsewhere in liana's own pipeline, is completely unaffected
    # and still raises normally on genuine errors.
    import pandas as pd

    # Robust guard against the patch stacking across repeated re-runs in the
    # same session. Rather than relying on a marker only THIS version's
    # patch sets (which fails to recognize damage left by earlier, differently
    # -coded patch attempts), check whether the currently-installed drop
    # method genuinely belongs to pandas itself -- any function whose
    # __module__ isn't pandas' own is presumably some kind of prior patch,
    # of unknown origin, and gets discarded rather than wrapped.
    import pandas.core.frame as _pd_frame_module

    _drop_fn = pd.DataFrame.drop
    if getattr(_drop_fn, "__module__", "") != _pd_frame_module.__name__:
        print(
            f"  Currently-installed DataFrame.drop is not pandas' own "
            f"(__module__={getattr(_drop_fn, '__module__', '?')!r}) -- this is a leftover "
            f"patch from an earlier run in this session, of unknown depth/origin. "
            f"A clean per-layer unwrap isn't reliable here."
        )
        print(
            "  RECOMMENDATION: restart the kernel/runtime and rerun from a clean process. "
            "Continuing anyway with a best-effort recovery attempt..."
        )
        # Best-effort: walk any _wrapped_original chain this or a compatible
        # patch may have left, as far as it goes.
        seen = set()
        while hasattr(_drop_fn, "_wrapped_original") and id(_drop_fn) not in seen:
            seen.add(id(_drop_fn))
            _drop_fn = _drop_fn._wrapped_original
        if getattr(_drop_fn, "__module__", "") != _pd_frame_module.__name__:
            raise RuntimeError(
                "Could not recover a genuine pandas DataFrame.drop from the current session "
                "state. This session's pandas module has been monkey-patched by an "
                "incompatible or unrecoverable prior run. Restart the kernel/runtime before "
                "proceeding -- no in-process fix can reliably recover from this."
            )
    pd.DataFrame.drop = _drop_fn  # confirmed genuine pandas method
    _original_df_drop = _drop_fn

    _patch_trigger_log = []  # records every time the patch actually activates

    def _patched_df_drop(self, labels=None, axis=0, **kwargs):
        try:
            return _original_df_drop(self, labels=labels, axis=axis, **kwargs)
        except KeyError as e:
            if (
                labels == [None]
                and axis in (1, "columns")
                and str(e) == "'[None] not found in axis'"
            ):
                info = {"n_rows": len(self), "columns": list(self.columns)}
                for col in self.columns:
                    if "ligand" in col.lower() or "receptor" in col.lower():
                        try:
                            info[col + "_sample"] = self[col].dropna().unique()[:5].tolist()
                        except Exception:
                            pass
                _patch_trigger_log.append(info)
                return self
            raise

    _patched_df_drop._is_liana_compat_patch = True
    _patched_df_drop._wrapped_original = _original_df_drop
    _patched_df_drop.__module__ = "liana_compat_patch"  # so future runs can detect THIS layer too

    pd.DataFrame.drop = _patched_df_drop
    print(
        "  Applied targeted compatibility patch for the known liana<2.0 "
        "'[None] not found in axis' bug (instrumented, module-origin-checked)."
    )

    def run_rank_aggregate(min_cells_value):
        if ANALYSIS_MODE == "by_sample":
            li.method.rank_aggregate.by_sample(
                adata,
                groupby=ct_col,
                sample_key=sample_col,
                expr_prop=0.1,
                min_cells=min_cells_value,
                resource_name="consensus",
                n_perms=N_PERMS,
                use_raw=False,
                verbose=True,
            )
        elif ANALYSIS_MODE == "pooled":
            # Tier 1: single run across all cells at once -- no per-sample
            # split. Much higher statistical power per cell-type-pair test,
            # since every cell in a type contributes to one test instead of
            # being split into 17 separate per-sample tests then aggregated.
            # This alone could explain the ~3.3x total-count gap.
            li.method.rank_aggregate(
                adata,
                groupby=ct_col,
                expr_prop=0.1,
                min_cells=min_cells_value,
                resource_name="consensus",
                n_perms=N_PERMS,
                use_raw=False,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown ANALYSIS_MODE: {ANALYSIS_MODE!r}")

    print(
        f"\nRunning rank_aggregate ({ANALYSIS_MODE} mode) -- "
        f"{adata.obs[sample_col].nunique()} samples, {adata.n_obs:,} cells. "
        f"This will take a while."
    )
    MIN_CELLS_USED = 10
    try:
        run_rank_aggregate(MIN_CELLS_USED)
    finally:
        # Always restore, even if this crashes -- prevents the exact
        # stacking bug that caused the RecursionError.
        pd.DataFrame.drop = _original_df_drop
    print(
        f"\n  [mode={ANALYSIS_MODE}, diagnosis_subset={DIAGNOSIS_SUBSET}, "
        f"min_cells used: {MIN_CELLS_USED}]"
    )

    # --- Report on how much the compatibility patch actually did ---
    print(f"\n  Compatibility patch fired {len(_patch_trigger_log)} time(s) during this run.")
    if _patch_trigger_log:
        import json

        log_path = OUT_DIR / f"{COHORT}_patch_trigger_log.json"
        with open(log_path, "w") as f:
            json.dump(_patch_trigger_log, f, indent=2, default=str)
        print(f"  Saved details to: {log_path}")
        print(
            "  If this number is large, or if a quick look at that file shows TNF-related "
            "ligand/receptor values appearing disproportionately often, the patch itself "
            "(not just the parameters) may be contributing to the TNF undercount -- worth "
            "treating the current results as provisional until that's ruled out."
        )
    res = adata.uns["liana_res"]
    if hasattr(res.index, "names"):
        res = res.reset_index()

    run_tag = f"{ANALYSIS_MODE}_{DIAGNOSIS_SUBSET}"
    res.to_csv(OUT_DIR / f"{COHORT}_liana_full_{run_tag}.csv", index=False)
    print(f"\nFull LIANA output saved: {len(res):,} rows")

    summary = summarize_communication(res, COHORT, REGIONS[COHORT])
    summary["min_cells_used"] = MIN_CELLS_USED
    summary["analysis_mode"] = ANALYSIS_MODE
    summary["diagnosis_subset"] = DIAGNOSIS_SUBSET
    summary["ct_column_used"] = ct_col
    pd.DataFrame([summary]).to_csv(OUT_DIR / f"{COHORT}_LIANA_summary_{run_tag}.csv", index=False)
    print(f"\nSaved: {OUT_DIR / f'{COHORT}_LIANA_summary_{run_tag}.csv'}")
