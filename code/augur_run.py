"""
augur_run.py

A single, fresh Augur run for Jin, Ma, and Nido — built to end the
four-way disagreement (manuscript text / FigS4_Augur_FIXED.png /
Ma_augur_results_FINAL.csv / Nido_augur_results.csv) by producing one
new, traceable result instead of asserting any of the old ones is right.

Two things this script does specifically because of what went wrong
before:

  1. RAW COUNTS ENFORCED, NOT ASSUMED. Every adata is checked for
     integer-like values before running Augur. If it looks already
     normalised (non-integer, small max), the run stops with an error
     instead of silently proceeding — this is exactly the failure mode
     described earlier (Augur run on log-normalised data gave every
     Jin cell type AUROC 0.997-1.000, a classic donor-identity
     overfitting signature, not real biology).

  2. BUILT-IN OVERFITTING CHECK. After each cohort's run, if more than
     half of its cell types score AUROC >= 0.95, the script prints a
     loud warning instead of treating the result as trustworthy. This
     is the same signature that flagged the earlier broken run — if it
     shows up again here, that's real evidence, not assumed.

Requires pertpy (pip install pertpy --break-system-packages).

Usage:
    python augur_run.py
    python augur_run.py --subsample 40000   # cap large cohorts for speed
"""

import os, io
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from pathlib import Path

DRIVE_FOLDER_ID = os.environ.get("LBD_DRIVE_FOLDER_ID", "")
TOKEN_PATH = Path("drive_token.json")
DATA_DIR = Path("./data_snrna")
OUT_DIR = Path("./results/augur_run")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_CONFIG = {
    "Jin": dict(
        remote_name="GSE235914_LBD_annotated_FIXED.h5ad",
        ct_col="broad_type",
        dx_col="condition",
        disease_keyword="lbd",
        control_keyword="control",
    ),
    "Ma": dict(
        remote_name="GSE253462_annotated.h5ad",
        ct_col="cell_type",
        dx_col="condition",
        disease_keyword="pd",
        control_keyword="control",
    ),
    "Nido": dict(
        remote_name="nido_analysis_ready.h5ad",
        ct_col="cell_type",
        dx_col="Diagnosis",
        disease_keyword="pd",
        control_keyword="control",
    ),
}

OVERFIT_AUROC_THRESHOLD = 0.95
OVERFIT_FRACTION_THRESHOLD = 0.5  # if >50% of cell types clear the threshold, warn
AUTO_SUBSAMPLE_THRESHOLD = 100_000  # cohorts larger than this get auto-capped
AUTO_SUBSAMPLE_CAP = 60_000  # same cap already used for Nido elsewhere in this project


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


def upload_to_drive(service, folder_id, local_path):
    from googleapiclient.http import MediaFileUpload

    local_path = Path(local_path)
    existing = (
        service.files()
        .list(
            q=f"name='{local_path.name}' and '{folder_id}' in parents and trashed = false",
            fields="files(id)",
        )
        .execute()
        .get("files", [])
    )
    media = MediaFileUpload(str(local_path), resumable=True)
    if existing:
        service.files().update(fileId=existing[0]["id"], media_body=media).execute()
    else:
        service.files().create(
            body={"name": local_path.name, "parents": [folder_id]}, media_body=media
        ).execute()


def verify_raw_counts(adata, cohort_name):
    """Checks .X first; if it's not raw counts, automatically checks
    adata.layers and adata.raw for a source that is, and copies that
    into .X if found — printing exactly what it did, so this is never
    a silent substitution. Only refuses to proceed if NOTHING raw-
    looking can be found anywhere on the object."""

    def _looks_raw(X):
        sample = X[:200, :200].toarray() if sps.issparse(X) else np.asarray(X[:200, :200])
        xmax = float(sample.max())
        looks_integer = np.allclose(sample, np.round(sample), atol=1e-6)
        return looks_integer and xmax >= 5, xmax, looks_integer

    ok, xmax, is_int = _looks_raw(adata.X)
    print(f"  [{cohort_name}] .X sample max={xmax:.3f}, integer-like={is_int}")
    if ok:
        return

    print(f"  [{cohort_name}] .X does not look like raw counts — checking layers/.raw...")
    for layer_name in adata.layers.keys():
        L = adata.layers[layer_name]
        ok, lmax, lint = _looks_raw(L)
        print(f"    layers['{layer_name}']: max={lmax:.3f}, integer-like={lint}")
        if ok:
            adata.X = L.copy()
            print(f"  [{cohort_name}] Using layers['{layer_name}'] as raw counts instead of .X")
            return

    if adata.raw is not None:
        ok, rmax, rint = _looks_raw(adata.raw.X)
        print(f"    adata.raw.X: max={rmax:.3f}, integer-like={rint}")
        if ok:
            adata.X = adata.raw.X.copy()
            print(f"  [{cohort_name}] Using adata.raw.X as raw counts instead of .X")
            return

    raise ValueError(
        f"{cohort_name}: no raw-counts source found in .X, any layer, or .raw. "
        f"Available layers: {list(adata.layers.keys())}, .raw present: {adata.raw is not None}. "
        f"This file may only have normalised data — check with whoever generated it "
        f"whether raw counts were ever retained."
    )


def run_augur_for_cohort(cohort, cfg, path, subsample_n=None, n_threads=1):
    import pertpy as pt

    print(f"\n{'=' * 70}\n{cohort}\n{'=' * 70}")
    adata = sc.read_h5ad(path)
    print(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    verify_raw_counts(adata, cohort)

    ct_col, dx_col = cfg["ct_col"], cfg["dx_col"]
    if ct_col not in adata.obs.columns or dx_col not in adata.obs.columns:
        raise KeyError(
            f"{cohort}: ct_col='{ct_col}' or dx_col='{dx_col}' not found. "
            f"Available: {list(adata.obs.columns)}"
        )

    raw_dx = adata.obs[dx_col].unique().tolist()
    dk, ck = cfg["disease_keyword"], cfg["control_keyword"]
    group_map = {
        v: ("control" if ck in str(v).lower() else "disease" if dk in str(v).lower() else None)
        for v in raw_dx
    }
    unmapped = [v for v, g in group_map.items() if g is None]
    if unmapped:
        raise ValueError(
            f"{cohort}: could not map these dx values: {unmapped}. "
            f"Fix disease_keyword/control_keyword in COHORT_CONFIG."
        )
    adata.obs["augur_label"] = adata.obs[dx_col].map(group_map)
    print(f"  Label counts: {adata.obs['augur_label'].value_counts().to_dict()}")

    if subsample_n and subsample_n > 0 and adata.n_obs > subsample_n:
        sc.pp.subsample(adata, n_obs=subsample_n, random_state=42)
        print(f"  Subsampled to {adata.n_obs:,} cells for speed (explicit --subsample)")
    elif subsample_n is None and adata.n_obs > AUTO_SUBSAMPLE_THRESHOLD:
        sc.pp.subsample(adata, n_obs=AUTO_SUBSAMPLE_CAP, random_state=42)
        print(
            f"  AUTO-SUBSAMPLED to {adata.n_obs:,} cells — this cohort exceeded "
            f"{AUTO_SUBSAMPLE_THRESHOLD:,} cells, which previously exhausted disk "
            f"space during Augur's cross-validation. Override with --subsample N "
            f"to set a different cap, or --subsample 0 to force the full dataset "
            f"(only if you've confirmed there's enough disk space)."
        )
    elif subsample_n == 0:
        print(
            f"  No subsampling applied (--subsample 0) — running full {adata.n_obs:,} "
            f"cells. This previously exhausted disk space for large cohorts; make "
            f"sure there's room before trusting this to finish."
        )

    # Standard normalisation for Augur's internal use — pertpy expects
    # this step to happen inside its own pipeline, not pre-normalised
    # input; raw counts go in, Augur handles scaling internally.
    ag = pt.tl.Augur("random_forest_classifier")
    loaded = ag.load(
        adata,
        label_col="augur_label",
        cell_type_col=ct_col,
        condition_label="control",
        treatment_label="disease",
    )
    print("  Running Augur (this can take several minutes for large cohorts)...")
    v_adata, results = ag.predict(loaded, subsample_size=20, n_threads=n_threads, random_state=42)

    # Don't assume the shape — pertpy returns multiple metrics per cell
    # type (auc, accuracy, precision, recall, f1, ...), not just AUC.
    # Forcing a fixed 2-column rename broke on exactly this earlier.
    summary_raw = results["summary_metrics"].T.reset_index()
    print(f"  Raw summary_metrics columns: {summary_raw.columns.tolist()}")
    cell_type_col = summary_raw.columns[0]  # the former index, always first after reset_index
    auc_col = next((c for c in summary_raw.columns if "auc" in str(c).lower()), None)
    if auc_col is None:
        raise KeyError(
            f"No AUC-like column found among {summary_raw.columns.tolist()} — "
            f"pertpy's output format may have changed; inspect summary_raw directly."
        )
    summary = summary_raw[[cell_type_col, auc_col]].copy()
    summary.columns = ["Cell_Type", "AUC"]
    summary = summary.sort_values("AUC", ascending=False).reset_index(drop=True)
    summary["Cohort"] = cohort

    n_high = (summary["AUC"] >= OVERFIT_AUROC_THRESHOLD).sum()
    frac_high = n_high / len(summary)
    print(f"\n  {cohort} AUGUR RESULTS (sorted by AUROC descending):")
    print(summary[["Cell_Type", "AUC"]].to_string(index=False))
    print(
        f"\n  Cell types with AUROC >= {OVERFIT_AUROC_THRESHOLD}: {n_high}/{len(summary)} "
        f"({frac_high:.0%})"
    )
    if frac_high > OVERFIT_FRACTION_THRESHOLD:
        print(
            f"  *** WARNING: >{OVERFIT_FRACTION_THRESHOLD:.0%} of cell types cleared "
            f"{OVERFIT_AUROC_THRESHOLD} AUROC. This is the exact signature of the"
        )
        print("  *** earlier donor-identity overfitting bug. Do NOT trust this result")
        print("  *** without checking: raw counts confirmed above, but also check for")
        print("  *** batch/donor leakage into cell-type labels, tiny cell-type Ns, or")
        print("  *** near-perfect case/control separation by donor alone.")
    else:
        print("  No overfitting signature detected — result looks plausible.")

    del adata
    import gc

    gc.collect()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Cap cells per cohort. Default: auto-caps any cohort over "
        "100,000 cells to 60,000 (this is what previously exhausted "
        "disk space). Pass an explicit number to override, or 0 to "
        "force the full dataset with no cap.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Parallel workers for Augur. Defaults to 1 (safest — avoids "
        "the memmap disk-space error seen with more workers on large "
        "cohorts). Raise this for speed once you've confirmed there's "
        "enough disk space.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PULLING FILES FROM GOOGLE DRIVE")
    print("=" * 70)
    service = get_drive_service()
    remote_index = list_all_drive_files(service, DRIVE_FOLDER_ID)

    paths = {}
    for cohort, cfg in COHORT_CONFIG.items():
        fname = cfg["remote_name"]
        if fname not in remote_index:
            close = [nm for nm in remote_index if cohort.lower() in nm.lower()]
            raise FileNotFoundError(f"'{fname}' not found. Closest: {close}")
        local_path = DATA_DIR / fname
        download_file(service, remote_index[fname], local_path)
        paths[cohort] = local_path

    all_results = []
    for cohort, cfg in COHORT_CONFIG.items():
        try:
            summary = run_augur_for_cohort(
                cohort, cfg, paths[cohort], subsample_n=args.subsample, n_threads=args.n_threads
            )
            all_results.append(summary)
        except Exception as e:
            print(f"\n  *** {cohort} FAILED: {e}")
            import traceback

            traceback.print_exc()

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = OUT_DIR / f"augur_fresh_results_{timestamp}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n{'=' * 70}\nSaved: {out_path}\n{'=' * 70}")
        print("\nFINAL SUMMARY (top cell type per cohort):")
        for cohort in COHORT_CONFIG:
            sub = combined[combined["Cohort"] == cohort]
            if len(sub):
                top = sub.iloc[0]
                print(f"  {cohort:6s}: {top['Cell_Type']} (AUC={top['AUC']:.3f})")

        print("\nUploading to Drive...")
        upload_to_drive(service, DRIVE_FOLDER_ID, out_path)
        print("Done. This filename is timestamped — compare it directly against")
        print("Ma_augur_results_FINAL.csv, Nido_augur_results.csv, and the manuscript")
        print("text to see which one this fresh run actually agrees with, if any.")
    else:
        print("\nNo cohorts completed successfully — see errors above.")
