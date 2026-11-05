"""Recompute the decoupleR / CollecTRI transcription-factor activities for ALL eight factors in the Ma and Nido cohorts, with the method the manuscript describes.

METHOD (the one stated in the Methods): decoupleR's univariate linear model (ULM) scores every nucleus on log-normalised expression (log1p of counts per 10,000, from the raw
counts), the scores of each transcription factor are averaged within each donor, and disease and control donors are compared with a two-sided Mann-Whitney U test. Each cohort is
analysed on its own; nothing is pooled.

SCOPE: secondary_analyses.py checks the two decoupleR values quoted in the Results (Nido ATF4 and Ma HSF1). This script computes all eight factors in
both cohorts, prints the decoupleR version (the Methods must state the version that produced the numbers), puts the results next to the values printed
in the manuscript, and flags a Mann-Whitney p-value that equals the p-value of a different analysis in the same cohort.

MEMORY: decoupleR turns the data it is given into a dense float64 array (277,831 nuclei x 33,000 genes = 73 GB for Nido), which kills a notebook kernel. ULM scores every nucleus
independently, so the nuclei are processed in batches with identical results. Do NOT narrow the genes instead: ULM uses the whole transcriptome as background.

Paste it into one notebook cell, or run it from a terminal:
  python decoupler_tf_activity.py                  reads data_snrna/ in the working folder, or downloads the two files from Google Drive
  python decoupler_tf_activity.py --local-only     never uses Google Drive
  python decoupler_tf_activity.py --cohorts Ma     one cohort only
It needs decoupler 2.x (dc.mt.ulm, dc.op.collectri). It writes results/decoupler_all_tfs/decoupler_all_tfs.csv.
"""

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats

# ---------------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------------
FILES = {"Ma": "GSE253462_annotated.h5ad", "Nido": "nido_analysis_ready.h5ad"}
DRIVE_FOLDER_ID = os.environ.get(
    "LBD_DRIVE_FOLDER_ID", ""
)  # the Google Drive folder that holds your input files (the long code at the end of its web address)
TFS = ["ATF4", "TP53", "NFE2L2", "XBP1", "ATF6", "HSF1", "STAT3", "NFKB1"]
BATCH = 5000  # nuclei per decoupleR call; any value gives the same answer

# Values printed in the manuscript: (disease minus control, p). QUOTED = decoupleR (Results); PANEL = curated target-gene panel (Results and Fig. 5B/C).
QUOTED = {("Nido", "ATF4"): (0.011, 0.782), ("Ma", "HSF1"): (0.793, 0.114)}
PANEL = {
    ("Nido", "ATF4"): (0.261, 0.156), ("Nido", "HSF1"): (0.001, 0.782), ("Nido", "NFKB1"): (None, 0.347), ("Nido", "TP53"): (None, 0.231),
    ("Ma", "ATF4"): (0.497, 0.114), ("Ma", "HSF1"): (0.755, 0.139), ("Ma", "NFKB1"): (None, 0.743), ("Ma", "TP53"): (None, 0.963),
}  # fmt: skip

OVERRIDE = {
    "donor": None,
    "diagnosis": None,
}  # set a column name here if automatic detection stops with an error
DONOR_CANDS = [
    "sample",
    "Sample",
    "sample_id",
    "SampleID",
    "donor",
    "Donor",
    "donor_id",
    "patient",
    "Patient",
    "subject",
    "individual",
    "orig.ident",
]
DIAG_CANDS = [
    "condition",
    "Condition",
    "diagnosis",
    "Diagnosis",
    "disease",
    "Disease",
    "Dx",
    "dx",
    "group",
    "Group",
    "status",
]
IN_NOTEBOOK = (
    "marimo" in sys.modules or "ipykernel" in sys.modules
)  # a notebook has its own command-line arguments, which must be ignored


def default_workdir():
    """The folder that holds data_snrna/: /marimo in a hosted notebook, otherwise the folder you are in."""
    return "/marimo" if Path("/marimo").is_dir() else "."


# ---------------------------------------------------------------------------------------------------------------------
# Finding the right columns and the raw counts (the same rules as the other scripts of this project)
# ---------------------------------------------------------------------------------------------------------------------
def looks_like_integer_counts(matrix, max_values=200000):
    """True if (almost) all stored values are whole numbers. Raw counts are integers; log-normalised expression essentially never is."""
    values = np.asarray(matrix.data if sps.issparse(matrix) else np.asarray(matrix).ravel())
    if values.size == 0:
        return False
    if values.size > max_values:
        values = values[
            np.random.default_rng(0).choice(values.size, size=max_values, replace=False)
        ]
    nonzero = values[values != 0]
    return bool(nonzero.size) and float(np.mean(np.abs(nonzero - np.round(nonzero)) < 1e-6)) > 0.99


def classify_diagnosis(values):
    """Map free-text diagnosis labels to PD (disease) or Control. It refuses to guess."""
    mapping = {}
    for value in pd.unique(pd.Series(values).astype(str)):
        text = value.strip().lower()
        if re.match(r"^(control|ctrl|ctl|healthy|hc|normal|non[- ]?pd|non[- ]?dlb)", text):
            mapping[value] = "Control"
        elif re.match(r"^(pd|dlb|lbd|lewy|dementia|parkinson|case|disease|patient)", text):
            mapping[value] = "PD"
        else:
            raise ValueError(
                f"Cannot classify the diagnosis label {value!r}. Set OVERRIDE['diagnosis'] or extend classify_diagnosis()."
            )
    if set(mapping.values()) != {"PD", "Control"}:
        raise ValueError(
            f"The labels did not give both a disease group and a control group: {mapping}"
        )
    return mapping


def resolve_columns(adata):
    """Return (diagnosis column, donor column, diagnosis per nucleus as PD/Control)."""
    diagnosis = None
    for column in [OVERRIDE["diagnosis"]] if OVERRIDE["diagnosis"] else DIAG_CANDS:
        if column in adata.obs.columns:
            try:
                classify_diagnosis(adata.obs[column])
                diagnosis = column
                break
            except ValueError:
                continue
    if diagnosis is None:
        raise KeyError(f"No usable diagnosis column. Columns present: {list(adata.obs.columns)}")
    dx = adata.obs[diagnosis].astype(str).map(classify_diagnosis(adata.obs[diagnosis]))
    donor = None
    for column in [OVERRIDE["donor"]] if OVERRIDE["donor"] else DONOR_CANDS:
        if column in adata.obs.columns:
            per_donor = (
                pd.DataFrame({"d": adata.obs[column].astype(str).values, "x": dx.values})
                .groupby("d")["x"]
                .nunique()
            )
            if 4 <= len(per_donor) <= 80 and (per_donor == 1).all():  # one diagnosis per donor
                donor = column
                break
    if donor is None:
        raise KeyError(
            f"No donor column with one diagnosis per donor. Columns present: {list(adata.obs.columns)}"
        )
    return diagnosis, donor, dx.values


def find_raw_counts(adata):
    """The matrix of raw counts: in X, in a layer, or in .raw. It stops rather than use normalised data."""
    candidates = [("X", adata.X)] + [(f"layers[{k!r}]", adata.layers[k]) for k in adata.layers]
    if adata.raw is not None:
        candidates.append(("raw.X", adata.raw.X))
    for name, matrix in candidates:
        if matrix.shape[1] == adata.n_vars and looks_like_integer_counts(matrix):
            return matrix, name
    raise ValueError(
        "No raw-count matrix found in X, layers or raw; refusing to continue on normalised data."
    )


# ---------------------------------------------------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------------------------------------------------
def _usable(module):
    """decoupler 2.x has dc.mt (methods) and dc.op (resources); an old 1.x copy or an empty leftover folder has neither."""
    return hasattr(module, "mt") and hasattr(module, "op")


def ensure_decoupler(install=True):
    """Import decoupler 2.x, installing or upgrading it first if it is missing or unusable (pip, then uv, as the project's other notebook cells do).
    Returns the module, or None after saying why not."""
    problem = "decoupler is not installed. "
    try:
        import decoupler as dc
    # Deliberately broad: a missing package and a broken old copy both mean 'not usable'.
    except Exception as error:  # noqa: BLE001
        dc = None
        if not isinstance(error, ImportError):
            problem = f"decoupler could not be imported ({type(error).__name__}: {error}). "
    if dc is not None and _usable(dc):
        return dc
    have = (
        problem
        if dc is None
        else f"decoupler {getattr(dc, '__version__', '(an empty leftover folder)')} is installed but this script needs decoupler 2.x (dc.mt.ulm, dc.op.collectri). "
    )
    manual = "Install it yourself with:  pip install -U 'decoupler>=2'   (or: uv pip install -U 'decoupler>=2'), then restart the kernel and run this cell again."
    if not install:
        print("STOPPED: " + have + "Automatic installation is switched off. " + manual)
        return None
    print(have + "Installing decoupler>=2 (about a minute)...")
    stale = {
        name: module
        for name, module in sys.modules.items()
        if name == "decoupler" or name.startswith("decoupler.")
    }
    problems = []
    for command in (
        [sys.executable, "-m", "pip", "install", "-q", "-U", "decoupler>=2"],
        ["uv", "pip", "install", "-q", "--python", sys.executable, "-U", "decoupler>=2"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            problems.append(f"{command[0]}: not found")
            continue
        if result.returncode == 0:
            break
        problems.append(
            f"{' '.join(command[:3])}: {(result.stderr or result.stdout).strip()[-300:]}"
        )
    else:
        print(
            "STOPPED: decoupler could not be installed automatically.\n  "
            + "\n  ".join(problems)
            + "\n"
            + manual
        )
        return None
    importlib.invalidate_caches()
    for (
        name,
        module,
    ) in stale.items():  # forget the old or empty copy that this session already loaded
        if sys.modules.get(name) is module:
            del sys.modules[name]
    try:
        import decoupler as dc
    except ImportError as error:
        print(
            f"decoupler was installed but cannot be loaded in this running session ({error}). Restart the kernel (the files on disk are kept) and run this cell again."
        )
        return None
    if not _usable(dc):
        print(
            "decoupler was installed, but this running session still sees an old copy. Restart the kernel (the files on disk are kept) and run this cell again."
        )
        return None
    print(f"  installed decoupler {dc.__version__}")
    return dc


def load_net():
    """The CollecTRI regulons (needs internet access to OmniPath the first time)."""
    import decoupler as dc

    return dc.op.collectri(organism="human", remove_complexes=False)


def donor_tf_effect(raw, totals, donors, dx, var_names, net, tfs, batch=BATCH):
    """Nucleus-level ULM activity of each TF (decoupleR, CollecTRI), averaged per donor, disease minus control, and a two-sided Mann-Whitney test."""
    import anndata as ad
    import decoupler as dc

    if not hasattr(dc, "mt"):
        raise RuntimeError("This script needs decoupler 2.x (dc.mt.ulm, dc.op.collectri).")
    sub = net[net["source"].isin(tfs)]
    names = pd.Index(var_names).astype(str)
    raw = sps.csr_matrix(raw)
    n = raw.shape[0]
    tot = np.maximum(np.asarray(totals, float), 1)
    nz = np.flatnonzero(
        np.asarray(raw.sum(axis=0)).ravel() > 0
    )  # genes with no counts anywhere are dropped once, for the whole matrix
    names_nz = names[nz]
    got = {tf: [] for tf in tfs}
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        x = sps.csr_matrix(raw[lo:hi][:, nz], dtype=np.float32, copy=True)
        x.data *= np.repeat((1e4 / tot[lo:hi]).astype(np.float32), np.diff(x.indptr))
        np.log1p(x.data, out=x.data)  # log1p of counts per 10,000
        a = ad.AnnData(
            x,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(lo, hi)]),
            var=pd.DataFrame(index=names_nz),
        )
        a.var_names_make_unique()
        dc.mt.ulm(
            data=a, net=sub, verbose=False, empty=False
        )  # empty=False: no per-batch dropping of empty genes or nuclei
        acts = dc.pp.get_obsm(adata=a, key="score_ulm")
        for tf in tfs:
            if tf in acts.var_names:
                column = acts[:, tf].X
                got[tf].append(
                    np.asarray(column.todense() if sps.issparse(column) else column).ravel()
                )
        del a, x, acts
    out = {}
    for tf in tfs:
        if not got[tf]:
            continue
        frame = pd.DataFrame(
            {
                "donor": np.asarray(donors).astype(str),
                "dx": np.asarray(dx).astype(str),
                "s": np.concatenate(got[tf]),
            }
        )
        per_donor = frame.groupby(["donor", "dx"]).s.mean().reset_index()
        case, control = (
            per_donor[per_donor.dx == "PD"].s.values,
            per_donor[per_donor.dx == "Control"].s.values,
        )
        test = stats.mannwhitneyu(case, control, alternative="two-sided")
        out[tf] = {
            "n_disease": len(case), "n_control": len(control), "disease_mean": float(case.mean()), "control_mean": float(control.mean()),
            "mean_diff": float(case.mean() - control.mean()), "U": float(test.statistic), "p": float(test.pvalue),
        }  # fmt: skip
    return out


def fmt_p(p):
    """A p-value as it should be written in a paper."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def same_p_elsewhere(cohort, tf, p, tol=0.0005):
    """Other analyses of the same cohort whose quoted p-value equals this one (a coincidence is possible, because small-sample p-values are discrete)."""
    return [
        t
        for (c, t), (_, q) in PANEL.items()
        if c == cohort and t != tf and q is not None and abs(q - p) < tol
    ]


def compare_rows(cohort, results):
    """One row per transcription factor, next to the values printed in the manuscript."""
    rows = []
    for tf, r in results.items():
        quoted, panel = QUOTED.get((cohort, tf)), PANEL.get((cohort, tf))
        agrees = (
            ""
            if quoted is None
            else (
                "AGREES"
                if round(r["mean_diff"], 3) == quoted[0] and round(r["p"], 3) == quoted[1]
                else "DIFFERS"
            )
        )
        other = same_p_elsewhere(cohort, tf, r["p"])
        rows.append(
            {
                "cohort": cohort, "TF": tf, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                "manuscript_decoupleR": "" if quoted is None else f"{quoted[0]:+.3f} (p = {quoted[1]})", "vs_manuscript": agrees,
                "manuscript_panel": "" if panel is None else ("" if panel[0] is None else f"{panel[0]:+.3f} ") + f"(p = {panel[1]})",
                "same_p_as_panel_analysis_of": ", ".join(other),
            }
        )  # fmt: skip
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# Google Drive (read only). Used only if a file is not on disk, a folder id is set and a login file is found.
# ---------------------------------------------------------------------------------------------------------------------
FOLDER_TYPE = "application/vnd.google-apps.folder"


def find_token():
    """Look for the Drive login file in the usual places. Returns a path or None."""
    for candidate in (
        os.environ.get("LBD_DRIVE_TOKEN"),
        "drive_token.json",
        "/marimo/drive_token.json",
        str(Path.home() / "drive_token.json"),
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _download_file(service, file_id, destination):
    from googleapiclient.http import MediaIoBaseDownload

    with open(destination, "wb") as handle:
        downloader = MediaIoBaseDownload(
            handle, service.files().get_media(fileId=file_id), chunksize=50 * 1024 * 1024
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()


class DriveFolder:
    """A Google Drive folder that we can search, sub-folders included, and download from."""

    def __init__(self, folder_id, token_path=None, service=None):
        self.folder_id = folder_id
        if service is None:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            service = build(
                "drive",
                "v3",
                credentials=Credentials.from_authorized_user_file(token_path),
                cache_discovery=False,
            )
        self.service = service

    def _children(self, folder_id):
        rows, page = [], None
        while True:
            answer = (
                self.service.files()
                .list(q=f"'{folder_id}' in parents and trashed=false", fields="nextPageToken, files(id, name, mimeType, modifiedTime)", pageSize=1000, pageToken=page,
                      supportsAllDrives=True, includeItemsFromAllDrives=True)
                .execute()
            )  # fmt: skip
            rows += answer.get("files", [])
            page = answer.get("nextPageToken")
            if not page:
                return rows

    def download(self, name, destination, max_depth=4):
        """Find the newest file with this name in the folder or its sub-folders and copy it to this computer. True if found."""
        matches, queue = [], [(self.folder_id, 0)]
        while queue:
            folder, depth = queue.pop(0)
            for row in self._children(folder):
                if row["name"] == name and row["mimeType"] != FOLDER_TYPE:
                    matches.append(row)
                elif row["mimeType"] == FOLDER_TYPE and depth < max_depth:
                    queue.append((row["id"], depth + 1))
        if not matches:
            return False
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        _download_file(
            self.service, max(matches, key=lambda r: r.get("modifiedTime", ""))["id"], destination
        )
        return True


# ---------------------------------------------------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="decoupleR / CollecTRI activity of all eight transcription factors in Ma and Nido."
    )
    parser.add_argument(
        "--workdir",
        default=default_workdir(),
        help="the folder that contains data_snrna/ (results go under results/)",
    )
    parser.add_argument(
        "--folder-id",
        default=os.environ.get("LBD_DRIVE_FOLDER_ID") or DRIVE_FOLDER_ID,
        help="Google Drive folder with the input files",
    )
    parser.add_argument("--local-only", action="store_true", help="do not use Google Drive at all")
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=list(FILES),
        choices=list(FILES),
        help="which cohorts to analyse",
    )
    parser.add_argument(
        "--no-install", action="store_true", help="do not try to install decoupler if it is missing"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH,
        help="nuclei per decoupleR call (any value gives the same answer)",
    )
    args = parser.parse_known_args(argv)[0] if IN_NOTEBOOK else parser.parse_args(argv)
    work = Path(args.workdir)

    dc = ensure_decoupler(install=not args.no_install)
    if dc is None:
        return 3
    print(
        f"=== decoupleR version: {dc.__version__}  <- the Methods, the Software list and the Fig. 5 legend must state this version"
    )
    net = load_net()
    print(
        f"    CollecTRI: {net['source'].nunique()} regulons, {len(net):,} edges; targets per factor: "
        + ", ".join(f"{t} {int((net['source'] == t).sum())}" for t in TFS)
    )

    import anndata

    rows = []
    for cohort in args.cohorts:
        print(f"\n=== {cohort}")
        path = work / "data_snrna" / FILES[cohort]
        if not path.is_file():
            token = None if args.local_only or not args.folder_id else find_token()
            if token is None or not DriveFolder(args.folder_id, token).download(
                FILES[cohort], path
            ):
                print(
                    f"  STOPPED: {path} is not on disk and could not be fetched from Google Drive (use --local-only to skip Drive; or set DRIVE_FOLDER_ID and add drive_token.json)."
                )
                return 3
            print(f"  downloaded {FILES[cohort]} from Google Drive")
        adata = anndata.read_h5ad(path)
        diagnosis, donor, dx = resolve_columns(adata)
        raw, source = find_raw_counts(adata)
        raw = sps.csr_matrix(raw)
        totals = np.asarray(raw.sum(axis=1)).ravel()
        donors = adata.obs[donor].values
        print(
            f"  {adata.n_obs:,} nuclei | diagnosis in '{diagnosis}' | donors in '{donor}' | raw counts from {source}"
        )
        results = donor_tf_effect(
            raw, totals, donors, dx, adata.var_names, net, TFS, batch=args.batch
        )
        rows += compare_rows(cohort, results)
        del adata, raw
        import gc

        gc.collect()

    table = pd.DataFrame(rows)
    out = work / "results" / "decoupler_all_tfs"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "decoupler_all_tfs.csv", index=False)
    show = [
        "cohort",
        "TF",
        "n_disease",
        "n_control",
        "mean_diff",
        "U",
        "p",
        "manuscript_decoupleR",
        "vs_manuscript",
        "manuscript_panel",
        "same_p_as_panel_analysis_of",
    ]
    print(
        "\n=== Every factor, both cohorts (mean_diff = disease minus control; p = two-sided Mann-Whitney on donor means)"
    )
    print(table[show].to_string(index=False))
    print("\n=== The four numbers the Results should report (ATF4 and HSF1 in both cohorts):")
    for cohort in args.cohorts:
        for tf in ("ATF4", "HSF1"):
            hit = table[(table.cohort == cohort) & (table.TF == tf)]
            if len(hit):
                r = hit.iloc[0]
                print(
                    f"  {cohort} {tf}: mean difference = {r.mean_diff:.3f}, p = {fmt_p(r.p)}  (U = {r.U:.0f}; {int(r.n_disease)} disease, {int(r.n_control)} control donors)"
                )
    flagged = table[table.same_p_as_panel_analysis_of != ""]
    if len(flagged):
        print(
            "\n=== Same rounded p-value as a DIFFERENT analysis of the same cohort (a coincidence is possible: small-sample p-values are discrete). Check the full-precision values and U in the CSV:"
        )
        for _, r in flagged.iterrows():
            print(
                f"  {r.cohort} decoupleR {r.TF}: p = {r.p:.4f} (U = {r.U:.0f}) equals the curated-panel p-value of {r.same_p_as_panel_analysis_of}"
            )
    bad = table[table.vs_manuscript == "DIFFERS"]
    print(
        "\n=== Result: "
        + (
            "the two decoupleR values printed in the manuscript are reproduced."
            if len(table[table.vs_manuscript == "AGREES"]) and not len(bad)
            else (
                "CHECK: "
                + ", ".join(f"{r.cohort} {r.TF}" for _, r in bad.iterrows())
                + " differ from the manuscript."
                if len(bad)
                else "no manuscript value to compare (run both cohorts)."
            )
        )
    )
    print(f"Table written to {out / 'decoupler_all_tfs.csv'}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code and IN_NOTEBOOK:
        print(f"\nThe run stopped with code {exit_code}. Read the message above for the reason.")
    elif exit_code:
        sys.exit(exit_code)
