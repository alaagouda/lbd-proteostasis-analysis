"""
make_figure7_augur.py

Figure 7 (Augur cell-type prioritisation) drawn from the saved results file of the Augur run (augur_run.py writes results/augur_run/augur_fresh_results_<timestamp>.csv
with columns Cell_Type, AUC, Cohort). The old figure7_augur.py typed the AUROC values in by hand.

CHECKS PRINTED BEFORE THE FIGURE IS DRAWN, against what the manuscript says:
  Jin  leading cell type microglia, AUROC 0.69      Ma  leading cell type oligodendrocytes, AUROC 0.85      Nido  AUROC range across the 17 subtypes: the TEXT says 0.47 to 0.61.
NOTE: the values typed into figure_augur_FRESH.py (the same run) give a Nido range of 0.471 to 0.613, i.e. 0.47 to 0.61, which is what the response letter said BEFORE it was changed to
  match the text. This script computes the range from the file; whichever it prints is the one to use in BOTH the text and the letter.
"""

import io, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sps

warnings.filterwarnings("ignore")

DATA_DIR = Path("./data_snrna")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILES = {
    "Jin": "GSE235914_LBD_annotated_FIXED.h5ad",
    "Ma": "GSE253462_annotated.h5ad",
    "Nido": "nido_analysis_ready.h5ad",
}
# Set a column explicitly here if automatic detection stops with an error. None = detect.
OVERRIDE = {c: {"donor": None, "diagnosis": None, "celltype": None} for c in FILES}
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
CT_CANDS = [
    "cell_type",
    "celltype",
    "CellType",
    "cell_type_fine",
    "fine_type",
    "subtype",
    "cell_subtype",
    "cell_type_broad",
    "broad_type",
    "majority_voting",
    "predicted_labels",
    "annotation",
]

DRIVE_FOLDER_ID = os.environ.get("LBD_DRIVE_FOLDER_ID", "")
DRIVE_TOKEN_PATH = Path("drive_token.json")
# Default True here, unlike the final network audit script: this is a
# diagnostic check being run to decide whether a manuscript passage needs
# correcting, not a locked final verification run where ambiguity about
# which exact file version was used would be a problem in itself.
ALLOW_DRIVE_FETCH = bool(DRIVE_FOLDER_ID)  # off unless LBD_DRIVE_FOLDER_ID is set


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


def classify_diagnosis(values):
    """Map free-text diagnosis labels to PD (disease) / Control, refusing to guess."""
    out = {}
    for v in pd.unique(pd.Series(values).astype(str)):
        s = v.strip().lower()
        if re.match(r"^(control|ctrl|ctl|healthy|hc|normal|non[- ]?pd|non[- ]?dlb)", s):
            out[v] = "Control"
        elif re.match(r"^(pd|dlb|lbd|lewy|dementia|parkinson|case|disease|patient)", s):
            out[v] = "PD"
        else:
            raise ValueError(
                f"Unclassifiable diagnosis label {v!r}; set OVERRIDE for this cohort or extend classify_diagnosis()."
            )
    if set(out.values()) != {"PD", "Control"}:
        raise ValueError(f"Labels did not give both classes: {out}")
    return out


def resolve_diag_and_donor(adata, cohort):
    ov = OVERRIDE.get(cohort, {})
    diag = None
    for c in ([ov["diagnosis"]] if ov.get("diagnosis") else DIAG_CANDS):
        if c in adata.obs.columns:
            try:
                classify_diagnosis(adata.obs[c])
                diag = c
                break
            except ValueError:
                continue
    if diag is None:
        raise KeyError(
            f"{cohort}: no usable diagnosis column. obs columns: {list(adata.obs.columns)}"
        )
    mapping = classify_diagnosis(adata.obs[diag])
    dx = adata.obs[diag].astype(str).map(mapping)
    donor = None
    for c in ([ov["donor"]] if ov.get("donor") else DONOR_CANDS):
        if c in adata.obs.columns:
            g = (
                pd.DataFrame({"d": adata.obs[c].astype(str).values, "x": dx.values})
                .groupby("d")["x"]
                .nunique()
            )
            if 4 <= len(g) <= 80 and (g == 1).all():
                donor = c
                break
    if donor is None:
        raise KeyError(
            f"{cohort}: no donor column with one diagnosis per donor. obs columns: {list(adata.obs.columns)}"
        )
    return diag, donor, dx.values


def find_raw_counts(adata):
    """Return (matrix, source). Hard-fails rather than continuing on unverified input."""
    cands = [("X", adata.X)] + [
        (f"layers[{k!r}]", adata.layers[k]) for k in adata.layers.keys() if k is not None
    ]
    if adata.raw is not None:
        cands.append(("raw.X", adata.raw.X))
    for name, M in cands:
        if M.shape[1] == adata.n_vars and looks_like_integer_counts(M):
            return M, name
    raise ValueError(
        "No raw-count matrix found in X, layers or raw; refusing to continue on normalised or unknown data."
    )


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1)
    adj = np.clip(np.minimum.accumulate(r[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[o] = adj
    return out


def resolve_celltype(adata, cohort):
    ov = OVERRIDE.get(cohort, {}).get("celltype")
    for c in ([ov] if ov else CT_CANDS):
        if c in adata.obs.columns:
            return c
    raise KeyError(f"{cohort}: no cell-type column. obs columns: {list(adata.obs.columns)}")


def load_cohort(cohort):
    """Load one cohort and everything the checks need, hard-failing on anything ambiguous."""
    import scanpy as sc

    path = resolve_input_path(DATA_DIR / FILES[cohort], f"{cohort} cohort", ALLOW_DRIVE_FETCH)
    adata = sc.read_h5ad(path)
    diag, donor, dx = resolve_diag_and_donor(adata, cohort)
    ctc = resolve_celltype(adata, cohort)
    raw, src = find_raw_counts(adata)
    raw = raw.tocsr() if sps.issparse(raw) else sps.csr_matrix(raw)
    print(
        f"  {cohort}: {adata.n_obs:,} nuclei | diagnosis '{diag}' | donor '{donor}' ({adata.obs[donor].nunique()} donors) | "
        f"cell type '{ctc}' ({adata.obs[ctc].nunique()} types) | raw counts from {src}"
    )
    return adata, dict(
        donor=adata.obs[donor].astype(str).values,
        dx=np.asarray(dx),
        ct=adata.obs[ctc].astype(str).values,
        raw=raw,
        totals=np.maximum(np.asarray(raw.sum(axis=1)).ravel().astype(float), 1.0),
    )


def gene_counts(adata, ctx, genes):
    """Dense cells x genes raw counts for the given genes; refuses to run if any gene is missing."""
    miss = [g for g in genes if g not in adata.var_names]
    if miss:
        raise KeyError(f"genes missing from the data: {miss}")
    idx = [adata.var_names.get_loc(g) for g in genes]
    return ctx["raw"][:, idx].toarray().astype(float)


def lognorm_in_place(adata, ctx):
    """Reset .X to the raw counts, then CP10k + log1p exactly once (clears the stale uns['log1p'] key)."""
    import scanpy as sc

    adata.X = ctx["raw"].astype(np.float32).copy()
    adata.raw = None
    adata.uns.pop("log1p", None)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def round_match(value, reported, decimals):
    return bool(np.isfinite(value)) and round(float(value), decimals) == round(
        float(reported), decimals
    )


# ---------------------------------------------------------------- journal figure style (Brain Communications: 170 x 210 mm area, 12 pt text, 600 dpi TIFF)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = dict(
    blue="#0072B2",
    orange="#E69F00",
    verm="#D55E00",
    sky="#56B4E9",
    green="#009E73",
    yellow="#F0E442",
    purple="#CC79A7",
    grey="#999999",
    black="#000000",
)
MAX_W_MM, MAX_H_MM = 170.0, 210.0


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 12,
        }
    )


def new_fig(w_mm, h_mm):
    if w_mm > MAX_W_MM or h_mm > MAX_H_MM:
        raise ValueError(f"{w_mm} x {h_mm} mm exceeds the {MAX_W_MM} x {MAX_H_MM} mm journal area")
    setup_style()
    return plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))


def panel_label(ax, letter, dx=-0.14, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


def save_figure(fig, prefix, alt_text, legend_text=None):
    """Vector PDF, 600 dpi LZW TIFF, PNG preview, alt text and (optionally) the legend, all from the same figure object."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    fig.savefig(f"{prefix}.pdf")
    fig.savefig(f"{prefix}.png", dpi=600)
    Image.open(f"{prefix}.png").convert("RGB").save(
        f"{prefix}.tiff", compression="tiff_lzw", dpi=(600, 600)
    )
    plt.close(fig)
    open(f"{prefix}_alt_text.txt", "w", encoding="utf-8").write(alt_text.strip() + "\n")
    if legend_text:
        open(f"{prefix}_legend.txt", "w", encoding="utf-8").write(legend_text.strip() + "\n")


import glob

PATTERN = "results/augur_run/augur_fresh_results_*.csv"
RESULTS_DIR = Path("results/figure7")
META7 = {
    "Jin": ("Jin: temporal cortex (DLB, n = 6 donors)", 6),
    "Ma": ("Ma: cingulate cortex (PD, n = 17 donors)", 17),
    "Nido": ("Nido: prefrontal cortex (PD, n = 33 donors)", 33),
}
TEXT_CLAIMS = {
    "Jin": dict(top="micro", auc=0.69),
    "Ma": dict(top="oligo", auc=0.85),
    "Nido": dict(lo=0.47, hi=0.61),
}


def load_augur(path):
    df = pd.read_csv(path)
    low = {str(c).lower(): c for c in df.columns}
    need = {
        "cell_type": next(
            (low[k] for k in low if k.replace(" ", "_") in ("cell_type", "celltype")), None
        ),
        "auc": next((low[k] for k in low if k == "auc" or "auc" in k), None),
        "cohort": next((low[k] for k in low if k == "cohort"), None),
    }
    if None in need.values():
        raise KeyError(
            f"cannot find Cell_Type / AUC / Cohort columns. Columns present: {list(df.columns)}"
        )
    out = df[[need["cohort"], need["cell_type"], need["auc"]]].copy()
    out.columns = ["cohort", "cell_type", "auc"]
    out["auc"] = pd.to_numeric(out["auc"], errors="coerce")
    out = out[
        ~out["cell_type"].astype(str).str.strip().str.lower().eq("unassigned")
    ]  # Unassigned nuclei are excluded from all downstream analyses (Fig. 3 legend), so they are not drawn
    return out.dropna()


def check_claims(df):
    rows = []
    for c, claim in TEXT_CLAIMS.items():
        d = df[df.cohort == c].sort_values("auc", ascending=False)
        if d.empty:
            rows.append(
                dict(cohort=c, check="data present", computed="none", manuscript="-", agrees=False)
            )
            continue
        if "top" in claim:
            rows.append(
                dict(
                    cohort=c,
                    check="leading cell type and AUROC",
                    computed=f"{d.iloc[0].cell_type} {d.iloc[0].auc:.3f}",
                    manuscript=f"{claim['top']}* {claim['auc']:.2f}",
                    agrees=claim["top"] in d.iloc[0].cell_type.lower()
                    and round_match(d.iloc[0].auc, claim["auc"], 2),
                )
            )
        else:
            rows.append(
                dict(
                    cohort=c,
                    check="AUROC range",
                    computed=f"{d.auc.min():.3f} to {d.auc.max():.3f}  ({d.auc.min():.2f} to {d.auc.max():.2f})",
                    manuscript=f"{claim['lo']:.2f} to {claim['hi']:.2f}",
                    agrees=round_match(d.auc.min(), claim["lo"], 2)
                    and round_match(d.auc.max(), claim["hi"], 2),
                )
            )
    return pd.DataFrame(rows)


AX_LEFT, AX_RIGHT = (
    0.44,
    0.93,
)  # horizontal extent of the bar axes as a fraction of the figure width


def make_figure7(df, prefix):
    cohorts = [c for c in ("Jin", "Ma", "Nido") if c in set(df.cohort)]
    counts = [int((df.cohort == c).sum()) for c in cohorts]
    gap, top_mm, bot_mm, bar_mm = (
        11.0,
        8.0,
        15.0,
        4.9,
    )  # mm: space between panels (holds the title), above the first panel, below the last (x label), per bar
    H = bar_mm * sum(counts) + gap * (len(cohorts) - 1) + top_mm + bot_mm
    fig = new_fig(170, H)
    gs = fig.add_gridspec(
        len(cohorts),
        1,
        height_ratios=counts,
        hspace=gap / (bar_mm * sum(counts) / len(cohorts)),
        left=AX_LEFT,
        right=AX_RIGHT,
        top=1 - top_mm / H,
        bottom=bot_mm / H,
    )
    for i, c in enumerate(cohorts):
        ax = fig.add_subplot(gs[i, 0])
        d = df[df.cohort == c].sort_values("auc", ascending=True)
        y = np.arange(len(d))
        top = d.auc.values.argmax()
        ax.barh(
            y,
            d.auc.values - 0.5,
            left=0.5,
            color=[OI["verm"] if k == top else OI["sky"] for k in range(len(d))],
            edgecolor="black",
            linewidth=0.5,
            height=0.72,
        )
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([s.replace("_", " ") for s in d.cell_type])
        for yi, v in zip(y, d.auc.values):
            ax.text(max(v, 0.5) + 0.008, yi, f"{v:.2f}", va="center", ha="left", fontsize=12)
        ax.set_xlim(0.40, 0.95)
        t = ax.set_title(META7[c][0], loc="left", fontsize=12, fontweight="bold")
        t.set_x(
            -(AX_LEFT - 0.02) / (AX_RIGHT - AX_LEFT)
        )  # anchor the title at the page edge, not at the axes (the long titles ran off the right edge)
        ax.set_xlabel("AUROC (0.5 = chance)" if i == len(cohorts) - 1 else "")
        if i < len(cohorts) - 1:
            ax.tick_params(labelbottom=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    top = {c: df[df.cohort == c].sort_values("auc", ascending=False).iloc[0] for c in cohorts}
    alt = (
        "Horizontal bar charts of Augur AUROC for each cell type in the "
        + ", ".join(
            f"{c} cohort ({counts[k]} cell types; highest: {top[c].cell_type.replace('_', ' ')}, {top[c].auc:.2f})"
            for k, c in enumerate(cohorts)
        )
        + ". A dashed line marks chance, AUROC 0.5."
    )
    leg = (
        "Fig. 7. Augur cell-type prioritisation across three cohorts. Area under the ROC curve (AUROC) per cell type, reflecting how well gene expression separates disease from control. "
        "Dashed line: AUROC 0.5 (chance); the highest cell type in each cohort is shown in vermillion. Cell types absent from the Jin panel had insufficient nuclei for Augur evaluation; unassigned nuclei are not shown. Augur classifies individual nuclei, so AUROC is descriptive and not a donor-level test. "
        "Analysis performed with pertpy v1.0.3; Nido was run on an automatic subsample of 60,000 of 277,831 nuclei."
    )
    save_figure(fig, prefix, alt, leg)


def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise FileNotFoundError(
            f"no file matches {PATTERN}. Copy the Augur results CSV there, or edit PATTERN."
        )
    print("using", files[-1])
    df = load_augur(files[-1])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chk = check_claims(df)
    chk.to_csv(RESULTS_DIR / "figure7_claims_check.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    print("\nData versus what the manuscript text says:")
    print(chk.to_string(index=False))
    if not chk.agrees.all():
        print(
            "\nAT LEAST ONE TEXT CLAIM DISAGREES WITH THE DATA: update the text (and the letter) to the 'computed' column."
        )
    make_figure7(df, str(RESULTS_DIR / "Figure7"))
    print(f"Figure 7 written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
