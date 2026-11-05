"""
communication_pairs.py

WHY: the communication tool found that the headline ratio depends on the pair definition. Under the STRICT definition (NLGN1 ligand with an NRXN receptor;
  TNF ligand with TNFRSF1A/1B) microglia-sourced significant pairs exist in only 3 of 15 Ma donors and 1 of 29 Nido donors. Under the LOOSE definition
  (any pair whose ligand is NLGN1 or whose receptor is an NRXN; any pair whose ligand is TNF or whose receptor is a TNFR-family member) nearly every donor has
  pairs, and TNFR-family pairs outnumber neurexin pairs. That raises two questions this script answers from the data:
    1. WHAT is being counted? Lists the pairs that make up each side (the LOOSE TNFR family includes TNFSF ligands, LTA, FAS-like and other receptors).
    2. Is the excess just the size of the pair family? A resource contains far more TNF-superfamily pairs than neurexin pairs, so counting significant pairs
       favours the larger family even if both are equally likely to be significant. The fair quantity is the FRACTION of tested pairs that are significant.
READS: Ma_liana_full.csv, Nido_liana_full.csv (sample + permutation p-value columns required). Nothing is written back to the manuscript; it informs a decision:
  - fractions similar between families, or neurexin higher  -> the "TNF dominance" is a resource-size artefact: do NOT present it as a finding.
  - TNFR-family fraction clearly and consistently higher    -> report it as a FAMILY-level result with the pairs named, never as "NLGN1-NRXN vs TNF-TNFR".
"""

import io, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats

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


def tokens(s):
    return set(t for t in re.split(r"[_&;,\s]+", str(s).upper()) if t)


def detect_columns(df, donor_ids):
    cols = {}
    for k, cands in COLS.items():
        hit = next((c for c in cands if c in df.columns), None)
        if hit is None:
            raise KeyError(
                f"no '{k}' column among {list(df.columns)}. Significance must be a permutation p-value, not a LIANA rank."
            )
        cols[k] = hit
    best, frac = None, 0.0
    for c in df.columns:
        if c in cols.values() or df[c].nunique() < 4 or df[c].nunique() > 80:
            continue
        f = len(set(df[c].astype(str)) & donor_ids) / max(len(set(df[c].astype(str))), 1)
        if f > frac:
            best, frac = c, f
    if best is None or frac < 0.5:
        raise KeyError(
            f"no sample column matches the donor ids of the cohort (best overlap {frac:.2f}). Columns: {list(df.columns)}"
        )
    cols["sample"] = best
    return cols


def balance_by_diagnosis(donor, dxmap):
    """Does the balance of significant pairs between the two families differ between disease and control donors?
    Mann-Whitney, two-sided, on each donor's TNFR-family fraction minus NRXN-family fraction (primary: 'balance'), and on each fraction alone. U is the smaller of the two U statistics, as in the Figure 4 legend.
    """
    d = donor.dropna(subset=["nrxn_fraction", "tnfr_fraction"]).copy()
    d["dx"] = [dxmap.get(str(i)) for i in d.index]
    out = {}
    for name, v in (
        ("balance", d.tnfr_fraction - d.nrxn_fraction),
        ("tnfr_fraction", d.tnfr_fraction),
        ("nrxn_fraction", d.nrxn_fraction),
    ):
        a, b = v[d.dx == "PD"].values.astype(float), v[d.dx == "Control"].values.astype(float)
        if len(a) < 2 or len(b) < 2 or np.ptp(np.r_[a, b]) == 0:
            out[name] = dict(
                n_pd=len(a),
                n_control=len(b),
                u=np.nan,
                p=np.nan,
                median_pd=np.nan,
                median_control=np.nan,
            )
            continue
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        u1 = float(mw.statistic)
        out[name] = dict(
            n_pd=len(a),
            n_control=len(b),
            u=min(u1, len(a) * len(b) - u1),
            p=float(mw.pvalue),
            median_pd=float(np.median(a)),
            median_control=float(np.median(b)),
        )
    return out


def classify_pairs(df, cols, definition):
    lig = df[cols["ligand"]].map(tokens)
    rec = df[cols["receptor"]].map(tokens)
    if definition == "STRICT":
        nl = lig.map(lambda s: "NLGN1" in s) & rec.map(
            lambda s: bool(s & {"NRXN1", "NRXN2", "NRXN3"})
        )
        tn = lig.map(lambda s: "TNF" in s) & rec.map(lambda s: bool(s & {"TNFRSF1A", "TNFRSF1B"}))
    else:  # the rule used by gap5_ma_liana.py
        nl = lig.map(lambda s: "NLGN1" in s) | rec.map(
            lambda s: any(t.startswith("NRXN") for t in s)
        )
        tn = lig.map(lambda s: s == {"TNF"}) | rec.map(
            lambda s: any(t.startswith("TNFR") for t in s)
        )
    return nl, tn


def obs_only(cohort):
    import scanpy as sc

    path = resolve_input_path(DATA_DIR / FILES[cohort], f"{cohort} cohort", ALLOW_DRIVE_FETCH)
    a = sc.read_h5ad(path, backed="r")
    diag, donor, dx = resolve_diag_and_donor(a, cohort)
    dxmap = dict(zip(a.obs[donor].astype(str), np.asarray(dx)))
    ids = set(dxmap)
    a.file.close()
    return dxmap, ids


LIANA_FILES = {"Ma": "Ma_liana_full.csv", "Nido": "Nido_liana_full.csv"}
RESULTS_DIR = Path("results/communication_pair_diagnostic")
COLS = {
    "source": ["source", "sender", "source_celltype"],
    "ligand": ["ligand_complex", "ligand", "ligand_symbol"],
    "receptor": ["receptor_complex", "receptor", "receptor_symbol"],
    "p": ["cellphone_pvals", "pval", "p_value", "pvalue", "p"],
}


def families(df, cols):
    lig = df[cols["ligand"]].map(tokens)
    rec = df[cols["receptor"]].map(tokens)
    out = pd.DataFrame(
        {
            "NRXN_family": lig.map(lambda s: "NLGN1" in s)
            | rec.map(lambda s: any(t.startswith("NRXN") for t in s)),
            "TNFR_family": lig.map(lambda s: s == {"TNF"})
            | rec.map(lambda s: any(t.startswith("TNFR") for t in s)),
            "strict_NLGN1_NRXN": lig.map(lambda s: "NLGN1" in s)
            & rec.map(lambda s: bool(s & {"NRXN1", "NRXN2", "NRXN3"})),
            "strict_TNF_TNFR1or2": lig.map(lambda s: "TNF" in s)
            & rec.map(lambda s: bool(s & {"TNFRSF1A", "TNFRSF1B"})),
        },
        index=df.index,
    )
    return out


def diagnose(df, cols, source_label, alpha=0.05):
    m = df[df[cols["source"]].astype(str) == source_label].copy()
    m = m[m[cols["p"]].notna()]
    fam = families(m, cols)
    m = pd.concat([m, fam], axis=1)
    m["sig"] = m[cols["p"]] < alpha
    m["pair"] = m[cols["ligand"]].astype(str) + " -> " + m[cols["receptor"]].astype(str)
    rows, tops = [], {}
    for f in ("NRXN_family", "TNFR_family", "strict_NLGN1_NRXN", "strict_TNF_TNFR1or2"):
        d = m[m[f]]
        pairs_tested = d["pair"].nunique()
        sig = d[d.sig]
        rows.append(
            dict(
                family=f,
                distinct_pairs_tested=pairs_tested,
                rows_tested=len(d),
                rows_significant=len(sig),
                distinct_pairs_significant=sig["pair"].nunique(),
                fraction_significant=(len(sig) / len(d)) if len(d) else np.nan,
                donors_with_a_significant_pair=sig[cols["sample"]].nunique(),
            )
        )
        tops[f] = (
            sig.groupby("pair")
            .agg(significant_rows=("sig", "size"), donors=(cols["sample"], "nunique"))
            .sort_values("significant_rows", ascending=False)
            .head(12)
        )
    donor = m.groupby(cols["sample"]).apply(
        lambda g: pd.Series(
            {
                "nrxn_tested": int(g.NRXN_family.sum()),
                "nrxn_sig": int((g.NRXN_family & g.sig).sum()),
                "tnfr_tested": int(g.TNFR_family.sum()),
                "tnfr_sig": int((g.TNFR_family & g.sig).sum()),
            }
        )
    )
    donor["nrxn_fraction"] = donor.nrxn_sig / donor.nrxn_tested.replace(0, np.nan)
    donor["tnfr_fraction"] = donor.tnfr_sig / donor.tnfr_tested.replace(0, np.nan)
    return pd.DataFrame(rows), tops, donor


def paired_test(donor):
    d = donor.dropna(subset=["nrxn_fraction", "tnfr_fraction"])
    diff = d.tnfr_fraction - d.nrxn_fraction
    nz = diff[diff != 0]
    wil = stats.wilcoxon(nz, alternative="two-sided") if len(nz) else None
    p2 = wil.pvalue if wil is not None else np.nan
    w_stat = wil.statistic if wil is not None else np.nan
    return dict(
        donors=len(d),
        tnfr_higher=int((diff > 0).sum()),
        nrxn_higher=int((diff < 0).sum()),
        median_nrxn_fraction=float(d.nrxn_fraction.median()),
        median_tnfr_fraction=float(d.tnfr_fraction.median()),
        wilcoxon_two_sided=float(p2),
        wilcoxon_w=float(w_stat),
        n_nonzero=int(len(nz)),
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


def keep_text_inside(fig, pad_pt=1.5, tries=10):
    """Shrink the axes area until every drawn piece of text lies inside the page. Matplotlib does not do this when the margins are set by hand, and fonts and matplotlib versions differ between machines:
    a label that overshoots the page by a fraction of a point passes on one machine and fails the journal check on another.
    """

    def drawn_ticklabels(axis, lim):
        lo, hi = min(lim), max(lim)
        tol = 1e-9 + 1e-6 * (hi - lo)
        return [
            lab
            for tick, lab in zip(axis.get_major_ticks(), axis.get_ticklabels())
            if lo - tol <= tick.get_loc() <= hi + tol
        ]  # ticks outside the view limits are not drawn

    for _ in range(tries):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        W, H = fig.bbox.width, fig.bbox.height
        pad = pad_pt * fig.dpi / 72
        texts = list(fig.texts)
        for ax in fig.axes:
            texts += (
                [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
                + list(ax.texts)
                + drawn_ticklabels(ax.xaxis, ax.get_xlim())
                + drawn_ticklabels(ax.yaxis, ax.get_ylim())
            )
            if ax.get_legend():
                texts += list(ax.get_legend().get_texts())
        boxes = [t.get_window_extent(r) for t in texts if t.get_visible() and t.get_text().strip()]
        if not boxes:
            return
        l, rt = max(0.0, pad - min(b.x0 for b in boxes)), max(
            0.0, max(b.x1 for b in boxes) - (W - pad)
        )
        bo, tp = max(0.0, pad - min(b.y0 for b in boxes)), max(
            0.0, max(b.y1 for b in boxes) - (H - pad)
        )
        if max(l, rt, bo, tp) <= 0:
            return
        for gs in {
            ax.get_subplotspec().get_gridspec()
            for ax in fig.axes
            if ax.get_subplotspec() is not None
        }:
            sp = gs.get_subplot_params(fig)
            gs.update(
                left=sp.left + l / W,
                right=sp.right - rt / W,
                bottom=sp.bottom + bo / H,
                top=sp.top - tp / H,
            )


def save_figure(fig, prefix, alt_text, legend_text=None):
    """Vector PDF, 600 dpi LZW TIFF, PNG preview, alt text and (optionally) the legend, all from the same figure object."""
    keep_text_inside(fig)
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


def fmt_p(p):
    """Exact p-value for a figure label, to one significant figure below 0.01 as the manuscript does: p = 0.004, p = 0.00006 (6.1e-05), p = 0.0001; two decimals from 0.01 up (p = 0.86).
    0.0039 must never print as 'p = 0.00', and an exact p of 6.1e-05 must never be rounded up to 0.0001.
    """
    if p >= 0.01:
        return f"p = {p:.2f}"
    if p < 1e-5:
        return "p < 0.00001"
    one = float(f"{p:.1g}")
    return f"p = {one:.{max(3, -int(np.floor(np.log10(one))))}f}"


def make_figure6_family(results, prefix):
    """Family-level communication figure. A: how many distinct pairs each family offers (the denominator). B: per donor, the FRACTION of tested pairs that are significant."""
    fig = new_fig(170, 105)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.3, 1.45], wspace=0.38, left=0.11, right=0.97, top=0.84, bottom=0.22
    )
    ca = fig.add_subplot(gs[0, 0])
    cb = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    xs = {"Ma": (0.0, 2.0), "Nido": (4.5, 6.5)}
    for co, (x1, x2) in xs.items():
        a = results[co]["summ"].set_index("family")
        v1, v2 = int(a.loc["NRXN_family", "distinct_pairs_tested"]), int(
            a.loc["TNFR_family", "distinct_pairs_tested"]
        )
        ca.bar(
            [x1, x2],
            [v1, v2],
            width=1.0,
            color=[OI["sky"], OI["verm"]],
            edgecolor="black",
            linewidth=0.5,
        )
        for xx, v in ((x1, v1), (x2, v2)):
            ca.text(xx, v, str(v), ha="center", va="bottom", fontsize=12)
    ca.set_xlim(-0.6, 7.0)
    ca.set_xticks([0.0, 2.0, 4.5, 6.5])
    ca.set_xticklabels(["Ma\nNRXN", "Ma\nTNFR", "Nido\nNRXN", "Nido\nTNFR"])
    # each bar says which family it is, as in panel B
    ca.set_ylabel("Distinct pairs available\n(microglia as sender)")
    ca.set_title("Pairs available", fontsize=12)
    for sp in ("top", "right"):
        ca.spines[sp].set_visible(False)
    panel_label(ca, "A", dx=-0.33, dy=1.10)
    xs2 = {"Ma": (0.0, 1.0), "Nido": (2.6, 3.6)}
    top = (
        max(
            0.5,
            max(
                float(results[co]["donor"][["nrxn_fraction", "tnfr_fraction"]].max().max())
                for co in results
            ),
        )
        + 0.06
    )
    for co, (x1, x2) in xs2.items():
        d = results[co]["donor"].dropna(subset=["nrxn_fraction", "tnfr_fraction"])
        for _, r in d.iterrows():
            cb.plot(
                [x1, x2],
                [r.nrxn_fraction, r.tnfr_fraction],
                color="#BBBBBB",
                linewidth=0.8,
                zorder=1,
            )
        cb.scatter(
            np.full(len(d), x1) + rng.uniform(-0.05, 0.05, len(d)),
            d.nrxn_fraction,
            color=OI["sky"],
            edgecolor="black",
            linewidth=0.4,
            s=26,
            zorder=3,
        )
        cb.scatter(
            np.full(len(d), x2) + rng.uniform(-0.05, 0.05, len(d)),
            d.tnfr_fraction,
            color=OI["verm"],
            edgecolor="black",
            linewidth=0.4,
            s=26,
            zorder=3,
        )
        cb.hlines(
            [d.nrxn_fraction.median()], x1 - 0.25, x1 + 0.25, color="black", linewidth=2, zorder=4
        )
        cb.hlines(
            [d.tnfr_fraction.median()], x2 - 0.25, x2 + 0.25, color="black", linewidth=2, zorder=4
        )
        pv = results[co]["test"]["wilcoxon_two_sided"]
        cb.text((x1 + x2) / 2, top, fmt_p(pv), ha="center", va="bottom", fontsize=12)
    cb.set_xticks([0, 1, 2.6, 3.6])
    cb.set_xticklabels(["Ma\nNRXN", "Ma\nTNFR", "Nido\nNRXN", "Nido\nTNFR"])
    cb.set_ylim(-0.02, top + 0.16)
    cb.set_ylabel("Fraction of tested pairs\nthat are significant")
    cb.set_title("Per donor, fraction significant", fontsize=12)
    for sp in ("top", "right"):
        cb.spines[sp].set_visible(False)
    panel_label(cb, "B", dx=-0.16, dy=1.10)
    tt = {co: results[co]["test"] for co in results}
    alt = (
        "Panel A: bars showing how many distinct ligand-receptor pairs are available in the neurexin family and the TNF-receptor family for microglia-sourced signalling in the Ma and Nido cohorts. "
        "Panel B: for each donor, the fraction of tested pairs that are significant in each family, joined by a line, with the paired two-sided Wilcoxon p-value."
    )
    leg = (
        "Fig. 6. Microglia-sourced signalling by pair family. (A) Distinct ligand-receptor pairs available to be tested in each family (neurexin family: NLGN1 ligand or NRXN receptor; TNF-receptor family: TNF ligand or TNFR-family receptor). "
        "A family with more pairs yields more significant pairs even at the same rate, so counts of significant pairs are not compared directly. (B) For each donor, the fraction of tested pairs in each family that is significant "
        "(CellPhoneDB permutation p < 0.05); grey lines join the two values of one donor; black bars, medians. "
        + "; ".join(
            f"{co}: {t['donors']} donors, TNFR family higher in {t['tnfr_higher']}, neurexin family higher in {t['nrxn_higher']}, paired two-sided Wilcoxon W = {t['wilcoxon_w']:g} ({t['n_nonzero']} donors with a non-zero difference), {fmt_p(t['wilcoxon_two_sided'])}"
            + (
                f"; disease versus control balance, Mann-Whitney U = {t['dx_u']:g}, {fmt_p(t['dx_p'])}, {int(t['dx_n_pd'])} versus {int(t['dx_n_control'])} donors"
                if np.isfinite(t.get("dx_p", np.nan))
                else ""
            )
            for co, t in tt.items()
        )
        + "."
    )
    save_figure(fig, prefix, alt, leg)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230, "display.max_columns", 30, "display.max_colwidth", 60)
    allres = {}
    for cohort, fname in LIANA_FILES.items():
        print(f"\n{'=' * 100}\n{cohort}\n{'=' * 100}")
        dxmap, ids = obs_only(cohort)
        path = resolve_input_path(DATA_DIR / fname, f"{cohort} LIANA output", ALLOW_DRIVE_FETCH)
        df = pd.read_csv(path)
        cols = detect_columns(df, ids)
        src = sorted({s for s in df[cols["source"]].astype(str) if "micro" in s.lower()})[0]
        summ, tops, donor = diagnose(df, cols, src)
        print(
            "\n1. HOW MANY PAIRS EXIST ON EACH SIDE, AND WHAT FRACTION IS SIGNIFICANT? (microglia-sourced, permutation p < 0.05)"
        )
        print(summ.round(4).to_string(index=False))
        for f in ("NRXN_family", "TNFR_family"):
            print(f"\n2. WHICH PAIRS MAKE UP THE {f.replace('_', ' ')} COUNT (top 12 significant)")
            print(tops[f].to_string())
        t = paired_test(donor)
        allres[cohort] = dict(summ=summ, donor=donor, test=t)
        print(
            "\n3. PER-DONOR FRACTION SIGNIFICANT, TNFR family vs NRXN family:",
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()},
        )
        bd = balance_by_diagnosis(donor, dxmap)
        t.update(
            dx_u=bd["balance"]["u"],
            dx_p=bd["balance"]["p"],
            dx_n_pd=bd["balance"]["n_pd"],
            dx_n_control=bd["balance"]["n_control"],
        )
        print(
            "\n4. DISEASE VERSUS CONTROL: does the balance of significant pairs between the families differ? (Mann-Whitney, two-sided; U = the smaller of the two)"
        )
        print(pd.DataFrame(bd).T.round(4).to_string())
        pd.DataFrame(bd).T.rename_axis("variant").reset_index().to_csv(
            RESULTS_DIR / f"{cohort}_diagnosis_balance_test.csv", index=False
        )
        summ.to_csv(RESULTS_DIR / f"{cohort}_family_summary.csv", index=False)
        donor.to_csv(RESULTS_DIR / f"{cohort}_per_donor_family_fractions.csv")
        for f, tp in tops.items():
            tp.to_csv(RESULTS_DIR / f"{cohort}_top_pairs_{f}.csv")
        a = summ.set_index("family")
        count_ratio = a.loc["NRXN_family", "rows_significant"] / max(
            a.loc["TNFR_family", "rows_significant"], 1
        )
        frac_ratio = (
            a.loc["NRXN_family", "fraction_significant"]
            / a.loc["TNFR_family", "fraction_significant"]
            if a.loc["TNFR_family", "fraction_significant"]
            else np.nan
        )
        print(
            f"\nINTERPRETATION {cohort}: NRXN:TNFR ratio of significant COUNTS = {count_ratio:.2f}; ratio of significant FRACTIONS = {frac_ratio:.2f}."
        )
        print(
            "  Counts and fractions disagree => the count ratio reflects how many pairs each family has, not how often they are significant."
            if (count_ratio < 1) != (frac_ratio < 1)
            or abs(np.log((count_ratio + 1e-9) / (frac_ratio + 1e-9))) > np.log(2)
            else "  Counts and fractions agree in direction and size."
        )
    make_figure6_family(allres, str(RESULTS_DIR / "Figure6_family_level"))
    print(
        "Family-level Figure 6 written (use it only if the fractions, not the counts, support a claim)."
    )
    print(f"\nFiles written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
