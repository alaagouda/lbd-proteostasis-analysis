"""
make_figure5_stouffer_tf.py

Figure 5 drawn FROM RESULTS. The old figure5_stouffer_tf.py said its per-cohort markers (panel A) and TF bars (panels B and C) were "read off the image".
  Panel A  Stouffer Z-scores of the seven primary HSP genes in each cohort, plus the combined estimate (weights = sqrt of the donor number).
           INPUT: the per-cohort gene-level meta-analysis table (Meta_Analysis_Results_DONOR_LEVEL.csv; NOT the ligand-receptor file cross_cohort_meta_analysis.csv). Needs one gene column, one Z column per cohort (Jin, Ma, Nido) and
           optionally a combined Z column. The combined Z is RECOMPUTED here and compared with the supplied one and with the seven values the manuscript marks as confirmed
           (HSPA1A 5.26, HSPA1B 4.04, CRYAB 3.44, HSPB1 3.37, DNAJB1 3.06, HSP90AA1 1.40, HSP90AB1 0.81). A gene with no Z in a cohort is combined from the others and
           gets a dagger.
  Panels B and C  donor-level TF activity differences from the TF tool (results/tf_activity_donor_level/tf_activity_results.csv), variant P3_lognorm by default.
If the Stouffer table's columns cannot be recognised the script stops and prints them; it never guesses numbers.
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


COHORTS = ["Jin", "Ma", "Nido"]
N_DONORS = {"Jin": 6, "Ma": 17, "Nido": 33}
GENES = ["CRYAB", "HSPB1", "DNAJB1", "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1"]
CONFIRMED_COMBINED = {
    "CRYAB": 3.436,
    "HSPB1": 3.371,
    "DNAJB1": 3.055,
    "HSPA1A": 5.260,
    "HSPA1B": 4.037,
    "HSP90AA1": 1.398,
    "HSP90AB1": 0.814,
}
META_FILE, TF_FILE = os.environ.get("FIG_META_FILE", "Meta_Analysis_Results_DONOR_LEVEL.csv"), Path(
    "results/tf_activity_donor_level/tf_activity_results.csv"
)
TF_VARIANT = "P3_lognorm"
RESULTS_DIR = Path("results/figure5")
TF_ORDER = ["NFKB1", "STAT3", "HSF1", "ATF6", "XBP1", "NFE2L2", "TP53", "ATF4"]
LOCKED_TF = {
    ("Nido", "ATF4"): (0.261, 0.156),
    ("Nido", "HSF1"): (0.001, 0.782),
    ("Nido", "NFKB1"): (-0.241, 0.347),
    ("Nido", "TP53"): (0.184, 0.231),
    ("Ma", "ATF4"): (0.497, 0.114),
    ("Ma", "HSF1"): (0.755, 0.139),
    ("Ma", "NFKB1"): (-0.309, 0.743),
    ("Ma", "TP53"): (-0.155, 0.963),
}  # (activity difference, p): the corrected donor-level analysis in the manuscript Results


def check_locked_tf(tab, variant=TF_VARIANT):
    """Problems that make a TF table unfit for Figure 5. An empty list means the table is the corrected analysis. This exists because the figure once showed pre-correction values (Nido ATF4 +0.575, p = 0.019)
    from an earlier table; the generator now refuses to draw from any table that disagrees with the manuscript.
    """
    sel = tab[tab.variant == variant]
    if sel.empty:
        return [
            f"no rows for variant {variant!r}; variants present: {sorted(tab.variant.unique())}"
        ]
    problems = []
    for (c, f), (d, p) in LOCKED_TF.items():
        r = sel[(sel.cohort == c) & (sel.TF == f)]
        if len(r) != 1:
            problems.append(f"{c} {f}: {len(r)} rows (expected 1)")
            continue
        got_d, got_p = float(r.mean_diff.iloc[0]), float(r.p.iloc[0])
        if round(got_d, 3) != d or round(got_p, 3) != p:
            problems.append(
                f"{c} {f}: the table has {got_d:+.3f} (p = {got_p:.3f}); the manuscript has {d:+.3f} (p = {p:.3f})"
            )
    missing = [
        (c, f)
        for c in ("Nido", "Ma")
        for f in TF_ORDER
        if sel[(sel.cohort == c) & (sel.TF == f)].empty
    ]
    if missing:
        problems.append(f"missing factors: {missing}")
    for c in ("Nido", "Ma"):
        sig = sel[(sel.cohort == c) & (sel.p < 0.05)].TF.tolist()
        if sig:
            problems.append(
                f"{c}: {sig} have p < 0.05, but the manuscript states that none of the eight factors reaches significance"
            )
    return problems


TF_LABELS = {
    "NFKB1": "NF-\u03baB1",
    "STAT3": "STAT3",
    "HSF1": "HSF1",
    "ATF6": "ATF6",
    "XBP1": "XBP1",
    "NFE2L2": "NRF2",
    "TP53": "TP53",
    "ATF4": "ATF4",
}
ZPAT = r"(^|[_\s.\-])z($|[_\s.\-])|zscore|z_score|z-score"


def detect_meta_columns(df):
    gene = next(
        (
            c
            for c in df.columns
            if str(c).lower() in ("gene", "genes", "gene_symbol", "symbol", "gene_name")
        ),
        df.columns[0],
    )
    zc = {}
    for co in COHORTS:
        c = [
            c
            for c in df.columns
            if co.lower() in str(c).lower() and re.search(ZPAT, str(c).lower())
        ]
        if c:
            zc[co] = c[0]
    comb = [
        c
        for c in df.columns
        if re.search(r"combined|meta|stouffer", str(c).lower())
        and re.search(ZPAT, str(c).lower())
        and not any(co.lower() in str(c).lower() for co in COHORTS)
    ]
    if len(zc) < 2:
        raise KeyError(
            f"cannot find per-cohort Z columns for at least two of {COHORTS}. Columns present: {list(df.columns)}"
        )
    return gene, zc, (comb[0] if comb else None)


def stouffer(z):
    w = {c: np.sqrt(N_DONORS[c]) for c in z if np.isfinite(z[c])}
    return (
        float(sum(w[c] * z[c] for c in w) / np.sqrt(sum(v**2 for v in w.values()))) if w else np.nan
    )


def meta_table(df):
    gene, zc, comb = detect_meta_columns(df)
    t = df[df[gene].astype(str).isin(GENES)].copy()
    missing = sorted(set(GENES) - set(t[gene].astype(str)))
    if missing:
        raise KeyError(f"genes missing from the meta-analysis table: {missing}")
    out = pd.DataFrame({"gene": t[gene].astype(str).values}).set_index("gene")
    for co in COHORTS:
        out[co] = pd.to_numeric(t[zc[co]], errors="coerce").values if co in zc else np.nan
    out["combined"] = [stouffer({co: out.loc[g, co] for co in COHORTS}) for g in out.index]
    out["n_cohorts"] = out[COHORTS].notna().sum(axis=1)
    out["combined_supplied"] = pd.to_numeric(t[comb], errors="coerce").values if comb else np.nan
    out["confirmed_in_manuscript"] = [CONFIRMED_COMBINED.get(g) for g in out.index]
    out["recomputed_matches_confirmed"] = [
        (bool(np.isfinite(a)) and b is not None and abs(float(a) - float(b)) <= 0.006)
        for a, b in zip(out.combined, out.confirmed_in_manuscript)
    ]
    return out.loc[GENES]


def place_legend(ax):
    """Put the legend where it covers no data point. Tries every corner and edge, one or four columns; if none is free, makes room on the right of the axis in steps. Stops the run if it cannot."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    xmax_data = max(
        float(np.nanmax(c.get_offsets()[:, 0])) for c in ax.collections if len(c.get_offsets())
    )
    pad = 7 * fig.dpi / 72
    for extra in (0.0, 1.5, 3.0, 4.5):
        ax.set_xlim(right=xmax_data + 0.5 + extra)
        fig.canvas.draw()
        pts = np.vstack(
            [
                ax.transData.transform(c.get_offsets())
                for c in ax.collections
                if len(c.get_offsets())
            ]
        )
        for ncol in (1, 4):
            for loc in (
                "lower right",
                "upper right",
                "center right",
                "lower center",
                "upper center",
                "center left",
                "upper left",
                "lower left",
            ):
                leg = ax.legend(
                    loc=loc, ncol=ncol, frameon=False, handletextpad=0.3, borderaxespad=0.2
                )
                fig.canvas.draw()
                bb = leg.get_window_extent(fig.canvas.get_renderer())
                bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
                if not (
                    (pts[:, 0] > bb.x0)
                    & (pts[:, 0] < bb.x1)
                    & (pts[:, 1] > bb.y0)
                    & (pts[:, 1] < bb.y1)
                ).any():
                    return loc
    raise RuntimeError(
        "Figure 5A: the legend cannot be placed without covering a data point, even after widening the axis."
    )


def draw_panel_a(ax, mt):
    order = mt.sort_values("combined").index.tolist()
    y = {g: i for i, g in enumerate(order)}
    style = {
        "Jin": ("o", OI["verm"], "Jin (n = 6)"),
        "Ma": ("s", OI["blue"], "Ma (n = 17)"),
        "Nido": ("^", OI["green"], "Nido (n = 33)"),
    }
    for co, (mk, colr, lab) in style.items():
        v = mt[co]
        ax.scatter(
            v.values,
            [y[g] for g in mt.index],
            marker=mk,
            s=55,
            color=colr,
            edgecolor="black",
            linewidth=0.5,
            label=lab,
            zorder=3,
        )
    ax.scatter(
        mt.combined.values,
        [y[g] for g in mt.index],
        marker="D",
        s=95,
        color="black",
        label="Combined",
        zorder=4,
    )
    ax.axvline(1.96, color=OI["grey"], linestyle="--", linewidth=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([g + ("\u2020" if mt.loc[g, "n_cohorts"] < 3 else "") for g in order])
    ax.set_xlabel("Stouffer Z-score")
    ax.set_xlim(left=min(0, float(np.nanmin(mt[COHORTS].values)) - 0.5))
    place_legend(ax)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_tf_panel(ax, tab, cohort, variant, lo, hi, show_labels):
    t = tab[(tab.cohort == cohort) & (tab.variant == variant)].set_index("TF")
    vals = [t.loc[f, "mean_diff"] for f in TF_ORDER]
    ps = [t.loc[f, "p"] for f in TF_ORDER]
    cols = [
        OI["blue"] if v < 0 else OI["verm"] if p < 0.05 else OI["grey"] for v, p in zip(vals, ps)
    ]
    ax.barh(range(len(TF_ORDER)), vals, color=cols, edgecolor="black", linewidth=0.5)
    for i, p in enumerate(ps):
        mark = "**" if p < 0.01 else "*" if p < 0.05 else ""
        if mark:
            ax.text(hi - 0.01, i, mark, va="center", ha="right", fontweight="bold")
    ax.set_yticks(range(len(TF_ORDER)))
    ax.set_yticklabels([TF_LABELS[f] for f in TF_ORDER] if show_labels else [""] * len(TF_ORDER))
    ax.set_xlim(lo, hi)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Activity difference\n(PD vs control)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return int(sum(p < 0.05 for p in ps)), int(sum(0.05 <= p < 0.15 for p in ps))


def make_figure5(mt, tab, prefix, variant=TF_VARIANT):
    fig = new_fig(170, 154)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1],
        hspace=0.78,
        wspace=0.18,
        left=0.16,
        right=0.97,
        top=0.95,
        bottom=0.14,
    )
    axA = fig.add_subplot(gs[0, :])
    draw_panel_a(axA, mt)
    panel_label(axA, "A", dx=-0.15, dy=0.99)
    sel = tab[(tab.variant == variant) & (tab.cohort.isin(["Nido", "Ma"]))]
    lo = min(0.0, float(sel.mean_diff.min())) - 0.06
    hi = max(0.0, float(sel.mean_diff.max())) + 0.20
    axB, axC = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    nB = draw_tf_panel(axB, tab, "Nido", variant, lo, hi, True)
    nC = draw_tf_panel(axC, tab, "Ma", variant, lo, hi, False)
    axB.set_title("Nido prefrontal (n = 33)", fontsize=12)
    axC.set_title("Ma cingulate (n = 17)", fontsize=12)
    panel_label(axB, "B", dx=-0.42, dy=1.12)
    panel_label(axC, "C", dx=-0.12, dy=1.12)
    alt = (
        f"Panel A: Stouffer Z-scores for seven heat-shock genes in the Jin, Ma and Nido cohorts with the combined estimate; combined Z ranges from {mt.combined.min():.2f} to {mt.combined.max():.2f}. "
        f"Panels B and C: donor-level transcription-factor activity differences (PD minus control) in Nido ({nB[0]} of 8 with p < 0.05) and Ma ({nC[0]} of 8 with p < 0.05)."
    )
    dag = [g for g in mt.index if mt.loc[g, "n_cohorts"] < 3]
    leg = (
        "Fig. 5. Stouffer meta-analysis and transcription factor activity. (A) Stouffer Z-scores for seven primary heat-shock genes. Circle, Jin (n = 6); square, Ma (n = 17); triangle, Nido (n = 33); "
        "black diamond, combined estimate weighted by the square root of the donor number. Dashed line: Z = 1.96 (p = 0.05)."
        + (
            f" \u2020 Combined from the cohorts in which the gene was detected ({', '.join(dag)})."
            if dag
            else ""
        )
        + f" (B, C) Donor-level transcription-factor activity differences (PD minus control) in Nido prefrontal and Ma cingulate cortex: donor mean of per-nucleus log-normalised expression ({variant}), each target gene z-scored "
        "across donors and averaged over the factor's panel (Supplementary Table 1); Mann-Whitney test. Markers: * p < 0.05, ** p < 0.01; none was reached. Bars: blue, negative; vermillion, p < 0.05; grey, other. PD, Parkinson's disease."
    )
    save_figure(fig, prefix, alt, leg)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = resolve_input_path(
        DATA_DIR / META_FILE, "cross-cohort meta-analysis table", ALLOW_DRIVE_FETCH
    )
    mt = meta_table(pd.read_csv(path))
    mt.to_csv(RESULTS_DIR / "stouffer_table_used.csv")
    pd.set_option("display.width", 220)
    print("\nStouffer table (combined Z recomputed with sqrt(n) weights):")
    print(mt.round(3).to_string())
    print(
        "\nRecomputed combined Z matches the seven values the manuscript marks as confirmed:",
        bool(mt.recomputed_matches_confirmed.all()),
    )
    if not mt.recomputed_matches_confirmed.all():
        print(
            "  -> if False, the supplied per-cohort Z columns are not the ones that produced the manuscript's combined values; check them before using panel A."
        )
    tab = pd.read_csv(TF_FILE)
    problems = check_locked_tf(tab)
    if problems and os.environ.get("LBD_ALLOW_UNLOCKED_TF") != "1":
        raise RuntimeError(
            "Figure 5 was not drawn: the TF table does not match the corrected analysis in the manuscript.\n  "
            + "\n  ".join(problems)
            + "\nRe-run tf_activity_donor_level.py. If the manuscript was changed on purpose, update LOCKED_TF and the text together (or set LBD_ALLOW_UNLOCKED_TF=1)."
        )
    make_figure5(mt, tab, str(RESULTS_DIR / "Figure5"))
    print(f"Figure 5 written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
