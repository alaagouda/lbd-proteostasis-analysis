"""
hypoxia_coupling.py

Correlation between the endothelial Hypoxia and Proteostasis module scores across donors in Ma and Nido (Supplementary Fig. 1).
The Hypoxia list is the 10-gene list of gene_modules.py (Supplementary Table 2). A donor's value is the mean module score of its
endothelial nuclei, scored either on all nuclei of the cohort (whole-cohort scoring) or within the endothelial nuclei only
(within-endothelial scoring). Donors with at least one endothelial nucleus are included, and the association is Spearman's rho.

Output: results/hypoxia_coupling/ (per-donor tables and Supplementary Fig. 1).
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


HYPOXIA_LISTS = {
    "H10_gene_modules_py": [
        "ADM",
        "BNIP3",
        "VEGFA",
        "SLC2A1",
        "LDHA",
        "PGK1",
        "PDK1",
        "ENO1",
        "HK2",
        "PFKFB3",
    ],
}
PROTEOSTASIS14 = [
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
RESULTS_DIR = Path("results/hypoxia_coupling")


def usable_hypoxia_lists(adata):
    """The hypoxia lists whose genes are all in the data. The 10-gene list and the 14 proteostasis genes are required by Supplementary Fig. 1; an alternative list (used only by the sensitivity grid)
    is skipped with a printed warning when one of its genes is absent, instead of stopping the figure.
    """
    present = set(adata.var_names)
    need = sorted((set(PROTEOSTASIS14) | set(HYPOXIA_LISTS["H10_gene_modules_py"])) - present)
    if need:
        raise KeyError(f"genes required for Supplementary Fig. 1 are missing from the data: {need}")
    ok = {}
    for k, v in HYPOXIA_LISTS.items():
        miss = [g for g in v if g not in present]
        if miss:
            print(
                f"  WARNING: alternative hypoxia list {k!r} skipped (it is not used in the figure): genes absent from the data: {miss}"
            )
        else:
            ok[k] = v
    return ok


def endothelial_mask(ct):
    labels = sorted({c for c in ct if "endo" in c.lower()})
    if not labels:
        raise KeyError(f"no endothelial label among {sorted(set(ct))[:25]}")
    return np.isin(ct, labels), labels


def nucleus_scores(adata, ctx, emask, lists=None):
    """{method: {list name: array, 'PROT': array}} for all nuclei (NaN outside the endothelium for within-endothelial scoring)."""
    import scanpy as sc

    lists = lists or HYPOXIA_LISTS
    allg = sorted(set(PROTEOSTASIS14) | {g for v in lists.values() for g in v})
    cnt = gene_counts(adata, ctx, allg)
    col = {g: i for i, g in enumerate(allg)}
    sets = dict(lists, PROT=PROTEOSTASIS14)
    out = {}
    lognorm_in_place(adata, ctx)
    out["score_whole"] = {}
    for k, v in sets.items():
        sc.tl.score_genes(adata, v, score_name="_s", ctrl_size=50, random_state=0, use_raw=False)
        out["score_whole"][k] = adata.obs["_s"].values.copy()
    sub = adata[emask].copy()
    out["score_within"] = {}
    for k, v in sets.items():
        sc.tl.score_genes(sub, v, score_name="_s", ctrl_size=50, random_state=0, use_raw=False)
        full = np.full(adata.n_obs, np.nan)
        full[emask] = sub.obs["_s"].values
        out["score_within"][k] = full
    return out, dict(cnt=cnt, col=col)


def donor_table(method, hname, minn, scores, aux, ctx, emask):
    d = pd.DataFrame({"donor": ctx["donor"][emask], "dx": ctx["dx"][emask]})
    d["hypoxia"] = scores[method][hname][emask]
    d["proteostasis"] = scores[method]["PROT"][emask]
    g = d.groupby("donor").agg(
        hypoxia=("hypoxia", "mean"),
        proteostasis=("proteostasis", "mean"),
        n_nuclei=("hypoxia", "size"),
    )
    return g[g.n_nuclei >= minn]


def grid_for_cohort(cohort, adata, ctx, return_tables=False):
    emask, labels = endothelial_mask(ctx["ct"])
    print(f"  {cohort}: endothelial labels {labels}: {int(emask.sum()):,} nuclei")
    lists = usable_hypoxia_lists(adata)
    scores, aux = nucleus_scores(adata, ctx, emask, lists)
    rows = []
    tables = {}
    dxmap = dict(zip(ctx["donor"], ctx["dx"]))
    for method in ("score_whole", "score_within"):
        for hname in lists:
            for minn in (1,):
                t = donor_table(method, hname, minn, scores, aux, ctx, emask)
                if len(t) < 4:
                    continue
                t = t.copy()
                t["dx"] = [dxmap.get(i, "NA") for i in t.index]
                tables[(method, hname, minn)] = t
                r, p = stats.spearmanr(t["hypoxia"], t["proteostasis"])
                rows.append(
                    dict(
                        cohort=cohort,
                        method=method,
                        hypoxia_list=hname,
                        min_nuclei=minn,
                        n_donors=len(t),
                        rho=r,
                        p=p,
                    )
                )
    g = pd.DataFrame(rows)
    return (g, tables) if return_tables else g


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


FIG_PIPELINES = [
    ("score_within", "H10_gene_modules_py", 1, "Within-endothelial scoring"),
    ("score_whole", "H10_gene_modules_py", 1, "Whole-cohort scoring"),
]


def make_supp_fig1(tables_by_cohort, prefix, pipelines=FIG_PIPELINES):
    """Two rows (the two scoring methods) x two columns (Ma, Nido), one point per donor. Both rows are shown because the coupling's significance depends on the scoring."""
    fig = new_fig(170, 150)
    gs = fig.add_gridspec(
        2, 2, hspace=1.05, wspace=0.42, left=0.20, right=0.97, top=0.83, bottom=0.10
    )
    col = {"PD": OI["orange"], "Control": OI["blue"]}
    stats_out = []
    for r, (method, hname, minn, label) in enumerate(pipelines):
        for c, cohort in enumerate(("Ma", "Nido")):
            ax = fig.add_subplot(gs[r, c])
            t = tables_by_cohort[cohort][(method, hname, minn)]
            rho, p = stats.spearmanr(t.hypoxia, t.proteostasis)
            stats_out.append((label, cohort, len(t), rho, p))
            ax.scatter(
                t.hypoxia,
                t.proteostasis,
                c=[col.get(d, OI["grey"]) for d in t.dx],
                edgecolor="black",
                linewidth=0.4,
                s=32,
                zorder=3,
            )
            ax.set_title(
                f"{label}\n{cohort} (n = {len(t)})\n\u03c1 = {rho:+.2f}, "
                + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}"),
                fontsize=12,
                pad=6,
            )
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            if c == 0:
                ax.set_ylabel("Endothelial proteostasis\nscore")
            if r == len(pipelines) - 1:
                ax.set_xlabel("Endothelial hypoxia score")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            panel_label(ax, "ABCD"[r * 2 + c], dx=-0.30 if c == 0 else -0.24, dy=1.02)
    sig = [f"{a} in {b}" for a, b, n, rho, p in stats_out if p < 0.05]
    alt = (
        "Four scatter plots with one point per donor: endothelial hypoxia score against endothelial proteostasis score in the Ma and Nido cohorts, scored within endothelial nuclei (top row) and on the whole cohort (bottom row). "
        + ("Nominally significant: " + "; ".join(sig) + "." if sig else "None reaches p < 0.05.")
    )
    leg = (
        "Supplementary Fig. 1. Endothelial hypoxia-proteostasis coupling is sensitive to how the module is scored. One point per donor: mean of the 10-gene Hypoxia module score (Supplementary Table 2) against "
        "the mean of the 14-gene Proteostasis module score in endothelial nuclei, both from Scanpy score_genes, within-endothelial scoring (A, B: score_genes run on the endothelial nuclei alone) or whole-cohort scoring (C, D: score_genes run on all nuclei, then restricted to endothelial nuclei). "
        "Spearman correlation across donors; "
        + "; ".join(
            f"{b} ({a.lower()}): rho = {rho:+.2f}, "
            + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")
            + f", n = {n}"
            for a, b, n, rho, p in stats_out
        )
        + ". Orange, PD; blue, control. Exploratory."
    )
    save_figure(fig, prefix, alt, leg)
    return stats_out


def main():
    grids, tabs = [], {}
    for cohort in ("Ma", "Nido"):
        print(f"Loading {cohort}...")
        adata, ctx = load_cohort(cohort)
        g, tb = grid_for_cohort(cohort, adata, ctx, return_tables=True)
        tabs[cohort] = tb
        grids.append(g)
        del adata
    grid = pd.concat(grids, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS_DIR / "correlations_by_cohort.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(grid.round(3).to_string(index=False))
    for cohort in tabs:
        for m, h, k, lab in FIG_PIPELINES:
            tabs[cohort][(m, h, k)].to_csv(
                RESULTS_DIR / f"donor_values_{cohort}_{m}_{h}_min{k}.csv"
            )
    make_supp_fig1(tabs, str(RESULTS_DIR / "SuppFig1_from_results"))
    print(
        "Supplementary Fig. 1 (both scorings, PDF, 600 dpi TIFF, alt text, legend) written from the per-donor tables."
    )
    print(f"\nFiles written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
