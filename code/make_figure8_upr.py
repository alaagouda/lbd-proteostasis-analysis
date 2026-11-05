"""
make_figure8_upr.py

Figure 8: (A) combined Stouffer Z of the seven HSP genes, (B) cell-type-level Unfolded Protein Response enrichment (NES, FDR) in Nido prefrontal cortex.

WARNING ABOUT THE OLD SCRIPT: figure8_upr.py contains the RETRACTED UPR values (endothelial NES 1.835 marked significant, astrocytes -1.157, microglia -1.063). The manuscript and the
  response letter report the corrected ones (endothelial +1.206, FDR 0.265; astrocytes +0.807, FDR 0.996; microglia -1.027, FDR 0.561; OPCs +0.792, FDR 0.914). This script REFUSES to draw
  a table that still contains the retracted values, and prints how the supplied table compares with the corrected ones.
INPUT B: a table with one row per broad cell-type category: columns for the category, NES and FDR (for categories made of several fine subtypes, the combined FDR the manuscript describes).
INPUT A: the same per-cohort meta-analysis table as Figure 5 (cross_cohort_meta_analysis.csv).
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


UPR_FILE = "Nido_celltype_UPR_summary.csv"
RESULTS_DIR = Path("results/figure8")
RETRACTED = {"Endothelial": 1.835, "Astrocytes": -1.157, "Microglia": -1.063}
CORRECTED = {
    "Astrocytes": (0.807, 0.996),
    "OPCs": (0.792, 0.914),
    "Microglia": (-1.027, 0.561),
    "Endothelial": (1.206, 0.265),
}
FAMILY = {
    "DNAJB1": ("HSP40", OI["purple"]),
    "HSPA1A": ("HSP70", OI["blue"]),
    "HSPA1B": ("HSP70", OI["blue"]),
    "HSP90AA1": ("HSP90", OI["grey"]),
    "HSP90AB1": ("HSP90", OI["grey"]),
    "HSPB1": ("small HSP", OI["orange"]),
    "CRYAB": ("small HSP", OI["orange"]),
}


def norm_name(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def load_upr(df):
    low = {str(c).lower(): c for c in df.columns}
    cat = next(
        (
            low[k]
            for k in low
            if k in ("cell_type", "celltype", "category", "broad", "broad_category", "cell type")
        ),
        df.columns[0],
    )
    nes = next((low[k] for k in low if "nes" in k), None)
    fdr = next((low[k] for k in low if "fdr" in k or "padj" in k or "qval" in k or k == "q"), None)
    if nes is None or fdr is None:
        raise KeyError(f"cannot find NES and FDR columns. Columns present: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "category": df[cat].astype(str),
            "NES": pd.to_numeric(df[nes], errors="coerce"),
            "FDR": pd.to_numeric(df[fdr], errors="coerce"),
        }
    )
    bas = next((low[k] for k in low if k in ("basis", "statistic", "stat")), None)
    out["basis"] = (
        df[bas].astype(str).values if bas else "FDR"
    )  # "FDR" or "combined p" (Stouffer's method across constituent subtypes)
    return out.dropna(subset=["NES", "FDR"])


def guard_retracted(upr):
    bad = []
    for _, r in upr.iterrows():
        for k, v in RETRACTED.items():
            if norm_name(k).rstrip("s") in norm_name(r.category) and abs(r.NES - v) < 0.01:
                bad.append(f"{r.category} NES {r.NES:+.3f}")
    if bad:
        raise ValueError(
            "this table still contains the RETRACTED UPR values ("
            + "; ".join(bad)
            + "). Rebuild it from the corrected GSEA output; refusing to draw the figure."
        )


def compare_corrected(upr):
    rows = []
    for k, (nes, fdr) in CORRECTED.items():
        m = upr[upr.category.map(norm_name).str.contains(norm_name(k).rstrip("s"))]
        if m.empty:
            rows.append(
                dict(
                    category=k,
                    corrected_NES=nes,
                    corrected_FDR=fdr,
                    in_table="missing",
                    agrees=False,
                )
            )
            continue
        r = m.iloc[0]
        rows.append(
            dict(
                category=k,
                corrected_NES=nes,
                corrected_FDR=fdr,
                in_table=f"NES {r.NES:+.3f}, FDR {r.FDR:.3f}",
                agrees=round_match(r.NES, nes, 2) and round_match(r.FDR, fdr, 2),
            )
        )
    return pd.DataFrame(rows)


def make_figure8(mt, upr, prefix):
    fig = new_fig(170, 150)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1, 1], hspace=0.62, left=0.27, right=0.96, top=0.95, bottom=0.10
    )
    a = fig.add_subplot(gs[0, 0])
    order = mt.sort_values("combined").index.tolist()
    y = np.arange(len(order))
    a.barh(
        y,
        mt.loc[order, "combined"].values,
        color=[FAMILY[g][1] for g in order],
        edgecolor="black",
        linewidth=0.5,
        height=0.7,
    )
    a.axvline(1.96, color=OI["grey"], linestyle="--", linewidth=1)
    a.set_yticks(y)
    a.set_yticklabels(
        [f"{g} ({FAMILY[g][0]})" + ("\u2020" if mt.loc[g, "n_cohorts"] < 3 else "") for g in order]
    )
    a.set_xlabel("Combined Stouffer Z-score")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    panel_label(a, "A", dx=-0.36, dy=1.02)
    b = fig.add_subplot(gs[1, 0])
    u = upr.sort_values("NES")
    yy = np.arange(len(u))
    hi = float(u.NES.max()) + 1.9
    lo = min(0.0, float(u.NES.min())) - 0.3
    b.barh(
        yy,
        u.NES.values,
        color=[OI["verm"] if v > 0 else OI["blue"] for v in u.NES],
        edgecolor="black",
        linewidth=0.5,
        height=0.7,
    )
    b.axvline(0, color="black", linewidth=0.7)
    b.set_yticks(yy)
    b.set_yticklabels([c.replace("_", " ") for c in u.category])
    b.set_xlim(lo, hi)
    b.set_xlabel("Normalised enrichment score (UPR hallmark)")
    for k, (_, r) in enumerate(u.iterrows()):
        b.text(
            hi - 0.05,
            k,
            (
                (f"p {r.FDR:.4f}" if r.FDR >= 0.001 else "p < 0.001")
                if "combined" in str(r.basis).lower()
                else f"FDR {r.FDR:.3f}"
            )
            + (" *" if r.FDR < 0.05 else ""),
            va="center",
            ha="right",
            fontsize=12,
        )
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    panel_label(b, "B", dx=-0.36, dy=1.02)
    sig = u[u.FDR < 0.05].category.tolist()
    W7 = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "all seven"}
    above = [g for g in reversed(order) if mt.loc[g, "combined"] > 1.96]
    below = [g for g in reversed(order) if mt.loc[g, "combined"] <= 1.96]
    where = (
        f"{W7[len(above)]} of the seven exceed the Z = 1.96 line ({', '.join(above)})"
        if len(above) < 7
        else "all seven exceed the Z = 1.96 line"
    ) + (f" and {' and '.join(below)} {'do' if len(below) > 1 else 'does'} not" if below else "")
    dag = [g for g in order if mt.loc[g, "n_cohorts"] < 3]
    dag_txt = (
        f"\u2020 Cohorts in which the gene was detected only ({', '.join(dag)} below detection in some cohorts). "
        if dag
        else ""
    )
    alt = (
        f"Panel A: bars of the combined Stouffer Z-score for seven heat-shock genes, coloured by gene family; {where}. Panel B: bars of the unfolded protein response normalised enrichment score "
        f"for {len(u)} cell-type categories in Nido prefrontal cortex; {len(sig)} reach FDR < 0.05"
        + (f" ({', '.join(sig)})." if sig else ".")
    )
    leg = (
        "Fig. 8. Stouffer meta-analysis Z-scores and UPR cell-type enrichment. (A) Combined Stouffer Z-scores for the seven primary HSP genes; bars coloured by gene family, which is also named on the axis. "
        "Dashed line: Z = 1.96. "
        + dag_txt
        + "(B) Unfolded Protein Response hallmark enrichment (NES; GSEApy preranked, "
        "Hallmarks 2020) for each cell-type category in Nido prefrontal cortex (n = 33). FDR from the category's single constituent subtype; for categories made of several subtypes (excitatory and inhibitory neurons) the combined p by Stouffer's method is shown. * FDR or combined p < 0.05; bars are orange for a positive and blue for a negative NES. Categories made of several subtypes show the Stouffer-combined p of the subtypes' signed nominal p-values, which is not adjusted for multiple testing."
    )
    save_figure(fig, prefix, alt, leg)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        mt = meta_table(
            pd.read_csv(
                resolve_input_path(DATA_DIR / META_FILE, "meta-analysis table", ALLOW_DRIVE_FETCH)
            )
        )
    except Exception as e:
        print(
            f"\n*** NOTICE: the per-cohort meta-analysis table could not be used ({type(e).__name__}: {str(e)[:120]})."
        )
        print(
            "*** Panel A is drawn from the combined Stouffer Z values stated in the manuscript (not recomputed)."
        )
        mt = pd.DataFrame(
            {
                "combined": pd.Series(CONFIRMED_COMBINED),
                "n_cohorts": pd.Series({g: 3 for g in GENES}),
            }
        ).loc[GENES]
    upr = load_upr(
        pd.read_csv(
            resolve_input_path(DATA_DIR / UPR_FILE, "cell-type UPR summary", ALLOW_DRIVE_FETCH)
        )
    )
    guard_retracted(upr)
    cmp_ = compare_corrected(upr)
    cmp_.to_csv(RESULTS_DIR / "figure8_vs_corrected_values.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nSupplied table versus the corrected values reported to the reviewers:")
    print(cmp_.to_string(index=False))
    if not cmp_.agrees.all():
        print(
            "\nTHE TABLE DOES NOT AGREE WITH THE CORRECTED VALUES. Resolve this before using the figure."
        )
    make_figure8(mt, upr, str(RESULTS_DIR / "Figure8"))
    print(f"Figure 8 written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
