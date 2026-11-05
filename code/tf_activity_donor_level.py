"""
tf_activity_donor_level.py

Donor-level activity of the curated transcription-factor target panels (Fig. 5B and 5C; Supplementary Table 1).

Method: the donor-level expression of a gene is the mean over that donor's nuclei of log1p(counts / library size x 10,000).
Each gene is z-scored across donors, the activity of a factor is the mean z over its panel genes, and the difference between
PD and control donors is tested with a two-sided Mann-Whitney U test. U is the smaller of the two U statistics.

Output: results/tf_activity_donor_level/tf_activity_results.csv and Figure 5 panels B and C. Jin (3 vs 3 donors) is computed
for completeness and is not used for inference.
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


TF_PANELS = {
    "ATF4": ["DDIT3", "ATF3", "ASNS", "PSAT1", "PHGDH", "SLC7A11", "HERPUD1"],
    "TP53": ["CDKN1A", "MDM2", "BBC3", "PMAIP1", "BAX", "GADD45A", "SESN2"],
    "NFE2L2": ["HMOX1", "NQO1", "GSTM1", "TXNRD1", "PRDX1", "TXN", "SRXN1"],
    "XBP1": ["HSPA5", "DNAJB9", "EDEM1", "PDIA4", "SEC61A1", "DERL1"],
    "ATF6": ["HSPA5", "DDIT3", "HERPUD1", "PDIA4", "DNAJB11"],
    "HSF1": [
        "HSPA1A",
        "HSPA1B",
        "HSPB1",
        "DNAJB1",
        "CRYAB",
        "HSP90AA1",
        "HSPA6",
        "HSPD1",
        "HSPE1",
        "DNAJB6",
        "HSPA2",
        "HSPB8",
        "HSPA8",
        "DNAJB2",
        "HSPH1",
        "BAG3",
        "STIP1",
    ],
    "STAT3": ["BCL2", "MYC", "CCND1", "MCL1", "VEGFA", "HIF1A", "TWIST1"],
    "NFKB1": ["TNF", "IL6", "IL1B", "CXCL10", "CCL2", "PTGS2", "ICAM1", "VCAM1", "BCL2", "BIRC3"],
}

VARIANTS = ["P3_lognorm"]
DEFAULT_VARIANT = "P3_lognorm"
RESULTS_DIR = Path("results/tf_activity_donor_level")


def activity_tables(adata, ctx, cohort):
    genes = sorted({g for v in TF_PANELS.values() for g in v})
    present = [g for g in genes if g in adata.var_names]
    miss = [g for g in genes if g not in adata.var_names]
    if miss:
        print(f"  WARNING {cohort}: panel genes absent from the data: {miss}")
    cnt = gene_counts(adata, ctx, present)
    tot = ctx["totals"]
    donor = ctx["donor"]
    dxd = pd.Series(ctx["dx"], index=donor).groupby(level=0).first()
    pb_by_variant = {"P3_lognorm": np.log1p(cnt / tot[:, None] * 1e4)}
    rows = []
    for v in VARIANTS:
        pb = pd.DataFrame(pb_by_variant[v], columns=present).groupby(donor).mean()
        dxs = dxd.loc[pb.index]
        for tf, tg in TF_PANELS.items():
            g = [x for x in tg if x in present]
            if len(g) < 3:
                continue
            z = (pb[g] - pb[g].mean()) / (pb[g].std() + 1e-8)
            act = z.mean(axis=1)
            a, b = act[dxs == "PD"].values, act[dxs == "Control"].values
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            p = mw.pvalue
            u_small = float(
                min(mw.statistic, len(a) * len(b) - mw.statistic)
            )  # the smaller of the two U statistics, as in the Figure 4 legend
            rows.append(
                dict(
                    cohort=cohort,
                    variant=v,
                    TF=tf,
                    n_genes=len(g),
                    n_PD=len(a),
                    n_control=len(b),
                    mean_PD=a.mean(),
                    mean_control=b.mean(),
                    mean_diff=a.mean() - b.mean(),
                    p=p,
                    U=u_small,
                )
            )
    return pd.DataFrame(rows)


LABELS = {
    "NFKB1": "NF-\u03baB1",
    "STAT3": "STAT3",
    "HSF1": "HSF1 (heat shock)",
    "ATF6": "ATF6 (ER)",
    "XBP1": "XBP1 (ER)",
    "NFE2L2": "NRF2",
    "TP53": "TP53",
    "ATF4": "ATF4 (ISR)",
}
ORDER = [
    "NFKB1",
    "STAT3",
    "HSF1",
    "ATF6",
    "XBP1",
    "NFE2L2",
    "TP53",
    "ATF4",
]  # same order as the original figure


def make_figure5_bc(tab, variant, out_prefix):
    """Panels B (Nido) and C (Ma) drawn from the results table: no number is typed in. Okabe-Ito palette, 12 pt text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 12, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.linewidth": 0.8}
    )
    sel = tab[(tab.variant == variant) & (tab.cohort.isin(["Nido", "Ma"]))]
    lo = min(0.0, float(sel.mean_diff.min())) - 0.06
    hi = max(0.0, float(sel.mean_diff.max())) + 0.20
    fig, axes = plt.subplots(1, 2, figsize=(170 / 25.4, 105 / 25.4), sharex=True)
    for ax, cohort, title, letter in (
        (axes[0], "Nido", "Nido prefrontal\n(n=33; disease-spared)", "B"),
        (axes[1], "Ma", "Ma cingulate\n(n=17; disease-affected)", "C"),
    ):
        t = tab[(tab.cohort == cohort) & (tab.variant == variant)].set_index("TF")
        vals = [t.loc[f, "mean_diff"] for f in ORDER]
        ps = [t.loc[f, "p"] for f in ORDER]
        cols = [
            "#0072B2" if v < 0 else "#D55E00" if p < 0.05 else "#E69F00" if p < 0.15 else "#999999"
            for v, p in zip(vals, ps)
        ]
        ax.barh(range(len(ORDER)), vals, color=cols, edgecolor="black", linewidth=0.5)
        for i, (v, p) in enumerate(zip(vals, ps)):
            mark = "**" if p < 0.01 else "*" if p < 0.05 else "\u2020" if p < 0.15 else ""
            if mark:
                ax.text(hi - 0.01, i, mark, va="center", ha="right", fontweight="bold")
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([LABELS[f] for f in ORDER] if cohort == "Nido" else [""] * len(ORDER))
        ax.set_xlim(lo, hi)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Activity difference\n(PD vs control)")
        ax.set_title(title, fontsize=12)
        ax.text(
            -0.02 if cohort == "Nido" else -0.04,
            1.14,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=600)
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    Image.open(f"{out_prefix}.png").convert("RGB").save(
        f"{out_prefix}.tiff", compression="tiff_lzw", dpi=(600, 600)
    )
    plt.close(fig)
    n_sig = {c: int(((sel.cohort == c) & (sel.p < 0.05)).sum()) for c in ("Nido", "Ma")}
    n_tr = {
        c: int(((sel.cohort == c) & (sel.p >= 0.05) & (sel.p < 0.15)).sum()) for c in ("Nido", "Ma")
    }
    open(f"{out_prefix}_alt_text.txt", "w").write(
        "Horizontal bar charts of the donor-level transcription-factor activity difference (PD minus control) for eight factors in the Nido prefrontal "
        f"cohort (panel B; {n_sig['Nido']} of 8 factors with p < 0.05, {n_tr['Nido']} with 0.05 <= p < 0.15) and the Ma cingulate cohort (panel C; "
        f"{n_sig['Ma']} of 8 with p < 0.05, {n_tr['Ma']} with 0.05 <= p < 0.15). Markers: dagger p < 0.15, * p < 0.05, ** p < 0.01. "
        f"Values come from the results table (variant {variant}); none is typed in.\n"
    )


def tf_cohorts():
    """Cohorts to load. Jin is computed for completeness only and is never used for inference, so the figure runner sets LBD_TF_COHORTS=Ma,Nido and avoids loading a third multi-gigabyte object."""
    got = tuple(
        c.strip() for c in os.environ.get("LBD_TF_COHORTS", "Ma,Nido,Jin").split(",") if c.strip()
    )
    if [c for c in got if c not in FILES] or not {"Ma", "Nido"} <= set(got):
        raise ValueError(
            f"LBD_TF_COHORTS must name known cohorts and include Ma and Nido; got {got}"
        )
    return got


def main():
    tabs = []
    for cohort in tf_cohorts():
        print(f"Loading {cohort}...")
        adata, ctx = load_cohort(cohort)
        tabs.append(activity_tables(adata, ctx, cohort))
        del adata
    tab = pd.concat(tabs, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS_DIR / "tf_activity_results.csv", index=False)
    pd.set_option("display.width", 230, "display.max_colwidth", 120)
    t = (
        tab[(tab.variant == DEFAULT_VARIANT) & (tab.cohort.isin(["Ma", "Nido"]))]
        .pivot(index="TF", columns="cohort", values=["mean_diff", "p"])
        .round(3)
    )
    print(t.to_string())
    make_figure5_bc(tab, DEFAULT_VARIANT, str(RESULTS_DIR / "Figure5_BC_from_results"))
    print(
        f"\nFigure 5 panels B and C written (PDF, PNG, 600 dpi TIFF, alt text). Files in {RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
