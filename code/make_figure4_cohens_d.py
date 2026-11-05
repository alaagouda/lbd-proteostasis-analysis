"""
make_figure4_cohens_d.py

Figure 4 (the proteostasis gradient) drawn FROM THE DATA. The old figure4_cohens_d.py typed its numbers in, and its confidence intervals were "read off the
original figure by eye": they disagree with the manuscript text for two cohorts (Jin -1.15 to 4.30 vs the text's -1.17 to 4.31; Ma -0.70 vs -0.69).

METHOD (the verified one): per-nucleus CPM of the 14 proteostasis genes, averaged across genes, then across a donor's nuclei -> one value per donor.
  Cohen's d = (mean disease - mean control) / pooled SD.  95% CI = d +/- t(df, 0.975) * sqrt((n1+n2)/(n1*n2) + d^2 / (2*df)), df = n1+n2-2.
  That formula reproduces all three CIs printed in the manuscript to two decimals (Jin -1.17 to 4.31, Ma -0.69 to 1.40, Nido -0.61 to 0.84).
  p = two-sided Mann-Whitney U on the donor values.
PANEL B shows every donor (the journal requires individual points when a group has fewer than 10), so Jin's 3 + 3 donors are visible.
MATCH RULE (fixed): d to 3 decimals, CI limits to 2 decimals, p to 3 decimals, group sizes exactly. If anything fails, the FIGURE is still drawn from the data and the
  TEXT must be updated to the printed table; nothing is forced to agree.
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


PROT14 = [
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
REPORTED = {
    "Jin": dict(d=1.568, ci=(-1.17, 4.31), p=0.100, n=(3, 3), u=0),
    "Ma": dict(d=0.357, ci=(-0.69, 1.40), p=0.606, n=(9, 8), u=30),
    "Nido": dict(d=0.114, ci=(-0.61, 0.84), p=0.927, n=(20, 13), u=127),
}
LABELS = {"Jin": "Jin\ntemporal", "Ma": "Ma\ncingulate", "Nido": "Nido\nprefrontal"}
DISEASE = {
    "Jin": "DLB",
    "Ma": "PD",
    "Nido": "PD",
}  # diagnosis of the cases in each cohort; Jin is dementia with Lewy bodies
RESULTS_DIR = Path("results/figure4")


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp)


def d_ci(d, n1, n2):
    df = n1 + n2 - 2
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * df))
    h = stats.t.ppf(0.975, df) * se
    return float(d - h), float(d + h)


def donor_scores(adata, ctx):
    cnt = gene_counts(adata, ctx, PROT14)
    score = (cnt / ctx["totals"][:, None] * 1e6).mean(axis=1)
    df = pd.DataFrame({"donor": ctx["donor"], "dx": ctx["dx"], "score": score})
    return df.groupby("donor").agg(
        score=("score", "mean"), dx=("dx", "first"), n_nuclei=("score", "size")
    )


def effect(tab):
    a, b = tab[tab.dx == "PD"].score.values, tab[tab.dx == "Control"].score.values
    d = cohens_d(a, b)
    lo, hi = d_ci(d, len(a), len(b))
    mw = stats.mannwhitneyu(
        a, b, alternative="two-sided"
    )  # the journal asks for the test statistic as well as p
    u1 = float(mw.statistic)
    return dict(
        d=d,
        ci_low=lo,
        ci_high=hi,
        p=float(mw.pvalue),
        u=min(u1, len(a) * len(b) - u1),
        u_disease=u1,
        n_pd=len(a),
        n_control=len(b),
    )  # u: the smaller of the two U statistics, as usually reported


def check(res, reported=REPORTED):
    rows = []
    for c, r in reported.items():
        x = res[c]
        ok = dict(
            d=round_match(x["d"], r["d"], 3),
            ci=round_match(x["ci_low"], r["ci"][0], 2) and round_match(x["ci_high"], r["ci"][1], 2),
            p=round_match(x["p"], r["p"], 3),
            n=(x["n_pd"], x["n_control"]) == r["n"],
            u=x["u"] == r["u"],
        )
        rows.append(
            dict(
                cohort=c,
                computed_d=round(x["d"], 3),
                reported_d=r["d"],
                computed_ci=f"{x['ci_low']:+.2f} to {x['ci_high']:+.2f}",
                reported_ci=f"{r['ci'][0]:+.2f} to {r['ci'][1]:+.2f}",
                computed_p=round(x["p"], 3),
                reported_p=r["p"],
                computed_u=x["u"],
                reported_u=r["u"],
                n=f"{x['n_pd']}+{x['n_control']}",
                **{f"match_{k}": v for k, v in ok.items()},
                ALL=all(ok.values()),
            )
        )
    return pd.DataFrame(rows)


def make_figure4(res, tables, prefix, order=("Jin", "Ma", "Nido")):
    fig = new_fig(170, 152)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.15, 1],
        hspace=1.05,
        wspace=0.42,
        left=0.17,
        right=0.97,
        top=0.95,
        bottom=0.075,
    )
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(order))
    d = np.array([res[c]["d"] for c in order])
    lo = np.array([res[c]["ci_low"] for c in order])
    hi = np.array([res[c]["ci_high"] for c in order])
    ax.axhline(0.8, color=OI["grey"], linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.plot(x, d, color=OI["blue"], linewidth=1.8, zorder=2)
    ax.errorbar(
        x,
        d,
        yerr=[d - lo, hi - d],
        fmt="o",
        color=OI["blue"],
        markersize=8,
        capsize=4,
        elinewidth=1.5,
        zorder=3,
    )
    ax.text(
        len(order) - 0.8,
        0.8,
        "large effect\n(d = 0.8)",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#555555",
    )  # to the right of the last point and its interval, so it cannot sit on an error bar
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{LABELS[c]}\n({res[c]['n_pd']} {DISEASE[c]} +\n{res[c]['n_control']} control)\nd = {res[c]['d']:.2f}\n[{res[c]['ci_low']:+.2f}, {res[c]['ci_high']:+.2f}]"
            for c in order
        ]
    )
    ax.set_xlim(-0.45, len(order))
    ax.set_ylim(min(lo.min(), 0) - 0.4, hi.max() + 0.5)
    ax.set_ylabel("Cohen's d\n(disease - control)\nwith 95% CI")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    panel_label(ax, "A", dx=-0.17, dy=0.98)
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _ext = [t.get_window_extent(_r) for t in ax.get_xticklabels()]
    _gap = min(b.x0 - a.x1 for a, b in zip(_ext, _ext[1:])) / fig.dpi * 25.4
    if _gap < 2.0:
        raise RuntimeError(
            f"Figure 4: the cohort labels under panel A are only {_gap:.1f} mm apart (need at least 2 mm); they overlap or nearly touch. Shorten them or widen the axis."
        )
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        a = fig.add_subplot(gs[1, i])
        t = tables[c]
        for j, (dxl, colr) in enumerate((("Control", OI["blue"]), ("PD", OI["orange"]))):
            v = t[t.dx == dxl].score.values
            a.scatter(
                j + rng.uniform(-0.12, 0.12, len(v)),
                v,
                c=colr,
                edgecolor="black",
                linewidth=0.4,
                s=30,
                zorder=3,
            )
            if len(v):
                a.hlines(np.median(v), j - 0.28, j + 0.28, color="black", linewidth=2, zorder=4)
        a.set_xticks([0, 1])
        a.set_xticklabels(["Ctrl", DISEASE[c]])
        a.set_xlim(-0.6, 1.6)
        a.set_title(f"{c}\np = {res[c]['p']:.3f}", fontsize=12)
        a.set_ylabel("Proteostasis score\n(CPM)" if i == 0 else "")
        a.ticklabel_format(axis="y", style="plain")
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if i == 0:
            panel_label(a, "B", dx=-0.62, dy=1.28)
    r = {c: res[c] for c in order}
    alt = (
        f"Panel A: Cohen's d with 95 percent confidence interval for the 14-gene proteostasis module, disease versus control, in Jin temporal cortex (d = {r['Jin']['d']:.2f}), "
        f"Ma cingulate cortex (d = {r['Ma']['d']:.2f}) and Nido prefrontal cortex (d = {r['Nido']['d']:.2f}), with a dashed line at the large-effect threshold d = 0.8. "
        "Panel B: every donor's proteostasis score by diagnosis in each cohort."
    )
    leg = (
        "Fig. 4. Proteostasis module activity follows regional Lewy body burden. (A) Cohen's d for the 14-gene proteostasis module, disease versus control, with 95% CI "
        "(d plus or minus t(df) x SE, SE = sqrt[(n1+n2)/(n1 n2) + d^2/(2 df)]). Dashed line: d = 0.8. "
        + "; ".join(
            f"{c}: d = {r[c]['d']:.3f} (95% CI {r[c]['ci_low']:.2f} to {r[c]['ci_high']:.2f}), n = {r[c]['n_pd']} {DISEASE[c]} + {r[c]['n_control']} control donors, two-sided Mann-Whitney U = {r[c]['u']:g} (the smaller of the two U statistics), p = {r[c]['p']:.3f}"
            for c in order
        )
        + ". (B) The donor values behind each estimate (one point per donor; line = median; per-nucleus CPM averaged across the 14 genes and then across each donor's nuclei). DLB, dementia with Lewy bodies; PD, Parkinson's disease."
    )
    save_figure(fig, prefix, alt, leg)


def main():
    res, tables = {}, {}
    for cohort in ("Jin", "Ma", "Nido"):
        print(f"Loading {cohort}...")
        adata, ctx = load_cohort(cohort)
        tables[cohort] = donor_scores(adata, ctx)
        res[cohort] = effect(tables[cohort])
        del adata
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chk = check(res)
    chk.to_csv(RESULTS_DIR / "figure4_reproduction_check.csv", index=False)
    for c, t in tables.items():
        t.to_csv(RESULTS_DIR / f"donor_scores_{c}.csv")
    pd.set_option("display.width", 220)
    print("\nDoes the data reproduce the numbers printed in the manuscript?")
    print(chk.to_string(index=False))
    print(
        "\nRESULT:",
        (
            "all numbers reproduce; the figure and the text agree."
            if chk.ALL.all()
            else "SOME NUMBERS DO NOT REPRODUCE. The figure is drawn from the data; update the text to the 'computed' columns above."
        ),
    )
    make_figure4(res, tables, str(RESULTS_DIR / "Figure4"))
    print(f"Figure 4 (PDF, 600 dpi TIFF, PNG, alt text, legend) written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
