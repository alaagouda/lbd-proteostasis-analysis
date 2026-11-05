"""
pde10a_snca_donor_level.py

Donor-level correlations behind Supplementary Fig. 3 and the PDE10A and SNCA Results.

Definitions (one value per donor: the mean per-nucleus CPM over the donor's nuclei):
  PDE10A  whole-tissue expression versus the Proteostasis module score (14 genes, whole tissue)
  SNCA    endothelial-nuclei expression versus the Proteostasis module score of the endothelial nuclei
Spearman rho; p by permutation (exact enumeration when n <= 8, otherwise 200,000 random permutations, two-sided, (b+1)/(N+1)),
and a second permutation that shuffles only within diagnosis. FDR: Benjamini-Hochberg across the four primary tests
(Ma and Nido, PDE10A and SNCA). Jin (n = 6) is descriptive only.

Output: results/pde10a_snca_donor_level/ (per-donor values, test results and Supplementary Fig. 3).
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
VARIANTS = ["D1_cpm"]
DEFAULT_VARIANT = "D1_cpm"
SCOPES = {"PDE10A": "whole", "SNCA": "endothelial"}
N_PERM = 200_000
RESULTS_DIR = Path("results/pde10a_snca_donor_level")


def spearman_perm(x, y, groups=None, n_perm=N_PERM, seed=0):
    """Spearman rho and a two-sided permutation p. Exact for n <= 8 without groups; otherwise Monte Carlo, optionally within groups."""
    import itertools

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xr, yr = stats.rankdata(x), stats.rankdata(y)
    xc, yc = xr - xr.mean(), yr - yr.mean()
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    obs = float((xc * yc).sum() / den)
    if groups is None and n <= 8:
        perms = np.array(list(itertools.permutations(range(n))))
        r = (yc[perms] @ xc) / den
        return obs, float((np.abs(r) >= abs(obs) - 1e-12).mean())
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    glist = (
        [np.where(groups == g)[0] for g in pd.unique(groups)]
        if groups is not None
        else [np.arange(n)]
    )
    while done < n_perm:
        b = min(50_000, n_perm - done)
        perm = np.tile(np.arange(n), (b, 1))
        for idx in glist:
            perm[:, idx] = idx[np.argsort(rng.random((b, len(idx))), axis=1)]
        r = (yc[perm] @ xc) / den
        ge += int((np.abs(r) >= abs(obs) - 1e-12).sum())
        done += b
    return obs, (ge + 1) / (n_perm + 1)


def endothelial_mask(ct):
    labels = sorted({c for c in ct if "endo" in c.lower()})
    if not labels:
        raise KeyError(f"no endothelial label among {sorted(set(ct))[:25]}")
    return np.isin(ct, labels)


def donor_values(adata, ctx):
    """{variant: DataFrame(donor, dx, PDE10A_whole, SNCA_endo, PROT_whole, PROT_endo)}"""
    genes = ["PDE10A", "SNCA"] + PROT14
    cnt = gene_counts(adata, ctx, genes)
    tot = ctx["totals"]
    col = {g: i for i, g in enumerate(genes)}
    donor = ctx["donor"]
    dx = pd.Series(ctx["dx"], index=donor).groupby(level=0).first()
    emask = endothelial_mask(ctx["ct"])
    out = {}
    pi = [col[g] for g in PROT14]
    for v in VARIANTS:
        M = cnt / tot[:, None] * 1e6
        f = lambda arr, mask: pd.Series(arr[mask]).groupby(donor[mask]).mean()
        allm = np.ones(len(donor), bool)
        pde = f(M[:, col["PDE10A"]], allm)
        sn = f(M[:, col["SNCA"]], emask)
        pw = f(M[:, pi].mean(axis=1), allm)
        pe = f(M[:, pi].mean(axis=1), emask)
        d = pd.DataFrame({"PDE10A_whole": pde, "SNCA_endo": sn, "PROT_whole": pw, "PROT_endo": pe})
        d.insert(0, "dx", dx.loc[d.index])
        out[v] = d
    return out


def tests_for(cohort, tables):
    rows = []
    for v, d in tables.items():
        for scope in ("whole", "endothelial"):
            for target, xcol, ycol in (
                ("PDE10A", "PDE10A_whole", "PROT_whole"),
                ("SNCA", "SNCA_endo", "PROT_whole" if scope == "whole" else "PROT_endo"),
            ):
                if target == "PDE10A" and scope == "endothelial":
                    continue
                z = d[[xcol, ycol, "dx"]].dropna()
                rho, p = spearman_perm(z[xcol], z[ycol])
                _, pw = (
                    spearman_perm(z[xcol], z[ycol], groups=z["dx"].values)
                    if len(z) > 8
                    else (np.nan, np.nan)
                )
                rows.append(
                    dict(
                        cohort=cohort,
                        variant=v,
                        proteostasis_scope=scope,
                        target=target,
                        n=len(z),
                        rho=rho,
                        perm_p=p,
                        within_dx_perm_p=pw,
                    )
                )
    return pd.DataFrame(rows)


def assemble(tests, variant, scopes):
    """One row per (cohort, target) for the chosen variant and per-target scope; BH across the four primary tests (Ma and Nido x PDE10A and SNCA)."""
    parts = []
    for target, sc in scopes.items():
        r = tests[
            (tests.variant == variant) & (tests.target == target) & (tests.proteostasis_scope == sc)
        ]
        if r.empty:
            raise KeyError(
                f"no results for {target} with proteostasis scope '{sc}' under variant {variant}"
            )
        parts.append(r)
    sel = pd.concat(parts, ignore_index=True)
    prim = sel.cohort.isin(["Ma", "Nido"]).values
    sel["BH_q_across_4"] = np.nan
    sel.loc[prim, "BH_q_across_4"] = benjamini_hochberg(sel.loc[prim, "perm_p"].values)
    return sel


UNITS = {"D1_cpm": "CPM"}


def make_supp_fig3(
    per_cohort, tests, out_prefix, variant=DEFAULT_VARIANT, snca_scope=SCOPES["SNCA"]
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {"font.size": 12, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.linewidth": 0.8}
    )
    unit = UNITS[variant]
    fig, axes = plt.subplots(2, 3, figsize=(170 / 25.4, 150 / 25.4))
    col = {"PD": "#D55E00", "Control": "#0072B2"}
    spec = (
        ("PDE10A", "PDE10A_whole", "PROT_whole", "A", f"PDE10A\n({unit})"),
        ("SNCA", "SNCA_endo", None, "B", f"Endothelial SNCA\n({unit})"),
    )
    for r, (target, xcol, ycol, letter, xlab) in enumerate(spec):
        for c, cohort in enumerate(("Ma", "Nido", "Jin")):
            ax = axes[r, c]
            d = per_cohort[cohort]
            yc = ycol or ("PROT_whole" if snca_scope == "whole" else "PROT_endo")
            z = d[[xcol, yc, "dx"]].dropna()
            ax.scatter(
                z[xcol],
                z[yc],
                c=[col.get(t, "#999999") for t in z.dx],
                edgecolor="black",
                linewidth=0.4,
                s=28,
                zorder=3,
            )
            tt = tests[(tests.cohort == cohort) & (tests.target == target)]
            if tt.empty:
                raise KeyError(f"no result for {target} in {cohort}; cannot draw the panel")
            t = tt.iloc[0]
            pt = f"{t.perm_p:.3f}" if t.perm_p >= 0.001 else "< 0.001"
            q = t.get("BH_q_across_4", np.nan)
            qt = (
                "" if not np.isfinite(q) else (f"\nq = {q:.3f}" if q >= 0.001 else "\nq < 0.001")
            )  # FDR across the four primary tests; not defined for descriptive Jin
            ax.set_title(
                f"{cohort} (n = {int(t.n)}){'*' if cohort == 'Jin' else ''}\n\u03c1 = {t.rho:+.2f}\np {'= ' + pt if t.perm_p >= 0.001 else pt}{qt}",
                fontsize=12,
                pad=6,
            )
            ax.xaxis.set_major_locator(MaxNLocator(2))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=12)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if c == 0:
                ax.set_ylabel(
                    (
                        f"Endothelial proteostasis\nscore ({unit})"
                        if (r == 1 and snca_scope == "endothelial")
                        else f"Proteostasis\nscore ({unit})"
                    )
                )
            ax.set_xlabel(xlab)
            if c == 0:
                ax.text(-0.55, 1.22, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout(h_pad=1.8, w_pad=1.2)
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=600)
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    Image.open(f"{out_prefix}.png").convert("RGB").save(
        f"{out_prefix}.tiff", compression="tiff_lzw", dpi=(600, 600)
    )
    plt.close(fig)
    open(f"{out_prefix}_alt_text.txt", "w").write(
        "Two rows of scatter plots with one point per donor. Row A: whole-tissue PDE10A against the donor proteostasis score in the Ma, Nido and Jin cohorts. "
        "Row B: endothelial SNCA against the donor proteostasis score. Each panel title gives the donor count, the Spearman correlation, the permutation p and, for Ma and Nido, the FDR q; "
        "* Jin (6 donors) is descriptive. Orange points are PD donors, blue points are controls.\n"
    )
    open(f"{out_prefix}_legend.txt", "w").write(
        f"Values are donor-level (one point per donor; {unit}). Spearman rho with permutation p (exact for n <= 8, otherwise 200,000 permutations, two-sided). "
        "q is the Benjamini-Hochberg FDR across the four primary tests (Ma and Nido, PDE10A and SNCA); it is not computed for Jin. Orange points, PD donors; blue points, control donors. "
        "No fitted line is drawn because the statistic is rank-based. * Descriptive only (n = 6). PD, Parkinson's disease.\n"
    )


# Values stated in the revised manuscript: (rho, permutation p, n donors, within-diagnosis permutation p, FDR q across the four primary tests).
# "<0.001" is an upper bound. The permutation seed is fixed (seed=0, 200,000 permutations), so p and q reproduce to within FINAL_P_TOL.
def main():
    per_cohort, all_tests = {}, []
    for cohort in ("Ma", "Nido", "Jin"):
        print(f"Loading {cohort}...")
        adata, ctx = load_cohort(cohort)
        tables = donor_values(adata, ctx)
        del adata
        per_cohort[cohort] = tables
        all_tests.append(tests_for(cohort, tables))
    tests = pd.concat(all_tests, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tests.to_csv(RESULTS_DIR / "all_tests.csv", index=False)
    v, scopes = DEFAULT_VARIANT, SCOPES
    sel = assemble(tests, v, scopes)
    sel.to_csv(RESULTS_DIR / "chosen_variant_results.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 140)
    cols = [
        "cohort",
        "target",
        "proteostasis_scope",
        "n",
        "rho",
        "perm_p",
        "within_dx_perm_p",
        "BH_q_across_4",
    ]
    print(sel[cols].round(4).to_string(index=False))
    donors = {c: per_cohort[c][v] for c in per_cohort}
    for c, d in donors.items():
        d.to_csv(RESULTS_DIR / f"donor_level_values_{c}.csv")
    make_supp_fig3(
        donors,
        sel,
        str(RESULTS_DIR / "SuppFig3_from_results"),
        variant=v,
        snca_scope=scopes["SNCA"],
    )
    print(
        f"\nSupplementary Fig. 3 (PDF, PNG, 600 dpi TIFF, alt text) written from the donor-level results. Files in {RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
