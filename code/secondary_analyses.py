#!/usr/bin/env python3
"""
secondary_analyses.py : independent re-check, from the raw count objects, of the manuscript claims that had not been re-derived.

  A  Hallmarks GSEA, NES and FDR (Ma, Nido). Also writes Supplementary Table 3: all 50 gene sets with the FDR < 0.05 and FDR < 0.25 thresholds.
  B  decoupleR / CollecTRI: Nido ATF4 and Ma HSF1.
  C  Autophagy module effect size (Nido), computed at donor level.
  D  The sequencing-depth screen (Tukey fence on the donor median UMI of all 56 donors) and the Ma sensitivity analysis. Also writes Supplementary Table 4.

Every number reported in the paper is typed below. Each check ends as AGREES, DIFFERS, INFO (reported, nothing to compare) or SKIPPED (a package or network resource was not available).
Definitions are not re-invented: donor scores and effect sizes come from make_figure4_cohens_d.py, donor-level SNCA tables and permutation tests from pde10a_snca_donor_level.py.
Calibration comes first: if the Figure 4 method does not reproduce d = 0.357 (Ma) and 0.114 (Nido), the depth sensitivity analysis is reported as CALIBRATION FAILED rather than trusted.

GSEA settings (stated, because the Methods do not state them): ranking = PyDESeq2 Wald statistic of donor-level pseudobulk (raw counts summed per donor over ALL nuclei, genes with < 10 counts removed),
disease versus control; GSEApy prerank; gene sets MSigDB_Hallmark_2020 (or data_snrna/hallmark.gmt if present); gene-set size 10-500; 1,000 permutations, seed 42 (the published settings), repeated with 10,000.

Usage (from the working folder that holds data_snrna/):   python secondary_analyses.py [--skip gsea,decoupler,depth,autophagy | --only gsea,depth]
"""

import sys, runpy, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
SEARCH = [Path("."), HERE]
OUT = Path("results/secondary_verification")
COHORTS = ("Jin", "Ma", "Nido")

# ---------------------------------------------------------------------------------------- values reported in the paper
# GSEA: the values quoted in the paper (donor pseudobulk over all nuclei). FDR "< 0.001" is entered as 0.0
GSEA_CLAIMS = {
    "Nido": [
        ("Cholesterol Homeostasis", 1.79, 0.002),
        ("mTORC1 Signaling", 1.50, 0.027),
        ("E2F Targets", 1.83, 0.003),
        ("G2-M Checkpoint", 1.62, 0.014),
        ("p53 Pathway", 1.44, 0.036),
        ("Hedgehog Signaling", -2.03, 0.001),
        ("Hypoxia", -1.27, 0.189),
    ],
    "Ma": [
        ("Interferon Alpha Response", -2.23, 0.001),
        ("Interferon Gamma Response", -2.19, 0.001),
        ("Unfolded Protein Response", -2.10, 0.001),
        ("TNF-alpha Signaling via NF-kB", -1.89, 0.001),
        ("IL-6/JAK/STAT3 Signaling", -1.75, 0.001),
        ("Notch Signaling", -1.18, 0.199),
    ],
}
NIDO_N_SIG_TEXT = None  # the text no longer states a count of significant sets
GSEA_NES_TOL = 0.15  # permutation noise plus a different ranking: NES within 0.15 and the same side of FDR 0.05
DECOUPLER_CLAIMS = {
    ("Nido", "ATF4"): (0.011, 0.782),
    ("Ma", "HSF1"): (0.793, 0.114),
}  # (mean difference PD - control of donor means, Mann-Whitney p)
AUTOPHAGY = ["MAP1LC3B", "SQSTM1", "ATG5", "ATG7", "BECN1", "ATG12", "LAMP1", "LAMP2", "TFEB"]
AUTOPHAGY_D_NIDO = 0.444  # donor level, the Figure 4 method
CALIBRATION = {"Ma": 0.357, "Nido": 0.114}  # the Figure 4 values, same method
DEPTH_CLAIM = dict(
    n_flagged=3,
    cohort="Ma",
    dx="Control",
    flagged=("Ma:15", "Ma:16", "Nido:PFCTX046"),
    mad=("Ma:15", "Ma:16"),
    medians=(690, 930, 1862),
    rest_min=2226,
    rest_max=15898,
    d=1.199,
    ci=(-0.05, 2.45),
    p=0.272,
)
GSEA_SETTINGS = dict(
    min_size=10, max_size=500, permutation_num=1000, seed=42, library="MSigDB_Hallmark_2020"
)
SKIPPABLE = ("gsea", "decoupler", "autophagy", "depth")


# ---------------------------------------------------------------------------------------- plumbing
def locate(script):
    for d in SEARCH:
        if (Path(d) / script).exists():
            return Path(d) / script
    hint = (
        (
            " This script needs the helper scripts make_figure4_cohens_d.py and pde10a_snca_donor_level.py next to it. marimo_cell_VERIFY_SECONDARY.py writes them for you: "
            "run THAT cell, do not paste this file on its own."
        )
        if script in ("make_figure4_cohens_d.py", "pde10a_snca_donor_level.py")
        else ""
    )
    raise FileNotFoundError(f"{script} not found in {[str(d) for d in SEARCH]}.{hint}")


def load_ns(script):
    return runpy.run_path(
        str(locate(script)), run_name="not_main"
    )  # definitions only; the script's own main() is not run


def row(check, expected, got, status, note=""):
    return dict(check=check, expected=expected, got=got, status=status, note=note)


def mem(label=""):
    """One line saying how much memory this process uses and how much the machine (or container) allows, so a kernel that dies can be diagnosed. Never raises."""
    try:
        rss = peak = None
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS:"):
                rss = int(l.split()[1]) / 1e6
            if l.startswith("VmHWM:"):
                peak = int(l.split()[1]) / 1e6
        lim = None
        for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                v = open(f).read().strip()
                lim = None if v == "max" else int(v) / 1e9
                break
            except Exception:
                continue
        if lim is None or lim > 1e4:
            for l in open("/proc/meminfo"):
                if l.startswith("MemTotal:"):
                    lim = int(l.split()[1]) / 1e6
                    break
        tag = f"    memory{(' ' + label) if label else ''}: "  # some sandboxes report no peak or no limit: show what is known
        return tag + (
            f"in use {rss:.1f} GB, peak {peak:.1f} GB, limit {lim:.1f} GB"
            if peak is not None and lim is not None
            else (
                f"in use {rss:.1f} GB, peak {'n/a' if peak is None else f'{peak:.1f} GB'}, limit {'unknown' if lim is None else f'{lim:.1f} GB'}"
                if rss is not None
                else "not available on this system"
            )
        )
    except Exception:
        return f"    memory{(' ' + label) if label else ''}: not available on this system"


def p_close(p, rep):
    return (
        (p <= 0.0002) if rep <= 0.0001 else abs(p - rep) <= max(0.004, 0.15 * rep)
    )  # the rule used for the permutation p-values elsewhere in this project


# ---------------------------------------------------------------------------------------- D: sequencing depth
def donor_umi(ctx):
    """Per donor: diagnosis, number of nuclei and the median total UMI count per nucleus."""
    d = pd.DataFrame(
        {
            "donor": np.asarray(ctx["donor"]).astype(str),
            "dx": np.asarray(ctx["dx"]).astype(str),
            "umi": np.asarray(ctx["totals"], float),
        }
    )
    return d.groupby("donor").agg(
        dx=("dx", "first"), n_nuclei=("umi", "size"), median_umi=("umi", "median")
    )


def tukey_flags(values, k=1.5):
    """Tukey fence on log-transformed values (the base does not matter). Returns (flags, lower fence, upper fence) on the original scale."""
    v = pd.Series(np.log10(np.asarray(values, float)), index=getattr(values, "index", None))
    q1, q3 = np.percentile(v, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (v < lo) | (v > hi), 10**lo, 10**hi


def mad_flags(values, thr=3.5):
    """The other standard convention: modified z-score 0.6745 * (x - median) / MAD on the log scale, flagged when |z| > 3.5."""
    v = pd.Series(np.log10(np.asarray(values, float)), index=getattr(values, "index", None))
    med = v.median()
    mad = (v - med).abs().median()
    return (0.6745 * (v - med) / mad).abs() > thr if mad > 0 else pd.Series(False, index=v.index)


def pooled_depth(per_cohort):
    """{cohort: donor_umi table} -> one table with a unique index, Tukey and MAD flags over the pooled donors."""
    t = pd.concat({c: d for c, d in per_cohort.items()}, names=["cohort", "donor"]).reset_index()
    t.index = t.cohort + ":" + t.donor
    t["tukey_flag"], lo, hi = tukey_flags(t.median_umi)
    t["mad_flag"] = mad_flags(t.median_umi)
    t.attrs["fence"] = (lo, hi)
    return t


def depth_by_cohort(t):
    rows = []
    for c, g in list(t.groupby("cohort")) + [("All donors", t)]:
        q1, med, q3 = np.percentile(g.median_umi, [25, 50, 75])
        rows.append(
            {
                "Cohort": c,
                "Donors": len(g),
                "Nuclei": int(g.n_nuclei.sum()),
                "Median of donor medians (UMI per nucleus)": med,
                "Q1": q1,
                "Q3": q3,
                "Minimum": g.median_umi.min(),
                "Maximum": g.median_umi.max(),
                "Flagged by Tukey fence": int(g.tukey_flag.sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- C and E: donor-level module effect and subgroup correlations
def module_donor_scores(ns4, adata, ctx, genes):
    """The Figure 4 method for any gene list: per nucleus the mean CPM of the genes, then the mean over each donor's nuclei."""
    cnt = ns4["gene_counts"](adata, ctx, genes)
    score = (cnt / np.asarray(ctx["totals"], float)[:, None] * 1e6).mean(axis=1)
    return (
        pd.DataFrame(
            {
                "donor": np.asarray(ctx["donor"]).astype(str),
                "dx": np.asarray(ctx["dx"]).astype(str),
                "score": score,
            }
        )
        .groupby("donor")
        .agg(score=("score", "mean"), dx=("dx", "first"), n_nuclei=("score", "size"))
    )


def pseudobulk(raw, donors, genes):
    """Raw counts summed per donor over all nuclei: DataFrame donors x genes (integers)."""
    donors = np.asarray(donors).astype(str)
    ids = pd.Index(sorted(set(donors)))
    pos = {x: i for i, x in enumerate(ids)}
    rows = np.array([pos[x] for x in donors])
    onehot = sps.csr_matrix(
        (np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(ids), len(rows))
    )
    M = onehot @ sps.csr_matrix(raw)
    df = pd.DataFrame(
        np.asarray(M.todense()).round().astype(int), index=ids, columns=pd.Index(genes).astype(str)
    )
    return df.loc[:, ~df.columns.duplicated()]


def wald_ranking(counts, dx, min_total=10, contrast=("condition", "PD", "Control")):
    """PyDESeq2 Wald statistic, disease versus control, on donor-level counts. Works with the 0.4 and the newer API."""
    import inspect
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts = counts.loc[:, counts.sum(axis=0) >= min_total]
    meta = pd.DataFrame({"condition": pd.Series(dx).loc[counts.index].values}, index=counts.index)
    kw = (
        dict(design="~condition")
        if "design" in inspect.signature(DeseqDataSet.__init__).parameters
        else dict(design_factors="condition")
    )
    import contextlib, io

    with contextlib.redirect_stdout(io.StringIO()):  # PyDESeq2 prints its whole results table
        dds = DeseqDataSet(counts=counts, metadata=meta, refit_cooks=True, **kw)
        dds.deseq2()
        st = DeseqStats(dds, contrast=list(contrast))
        st.summary()
    s = st.results_df["stat"].dropna()
    return s[~s.index.duplicated()].sort_values(ascending=False)


GMT_CANDIDATES = [
    "data_snrna/hallmark.gmt",
    "results/pre_submission/Ma_Hallmarks_GSEA/gene_sets.gmt",
]  # the second is the file the stored Ma run used


def gene_sets(candidates=None):
    """A local Hallmark GMT if one exists (a previously downloaded file), otherwise the Enrichr library named in GSEA_SETTINGS (needs network access)."""
    for p in candidates or GMT_CANDIDATES:
        if Path(p).exists():
            return str(p)
    import gseapy as gp

    try:
        return gp.get_library(name=GSEA_SETTINGS["library"], organism="Human")
    except TypeError:
        return gp.get_library(GSEA_SETTINGS["library"])


def gsea(ranked, sets, permutation_num=1000, seed=42):
    import gseapy as gp

    rnk = ranked.reset_index()
    rnk.columns = ["gene_name", "score"]
    r = gp.prerank(
        rnk=rnk,
        gene_sets=sets,
        min_size=GSEA_SETTINGS["min_size"],
        max_size=GSEA_SETTINGS["max_size"],
        permutation_num=permutation_num,
        seed=seed,
        outdir=None,
        verbose=False,
    ).res2d
    r = r[["Term", "NES", "NOM p-val", "FDR q-val"]].copy()
    for c in ("NES", "NOM p-val", "FDR q-val"):
        r[c] = r[c].astype(float)
    return r.sort_values("NES", ascending=False).reset_index(drop=True)


def gsea_rows(cohort, res, claims=None, n_sig_text=None):
    rows = []
    for term, nes, fdr in (
        GSEA_CLAIMS[cohort] if claims is None else claims
    ):  # an empty list means 'no individual claims', not 'use the defaults'
        m = res[res.Term.str.lower() == term.lower()]
        if m.empty:
            rows.append(
                row(
                    f"GSEA {cohort}: {term}",
                    f"NES {nes:+.3f}, FDR {fdr}",
                    "gene set absent",
                    "DIFFERS",
                )
            )
            continue
        r = m.iloc[0]
        same_class = (r["FDR q-val"] < 0.05) == (fdr < 0.05)
        ok = abs(r["NES"] - nes) <= GSEA_NES_TOL and same_class
        rows.append(
            row(
                f"GSEA {cohort}: {term}",
                f"NES {nes:+.3f}, FDR {'<' if fdr <= 0.001 else '='} {fdr}",
                f"NES {r['NES']:+.3f}, FDR {r['FDR q-val']:.3f}",
                "AGREES" if ok else "DIFFERS",
                (
                    ""
                    if ok
                    else f"|dNES| = {abs(r['NES'] - nes):.3f}; significant at 0.05: text {fdr < 0.05}, re-run {bool(r['FDR q-val'] < 0.05)}"
                ),
            )
        )
    if n_sig_text is not None:
        n = int((res["FDR q-val"] < 0.05).sum())
        rows.append(
            row(
                f"GSEA {cohort}: gene sets with FDR < 0.05",
                n_sig_text,
                n,
                "AGREES" if n == n_sig_text else "DIFFERS",
                "" if n == n_sig_text else "the text count should be replaced by the re-run count",
            )
        )
    return rows


# ---------------------------------------------------------------------------------------- B: decoupleR
def donor_tf_effect(raw, totals, donors, dx, var_names, net, tfs, batch=5000, lognorm=False):
    """Nucleus-level ULM activity of each TF (decoupleR, CollecTRI), averaged per donor; disease minus control and a two-sided Mann-Whitney p. The method of verify_decoupler_claim.py.
    MEMORY: decoupleR turns the data it is given into a DENSE float64 array (277,831 nuclei x 33,000 genes = 73 GB for Nido), which kills the kernel. ULM scores every nucleus independently, so the nuclei are
    processed in batches of `batch` (a few GB at most) with identical results. Do NOT narrow the genes instead: ULM uses the whole transcriptome as background, so slicing the target columns changes the
    activities (tested: differences up to 0.9 on a scale of about +-4). Two details keep the batches equivalent to one big call: the genes that are zero in EVERY nucleus (which decoupleR drops, once,
    for the whole matrix) are removed once, globally; and empty=False stops decoupleR dropping genes or nuclei that are empty only within one batch.
    """
    import anndata as ad, decoupler as dc

    if not hasattr(dc, "mt"):
        raise RuntimeError(
            "secondary_analyses.py needs decoupler>=2.0 (dc.op.collectri, dc.mt.ulm); install it, see environment.yml"
        )
    sub = net[net["source"].isin(tfs)]
    names = pd.Index(var_names).astype(str)
    raw = sps.csr_matrix(raw)
    n = raw.shape[0]
    tot = np.maximum(np.asarray(totals, float), 1)
    nz = np.flatnonzero(np.asarray(raw.sum(axis=0)).ravel() > 0)
    names_nz = names[nz]
    got = {tf: [] for tf in tfs}
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        X = sps.csr_matrix(raw[lo:hi][:, nz], dtype=np.float32, copy=True)
        if not lognorm:
            X.data *= np.repeat((1e4 / tot[lo:hi]).astype(np.float32), np.diff(X.indptr))
            np.log1p(
                X.data, out=X.data
            )  # lognorm=True: `raw` already holds log-normalised values (an object's own layer)
        a = ad.AnnData(
            X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(lo, hi)]),
            var=pd.DataFrame(index=names_nz),
        )
        a.var_names_make_unique()
        dc.mt.ulm(data=a, net=sub, verbose=False, empty=False)
        acts = dc.pp.get_obsm(adata=a, key="score_ulm")
        for tf in tfs:
            if tf in acts.var_names:
                got[tf].append(
                    np.asarray(
                        acts[:, tf].X.todense() if sps.issparse(acts.X) else acts[:, tf].X
                    ).ravel()
                )
        del a, X, acts
    out = {}
    for tf in tfs:
        if not got[tf]:
            continue
        m = (
            pd.DataFrame(
                {
                    "donor": np.asarray(donors).astype(str),
                    "dx": np.asarray(dx).astype(str),
                    "s": np.concatenate(got[tf]),
                }
            )
            .groupby(["donor", "dx"])
            .s.mean()
            .reset_index()
        )
        pd_, ct = m[m.dx == "PD"].s.values, m[m.dx == "Control"].s.values
        out[tf] = dict(
            mean_diff=float(pd_.mean() - ct.mean()),
            p=float(stats.mannwhitneyu(pd_, ct, alternative="two-sided").pvalue),
            n_pd=len(pd_),
            n_control=len(ct),
        )
    return out


def looks_integer(X, n=2000):
    """True if a sample of nuclei, spread over the matrix, holds only whole numbers (raw counts)."""
    rows = np.unique(np.linspace(0, X.shape[0] - 1, min(n, X.shape[0])).astype(int))
    v = X[rows]
    v = np.asarray(v.data if sps.issparse(v) else v).ravel()
    v = v[v != 0]
    return bool(v.size == 0 or np.allclose(v, np.round(v)))


def exclusion_rows(effect, prot, flagged, cohorts=("Nido",)):
    """The post hoc sensitivity analysis that the two Ma controls get (Discussion), for the flagged donors of the other cohorts. INFO rows: the primary estimate stays the full cohort."""
    rows = []
    for c in cohorts:
        drop = [i.split(":", 1)[1] for i in flagged.index if i.startswith(c + ":")]
        if not drop or c not in prot:
            continue
        e = effect(prot[c].drop(index=drop, errors="ignore"))
        rows.append(
            row(
                f"{c} proteostasis d after excluding the flagged donor{'s' if len(drop) > 1 else ''} ({', '.join(drop)}): post hoc",
                "",
                f"{e['d']:+.3f} [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}], Mann-Whitney p = {e['p']:.3f} ({e['n_pd']} + {e['n_control']} donors)",
                "INFO",
                "report as a post hoc sensitivity analysis; the primary estimate stays the full cohort",
            )
        )
    return rows


def decoupler_row(cohort, tf, got, claim=None):
    mean_diff, p = claim or DECOUPLER_CLAIMS[(cohort, tf)]
    check = f"decoupleR/CollecTRI {cohort} {tf}: mean difference, p"
    if got is None:
        return row(check, f"{mean_diff:+.3f} (p = {p})", "TF not in the network output", "SKIPPED")
    exact = round(got["mean_diff"], 3) == mean_diff and abs(got["p"] - p) <= 0.005
    ok = exact or (
        np.sign(got["mean_diff"]) == np.sign(mean_diff)
        and abs(got["mean_diff"] - mean_diff) <= 0.05
        and (got["p"] < 0.05) == (p < 0.05)
        and abs(got["p"] - p) <= 0.1
    )
    return row(
        check,
        f"{mean_diff:+.3f} (p = {p})",
        f"{got['mean_diff']:+.3f} (p = {got['p']:.3f})",
        "AGREES" if ok else "DIFFERS",
        (
            "exact"
            if exact
            else (
                "close (a different decoupleR/CollecTRI version changes the third decimal)"
                if ok
                else ""
            )
        ),
    )


# ---------------------------------------------------------------------------------------- tables for the supplement
def write_table3(results, path, settings=GSEA_SETTINGS):
    """results: {cohort: (res_1000, res_10000 or None)}"""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    wb.remove(wb.active)
    for cohort, (r1, r2) in results.items():
        ws = wb.create_sheet(cohort)
        ws["A1"] = (
            f"Supplementary Table 3. Hallmarks gene-set enrichment, {cohort} cohort: all {len(r1)} gene sets"
        )
        ws["A1"].font = Font(bold=True)
        ws["A2"] = (
            "Genes ranked by the PyDESeq2 Wald statistic of donor-level pseudobulk (raw counts summed per donor over all nuclei), disease versus control; GSEApy preranked against MSigDB Hallmark 2020, "
            f"gene-set size {settings['min_size']}-{settings['max_size']}, {settings['permutation_num']} permutations, seed {settings['seed']}. Positive NES: enriched in disease."
        )
        t = r1.rename(columns={"Term": "Gene set", "NOM p-val": "Nominal p", "FDR q-val": "FDR q"})
        t["FDR < 0.05"] = np.where(t["FDR q"] < 0.05, "yes", "no")
        t["FDR < 0.25"] = np.where(t["FDR q"] < 0.25, "yes", "no")
        if r2 is not None:
            m = r2.set_index("Term")[["NES", "FDR q-val"]]
            t["NES (10,000 permutations)"] = t["Gene set"].map(m["NES"])
            t["FDR q (10,000 permutations)"] = t["Gene set"].map(m["FDR q-val"])
            t["FDR < 0.05 (10,000 permutations)"] = np.where(
                t["FDR q (10,000 permutations)"] < 0.05, "yes", "no"
            )
        for j, c in enumerate(t.columns, 1):
            ws.cell(row=4, column=j, value=c).font = Font(bold=True)
        for i, rec in enumerate(t.itertuples(index=False), 5):
            for j, v in enumerate(rec, 1):
                ws.cell(
                    row=i,
                    column=j,
                    value=(round(float(v), 4) if isinstance(v, (float, np.floating)) else v),
                )
        n05, n25 = int((t["FDR q"] < 0.05).sum()), int((t["FDR q"] < 0.25).sum())
        ws.cell(
            row=len(t) + 6,
            column=1,
            value=f"Gene sets with FDR < 0.05: {n05}; with FDR < 0.25: {n25}; tested: {len(t)}.",
        )
        for col, w in zip("ABCDEFGHIJ", (34, 10, 11, 10, 11, 11, 18, 18, 22, 8)):
            ws.column_dimensions[col].width = w
    wb.save(path)


def write_table4(t, by_cohort, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("By cohort")
    ws["A1"] = (
        "Supplementary Table 4. Sequencing depth: median total UMI count per nucleus, by cohort and by donor"
    )
    ws["A1"].font = Font(bold=True)
    lo, hi = t.attrs.get("fence", (np.nan, np.nan))
    ws["A2"] = (
        f"For each donor the median total UMI count per nucleus was taken; the cohort rows summarise those donor medians. Pooled over all {len(t)} donors, a Tukey fence "
        f"(1.5 x IQR on log10 values) flags donors below {lo:,.0f} or above {hi:,.0f} UMI per nucleus. This post hoc screen was not used to define the primary analysis."
    )
    for j, c in enumerate(by_cohort.columns, 1):
        ws.cell(row=4, column=j, value=c).font = Font(bold=True)
    for i, rec in enumerate(by_cohort.itertuples(index=False), 5):
        for j, v in enumerate(rec, 1):
            ws.cell(
                row=i,
                column=j,
                value=(
                    round(float(v), 1)
                    if isinstance(v, (float, np.floating))
                    else (int(v) if isinstance(v, (np.integer,)) else v)
                ),
            )
    w2 = wb.create_sheet("By donor")
    cols = [
        "Cohort",
        "Donor",
        "Diagnosis",
        "Nuclei",
        "Median UMI per nucleus",
        "Flagged (Tukey fence)",
        "Flagged (modified z, MAD)",
    ]
    for j, c in enumerate(cols, 1):
        w2.cell(row=1, column=j, value=c).font = Font(bold=True)
    for i, r in enumerate(t.sort_values(["cohort", "median_umi"]).itertuples(), 2):
        for j, v in enumerate(
            (
                r.cohort,
                r.donor,
                r.dx,
                int(r.n_nuclei),
                round(float(r.median_umi), 1),
                "yes" if r.tukey_flag else "no",
                "yes" if r.mad_flag else "no",
            ),
            1,
        ):
            w2.cell(row=i, column=j, value=v)
    for col, w in zip("ABCDEFGHI", (14, 22, 12, 12, 24, 12, 12, 12, 22)):
        ws.column_dimensions[col].width = w
    wb.save(path)


# ---------------------------------------------------------------------------------------- orchestration
def analyse(
    skip=frozenset(), perms=(1000, 10000), net=None, sets=None, loader=None, checkpoint=None
):
    """Returns (rows, extras). `loader(cohort)` -> (adata, ctx) is injectable for tests."""
    ns4 = load_ns("make_figure4_cohens_d.py")
    load = loader or ns4["load_cohort"]
    rows, ex = [], dict(umi={}, gsea={}, prot={})
    if sets is None and "gsea" not in skip:
        try:
            sets = gene_sets()
            rows.append(
                row(
                    "GSEA gene-set source",
                    GSEA_SETTINGS["library"],
                    sets if isinstance(sets, str) else f"Enrichr library, {len(sets)} gene sets",
                    "INFO",
                    "record this in the Methods",
                )
            )
        except Exception as e:
            rows.append(
                row(
                    "GSEA gene sets",
                    "MSigDB Hallmark 2020",
                    f"not available ({type(e).__name__})",
                    "SKIPPED",
                    "put the Hallmark GMT in data_snrna/hallmark.gmt or allow network access",
                )
            )
            skip = set(skip) | {"gsea"}
    for c in COHORTS:
        print(f"\n=== {c}: loading")
        adata, ctx = load(c)
        t0 = time.time()
        print(mem("after loading"))
        ex["umi"][c] = donor_umi(ctx)
        ex["prot"][c] = ns4["donor_scores"](adata, ctx)
        if c == "Nido" and "autophagy" not in skip:
            try:
                tab = module_donor_scores(ns4, adata, ctx, AUTOPHAGY)
                e = ns4["effect"](tab)
                ex["autophagy"] = e
                rows.append(
                    row(
                        "Autophagy module, Nido: Cohen's d (donor level)",
                        f"{AUTOPHAGY_D_NIDO} (text: donor level, the Figure 4 method)",
                        f"{e['d']:+.3f}  [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}], Mann-Whitney p = {e['p']:.3f}, {e['n_pd']} + {e['n_control']} donors",
                        "AGREES" if abs(e["d"] - AUTOPHAGY_D_NIDO) <= 0.02 else "DIFFERS",
                        (
                            "the text value differs from the donor-level computation: report the computed one"
                            if abs(e["d"] - AUTOPHAGY_D_NIDO) > 0.02
                            else ""
                        ),
                    )
                )
            except Exception as ex_:
                rows.append(
                    row(
                        "Autophagy module, Nido",
                        AUTOPHAGY_D_NIDO,
                        f"{type(ex_).__name__}: {ex_}",
                        "SKIPPED",
                        "a module gene is missing from the data?",
                    )
                )
        if c in ("Ma", "Nido") and "gsea" not in skip:
            try:
                counts = pseudobulk(ctx["raw"], ctx["donor"], adata.var_names)
                dxmap = (
                    pd.Series(
                        np.asarray(ctx["dx"]).astype(str),
                        index=np.asarray(ctx["donor"]).astype(str),
                    )
                    .groupby(level=0)
                    .first()
                )
                ranked = wald_ranking(counts, dxmap)
                r1 = gsea(ranked, sets, perms[0], GSEA_SETTINGS["seed"])
                r2 = gsea(ranked, sets, perms[1], GSEA_SETTINGS["seed"]) if len(perms) > 1 else None
                ex["gsea"][c] = (r1, r2)
                rows += gsea_rows(c, r1, n_sig_text=NIDO_N_SIG_TEXT if c == "Nido" else None)
                rows.append(
                    row(
                        f"GSEA {c}: gene sets at FDR < 0.05 / < 0.25",
                        "",
                        f"{int((r1['FDR q-val'] < 0.05).sum())} / {int((r1['FDR q-val'] < 0.25).sum())} of {len(r1)}"
                        + (
                            f" ({int((r2['FDR q-val'] < 0.05).sum())} / {int((r2['FDR q-val'] < 0.25).sum())} with 10,000 permutations)"
                            if r2 is not None
                            else ""
                        ),
                        "INFO",
                    )
                )
            except Exception as ex_:
                rows.append(
                    row(
                        f"GSEA {c}",
                        "",
                        f"{type(ex_).__name__}: {ex_}",
                        "SKIPPED",
                        "pydeseq2/gseapy missing, or no gene sets",
                    )
                )
        if c in ("Ma", "Nido") and "decoupler" not in skip:
            try:
                n_ = (
                    net
                    if net is not None
                    else __import__("decoupler").op.collectri(
                        organism="human", remove_complexes=False
                    )
                )
                tfs = [tf for (co, tf) in DECOUPLER_CLAIMS if co == c]
                got = donor_tf_effect(
                    ctx["raw"], ctx["totals"], ctx["donor"], ctx["dx"], adata.var_names, n_, tfs
                )
                for tf in tfs:
                    rows.append(decoupler_row(c, tf, got.get(tf)))
            except Exception as ex_:
                rows.append(
                    row(
                        f"decoupleR/CollecTRI {c}",
                        "",
                        f"{type(ex_).__name__}: {str(ex_)[:120]}",
                        "SKIPPED",
                        "decoupler not installed, or the CollecTRI network could not be downloaded",
                    )
                )
        del adata, ctx
        import gc

        gc.collect()
        print(f"    {c} done in {time.time() - t0:.0f} s")
        print(mem("after the cohort"))
        if checkpoint is not None:
            pd.DataFrame(rows).to_csv(
                checkpoint, index=False
            )  # what is finished survives a later crash, if the disk does
    if "depth" not in skip:
        t = pooled_depth(ex["umi"])
        ex["depth"] = t
        by = depth_by_cohort(t)
        ex["depth_by_cohort"] = by
        fl = t[t.tukey_flag]
        cal_ok = True
        for c, want in CALIBRATION.items():
            e = ns4["effect"](ex["prot"][c])
            ok = round(e["d"], 3) == want
            cal_ok &= ok
            rows.append(
                row(
                    f"Calibration: {c} proteostasis d (Figure 4 method)",
                    want,
                    f"{e['d']:+.3f}",
                    "AGREES" if ok else "CALIBRATION FAILED",
                    "" if ok else "the depth sensitivity below cannot be trusted",
                )
            )
        k = DEPTH_CLAIM
        want_ids = set(k["flagged"]) if "flagged" in k else None
        ok1 = (
            (set(fl.index) == want_ids)
            if want_ids
            else (
                len(fl) == k["n_flagged"]
                and (fl.cohort == k["cohort"]).all()
                and (fl.dx == k["dx"]).all()
            )
        )
        rows.append(
            row(
                "Depth screen: donors flagged by the pooled Tukey fence",
                f"{k['n_flagged']} ("
                + (", ".join(sorted(k["flagged"])) if want_ids else f"{k['cohort']} {k['dx']}")
                + ")",
                (
                    f"{len(fl)}: "
                    + "; ".join(f"{i} ({r.dx}, {r.median_umi:,.0f})" for i, r in fl.iterrows())
                    if len(fl)
                    else "0"
                ),
                "AGREES" if ok1 else "DIFFERS",
            )
        )
        if len(fl):
            rows.append(
                row(
                    "Depth screen: median UMI of the flagged donors",
                    " and ".join(str(v) for v in k["medians"]),
                    " and ".join(f"{v:,.0f}" for v in sorted(fl.median_umi)),
                    (
                        "AGREES"
                        if sorted(round(v) for v in fl.median_umi) == sorted(k["medians"])
                        else "DIFFERS"
                    ),
                )
            )
        rest = t[~t.tukey_flag].median_umi
        rows.append(
            row(
                "Depth screen: range of the remaining donors",
                f"{k['rest_min']:,}-{k['rest_max']:,}",
                f"{rest.min():,.0f}-{rest.max():,.0f}",
                (
                    "AGREES"
                    if (round(rest.min()), round(rest.max())) == (k["rest_min"], k["rest_max"])
                    else "DIFFERS"
                ),
            )
        )
        mad_ids = set(t.index[t.mad_flag])
        want_mad = set(k["mad"]) if "mad" in k else None
        rows.append(
            row(
                "Depth screen: donors flagged by the MAD rule",
                (
                    ("the same two: " + ", ".join(sorted(want_mad)))
                    if want_mad
                    else "not the 2 Ma controls"
                ),
                ", ".join(sorted(mad_ids)) or "none",
                (
                    "AGREES"
                    if (
                        mad_ids == want_mad
                        if want_mad is not None
                        else not (mad_ids & set(fl.index))
                    )
                    else "DIFFERS"
                ),
            )
        )
        keep = ex["prot"]["Ma"].drop(
            index=[i.split(":", 1)[1] for i in fl.index if i.startswith("Ma:")], errors="ignore"
        )
        e = ns4["effect"](keep)
        if cal_ok:
            rows.append(
                row(
                    "Ma proteostasis d after excluding the flagged donors",
                    f"{k['d']} [{k['ci'][0]:+.2f}, {k['ci'][1]:+.2f}], p = {k['p']}",
                    f"{e['d']:+.3f} [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}], p = {e['p']:.3f} ({e['n_pd']} + {e['n_control']} donors)",
                    (
                        "AGREES"
                        if round(e["d"], 3) == k["d"]
                        and abs(e["ci_low"] - k["ci"][0]) <= 0.01
                        and abs(e["ci_high"] - k["ci"][1]) <= 0.01
                        and abs(e["p"] - k["p"]) <= 0.002
                        else "DIFFERS"
                    ),
                )
            )
        else:
            rows.append(
                row(
                    "Ma proteostasis d after excluding the flagged donors",
                    f"{k['d']}",
                    f"{e['d']:+.3f}",
                    "CALIBRATION FAILED",
                )
            )
        if cal_ok:
            rows += exclusion_rows(ns4["effect"], ex["prot"], fl)
    return rows, ex


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    skip = set()
    tag = ""
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            tag = "_" + "+".join(
                sorted(x for x in argv[i + 1].split(",") if x)
            )  # a staged run writes its own verdict file, so the next stage cannot overwrite it
        if a == "--skip" and i + 1 < len(argv):
            skip = {s for s in argv[i + 1].split(",") if s}
        if a == "--only" and i + 1 < len(argv):
            only = {s for s in argv[i + 1].split(",") if s}
            bad_only = only - set(SKIPPABLE)
            skip = set(SKIPPABLE) - only if not bad_only else skip
    bad = (
        skip
        | {
            s
            for i, a in enumerate(argv)
            if a == "--only" and i + 1 < len(argv)
            for s in argv[i + 1].split(",")
            if s
        }
    ) - set(SKIPPABLE)
    if bad:
        print(f"unknown --skip/--only names {sorted(bad)}; choose from {list(SKIPPABLE)}")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ex = analyse(skip, checkpoint=OUT / "partial_verdicts.csv")
    v = pd.DataFrame(rows, columns=["check", "expected", "got", "status", "note"])
    v.to_csv(OUT / f"verification_of_secondary_claims{tag}.csv", index=False)
    if ex.get("gsea"):
        write_table3(ex["gsea"], OUT / "Supplementary_Table_3_Hallmarks_GSEA_all_gene_sets.xlsx")
        for c, (r1, r2) in ex["gsea"].items():
            r1.to_csv(OUT / f"gsea_{c}_1000_permutations.csv", index=False)
    if "depth" in ex:
        write_table4(
            ex["depth"], ex["depth_by_cohort"], OUT / "Supplementary_Table_4_sequencing_depth.xlsx"
        )
        ex["depth"].to_csv(OUT / "donor_median_umi.csv")
    pd.set_option("display.width", 250, "display.max_colwidth", 95, "display.max_rows", 200)
    print(
        "\nVERIFICATION OF THE SECONDARY CLAIMS (v29 numbers against recomputation from the raw count objects)\n"
    )
    print(v[["check", "expected", "got", "status"]].to_string(index=False))
    n = v.status.value_counts().to_dict()
    print("\nsummary:", n)
    notes = v[v.note != ""]
    if len(notes):
        print("\nnotes:")
        [print(f"  - {r.check}: {r.note}") for r in notes.itertuples()]
    print(f"\nFiles in {OUT.resolve()}")
    bad = v.status.isin(["DIFFERS", "CALIBRATION FAILED"]).sum()
    print(
        "\n"
        + (
            "EVERY CHECKED CLAIM AGREES."
            if not bad and not n.get("SKIPPED")
            else f"{bad} claim(s) DIFFER or could not be calibrated, {n.get('SKIPPED', 0)} check(s) were SKIPPED: the manuscript text for those must be revisited before submission."
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    code = main()
    if any(m in sys.modules for m in ("marimo", "ipykernel")):
        sys.stdout.write(
            f"[secondary_analyses finished with status {code}]\n"
        )  # a notebook shows SystemExit as an error, so only report the status
    else:
        sys.exit(code)
