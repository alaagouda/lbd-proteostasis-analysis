#!/usr/bin/env python3
"""
figures_2c_and_8b.py : the two figure panels whose numbers had no source in the repository.

Figure 8B, Unfolded Protein Response by cell type (Nido prefrontal cortex). make_figure8_upr.py only DRAWS a table it is given. This script computes that table from the raw counts: for each of the 17 fine
cell types, donor-level pseudobulk -> PyDESeq2 Wald statistic -> GSEApy preranked on the Hallmark gene sets (size 10-500, 1,000 permutations, seed 42, the same gene-set library as the rest of the manuscript)
-> the NES, nominal p and FDR of the Unfolded Protein Response set. The fine types are then grouped into the seven broad categories of the figure. A category with ONE fine type keeps that type's NES and FDR
("direct"). A category with several (excitatory and inhibitory neurons) gets the mean NES, and a Stouffer-combined p from the signed Z of the subtypes' nominal p (as for the HSP panel); the legend calls this
the "combined p". The result is compared with the seven values in the manuscript. Output: figure8b_source_table.csv (columns category, NES, FDR, basis: the input make_figure8_upr.py expects).
If the recomputed values differ from the manuscript, the manuscript text and Figure 8 must be updated to the recomputed ones (and the method stated), as was done for the other GSEA values.

Figure 2C, cell-type proportions per cohort from the FULL (unsubsampled) objects, drawn in the colours of panel A and, if the existing panels A and B are given (--ab), stacked under them as one figure.
Segments below 3 % of a cohort are not labelled. The label column of each cohort is fixed below and every label must map to a panel-A category, otherwise the script stops and lists the labels.

Usage (from the working folder that holds data_snrna/):   python figures_2c_and_8b.py [--only figure8b,figure2c] [--ab Figure2_AB_FINAL.png] [--perms 1000]
"""

import sys, re, runpy, time, warnings, gc, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
SEARCH = [Path("."), HERE]
OUT = Path("results/figures_2c_and_8b")
UPR = "Unfolded Protein Response"
CATEGORIES = [
    "Oligodendrocytes",
    "Endothelial",
    "Excitatory neurons",
    "Astrocytes",
    "OPCs",
    "Microglia",
    "Inhibitory neurons",
]
FIG8_CLAIMS = {
    "Oligodendrocytes": (1.605, 0.011),
    "Endothelial": (1.206, 0.265),
    "Excitatory neurons": (1.065, 0.0002),
    "Astrocytes": (0.807, 0.996),
    "OPCs": (0.792, 0.914),
    "Microglia": (-1.027, 0.561),
    "Inhibitory neurons": (-1.286, 0.0023),
}  # (NES, FDR or Stouffer-combined p) as in Figure 8B and the Results
NES_TOL, P_FLOOR = (
    0.15,
    1e-3,
)  # a nominal p of 0 from 1,000 permutations is entered as 1e-3 before combining
FIG2_N = {"Jin": 41899, "Ma": 64370, "Nido": 277831}
FIG2_TOTAL = 384100  # Figure 2 legend
LABEL_COL = {"Jin": "broad_type", "Ma": "cell_type", "Nido": "cell_type_broad"}
HARM = [
    "Astrocyte",
    "Endothelial",
    "Excitatory",
    "Inhibitory",
    "Microglia",
    "OPC",
    "Oligodendrocyte",
    "Pericyte",
    "Unassigned",
]  # the legend of panel A
COLORS = dict(
    zip(
        HARM,
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#cccccc",
        ],
    )
)  # matplotlib tab10, as in panel A
RULES = (
    ("oligodendrocyte precursor", "OPC"),
    ("astro", "Astrocyte"),
    ("endo", "Endothelial"),
    ("excit", "Excitatory"),
    ("inhib", "Inhibitory"),
    ("micro", "Microglia"),
    ("opc", "OPC"),
    ("oligo", "Oligodendrocyte"),
    ("pericyte", "Pericyte"),
    ("unassigned", "Unassigned"),
    ("unknown", "Unassigned"),
)


def locate(script):
    for d in SEARCH:
        if (Path(d) / script).exists():
            return Path(d) / script
    raise FileNotFoundError(
        f"{script} not found in {[str(d) for d in SEARCH]}. Run marimo_cell_CLOSE_FIGURES_2C_8B.py: it writes the helper scripts for you; do not paste this file on its own."
    )


def load_ns(script):
    return runpy.run_path(str(locate(script)), run_name="not_main")


def row(check, expected, got, status, note=""):
    return dict(check=check, expected=expected, got=got, status=status, note=note)


# ---------------------------------------------------------------------------------------- Figure 8B
def broad_category(fine):
    """Broad category of a Nido fine cell type from its name (L2-3 ..., L6 ... are excitatory; InN ... inhibitory)."""
    f = str(fine).strip().lower()
    if re.match(r"^l\d", f) or f.startswith(("exn", "excit")):
        return "Excitatory neurons"
    if f.startswith(("inn", "inhib")):
        return "Inhibitory neurons"
    for k, v in (
        ("astro", "Astrocytes"),
        ("endo", "Endothelial"),
        ("micro", "Microglia"),
        ("opc", "OPCs"),
        ("oligo", "Oligodendrocytes"),
    ):
        if f.startswith(k):
            return v
    raise ValueError(f"cannot assign the cell type '{fine}' to one of {CATEGORIES}")


def fine_to_broad(fine_types, obs=None, fine_col=None, broad_col="cell_type_broad"):
    """{fine type: broad category}. From the object's own broad column when it has one (it must give each fine type exactly one broad label), otherwise from the names. Both are compared when available."""
    by_name = {f: broad_category(f) for f in fine_types}
    if obs is None or broad_col not in getattr(obs, "columns", []) or fine_col not in obs.columns:
        return by_name
    g = obs[[fine_col, broad_col]].astype(str).drop_duplicates()
    bad = g[g[fine_col].duplicated(keep=False)]
    if len(bad):
        raise ValueError(f"fine types with more than one broad label: {sorted(set(bad[fine_col]))}")
    canon = {c.lower(): c for c in CATEGORIES}
    out = {}
    for f, b in zip(g[fine_col], g[broad_col]):
        if f not in by_name:
            continue
        out[f] = canon.get(b.lower(), canon.get(b.lower().rstrip("s") + "s", by_name[f]))
        if out[f] != by_name[f]:
            raise ValueError(
                f"'{f}': the object says '{b}' but its name says '{by_name[f]}'; resolve this before combining"
            )
    return {f: out.get(f, by_name[f]) for f in fine_types}


def upr_by_celltype(S, raw, donors, dx, ct, genes, sets, perms=1000, seed=42, min_donors=4):
    """One row per fine cell type: the NES, nominal p and FDR of the Unfolded Protein Response set from that type's own donor-level ranking."""
    ct, donors, dx = (
        np.asarray(ct).astype(str),
        np.asarray(donors).astype(str),
        np.asarray(dx).astype(str),
    )
    rows = []
    for t in sorted(set(ct)):
        m = ct == t
        dxm = pd.Series(dx[m], index=donors[m]).groupby(level=0).first()
        base = dict(fine_type=t, n_nuclei=int(m.sum()), n_donors=len(dxm))
        if len(dxm) < min_donors or (dxm.value_counts() < 2).any() or dxm.nunique() < 2:
            rows.append(
                dict(
                    base,
                    NES=np.nan,
                    nom_p=np.nan,
                    FDR=np.nan,
                    note="skipped: too few donors in a group",
                )
            )
            continue
        try:
            counts = S["pseudobulk"](raw[m], donors[m], genes).loc[dxm.index]
            res = S["gsea"](S["wald_ranking"](counts, dxm), sets, perms, seed)
            r = res[res.Term == UPR]
            if r.empty:
                rows.append(
                    dict(
                        base,
                        NES=np.nan,
                        nom_p=np.nan,
                        FDR=np.nan,
                        note="the gene set is outside the size limits for this ranking",
                    )
                )
                continue
            r = r.iloc[0]
            rows.append(
                dict(
                    base,
                    NES=float(r["NES"]),
                    nom_p=float(r["NOM p-val"]),
                    FDR=float(r["FDR q-val"]),
                    note="",
                )
            )
        except Exception as e:
            rows.append(
                dict(
                    base,
                    NES=np.nan,
                    nom_p=np.nan,
                    FDR=np.nan,
                    note=f"{type(e).__name__}: {str(e)[:80]}",
                )
            )
        print(
            f"    {t}: {rows[-1]['NES'] if rows[-1]['note'] == '' else rows[-1]['note']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def stouffer_p(nom_p, nes, floor=P_FLOOR):
    """Two-sided Stouffer p of the signed Z of each subtype (sign = sign of its NES; |Z| from its nominal p, floored)."""
    p = np.clip(np.asarray(nom_p, float), floor, 1.0)
    z = np.sign(np.asarray(nes, float)) * stats.norm.isf(p / 2)
    zc = z.sum() / np.sqrt(len(z))
    return float(2 * stats.norm.sf(abs(zc)))


def category_table(fine, mapping):
    """Seven rows: category, NES, FDR (direct FDR for a single subtype; Stouffer-combined p for several), basis, n_subtypes."""
    f = fine.dropna(subset=["NES"]).copy()
    f["category"] = f.fine_type.map(mapping)
    rows = []
    for cat in CATEGORIES:
        g = f[f.category == cat]
        if g.empty:
            continue
        if len(g) == 1:
            rows.append(
                dict(
                    category=cat,
                    NES=float(g.NES.iloc[0]),
                    FDR=float(g.FDR.iloc[0]),
                    basis="FDR",
                    n_subtypes=1,
                )
            )
        else:
            rows.append(
                dict(
                    category=cat,
                    NES=float(g.NES.mean()),
                    FDR=stouffer_p(g.nom_p.values, g.NES.values),
                    basis="combined p",
                    n_subtypes=len(g),
                )
            )
    return pd.DataFrame(rows)


def compare_fig8(cat, claims=FIG8_CLAIMS, tol=NES_TOL):
    rows = []
    C = cat.set_index("category")
    for k, (nes, val) in claims.items():
        if k not in C.index:
            rows.append(
                row(
                    f"Figure 8B {k}: NES, FDR or combined p",
                    f"{nes:+.3f}, {val}",
                    "category not computed",
                    "DIFFERS",
                )
            )
            continue
        r = C.loc[k]
        ok = abs(r.NES - nes) <= tol and (r.FDR < 0.05) == (val < 0.05)
        rows.append(
            row(
                f"Figure 8B {k}: NES, {r.basis}",
                f"{nes:+.3f}, {val}",
                f"{r.NES:+.3f}, {r.FDR:.4g}"
                + (f" ({int(r.n_subtypes)} subtypes)" if r.n_subtypes > 1 else ""),
                "AGREES" if ok else "DIFFERS",
                (
                    ""
                    if ok
                    else f"|dNES| = {abs(r.NES - nes):.3f}; significant at 0.05: manuscript {val < 0.05}, recomputed {bool(r.FDR < 0.05)}: update the text and the figure to the recomputed value"
                ),
            )
        )
    return rows


# ---------------------------------------------------------------------------------------- Figure 2C
def harmonise(label):
    l = str(label).strip().lower()
    for k, v in RULES:
        if l.startswith(k):
            return v
    raise ValueError(label)


def harmonise_all(series):
    labels = sorted(set(series.astype(str)))
    mapping, bad = {}, []
    for l in labels:
        try:
            mapping[l] = harmonise(l)
        except ValueError:
            bad.append(l)
    if bad:
        raise ValueError(
            f"cannot assign {bad} to a panel-A category {HARM}; add a rule or choose another label column"
        )
    return series.astype(str).map(mapping)


def proportions(labels_by_cohort):
    """Fraction of each cohort's nuclei in each panel-A category, and the number of nuclei."""
    props, counts = {}, {}
    for c, s in labels_by_cohort.items():
        h = harmonise_all(s)
        vc = h.value_counts()
        counts[c] = int(len(h))
        props[c] = {k: float(vc.get(k, 0)) / len(h) for k in HARM}
    return pd.DataFrame(props).T[HARM], pd.Series(counts)


def _text_color(hexc):
    r, g, b = (int(hexc[i : i + 2], 16) for i in (1, 3, 5))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "white"


def draw_panel_c(props, counts, out_png, width_px, min_label=0.03):
    """Horizontal stacked bars (one per cohort). The figure is 18 inches wide like the two UMAP panels, so type sizes match when it is placed under them. Returns the number of labelled segments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(18, 4.8), dpi=width_px / 18)
    cohorts = list(props.index)
    y = np.arange(len(cohorts))[::-1]
    n_lab = 0
    for i, c in enumerate(cohorts):
        left = 0.0
        for k in HARM:
            v = float(props.loc[c, k])
            ax.barh(
                y[i],
                v * 100,
                left=left * 100,
                color=COLORS[k],
                edgecolor="white",
                linewidth=0.6,
                height=0.62,
            )
            if v >= min_label:
                ax.text(
                    (left + v / 2) * 100,
                    y[i],
                    f"{v * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=_text_color(COLORS[k]),
                )
                n_lab += 1
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}\n(n={int(counts[c]):,})" for c in cohorts], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of nuclei", fontsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_title(
        f"C. Cell-type proportions per cohort (full data, n={int(counts.sum()):,})", fontsize=13
    )
    [ax.spines[s].set_visible(False) for s in ("top", "right")]
    ax.legend(
        handles=[Patch(facecolor=COLORS[k], label=k) for k in HARM],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(HARM),
        frameon=False,
        fontsize=11,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=width_px / 18)
    plt.close(fig)
    return n_lab


def composite(ab_png, c_png, out_png, out_pdf=None):
    from PIL import Image

    a, c = Image.open(ab_png).convert("RGB"), Image.open(c_png).convert("RGB")
    if c.width != a.width:
        c = c.resize((a.width, round(c.height * a.width / c.width)))
    canvas = Image.new("RGB", (a.width, a.height + c.height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(c, (0, a.height))
    canvas.save(out_png, dpi=(300, 300))
    if out_pdf:
        canvas.save(out_pdf, "PDF", resolution=300.0)
    return canvas.size


def read_obs_labels(cohort, F4):
    import anndata as ad

    path = F4["resolve_input_path"](
        F4["DATA_DIR"] / F4["FILES"][cohort], f"{cohort} cohort", F4["ALLOW_DRIVE_FETCH"]
    )
    a = ad.read_h5ad(path, backed="r")
    try:
        col = LABEL_COL[cohort]
        if col not in a.obs.columns:
            raise KeyError(
                f"{cohort}: column '{col}' not in the object; columns: {list(a.obs.columns)[:30]}"
            )
        return a.obs[col].astype(str).copy()
    finally:
        a.file.close()


# ---------------------------------------------------------------------------------------- orchestration
def analyse(
    only=("figure8b", "figure2c"),
    loader=None,
    obs_loader=None,
    sets=None,
    ab=None,
    out=None,
    perms=1000,
    checkpoint=None,
):
    S = load_ns("secondary_analyses.py")
    F4 = load_ns("make_figure4_cohens_d.py")
    load = loader or F4["load_cohort"]
    out = Path(out) if out else None
    rows, ex = [], {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    if "figure8b" in only:
        try:
            print("\n=== Figure 8B: Nido, UPR by fine cell type")
            adata, ctx = load("Nido")
            t0 = time.time()
            sets = sets if sets is not None else S["gene_sets"]()
            fine = upr_by_celltype(
                S,
                ctx["raw"],
                ctx["donor"],
                ctx["dx"],
                ctx["ct"],
                adata.var_names,
                sets,
                perms=perms,
            )
            obs = getattr(adata, "obs", None)
            mapping = fine_to_broad(list(fine.fine_type), obs, "cell_type")
            fine["category"] = fine.fine_type.map(mapping)
            cat = category_table(fine, mapping)
            ex["fine"], ex["cat"] = fine, cat
            rows += compare_fig8(cat)
            rows.append(
                row(
                    "Figure 8B: fine cell types tested / skipped",
                    "17 tested",
                    f"{int(fine.NES.notna().sum())} tested, {int(fine.NES.isna().sum())} skipped"
                    + (
                        ": "
                        + "; ".join(
                            f"{r.fine_type} ({r.note})" for r in fine[fine.NES.isna()].itertuples()
                        )
                        if fine.NES.isna().any()
                        else ""
                    ),
                    "INFO",
                )
            )
            rows.append(
                row(
                    "Figure 8B: fine type to category",
                    "",
                    "; ".join(
                        f"{c}: {', '.join(sorted(k for k, v in mapping.items() if v == c))}"
                        for c in CATEGORIES
                        if c in mapping.values()
                    ),
                    "INFO",
                    "combined p = Stouffer on the signed Z of the subtypes' nominal p (floor 0.001); category NES = mean of the subtypes' NES",
                )
            )
            if out is not None:
                cat[["category", "NES", "FDR", "basis"]].to_csv(
                    out / "figure8b_source_table.csv", index=False
                )
                fine.to_csv(out / "figure8b_per_subtype.csv", index=False)
            del adata, ctx
            gc.collect()
            print(f"    done in {time.time() - t0:.0f} s")
        except Exception as e:
            rows.append(
                row(
                    "Figure 8B",
                    "",
                    f"{type(e).__name__}: {str(e)[:160]}",
                    "SKIPPED",
                    "pydeseq2/gseapy missing, no gene sets, or an unmapped cell type",
                )
            )
        if checkpoint is not None:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
    if "figure2c" in only:
        try:
            print("\n=== Figure 2C: cell-type proportions")
            get = obs_loader or (lambda c: read_obs_labels(c, F4))
            labs = {c: get(c) for c in ("Jin", "Ma", "Nido")}
            props, counts = proportions(labs)
            ex["props"], ex["counts"] = props, counts
            for c, n in FIG2_N.items():
                rows.append(
                    row(
                        f"Figure 2C {c}: nuclei",
                        f"{n:,}",
                        f"{int(counts[c]):,}",
                        "AGREES" if int(counts[c]) == n else "DIFFERS",
                    )
                )
            rows.append(
                row(
                    "Figure 2C: all nuclei",
                    f"{FIG2_TOTAL:,}",
                    f"{int(counts.sum()):,}",
                    "AGREES" if int(counts.sum()) == FIG2_TOTAL else "DIFFERS",
                )
            )
            rows.append(
                row(
                    "Figure 2C: label column per cohort",
                    "",
                    ", ".join(
                        f"{c}: '{LABEL_COL[c]}' -> {int(labs[c].nunique())} labels" for c in labs
                    ),
                    "INFO",
                    "the harmonised categories are those of panel A; check that panel A used the same grouping",
                )
            )
            if out is not None:
                tab = props.copy()
                tab.insert(0, "n_nuclei", counts)
                tab.to_csv(out / "figure2c_proportions.csv")
                width = 3600
                if ab and Path(ab).exists():
                    from PIL import Image

                    width = Image.open(ab).width
                n_lab = draw_panel_c(props, counts, out / "Figure2C_panel.png", width)
                rows.append(
                    row(
                        "Figure 2C: panel drawn",
                        "",
                        f"{n_lab} segments labelled (>= 3 % of a cohort), width {width}px",
                        "INFO",
                    )
                )
                if ab and Path(ab).exists():
                    size = composite(
                        ab,
                        out / "Figure2C_panel.png",
                        out / "Figure2_ABC.png",
                        out / "Figure2_ABC.pdf",
                    )
                    rows.append(
                        row(
                            "Figure 2: panels A and B with C",
                            "",
                            f"Figure2_ABC.png {size[0]}x{size[1]}px (and .pdf)",
                            "INFO",
                            "check the panel by eye; re-export it from the original figure source if the journal needs vector graphics",
                        )
                    )
                else:
                    rows.append(
                        row(
                            "Figure 2: panels A and B with C",
                            "",
                            "no --ab file given: only the panel was written",
                            "INFO",
                        )
                    )
        except Exception as e:
            rows.append(
                row(
                    "Figure 2C",
                    "",
                    f"{type(e).__name__}: {str(e)[:200]}",
                    "SKIPPED",
                    "a label did not map to a panel-A category, or an object could not be read",
                )
            )
    return rows, ex


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", default="figure8b,figure2c")
    ap.add_argument("--ab", default=None)
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args(argv)
    only = tuple(x for x in a.only.split(",") if x)
    bad = set(only) - {"figure8b", "figure2c"}
    if bad:
        print(f"unknown --only names {sorted(bad)}; choose figure8b, figure2c")
        return 2
    ab = a.ab or next(
        (
            str(p)
            for p in (Path("Figure2_AB_FINAL.png"), Path("/marimo/Figure2_AB_FINAL.png"))
            if p.exists()
        ),
        None,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ex = analyse(only, ab=ab, out=OUT, perms=a.perms, checkpoint=OUT / "partial_verdicts.csv")
    v = pd.DataFrame(rows, columns=["check", "expected", "got", "status", "note"])
    v.to_csv(OUT / "verification_figures_2c_and_8b.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print("\nFIGURES 8B AND 2C FROM THE RAW OBJECTS\n")
    print(v[["check", "expected", "got", "status"]].to_string(index=False))
    n = v.status.value_counts().to_dict()
    print("\nsummary:", n)
    for r in v[v.note != ""].itertuples():
        print(f"  - {r.check}: {r.note}")
    bad = int((v.status == "DIFFERS").sum())
    print(f"\nFiles in {OUT.resolve()}\n")
    print(
        "EVERY CHECKED VALUE AGREES."
        if not bad and not n.get("SKIPPED")
        else f"{bad} value(s) DIFFER, {n.get('SKIPPED', 0)} check(s) SKIPPED."
    )
    return 1 if bad or n.get("SKIPPED") else 0


if __name__ == "__main__":
    code = main()
    if any(m in sys.modules for m in ("marimo", "ipykernel")):
        sys.stdout.write(f"[figures_2c_and_8b finished with status {code}]\n")
    else:
        sys.exit(code)
