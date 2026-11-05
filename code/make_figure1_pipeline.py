#!/usr/bin/env python3
"""
make_figure1_pipeline.py : Figure 1, the analytical pipeline, drawn from one table of facts and wording.

WHY THIS SCRIPT EXISTS
  The previous Figure 1 was a hand-made graphic that had drifted away from the manuscript: it still said "28-fold NLGN1:TNF regional inversion" and "ATF4 + TP53 active", named SingleR for annotation,
  gave Ma as "~57K" nuclei (64,370), called Jin "LBD" (the manuscript says DLB), and sent every cohort through CellBender and DoubletFinder. Its text was also below the journal's 12 pt minimum.

HOW IT WORKS
  * Everything the figure says is in the block "FACTS AND WORDING" below. Edit that block, not the drawing code.
  * Text is wrapped by its MEASURED width, box heights follow from the text, and the figure height follows from the boxes (it must not exceed 210 mm).
  * Before anything is written the script checks: all text is 12 pt; every line lies inside its box and inside the page; the facts add up (56 donors, 384,100 nuclei); the figure fits 170 x 210 mm.
  * Needs no data and no network: python make_figure1_pipeline.py     (writes results/figure1/Figure1.*; set LBD_FIG1_OUT to change the folder)
OUTPUT  Figure1.pdf (vector, fonts embedded), Figure1.svg (text stays editable text), Figure1.png and Figure1.tiff (600 dpi), Figure1_alt_text.txt, Figure1_legend.txt
"""

import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath

OUT_DIR = Path(os.environ.get("LBD_FIG1_OUT", "results/figure1"))
PREFIX = "Figure1"
W_MM, MAX_H_MM, FONT_PT = (
    170.0,
    210.0,
    12,
)  # journal typeset width and height limit; minimum text size
MARGIN, PAD, GAP, LINE_FACTOR = (
    2.0,
    2.6,
    4.6,
    1.17,
)  # mm, mm, mm between rows, line spacing as a multiple of the font size
FONT = "DejaVu Sans"  # ships with matplotlib, so the figure is reproducible on any machine

# =====================================================================================================================
# FACTS AND WORDING: the only part to edit when the manuscript changes
# =====================================================================================================================
COHORTS = {  # order = temporal, cingulate, prefrontal (heavy to spared Lewy pathology)
    "Jin": dict(
        accession="GSE235914",
        region="Temporal cortex",
        disease="DLB",
        cases=3,
        controls=3,
        nuclei=41899,
        burden="Heavily affected",
    ),
    "Ma": dict(
        accession="GSE253462",
        region="Cingulate cortex",
        disease="PD",
        cases=9,
        controls=8,
        nuclei=64370,
        burden="Intermediate",
    ),
    "Nido": dict(
        accession="EGAD50000000430",
        region="Prefrontal cortex",
        disease="PD",
        cases=20,
        controls=13,
        nuclei=277831,
        burden="Relatively spared",
    ),
}
EXPECTED_TOTALS = dict(donors=56, nuclei=384100)  # abstract: "56 donors, 384,100 nuclei"

BLOCKS = [  # (key, title, paragraphs, bullets)
    (
        "pre",
        "Preprocessing",
        [
            "Jin, Ma: Cell Ranger matrices (GEO). Nido: CellBender matrices.",
            "QC filter; Scrublet doublets (all), DoubletFinder (Nido).",
            "Log-normalisation; Harmony (UMAP and annotation only).",
        ],
        False,
    ),
    (
        "ann",
        "Cell-type annotation",
        [
            "Jin, Ma: marker scores. Nido: CellTypist, then marker check.",
            "All cohorts: manual curation with canonical markers.",
        ],
        False,
    ),
    (
        "pri",
        "Primary analyses",
        [
            "Module scoring; pseudobulk DE (PyDESeq2); GSEA and GO enrichment; gene network with STRING; LIANA+ communication."
        ],
        False,
    ),
    (
        "sec",
        "Secondary analyses",
        [
            "Augur ranking; TF activity (curated panel, decoupleR check); PDE10A and SNCA donor-level correlations; endothelial hypoxia and proteostasis (exploratory); Stouffer weighted-Z meta-analysis."
        ],
        False,
    ),
    (
        "fin",
        "Principal findings",
        [
            "Proteostasis gradient: temporal > cingulate > prefrontal",
            "Cell-type ranking differed by cohort; no clear leader in prefrontal cortex",
            "Microglia-sourced signalling: no diagnosis-dependent shift",
            "ATF4-associated signal: hypothesis-generating candidate",
            "Secondary: PDE10A and endothelial SNCA correlate with proteostasis (donor level)",
        ],
        True,
    ),
]
COLOURS = {
    "data": ("#CFE0F3", "#2F5D8C"),
    "pre": ("#D9EAC2", "#5B8A2E"),
    "ann": ("#D9EAC2", "#5B8A2E"),
    "pri": ("#FBDDBA", "#B8691A"),
    "sec": ("#FBDDBA", "#B8691A"),
    "fin": ("#DCCFEA", "#6B4C93"),
}
TEXT, ARROW = "#111111", "#4A6FA5"

# =====================================================================================================================
# derived facts and checks
# =====================================================================================================================
TOTAL_DONORS = sum(c["cases"] + c["controls"] for c in COHORTS.values())
TOTAL_NUCLEI = sum(c["nuclei"] for c in COHORTS.values())
LINE_MM = FONT_PT * LINE_FACTOR * 25.4 / 72


def check_facts():
    got = dict(donors=TOTAL_DONORS, nuclei=TOTAL_NUCLEI)
    if got != EXPECTED_TOTALS:
        raise SystemExit(
            f"Figure 1 facts do not add up: {got} versus the manuscript's {EXPECTED_TOTALS}"
        )


def width_mm(s, bold=False):
    """Rendered width of a string in mm, from the font outlines (no canvas needed)."""
    return (
        TextPath(
            (0, 0),
            s,
            size=FONT_PT,
            prop=FontProperties(family=FONT, weight="bold" if bold else "normal"),
        )
        .get_extents()
        .width
        * 25.4
        / 72
    )


def wrap(text, max_mm, bold=False):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if cur and width_mm(trial, bold) > max_mm:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur]


def layout():
    """All text lines and boxes in mm, top to bottom. Returns (boxes, total_height)."""
    boxes, y = [], MARGIN
    head = f"Three post-mortem cohorts: {TOTAL_DONORS} donors, {TOTAL_NUCLEI:,} nuclei"
    boxes.append(
        dict(
            kind="head",
            x=MARGIN,
            y=y,
            w=W_MM - 2 * MARGIN,
            h=LINE_MM,
            lines=[(head, True)],
            align="center",
        )
    )
    y += LINE_MM + 1.6
    bw = (W_MM - 2 * MARGIN - 2 * 3.0) / 3
    rows = []
    for k, (name, c) in enumerate(COHORTS.items()):
        lines = [
            (name, True),
            (c["accession"], False),
            (c["region"], False),
            (f"{c['cases']} {c['disease']} + {c['controls']} control", False),
            (f"{c['nuclei']:,} nuclei", False),
            (c["burden"], False),
        ]
        rows.append(
            dict(kind="data", x=MARGIN + k * (bw + 3.0), y=y, w=bw, lines=lines, align="center")
        )
    h = max(len(r["lines"]) for r in rows) * LINE_MM + 2 * PAD
    for r in rows:
        r["h"] = h
        boxes.append(r)
    y += h
    for key, title, paras, bullets in BLOCKS:
        inner = W_MM - 2 * MARGIN - 2 * PAD - 1.0
        lines = [(title, True)]
        for p in paras:
            first = True
            for l in wrap(("\u2022 " if bullets else "") + p, inner):
                lines.append((l if first or not bullets else "  " + l, False))
                first = False
        h = len(lines) * LINE_MM + 2 * PAD
        y += GAP
        boxes.append(
            dict(kind=key, x=MARGIN, y=y, w=W_MM - 2 * MARGIN, h=h, lines=lines, align="left")
        )
        y += h
    return boxes, y + MARGIN


def draw(boxes, H):
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": FONT_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(W_MM / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_MM)
    ax.set_ylim(H, 0)
    ax.axis("off")
    placed = []
    for b in boxes:
        if b["kind"] != "head":
            fc, ec = COLOURS[b["kind"]]
            ax.add_patch(
                FancyBboxPatch(
                    (b["x"], b["y"]),
                    b["w"],
                    b["h"],
                    boxstyle="round,pad=0,rounding_size=2.2",
                    fc=fc,
                    ec=ec,
                    lw=1.5,
                )
            )
        for i, (s, bold) in enumerate(b["lines"]):
            yy = b["y"] + (PAD if b["kind"] != "head" else 0) + (i + 0.5) * LINE_MM
            if b["align"] == "center":
                t = ax.text(
                    b["x"] + b["w"] / 2,
                    yy,
                    s,
                    ha="center",
                    va="center",
                    color=TEXT,
                    fontweight="bold" if bold else "normal",
                    fontsize=FONT_PT,
                )
            else:
                t = ax.text(
                    b["x"] + PAD,
                    yy,
                    s,
                    ha="left",
                    va="center",
                    color=TEXT,
                    fontweight="bold" if bold else "normal",
                    fontsize=FONT_PT,
                )
            placed.append((t, b))
    data = [b for b in boxes if b["kind"] == "data"]
    nxt = next(b for b in boxes if b["kind"] == "pre")
    for b in data:
        ax.annotate(
            "",
            xy=(b["x"] + b["w"] / 2, nxt["y"]),
            xytext=(b["x"] + b["w"] / 2, b["y"] + b["h"]),
            arrowprops=dict(
                arrowstyle="-|>", lw=1.8, color=ARROW, mutation_scale=16, shrinkA=0, shrinkB=0
            ),
        )
    flow = [b for b in boxes if b["kind"] in ("pre", "ann", "pri", "sec", "fin")]
    for a, c in zip(flow, flow[1:]):
        ax.annotate(
            "",
            xy=(W_MM / 2, c["y"]),
            xytext=(W_MM / 2, a["y"] + a["h"]),
            arrowprops=dict(
                arrowstyle="-|>", lw=1.8, color=ARROW, mutation_scale=16, shrinkA=0, shrinkB=0
            ),
        )
    return fig, ax, placed


def verify(fig, ax, placed):
    """Every text is 12 pt and lies inside its box and the page. A figure that fails is never written."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for t, b in placed:
        (x0, y0), (x1, y1) = inv.transform(t.get_window_extent(r).get_points())
        ymin, ymax = min(y0, y1), max(y0, y1)
        bx0, bx1 = (b["x"], b["x"] + b["w"]) if b["kind"] != "head" else (0.0, W_MM)
        if t.get_fontsize() < FONT_PT - 1e-6:
            bad.append(f"{t.get_text()!r}: {t.get_fontsize()} pt")
        if x0 < bx0 + 0.6 or x1 > bx1 - 0.6:
            bad.append(
                f"{t.get_text()!r}: leaves its box horizontally ({x0:.1f} to {x1:.1f} mm, box {bx0:.1f} to {bx1:.1f})"
            )
        if ymin < b["y"] - 0.2 or ymax > b["y"] + b["h"] + 0.2:
            bad.append(f"{t.get_text()!r}: leaves its box vertically")
    if bad:
        raise SystemExit("Figure 1 layout check failed:\n  " + "\n  ".join(bad))


def alt_text():
    c = COHORTS
    parts = "; ".join(
        f"{n}, {v['region'].lower()}, {v['disease']}, {v['cases']} + {v['controls']} donors, {v['nuclei']:,} nuclei"
        for n, v in c.items()
    )
    findings = "; ".join(p for k, t, ps, bl in BLOCKS if k == "fin" for p in ps)
    return (
        f"Flow diagram of the analytical pipeline. Three post-mortem cohorts ({parts}; {TOTAL_DONORS} donors and {TOTAL_NUCLEI:,} nuclei in total) pass through preprocessing, cell-type annotation, "
        f"primary analyses and secondary analyses to five principal findings: {findings}."
    )


def legend_text():
    c = COHORTS
    return (
        "Fig. 1. Analytical pipeline overview. Three post-mortem snRNA-seq cohorts from regions with different Lewy pathology burden ("
        + f"{TOTAL_DONORS} donors, {TOTAL_NUCLEI:,} nuclei"
        + ") pass through preprocessing, cell-type annotation, "
        "primary and secondary analyses to the principal findings. Cohort details: "
        + ", ".join(
            f"{n} ({v['accession']}, n={v['cases']}+{v['controls']}, {v['nuclei']:,} nuclei, {'sporadic ' if v['disease'] == 'PD' else ''}{v['disease']} {v['region'].lower()})"
            for n, v in c.items()
        )
        + ". Jin and Ma matrices were processed with Cell Ranger by the data-generating studies; Nido matrices were CellBender-corrected by the data-generating laboratory. Doublets were removed with Scrublet in all cohorts and with DoubletFinder in Nido. "
        "Cell-type annotation used a marker-score-based approach (Scanpy score_genes) for Jin and Ma, with CellTypist as an initial classification step for Nido, followed by manual curation using established marker genes in all cohorts. "
        "Harmony-corrected embeddings were used for visualisation and annotation only; downstream statistics used the uncorrected normalised counts. The ATF4-associated signal is a hypothesis-generating candidate. All analyses performed in Python 3.11.9."
    )


def save(fig):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / PREFIX
    fig.savefig(f"{p}.pdf")
    fig.savefig(f"{p}.svg")
    fig.savefig(f"{p}.png", dpi=600)
    Image.open(f"{p}.png").convert("RGB").save(f"{p}.tiff", compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)
    (OUT_DIR / f"{PREFIX}_alt_text.txt").write_text(alt_text() + "\n", encoding="utf-8")
    (OUT_DIR / f"{PREFIX}_legend.txt").write_text(legend_text() + "\n", encoding="utf-8")
    return p


def main():
    check_facts()
    boxes, H = layout()
    if H > MAX_H_MM:
        raise SystemExit(
            f"Figure 1 is {H:.1f} mm tall; the journal limit is {MAX_H_MM:.0f} mm. Shorten the wording in BLOCKS."
        )
    fig, ax, placed = draw(boxes, H)
    verify(fig, ax, placed)
    p = save(fig)
    print(
        f"Figure 1 written: {W_MM:.0f} x {H:.1f} mm, {len(placed)} text lines at {FONT_PT} pt, all inside their boxes. Files: {p}.pdf .svg .png .tiff and the alt-text and legend files in {OUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
