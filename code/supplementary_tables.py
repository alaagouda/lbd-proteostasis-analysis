#!/usr/bin/env python3
"""
supplementary_tables.py : write Supplementary Table 1 (TF panels) and Supplementary Table 2 (module gene lists) from the code of record, so the tables cannot drift from the analysis.

    python supplementary_tables.py [--out DIR] [--gene-modules PATH] [--tf-script PATH]

  Table 1  the TF_PANELS dictionary of tf_activity_donor_level.py (the script that produced the donor-level activities in the manuscript)
  Table 2  MODULES (the seven non-proteostasis modules) and HYPOXIA from gene_modules.py. The 14-gene proteostasis list is in the main text and is not repeated.

Why generated: an earlier Table 2 lacked the Hypoxia list that the Methods and the Supplementary Fig. 1 legend cite, a later copy said "seven modules" above eight rows, and a copy of Table 1 described the
z-scoring wrongly. Writing the tables from the code makes the gene lists and counts true by construction; tests/test_supplementary_tables.py checks the wording.
"""

import argparse, ast, math
from pathlib import Path
import openpyxl
import openpyxl.worksheet.properties
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
TITLE_1 = "Supplementary Table 1. Curated target-gene panels used for transcription-factor activity analysis."
NOTE_1 = (
    "Literature-validated direct target genes assembled per transcription factor for the primary curated-panel method (see Methods). For each donor, the value of a gene is the mean over the donor's nuclei of "
    "per-nucleus log-normalised expression (counts per 10,000, log1p). Each target gene is then z-scored across donors, and the z-scores are averaged within the panel to give the donor-level activity of the "
    "transcription factor. The activity difference reported in the Results is the mean in disease minus the mean in control donors."
)
TITLE_2 = "Supplementary Table 2. Gene compositions of the seven non-proteostasis expression modules and the Hypoxia module."
NOTE_2 = (
    "The seven non-proteostasis modules were scored per nucleus with Scanpy score_genes, as was the 14-gene proteostasis module, whose gene list is given in full in the main text and is not repeated here. "
    "The Hypoxia module was used only for the exploratory endothelial hypoxia-proteostasis sensitivity analysis described in the Results and Supplementary Fig. 1."
)
DISPLAY = {"Oxidative_Stress": "Oxidative Stress", "ER_Stress": "ER Stress"}


def find(name, extra=None):
    for d in ([Path(extra)] if extra else []) + [
        HERE,
        HERE.parent / "scripts",
        HERE.parent / "revision",
        HERE.parent / "closing_kit",
        Path("."),
        Path("scripts"),
        Path("revision"),
    ]:
        p = d if d.is_file() else d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found; pass its path explicitly")


def read_modules(path):
    """MODULES and HYPOXIA from gene_modules.py. The file is plain data, so it is executed in an empty namespace."""
    ns = {}
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), ns)
    mods = {k: list(v) for k, v in ns["MODULES"].items() if k != "Proteostasis"}
    return mods, list(ns["HYPOXIA"])


def read_tf_panels(path):
    """TF_PANELS from the verification script, read from the source without importing it (the script imports Scanpy)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TF_PANELS" for t in node.targets
        ):
            return {k: list(v) for k, v in ast.literal_eval(node.value).items()}
    raise KeyError("TF_PANELS not found")


def _write(path, title, note, header, rows, widths):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TF gene panels" if "Table 1" in title else "Module gene compositions"
    n = len(header)
    ws["A1"] = title
    ws["A1"].font = Font(name="Arial", size=12, bold=True)
    ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n)
    ws.row_dimensions[1].height = (
        17 * math.ceil(len(title) / (sum(widths) * 0.85)) + 3
    )  # the title wraps instead of running off the page
    ws["A2"] = note
    ws["A2"].font = Font(name="Arial", size=10, italic=True)
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=n)
    ws.row_dimensions[2].height = 14 * math.ceil(len(note) / (sum(widths) * 0.95)) + 6
    for j, h in enumerate(header, 1):
        c = ws.cell(row=4, column=j, value=h)
        c.font = Font(name="Arial", size=11, bold=True)
        c.fill = PatternFill("solid", fgColor="D9E1F2")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[4].height = 27.6
    for i, (name, count, genes) in enumerate(rows, 5):
        ws.cell(row=i, column=1, value=name).font = Font(name="Arial", size=11, bold=True)
        ws.cell(row=i, column=2, value=count).font = Font(name="Arial", size=11)
        g = ws.cell(row=i, column=3, value=", ".join(genes))
        g.font = Font(name="Arial", size=11)
        g.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row=i, column=1).alignment = Alignment(vertical="top")
        ws.cell(row=i, column=2).alignment = Alignment(horizontal="center", vertical="top")
        ws.row_dimensions[i].height = (
            15 * math.ceil(len(", ".join(genes)) / (widths[2] * 0.80)) + 6
        )  # generous: wrapped rows must not clip
    for col, w in zip("ABC", widths):
        ws.column_dimensions[col].width = w
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr = openpyxl.worksheet.properties.PageSetupProperties(
        fitToPage=True
    )  # one page wide when printed or converted to PDF
    wb.save(path)


def build(out, gene_modules=None, tf_script=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mods, hyp = read_modules(find("gene_modules.py", gene_modules))
    panels = read_tf_panels(find("tf_activity_donor_level.py", tf_script))
    for name, genes in list(mods.items()) + [("Hypoxia", hyp)] + list(panels.items()):
        if len(genes) != len(set(genes)):
            raise ValueError(f"{name}: duplicate genes in {genes}")
    t1, t2 = (
        out / "Supplementary_Table_1_TF_gene_panels_FINAL.xlsx",
        out / "Supplementary_Table_2_module_gene_lists_FINAL.xlsx",
    )
    _write(
        t1,
        TITLE_1,
        NOTE_1,
        ["Transcription factor", "Number of target genes", "Target genes"],
        [(tf, len(g), g) for tf, g in panels.items()],
        (22, 18, 70),
    )
    _write(
        t2,
        TITLE_2,
        NOTE_2,
        ["Module", "Number of genes", "Genes"],
        [(DISPLAY.get(k, k), len(g), g) for k, g in mods.items()] + [("Hypoxia", len(hyp), hyp)],
        (20, 16, 75),
    )
    return t1, t2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/supplementary_tables")
    ap.add_argument("--gene-modules")
    ap.add_argument("--tf-script")
    a = ap.parse_args()
    for p in build(a.out, a.gene_modules, a.tf_script):
        print("written", p)
