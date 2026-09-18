# Cortical proteostasis in Lewy body disorders - analysis code

Scripts used for the pseudobulk differential expression, module scoring,
network, transcription factor activity, and cell-cell communication
analyses in Gouda et al., "Cortical proteostasis signatures track
α-synuclein pathology and ATF4-driven stress signalling in Lewy body
disorders," together with the additional scripts written in response to
peer review (see "Revision analyses" below).

## Data

The three snRNA-seq datasets analysed here are publicly available and are
not included in this repository:

- Jin cohort: GEO [GSE235914](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235914)
- Ma cohort: GEO [GSE253462](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE253462)
- Nido cohort: EGA EGAD50000000430

Each script expects an annotated `.h5ad` file as input, with per-nucleus
cell type, diagnosis, and donor/sample columns. Column names vary slightly
between cohorts depending on the annotation pass that produced the file;
`common.py` resolves them automatically and prints what it found, so check
that output before trusting a run on a new file.

## Scripts

| Script | Purpose |
|---|---|
| `pseudobulk_de.py` | Per-cell-type pseudobulk differential expression (PyDESeq2) |
| `module_scoring.py` | Donor-level Cohen's d for all eight gene modules |
| `stouffer_meta_analysis.py` | Weighted-Z meta-analysis of the seven pre-specified heat shock genes across cohorts |
| `gene_network.py` | Gene-gene correlation network with STRING/BioGRID validation |
| `augur_prioritisation.py` | Cell-type ranking by disease/control classifiability |
| `tf_activity.py` | Transcription factor regulon activity via decoupleR/CollecTRI, the orthogonal check against `03_tf_curated_panel.py`'s primary method |
| `go_enrichment.py` | GO:0006986 pre-specified enrichment test |
| `hallmarks_gsea.py` | MSigDB Hallmarks preranked GSEA |
| `cell_cell_communication.py` | Ligand-receptor analysis (LIANA+) |
| `endothelial_coupling.py` | Endothelial hypoxia-proteostasis correlation |
| `integration_qc.py` | Harmony batch correction quality metrics |

`common.py` and `gene_modules.py` hold shared helper functions and gene
set definitions used across the above.

## Verification and QC utilities

Scripts `01` through `15` cover verification, quality control, and
figure generation. They're numbered in the rough order they were
written rather than grouped by topic. Each takes its input as a local
file or directory path (`--input`, `--jin`/`--ma`/`--nido`, or
`--checkpoint-dir` depending on the script - run any of them with
`--help` to see its exact arguments), so nothing here depends on how or
where the underlying data happens to be stored on your own machine.

Two scripts (`04` and `05`) don't take a single h5ad file at all. They
work by scanning a directory of intermediate CSV outputs already
produced by an earlier run of `01_main_pipeline.py`, since their job is
to trace a specific reported number back to whichever run actually
computed it, not to run a fresh analysis themselves - point their
`--checkpoint-dir` argument at wherever those CSVs live locally.

| Script | What it was for |
|---|---|
| `01_main_pipeline.py` | Checkpointed rerun of the six core analyses (pseudobulk DE, Hallmarks GSEA, TNF/TNFR, Augur, per-cell-type UPR, per-cell-type TF activity) after the original run died partway through; every unit of work is saved to a local checkpoint directory as it finishes so a restart never repeats work |
| `03_tf_curated_panel.py` | Curated target-gene panel TF activity per donor, the primary method for the transcription-factor results reported in the manuscript |
| `04_gsea_source_search.py` | Traces which analysis run produced a given NES value, by searching a directory of checkpointed CSVs |
| `05_tf_kruskal_wallis.py` | Cross-cohort Kruskal-Wallis test on the per-cell-type TF activity checkpoints, testing whether ATF4/HSF1 activity differs significantly by cohort |
| `06_nuclei_count_search.py` | Verifies the total nucleus count across all three cohorts directly from the h5ad files |
| `07_snca_pde10a_correlation.py` | Computes the SNCA-PDE10A correlation by cell type for all three cohorts, on raw (not log-normalised) expression, appropriate for this specific comparison |

| `09_figure_pde10a_snca.py` | Builds the new supplementary figure for the PDE10A/SNCA/proteostasis correlations, requested by a reviewer who noted the existing text had no supporting figure |
| `10_figure_proteostasis_gradient.py` | Builds the main proteostasis-gradient figure (Cohen's d across cohorts) |
| `11_figure_augur_ranking.py` | Builds the Augur cell-type ranking figure from raw-count AUROC values |
| `12_qc_figure_specs.py` | Checks every exported figure file against the journal's size and resolution requirements |
| `13_qc_axis_labels.py` | Scans plotting scripts for axes that are missing a label or a unit |
| `14_qc_legend_abbreviations.py` | Scans the manuscript text itself for any abbreviation used in a figure legend that isn't defined in that same legend |
| `15_graphical_abstract_template.py` | Generates a blank canvas at the journal's exact required size and resolution for the graphical abstract, as a starting point rather than a finished figure |

A few of these (`02`, `03`, `04`, `06`) are investigation scripts kept as
part of the record of how a specific number was traced back to its
source, not because they're meant to be rerun routinely - `04_gsea_source_search.py`,
for instance, exists to answer one specific question about one specific
value, and says so in its own docstring.

## A note on Cohen's d

Effect sizes throughout are computed at the donor level: per-cell scores
are averaged within each sample first, and the group comparison runs on
those per-donor means. Computing Cohen's d directly from per-cell scores
treats each nucleus as an independent observation, which inflates sample
size and gives an effect size that doesn't reflect the actual number of
biological replicates. `module_scoring.py` and `go_enrichment.py` both
aggregate to donor level before any statistic is computed - this matters
if you're adapting these scripts for other data.

## Environment

See `environment.yml`. Built and run on Python 3.11. Package versions
are pinned to match those stated in the manuscript's Methods (Scanpy
1.10.1, PyDESeq2 0.4, GSEApy 1.1.3, decoupleR 1.8.0, LIANA+ 1.1.0,
CellTypist 1.7.1, pertpy 1.0.3, NetworkX 3.6.1).

```
conda env create -f environment.yml
conda activate lbd-proteostasis
```

## Usage

Each script takes `--input`, `--cohort`, and `--outdir` (some take
additional arguments - run with `--help` to see them). Typical order:

```
python pseudobulk_de.py --input <h5ad> --cohort Ma --outdir results/ma_de
python module_scoring.py --input <h5ad> --cohort Ma --outdir results/ma_modules
python stouffer_meta_analysis.py --jin ... --ma ... --nido ... --outdir results/meta
python gene_network.py --input <h5ad> --cohort Ma --outdir results/ma_network
python augur_prioritisation.py --input <h5ad> --cohort Ma --outdir results/ma_augur
python tf_activity.py --input <h5ad> --cohort Ma --outdir results/ma_tf
python go_enrichment.py --input <h5ad> --cohort Ma --outdir results/ma_go
python hallmarks_gsea.py --de-dir results/ma_de --cohort Ma --outdir results/ma_gsea
python cell_cell_communication.py --input <h5ad> --cohort Ma --outdir results/ma_liana
python endothelial_coupling.py --input <h5ad> --cohort Ma --outdir results/ma_endo
python integration_qc.py --input <h5ad> --outdir results/qc
```

`gene_network.py` optionally uses a BioGRID API key
(`BIOGRID_API_KEY` environment variable) for experimental interaction
cross-validation; it runs without one but skips that check.
