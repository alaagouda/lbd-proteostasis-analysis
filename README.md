# Chaperones on a gradient: cortical proteostasis signatures track α-synuclein pathology

Analysis code for Gouda et al., Brain Communications (under review). The study compares proteostasis gene expression in three published post-mortem single-nucleus cohorts: Jin (superior temporal cortex, dementia with Lewy bodies), Ma (cingulate cortex, Parkinson's disease with dementia) and Nido (prefrontal cortex, Parkinson's disease). Every analysis uses the donor as the unit of inference.

## Data

| Cohort | Source | Access |
|---|---|---|
| Jin | GEO GSE235914 | public |
| Ma | GEO GSE253462 | public |
| Nido | EGA EGAD50000000430 | controlled access, through the data access committee |

The scripts read annotated AnnData objects from a folder named `data_snrna/` in the working directory. The expected file names are set in the `FILES` dictionary at the top of each script: `GSE235914_LBD_annotated_FIXED.h5ad`, `GSE253462_annotated.h5ad` and `nido_analysis_ready.h5ad`. Figure 6 also reads `Ma_liana_full.csv` and `Nido_liana_full.csv`, which `liana_pipeline.py` writes; copy them into `data_snrna/`.

## Environment

```
conda env create -f environment.yml
conda activate lbd-proteostasis
```

`environment.yml` pins the versions used for the analyses. The one exception is `liana`, used only by `liana_pipeline.py`: the Methods do not state its version, so the file installs the current release. Record the version you install with `pip show liana`.

## Running

Run every script from the working directory that contains `data_snrna/`, for example:

```
python code/three_cohort_gradient.py
python code/make_figure4_cohens_d.py
```

Results are written to `results/<analysis>/`. Gene sets for GSEA come from the MSigDB Hallmark 2020 library (Enrichr, which needs network access) or from `data_snrna/hallmark.gmt` when that file exists. The scripts can fetch missing input files from a Google Drive folder when `LBD_DRIVE_FOLDER_ID` is set and a token is available; without it they use local files only.

## Scripts and the results they produce

| Paper item | Scripts |
|---|---|
| Fig. 1, pipeline overview | `make_figure1_pipeline.py` |
| Fig. 2C, cell-type composition, and Fig. 8B, unfolded protein response by cell type | `figures_2c_and_8b.py` |
| Gene modules and module scoring (shared by most analyses) | `gene_modules.py`, `module_scoring.py`, `common.py` |
| Fig. 4, donor-level effect sizes of the proteostasis module | `three_cohort_gradient.py`, `make_figure4_cohens_d.py` |
| Proteostasis specificity | `proteostasis_specificity.py` |
| Pseudobulk differential expression | `pseudobulk_de.py` |
| Hallmarks GSEA | `hallmarks_gsea.py` |
| Supplementary Tables 3 and 4 (GSEA, sequencing depth) and the decoupleR checks | `secondary_analyses.py` |
| GO:0006986 enrichment | `go_enrichment.py` |
| Seven-gene Stouffer meta-analysis, Fig. 5A and 8A | `stouffer_meta_analysis.py`, `stouffer_from_counts.py`, `make_figure5_stouffer_tf.py`, `make_figure8_upr.py` |
| Transcription factor activity, Fig. 5B and 5C | `tf_activity_donor_level.py`, `tf_curated_panel.py`, `tf_activity.py` |
| decoupleR with CollecTRI, all eight factors | `decoupler_tf_activity.py` |
| Cell-cell communication, Fig. 6 | `liana_pipeline.py`, `communication_pairs.py` |
| Augur cell-type prioritisation, Fig. 7 | `augur_run.py`, `make_figure7_augur.py` |
| Unfolded protein response analysis, Fig. 8 | `main_pipeline.py`, `make_figure8_upr.py` |
| Gene network and STRING support | `gene_network.py`, `string_network_query.py` |
| Harmony integration metrics | `integration_qc.py` |
| Endothelial hypoxia and proteostasis, Supplementary Fig. 1 | `hypoxia_coupling.py` |
| PDE10A and SNCA, Supplementary Fig. 3 | `pde10a_snca_donor_level.py` |
| SNCA and neuroinflammation, supplementary table | `snca_neuroinflammation_primary.py`, `snca_neuroinflammation_donor_level.py`, `snca_neuroinflammation_scoregenes.py`, `supplementary_table_snca_neuroinflammation.py` |
| Supplementary Tables 1 and 2 (transcription factor panels, module gene lists) | `supplementary_tables.py` |

## Reference values

A run should give these values, which are the ones reported in the paper.

| Analysis | Result |
|---|---|
| Fig. 4, Cohen's d (95% CI) | Jin 1.568 (-1.17 to 4.31); Ma 0.357 (-0.69 to 1.40); Nido 0.114 (-0.61 to 0.84) |
| Seven-gene Stouffer Z | HSPA1A 5.26, HSPA1B 4.04, CRYAB 3.44, HSPB1 3.37, DNAJB1 3.06, HSP90AA1 1.40, HSP90AB1 0.81 |
| decoupleR, mean difference, U, p | Ma HSF1 +0.793, 19, 0.114; Ma ATF4 +0.366, 26, 0.370; Nido ATF4 +0.011, 122, 0.782; Nido HSF1 -0.057, 115, 0.593 |
| Communication balance, disease versus control | Ma U = 21, p = 0.53; Nido U = 7, p = 0.31 |
| PDE10A against proteostasis, rho | Ma -0.539; Nido -0.351 |
| Endothelial SNCA against proteostasis, rho | Ma +0.525; Nido +0.715 |
| Hypoxia against proteostasis, whole-cohort scoring | Ma rho +0.248, p 0.338; Nido rho +0.314, p 0.075 |
| STRING functional network (release 12.0) | 53 of 60 Tier 1 edges present |

## Citation

Gouda et al., Chaperones on a gradient: cortical proteostasis signatures track α-synuclein pathology. Brain Communications, under review. A DOI will be added when the code is archived.

## Licence

MIT License. Copyright (c) 2026 the authors of Gouda et al.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
