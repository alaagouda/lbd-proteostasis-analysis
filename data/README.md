# Data

Raw and processed single-nucleus RNA-seq data are not included in this
repository.

The analyses use three publicly available datasets:

- Jin cohort: GEO [GSE235914](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235914)
- Ma cohort: GEO [GSE253462](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE253462)
- Nido cohort: EGA EGAD50000000430

Please obtain the datasets from their original repositories and follow
the access and data-use requirements attached to each. The EGA-hosted
Nido cohort in particular requires a data access request through EGA's
own process; it isn't a direct download.

## Expected file format

The analysis scripts expect annotated AnnData (`.h5ad`) files. At minimum,
`adata.obs` needs:

- a cell type or cluster annotation column
- a diagnosis/condition column distinguishing disease from control
- a donor or sample ID column, since every effect size in this project is
  computed at the donor level, not the per-nucleus level

Column names differ across the three cohorts depending on which
annotation pipeline produced the file. `common.py` (used by the core
analysis scripts) resolves common naming variants automatically and
prints what it found; check that output before trusting a run on a file
it hasn't seen before, since a silently wrong column match is worse than
a script that just fails to find one.

`adata.X` should hold either raw counts or log-normalised expression -
several scripts check `X.max()` and normalise automatically if it still
looks like raw counts, and print a message when they do this. Two of the
revision scripts (see the main README's "Revision analyses" section)
deliberately skip this check, since matching an earlier analysis's exact
method - including a normalisation step it didn't have - was the whole
point of running them.
