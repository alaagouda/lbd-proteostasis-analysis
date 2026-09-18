"""
Verifies the total nucleus count across all three cohorts by summing
each cohort's raw, as-loaded nucleus count directly from the h5ad
files, with no cell-type exclusion applied to any of them.

Usage:
    python 06_nuclei_count_search.py --jin path/to/jin.h5ad \\
        --ma path/to/ma.h5ad --nido path/to/nido.h5ad
"""
import argparse
from pathlib import Path

import scanpy as sc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jin", type=Path, required=True)
    parser.add_argument("--ma", type=Path, required=True)
    parser.add_argument("--nido", type=Path, required=True)
    args = parser.parse_args()

    counts = {}
    for cohort, path in [("Jin", args.jin), ("Ma", args.ma), ("Nido", args.nido)]:
        adata = sc.read_h5ad(path, backed="r")
        counts[cohort] = adata.n_obs
        print(f"  {cohort} ({path.name}): n_obs = {adata.n_obs}")

    total = sum(counts.values())
    print(f"\nTotal across all three cohorts: {total}")
