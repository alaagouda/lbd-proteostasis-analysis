"""
Searches a local directory of checkpointed CSV files for a NES value
matching a reported figure, to trace which analysis run produced it.
Does not recompute anything - it scans every numeric column of every
CSV already sitting in the given directory. Note that tracing a
value's origin and confirming its correctness are different questions;
this script answers only the first, by design.

Usage:
    python 04_gsea_source_search.py --checkpoint-dir path/to/checkpoints \\
        --target 0.807 --tolerance 0.01
"""
import argparse
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                         help="Local directory containing previously saved checkpoint CSVs")
    parser.add_argument("--target", type=float, required=True,
                         help="The NES value to search for")
    parser.add_argument("--tolerance", type=float, default=0.01,
                         help="How close a match needs to be (default 0.01)")
    parser.add_argument("--name-filter", type=str, default=None,
                         help="Optional substring (e.g. 'astro') to flag matching filenames separately")
    args = parser.parse_args()

    csv_files = sorted(args.checkpoint_dir.glob("*.csv"))
    print(f"{len(csv_files)} CSV files found in {args.checkpoint_dir}")

    print(f"\nSearching for NES values near {args.target} (+/- {args.tolerance})...")
    found_any = False
    filter_matches_checked = []

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  could not read {path.name}: {e}")
            continue

        if args.name_filter and args.name_filter.lower() in path.name.lower():
            filter_matches_checked.append(path.name)

        for col in df.columns:
            if "nes" not in col.lower():
                continue
            try:
                matches = df[(df[col] - args.target).abs() < args.tolerance]
            except TypeError:
                continue
            if len(matches):
                found_any = True
                print(f"\n  FOUND in {path.name} (column '{col}'):")
                print(matches.to_string())

    if args.name_filter:
        print(f"\nFiles matching '{args.name_filter}' checked ({len(filter_matches_checked)}):")
        for f in filter_matches_checked:
            print(f"  {f}")

    if not found_any:
        print(f"\nNo file in {args.checkpoint_dir} contains a NES value near {args.target}.")
        print("Either the source file hasn't been added to this checkpoint directory yet,")
        print("or the original computation's output was never saved (terminal-only output,")
        print("now unrecoverable). Check for a more complete copy of the checkpoint set before")
        print("concluding the value is genuinely untraceable.")
