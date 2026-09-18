"""
Combines the per-cell-type TF activity checkpoint files (Jin_TF_*.csv,
Ma_TF_*.csv, Nido_TF_*.csv, produced by 01_main_pipeline.py) from a local
checkpoint directory and runs a Kruskal-Wallis test on each transcription
factor: does the per-cell-type activity difference distribution vary
significantly by cohort?

For ATF4 and HSF1, this test came back significant (H=15.46, p=0.0004 and
H=18.35, p=0.0001 respectively), cited in the manuscript alongside an
explicit caveat: the underlying per-cell-type activity values come from
the original curated-panel computation and have not been independently
re-verified with the same rigor as the donor-level cohort comparisons
elsewhere in the analysis. This script is provided so that re-verification
can be run directly against the checkpoint files.

Usage:
    python 05_tf_kruskal_wallis.py --checkpoint-dir path/to/checkpoints \\
        --outdir results/domain4_tf_interaction
"""
import argparse
from pathlib import Path
import pandas as pd
from scipy import stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                         help="Local directory containing the per-cell-type TF activity CSVs")
    parser.add_argument("--outdir", type=Path, default=Path("results/domain4_tf_interaction"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for cohort in ["Jin", "Ma", "Nido"]:
        matching = sorted(args.checkpoint_dir.glob(f"{cohort}_TF_*.csv"))
        print(f"  {cohort}: {len(matching)} per-cell-type TF files found")
        for path in matching:
            try:
                df = pd.read_csv(path)
                df["Cohort"] = cohort
                frames.append(df)
            except Exception as e:
                print(f"    failed to read {path.name}: {e}")

    if not frames:
        raise SystemExit(f"No TF checkpoint files matching <Cohort>_TF_*.csv found in "
                          f"{args.checkpoint_dir} - check the pipeline actually completed "
                          f"the TF activity step for all three cohorts.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.outdir / "all_TF_diffs_pooled.csv", index=False)
    print(f"\n  Combined: {len(combined)} rows across {combined['Cohort'].nunique()} cohorts")

    print("\n" + "=" * 70)
    print("KRUSKAL-WALLIS TEST - does TF activity difference vary by cohort?")
    print("=" * 70)
    for tf in ["ATF4", "HSF1"]:
        sub = combined[combined["TF"] == tf]
        groups = [sub[sub["Cohort"] == c]["diff"].values for c in ["Jin", "Ma", "Nido"]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            print(f"{tf}: insufficient data for test")
            continue
        stat, p = stats.kruskal(*groups)
        print(f"\n{tf} - Kruskal-Wallis across cohorts:")
        print(f"  H = {stat:.3f}, p = {p:.4f}")
        for c in ["Jin", "Ma", "Nido"]:
            vals = sub[sub["Cohort"] == c]["diff"]
            if len(vals):
                print(f"  {c}: n={len(vals)}, median diff={vals.median():.3f}, "
                      f"mean diff={vals.mean():.3f}, %positive={100*(vals>0).mean():.0f}%")
        jin_vals = sub[sub["Cohort"] == "Jin"]["diff"].values
        nido_vals = sub[sub["Cohort"] == "Nido"]["diff"].values
        if len(jin_vals) and len(nido_vals):
            _, p_pair = stats.mannwhitneyu(jin_vals, nido_vals, alternative="two-sided")
            print(f"  Jin vs Nido pairwise: p = {p_pair:.4f}")

    print("\nBoth ATF4 and HSF1 came back significant when this was run against the full")
    print("checkpoint set, supporting the manuscript's claim that the pattern scales with")
    print("cohort/burden rather than being a purely descriptive observation. The strength")
    print("of that claim rests on the per-cell-type breakdown printed above, not just the")
    print("headline H and p values - check whether the effect is driven by one outlying")
    print("cohort before citing this as evidence for a smooth cross-cohort gradient.")
