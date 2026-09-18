"""
C11 supplementary figure - PDE10A, SNCA, and endothelial correlation results.

Reproduces, with real underlying data, the three relationships already
described in the manuscript's Methods/Results text but never shown:
  Panel A: PDE10A vs Proteostasis (Nido, cell-level, all nuclei)
  Panel B: SNCA vs Proteostasis, endothelial cells only (Nido)
  Panel C: SNCA-PDE10A correlation by cell type, Nido vs Ma side by side
           (the "reversal from inverse to positive" finding discussed
           in the manuscript's Discussion)

Requires a 'Proteostasis' module score already present in .obs (Nido)
or computes one fresh via scanpy score_genes (Ma, if not present).

Usage:
    python 09_figure_pde10a_snca.py --nido path/to/nido.h5ad --ma path/to/ma.h5ad \\
        --outdir results/c11_pde10a_snca_figure
"""
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

PROTEOSTASIS_GENES = [
    "HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "DNAJB4", "HSP90AA1", "HSP90AB1",
    "DNAJA1", "DNAJA4", "CHORDC1", "STIP1", "CRYAB", "BAG3", "HSPH1",
]


def ensure_lognorm_and_score(adata):
    X = adata.X
    xmax = X.data.max() if sps.issparse(X) else float(np.asarray(X).max())
    if xmax > 20:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if "Proteostasis_score" not in adata.obs.columns and "Proteostasis" not in adata.obs.columns:
        avail = [g for g in PROTEOSTASIS_GENES if g in adata.var_names]
        sc.tl.score_genes(adata, avail, score_name="Proteostasis_score")
        return "Proteostasis_score"
    return "Proteostasis_score" if "Proteostasis_score" in adata.obs.columns else "Proteostasis"


def gene_vector(adata, gene):
    if gene not in adata.var_names:
        return None
    X = adata[:, gene].X
    return np.asarray(X.todense()).flatten() if sps.issparse(X) else np.asarray(X).flatten()


def format_pvalue(p):
    """Never display a p-value as exactly 0. scipy's spearmanr underflows
    to literal 0.0 for very large n with a strong correlation -- this is
    a floating-point artifact, not a true result, since no real p-value
    is exactly zero. Report the smallest representable bound instead of
    a false exact-zero claim."""
    if p == 0.0:
        tiny = np.finfo(float).tiny  # smallest positive representable float
        return f"< {tiny:.0e}"
    return f"= {p:.2e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nido", type=Path, required=True, help="Path to the Nido cohort h5ad")
    parser.add_argument("--ma", type=Path, required=True, help="Path to the Ma cohort h5ad")
    parser.add_argument("--outdir", type=Path, default=Path("results/c11_pde10a_snca_figure"))
    args = parser.parse_args()
    OUT_DIR = args.outdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {"Nido": args.nido, "Ma": args.ma}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel A: PDE10A vs Proteostasis, Nido, all nuclei ---
    print("\nLoading Nido for Panel A/B...")
    nido = sc.read_h5ad(paths["Nido"])
    prot_col = ensure_lognorm_and_score(nido)
    pde10a = gene_vector(nido, "PDE10A")
    prot_scores = nido.obs[prot_col].values

    ax = axes[0]
    if pde10a is not None:
        rho, p = stats.spearmanr(pde10a, prot_scores)
        ax.scatter(pde10a, prot_scores, s=2, alpha=0.15, color="steelblue", rasterized=True)
        z = np.polyfit(pde10a, prot_scores, 1)
        xs = np.linspace(pde10a.min(), pde10a.max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color="black", linewidth=1.5)
        ax.set_title("A. PDE10A vs Proteostasis (Nido prefrontal cortex)")
        ax.text(0.97, 0.95, f"$\\rho$ = {rho:.3f}\np {format_pvalue(p)}\nn = {len(pde10a):,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.set_xlabel("PDE10A expression (log-normalised)")
        ax.set_ylabel("Proteostasis module score")
        print(f"  Panel A: rho={rho:.4f}, p{format_pvalue(p)} (n={len(pde10a)})")
    else:
        ax.text(0.5, 0.5, "PDE10A not detected", ha="center")

    # --- Panel B: SNCA vs Proteostasis, endothelial cells only, Nido ---
    ax = axes[1]
    ct_col = "cell_type" if "cell_type" in nido.obs.columns else None
    if ct_col:
        endo_mask = nido.obs[ct_col].astype(str).str.contains("Endo", case=False, na=False)
        snca = gene_vector(nido, "SNCA")
        if snca is not None and endo_mask.sum() > 10:
            snca_endo = snca[endo_mask.values]
            prot_endo = prot_scores[endo_mask.values]
            rho, p = stats.spearmanr(snca_endo, prot_endo)
            ax.scatter(snca_endo, prot_endo, s=6, alpha=0.3, color="firebrick")
            z = np.polyfit(snca_endo, prot_endo, 1)
            xs = np.linspace(snca_endo.min(), snca_endo.max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), color="black", linewidth=1.5)
            ax.set_title("B. SNCA vs Proteostasis, endothelial cells (Nido)")
            ax.text(0.97, 0.95, f"$\\rho$ = {rho:.3f}\np {format_pvalue(p)}\nn = {endo_mask.sum():,}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
            ax.set_xlabel("SNCA expression (log-normalised)")
            ax.set_ylabel("Proteostasis module score")
            print(f"  Panel B: rho={rho:.4f}, p{format_pvalue(p)} (n={endo_mask.sum()})")
    del nido
    import gc; gc.collect()

    # --- Panel C: SNCA-PDE10A correlation by cell type, Nido vs Ma ---
    print("\nLoading Ma for Panel C...")
    ma = sc.read_h5ad(paths["Ma"])
    ensure_lognorm_and_score(ma)
    ma_ct_col = "cell_type" if "cell_type" in ma.obs.columns else None

    results = []
    for cohort, adata, ctc in [("Nido", None, ct_col), ("Ma", ma, ma_ct_col)]:
        if cohort == "Nido":
            nido2 = sc.read_h5ad(paths["Nido"])
            ensure_lognorm_and_score(nido2)
            adata = nido2
        snca_v = gene_vector(adata, "SNCA")
        pde_v = gene_vector(adata, "PDE10A")
        if snca_v is None or pde_v is None or ctc is None:
            continue
        for ct in adata.obs[ctc].unique():
            mask = (adata.obs[ctc] == ct).values
            if mask.sum() < 30:
                continue
            rho, p = stats.spearmanr(snca_v[mask], pde_v[mask])
            results.append({"Cohort": cohort, "Cell_Type": str(ct), "rho": rho, "p": p, "n": mask.sum()})
        if cohort == "Nido":
            del nido2
            gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "snca_pde10a_by_celltype.csv", index=False)
    print("\nPanel C data (SNCA-PDE10A correlation by cell type, cell-level):")
    print(results_df.to_string(index=False))

    # --- Diagnostic: test whether Ma's published numbers used DONOR-LEVEL
    # pseudobulk instead of cell-level Spearman (the manuscript's stated
    # method). This is the leading hypothesis for why Ma's cell-level
    # numbers don't reproduce the published "reversal" claim.
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Ma donor-level pseudobulk SNCA-PDE10A correlation")
    print("(tests whether published Ma numbers used pseudobulk, not cell-level)")
    print("=" * 70)
    ma_sample_col = "sample" if "sample" in ma.obs.columns else None
    pseudobulk_results = []
    if ma_sample_col and ma_ct_col:
        snca_v = gene_vector(ma, "SNCA")
        pde_v = gene_vector(ma, "PDE10A")
        df = pd.DataFrame({
            "SNCA": snca_v, "PDE10A": pde_v,
            "sample": ma.obs[ma_sample_col].values,
            "cell_type": ma.obs[ma_ct_col].values,
        })
        for ct in df["cell_type"].unique():
            sub = df[df["cell_type"] == ct]
            pb = sub.groupby("sample")[["SNCA", "PDE10A"]].mean()
            if len(pb) < 5:
                continue
            rho, p = stats.spearmanr(pb["SNCA"], pb["PDE10A"])
            pseudobulk_results.append({"Cell_Type": str(ct), "rho_pseudobulk": rho,
                                        "p_pseudobulk": p, "n_donors": len(pb)})
        pb_df = pd.DataFrame(pseudobulk_results)
        pb_df.to_csv(OUT_DIR / "ma_snca_pde10a_pseudobulk.csv", index=False)
        print(pb_df.to_string(index=False))
        print("\nDonor-level pseudobulk results, for reference alongside the")
        print("cell-level correlations above (Panel C uses cell-level).")
    del ma
    gc.collect()

    ax = axes[2]
    if len(results_df):
        pivot = results_df.pivot_table(index="Cell_Type", columns="Cohort", values="rho")
        pivot = pivot.dropna(how="all").sort_values("Nido", na_position="last")
        pivot.plot(kind="barh", ax=ax, color={"Nido": "steelblue", "Ma": "firebrick"})
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("C. SNCA-PDE10A correlation by cell type")
        ax.set_xlabel("Spearman rho")
        ax.legend(title="Cohort")

    plt.tight_layout()
    out_path = OUT_DIR / "Supplementary_Fig_PDE10A_SNCA.pdf"
    png_path = OUT_DIR / "Supplementary_Fig_PDE10A_SNCA.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to {out_path} and {png_path}")
    print("Cell-level Spearman correlations only, no unverified comparison annotations.")
