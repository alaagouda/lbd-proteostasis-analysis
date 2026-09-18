"""
Regenerates Figure 7 (Augur cell-type prioritisation) from raw-count
AUROC values across all three cohorts.
"""
import matplotlib.pyplot as plt

# ---- Augur AUROC values (raw counts), per cohort ----
JIN = {
    "Microglia": 0.692721, "Inhib. Neuron": 0.645522, "Astrocyte": 0.644433,
    "Excit. Neuron": 0.642517, "OPC": 0.627766, "Oligodendrocyte": 0.621304,
    "Endothelial": 0.573968,
}
MA = {
    "Oligodendrocytes": 0.847540, "Endothelial": 0.783946, "Pericytes": 0.607721,
    "Unassigned": 0.606995, "Microglia": 0.581531, "Excitatory_Neurons": 0.568980,
    "Inhibitory_Neurons": 0.568662, "Astrocytes": 0.556916, "OPCs": 0.555726,
}
NIDO = {
    "L6 OPRK1 SMYD1 SNTB1": 0.595136, "L6 FEZF2 SYT6 PTPRT": 0.584705,
    "Oligo MOG OPALIN": 0.577290, "OPC PDGFRA PCDH15": 0.561803,
    "L6 OPRK1 THEMIS RGS6": 0.559943, "Endo CLDN5 SLC7A5": 0.543866,
    "L3-5 RORB GABRG1 KCNH7": 0.539331, "L3-5 RORB MKX GRIN3A": 0.535748,
    "InN SST FREM1": 0.534626, "InN PVALB PDE3A": 0.529796,
    "L2-3 CUX2 ACVR1C THSD7A": 0.529172, "InN PVALB HPSE": 0.516122,
    "Micro P2RY12 APBB1IP": 0.513526, "Astro AQP4 SLC1A2": 0.511190,
    "InN VIP HS3ST3B1": 0.507596, "InN LAMP5 BMP7": 0.499172,
    "L5-6 FEZF2 NXPH2 CDH8": 0.475748,
}

PANELS = [
    ("A", "Jin \u2014 Temporal (DLB)", JIN, "#c66767"),
    ("B", "Ma \u2014 Cingulate (sPD)", MA, "#d99a5b"),
    ("C", "Nido \u2014 Prefrontal (sPD)", NIDO, "#6f8fc7"),
]

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), gridspec_kw={"width_ratios": [1, 1, 1.3]})

    for ax, (label, title, data, color) in zip(axes, PANELS):
        items = sorted(data.items(), key=lambda kv: kv[1])  # ascending, so highest ends up at top
        names = [k for k, v in items]
        values = [v for k, v in items]

        ax.barh(names, values, color=color, edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0.45, 0.90)
        ax.set_xlabel("AUC")
        ax.set_title(title, fontsize=13, fontweight="bold", color=color)
        ax.text(-0.02, 1.05, label, transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.02,
              "Augur cell-type prioritisation, raw counts (corrected). AUC=0.5: random chance. "
              "Higher AUC = gene expression more reliably distinguishes disease from control in that cell type.",
              ha="center", fontsize=10, style="italic", color="dimgray")

    plt.tight_layout()
    plt.savefig("Figure7_REGENERATED.png", dpi=300, bbox_inches="tight")
    plt.savefig("Figure7_REGENERATED.pdf", bbox_inches="tight")
    print("Saved Figure7_REGENERATED.png and .pdf")
    print("\nAll three cohorts sit near/above chance in a coherent gradient")
    print("(Ma highest, Jin intermediate, Nido lowest).")
