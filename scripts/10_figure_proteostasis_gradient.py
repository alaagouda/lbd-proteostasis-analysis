"""
Regenerates Figure 2C: the proteostasis module Cohen's d gradient
across the three cohorts.

Effect sizes are computed by the mean-of-ratios donor-level pseudobulk
method: each cell's CPM is computed first, then averaged per donor,
and Cohen's d is calculated on the resulting donor-level means. Sample
sizes are read directly from each cohort's h5ad .obs metadata.

Run this locally -- it needs only matplotlib, nothing else.
"""
import matplotlib.pyplot as plt
import numpy as np

COHORTS = ["Jin\n(temporal)", "Ma\n(cingulate)", "Nido\n(prefrontal)"]
COHENS_D = [1.568, 0.357, 0.114]
N_NUCLEI = [41899, 64370, 277831]
N_DONORS = ["n=6 (3+3)", "n=17 (9+8)", "n=33 (20+13)"]

EFFECT_SIZE_BANDS = {
    "Large (d\u22650.8)": 0.8,
    "Medium (d\u22650.5)": 0.5,
}

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    x = np.arange(len(COHORTS))

    for label, threshold in EFFECT_SIZE_BANDS.items():
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(2.62, threshold + 0.02, label, fontsize=8, color="gray", ha="right")

    ax.plot(x, COHENS_D, "o-", color="steelblue", markersize=10, linewidth=2, zorder=3)
    label_offsets = [-36, -52, -36]  # Ma's label pushed further down to clear the "Medium" reference line
    for i, (d, n, donors) in enumerate(zip(COHENS_D, N_NUCLEI, N_DONORS)):
        ax.annotate(f"d = {d:.3f}", (x[i], d), textcoords="offset points",
                     xytext=(0, 16), ha="center", fontsize=11, fontweight="bold",
                     color="steelblue")
        ax.annotate(f"{donors}\n{n:,} nuclei", (x[i], d), textcoords="offset points",
                     xytext=(0, label_offsets[i]), ha="center", fontsize=8, color="dimgray",
                     zorder=4, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(COHORTS)
    # Padding on both sides so the first/last points and their text labels
    # never sit flush against the axis lines or tick labels.
    ax.set_xlim(-0.45, len(COHORTS) - 1 + 0.45)
    ax.set_xlabel("Brain region (ordered by Lewy body burden, high \u2192 low)")
    ax.set_ylabel("Cohen's d (proteostasis module vs control, pseudobulk)")
    # Extra headroom above the highest point so its "d = ..." label and the
    # top y-tick (2.0) never overlap.
    ax.set_ylim(-0.65, 2.35)
    ax.set_title("C", loc="left", fontweight="bold", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("Figure2C_REGENERATED.png", dpi=300, bbox_inches="tight")
    plt.savefig("Figure2C_REGENERATED.pdf", bbox_inches="tight")
    print("Saved Figure2C_REGENERATED.png and .pdf")
