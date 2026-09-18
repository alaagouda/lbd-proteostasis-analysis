"""
Gene sets used for module scoring. The proteostasis module is the
core 14-gene set analysed in most of the pipeline; the other seven
were scored alongside it during exploratory work and are kept here
for the module comparison in module_scoring.py.
"""

PROTEOSTASIS = [
    "HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "DNAJB4", "HSP90AA1", "HSP90AB1",
    "DNAJA1", "DNAJA4", "CHORDC1", "STIP1", "CRYAB", "BAG3", "HSPH1",
]

# Subset pre-specified for pseudobulk DE and Stouffer meta-analysis
# (documented stress-inducibility, detected above threshold in all
# three cohorts)
HEAT_SHOCK_PRIMARY = [
    "HSPA1A", "HSPA1B", "HSPB1", "DNAJB1", "CRYAB", "HSP90AA1", "HSP90AB1",
]

MODULES = {
    "Proteostasis": PROTEOSTASIS,
    "Neuroinflammation": [
        "TNF", "IL1B", "IL6", "NFKB1", "CCL2", "CXCL10",
        "CD68", "AIF1", "TREM2", "C1QA", "C1QB", "C1QC", "TYROBP",
    ],
    "Oxidative_Stress": [
        "SOD1", "SOD2", "CAT", "GPX1", "GPX4", "PRDX1", "PRDX2", "NQO1", "HMOX1",
    ],
    "Mitochondrial": [
        "MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND4", "MT-ATP6",
    ],
    "ER_Stress": [
        "HSPA5", "ATF4", "ATF6", "DDIT3", "XBP1", "ERN1", "EIF2AK3", "PPP1R15A", "DNAJC3", "HYOU1",
    ],
    "Autophagy": [
        "MAP1LC3B", "SQSTM1", "ATG5", "ATG7", "BECN1", "ATG12", "LAMP1", "LAMP2", "TFEB",
    ],
    "Synaptic": [
        "SYP", "SYN1", "SYN2", "SNAP25", "STX1A", "VAMP2", "SYT1", "DLG4", "GRIN1", "GRIA1", "NLGN1",
    ],
    "Complement": [
        "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CFH", "CFB", "C5AR1", "CR1",
    ],
}

HYPOXIA = [
    "ADM", "BNIP3", "VEGFA", "SLC2A1", "LDHA", "PGK1", "PDK1", "ENO1", "HK2", "PFKFB3",
]

CELL_TYPE_MARKERS = {
    "Excitatory Neuron": ["SLC17A7", "CAMK2A", "SATB2", "NRGN"],
    "Inhibitory Neuron": ["GAD1", "GAD2", "SLC32A1", "PVALB"],
    "Astrocyte": ["AQP4", "GFAP", "SLC1A2", "GJA1"],
    "Microglia": ["C1QA", "CSF1R", "TMEM119", "P2RY12"],
    "Oligodendrocyte": ["MBP", "MOG", "PLP1"],
    "OPC": ["PDGFRA", "CSPG4"],
    "Endothelial": ["CLDN5", "FLT1"],
    "Pericyte": ["PDGFRB", "RGS5"],
}
