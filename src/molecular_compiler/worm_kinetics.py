"""M5 kinetics library for C. elegans from curated literature records.

Revision 8 (spec Section 10.2): kinetic priors are assigned by curated
functional family first; PLM nearest neighbors are used only within a family.
Every family record cites its measured source. Families without a measured
worm source are marked `measured: false` and get inflated covariance.
"""

import re

import numpy as np

# Functional families. Ligand-gated receptors are split by ligand and ion
# selectivity because selectivity sets the reversal potential, and therefore
# the sign (M6-R1, M6-R3).
FAMILY_PATTERNS = [
    ("Kv_shaker", r"shk-1|egl-36|kvs-\d+|shw-\d+"),
    ("Kv_shal", r"shl-1"),
    ("KCNQ", r"kqt-\d+"),
    ("ERG_EAG", r"unc-103|egl-2"),
    ("Kir", r"irk-\d+"),
    ("BK", r"slo-1"),
    ("Slack", r"slo-2"),
    ("SK", r"kcnl-\d+"),
    ("K2P", r"twk-\d+|sup-9|unc-58|egl-23"),
    ("CaV1", r"egl-19"),
    ("CaV2", r"unc-2"),
    ("CaV3", r"cca-1"),
    ("NALCN", r"nca-[12]"),
    ("CNG", r"cng-\d+|tax-[24]"),
    ("TRP", r"osm-9|ocr-\d+|trp-\d+|trpa-1|gon-2|gtl-\d+|pkd-2|lov-1"),
    (
        "DEG_ENaC",
        r"deg-1|mec-4|mec-10|unc-8|del-\d+|asic-\d+|egas-\d+|flr-1|acd-\d+|unc-105",
    ),
    ("ClC", r"clh-\d+"),
    ("bestrophin", r"best-\d+"),
    ("TMC", r"tmc-\d+"),
    ("Piezo", r"pezo-1"),
    (
        "nAChR_cation",
        r"acr-\d+|lev-[18]|unc-29|unc-38|unc-63|deg-3|des-2|eat-2",
    ),
    ("ACh_anion", r"acc-\d+|lgc-4[789]"),
    ("iGluR_cation", r"glr-\d+|nmr-\d+"),
    ("GluCl_anion", r"glc-\d+|avr-1[45]"),
    ("GABA_anion", r"unc-49|gab-1|ggr-\d+|lgc-3[678]|lgc-5[67]"),
    ("GABA_cation", r"exp-1|lgc-35"),
    ("amine_LGC", r"mod-1|lgc-\d+"),
    ("cation_chloride_cotransporter", r"kcc-\d+|nkcc-1"),
    ("transporter_other", r"abts-\d+|vglu-\d+|eat-4|unc-47|unc-17|cat-1|snf-\d+"),
    ("transporter_other", r"mod-5|dat-1|eat-6|glt-\d+|ncx-\d+"),
    ("innexin", r"inx-\d+|unc-7|unc-9|eat-5"),
]
_COMPILED = [(f, re.compile(p)) for f, p in FAMILY_PATTERNS]


def kinetic_family(name, molecule_class):
    """Curated family; GPCRs and unlisted genes fall back to their class."""
    for family, pattern in _COMPILED:
        if pattern.fullmatch(name):
            return family
    return molecule_class


# Curated family priors. Values were read from the cited sources during Phase 1
# curation (records with quotes in data/cache/kinetics_curation.json, local).
# The simulator's generic HH form has one activation gate (steady state
# sigmoid((V - v_half) / slope), time constant tau) and an optional single
# inactivation gate, so multi-gate published models are reduced to one m and
# one h gate with voltage-independent time constants (the published tau
# function evaluated near -30 mV, or the legend value). Ca2+ gating, U-shaped
# inactivation and multi-component gates are dropped; all are declared
# approximations. "measured" is True only when
# the gating or decay values come from a C. elegans measurement or a model
# fitted to one. gbar is the median over the neuron models in the source.
NICOLETTI_2019 = "Nicoletti et al. 2019 PLoS ONE 14:e0218738 (CC BY 4.0)"
NICOLETTI_2024 = "Nicoletti et al. 2024 PLoS ONE 19:e0298105 (CC BY 4.0)"
ALWAYS_OPEN = {"v_half_mV": -200.0, "slope_mV": 10.0, "tau_s": 0.001}
K, CA, CL = {"K": 1.0}, {"Ca": 1.0}, {"Cl": 1.0}
CATION = {"Na": 1.0, "K": 1.0}

CURATED_FAMILIES = {
    "Kv_shal": {
        "ion": K,
        "reversal": -80.0,
        "gbar": 1.7,
        "v_half": -6.8,
        "slope": 14.1,
        "tau": 0.00073,
        "h_v_half": -33.1,
        "h_slope": 8.3,
        "h_tau": 0.088,
        "m_power": 3,
        "measured": True,
        "source": NICOLETTI_2019 + " SHL-1",
    },
    "Kv_shaker": {
        "ion": K,
        "reversal": -80.0,
        "gbar": 0.4,
        "v_half": 2.0,
        "slope": 10.0,
        "tau": 0.0149,
        "h_v_half": -6.95,
        "h_slope": 5.8,
        "h_tau": 1.4,
        "measured": True,
        "source": NICOLETTI_2024 + " SHK-1",
    },
    "KCNQ": {
        "ion": K,
        "reversal": -80.0,
        "gbar": 0.38,
        "v_half": 7.7,
        "slope": 15.8,
        "tau": 0.107,
        "measured": True,
        "source": NICOLETTI_2019 + " KQT-3",
    },
    "ERG_EAG": {
        "ion": K,
        "reversal": -80.0,
        "gbar": 0.15,
        "v_half": -6.9,
        "slope": 14.9,
        "tau": 1.0,
        "measured": True,
        "source": NICOLETTI_2024 + " EGL-2",
    },
    "Kir": {
        "ion": K,
        "reversal": -80.0,
        "gbar": 0.295,
        "v_half": -82.0,
        "slope": -13.0,
        "tau": 0.008,
        "measured": True,
        "source": NICOLETTI_2024 + " IRK",
    },
    "BK": {
        "ion": K,
        "reversal": -80.0,
        "gbar": 0.3,
        "v_half": -20.0,
        "slope": 10.0,
        "tau": 0.008,
        "measured": False,
        "source": NICOLETTI_2019 + " SLO-1 gbar and tau; Ca2+ gating replaced"
        " by a placeholder voltage gate",
    },
    "Slack": {
        "ion": K,
        "reversal": -80.0,
        "gbar": 1.0,
        "v_half": -20.0,
        "slope": 10.0,
        "tau": 0.008,
        "measured": False,
        "source": NICOLETTI_2024 + " SLO-2 gbar; placeholder voltage gate",
    },
    "SK": dict(
        ion=K,
        reversal=-80.0,
        gbar=0.06,
        **{**ALWAYS_OPEN, "tau_s": 0.0063},
        measured=False,
        source=NICOLETTI_2019 + " KCNL gbar and tau; Ca2+ gating omitted",
    ),
    "K2P": dict(
        ion=K,
        reversal=-80.0,
        gbar=0.1,
        **ALWAYS_OPEN,
        measured=False,
        source="unmeasured: K+ leak placeholder",
    ),
    "CaV1": {
        "ion": CA,
        "reversal": 60.0,
        "gbar": 0.1,
        "v_half": -4.4,
        "slope": 7.5,
        "tau": 0.006,
        "h_v_half": 14.9,
        "h_slope": 12.0,
        "h_tau": 0.0446,
        "measured": True,
        "source": NICOLETTI_2019 + " EGL-19",
    },
    "CaV2": {
        "ion": CA,
        "reversal": 60.0,
        "gbar": 0.615,
        "v_half": -37.2,
        "slope": 4.0,
        "tau": 0.0024,
        "h_v_half": -77.5,
        "h_slope": 5.6,
        "h_tau": 0.12,
        "measured": True,
        "source": NICOLETTI_2019 + " UNC-2",
    },
    "CaV3": {
        "ion": CA,
        "reversal": 60.0,
        "gbar": 0.785,
        "v_half": -57.7,
        "slope": 2.4,
        "tau": 0.019,
        "h_v_half": -73.0,
        "h_slope": 8.1,
        "h_tau": 0.02,
        "m_power": 2,
        "measured": True,
        "source": NICOLETTI_2019 + " CCA-1",
    },
    "NALCN": dict(
        ion={"Na": 1.0},
        reversal=30.0,
        gbar=0.05,
        **ALWAYS_OPEN,
        measured=True,
        source=NICOLETTI_2024 + " NCA leak",
    ),
    "nAChR_cation": {
        "ion": CATION,
        "reversal": 0.0,
        "tau": 0.0065,
        "measured": True,
        "source": "NMJ ACh PSC decay 5.6-7.4 ms (receptor curation);"
        " reversal assumed 0 mV",
    },
    "ACh_anion": {
        "ion": CL,
        "tau": 0.02,
        "measured": False,
        "source": "unmeasured decay; Cl- selectivity from family",
    },
    "iGluR_cation": {
        "ion": CATION,
        "reversal": 0.0,
        "tau": 0.004,
        "measured": True,
        "source": "Wang et al. 2012 Neuron, GLR-1 desensitization"
        " ~4 ms in AVA; reversal assumed 0 mV",
    },
    "GluCl_anion": {
        "ion": CL,
        "tau": 0.02,
        "measured": False,
        "source": "unmeasured decay; Cl- selectivity (Horoszok 2001)",
    },
    "GABA_anion": {
        "ion": CL,
        "tau": 0.0248,
        "measured": True,
        "source": "NMJ GABA IPSC decay 24.8 ms (receptor curation)",
    },
    "GABA_cation": {
        "ion": CATION,
        "reversal": 0.0,
        "tau": 0.02,
        "measured": False,
        "source": "EXP-1/LGC-35 cation selectivity; decay unmeasured",
    },
    "amine_LGC": {
        "ion": CL,
        "tau": 0.132,
        "measured": False,
        "source": "MOD-1 desensitization 132 ms used as decay"
        " (Hardege et al. 2022); family selectivity mixed",
    },
    "receptor": {
        "ion": CL,
        "tau": 0.02,
        "measured": False,
        "source": "unmeasured receptor placeholder",
    },
    "gpcr": {
        "tau": 1.0,
        "measured": False,
        "source": "unmeasured: no C. elegans GPCR timescale found",
    },
    "innexin": {"measured": False, "source": "gap conductance comes from M4 rules"},
    "cation_chloride_cotransporter": {
        "chloride": 5.0,
        "measured": False,
        "source": "no measured neuronal [Cl-]i; lower bound of policy prior",
    },
    "transporter_other": {
        "chloride": 10.0,
        "measured": False,
        "source": "neutral: equals the basal [Cl-]i, so no ionic effect",
    },
}
for _family in ("CNG", "TRP", "DEG_ENaC", "TMC", "Piezo"):
    CURATED_FAMILIES[_family] = dict(
        ion=CATION,
        reversal=0.0,
        gbar=0.01,
        **ALWAYS_OPEN,
        measured=False,
        source="unmeasured sensory/mechanosensory cation channel: small leak",
    )
for _family in ("ClC", "bestrophin"):
    CURATED_FAMILIES[_family] = dict(
        ion=CL,
        gbar=0.01,
        **ALWAYS_OPEN,
        measured=False,
        source="unmeasured Cl- channel: small leak",
    )
CURATED_FAMILIES["channel"] = dict(
    ion=K,
    reversal=-80.0,
    gbar=0.01,
    **ALWAYS_OPEN,
    measured=False,
    source="unmeasured channel placeholder",
)

# Neuron passive properties: medians over the Nicoletti 2019/2024 neuron models
# (Cm: AIY, AVAL, AVAR, RIM, VA5, VB6, VD5, AWCon; leak: the same plus RMD).
PASSIVE = {"capacitance_pF": 4.68, "leak_nS": 0.15, "leak_reversal_mV": -70.0}
UNMEASURED_COVARIANCE_SCALE = 16.0


def _record(gene, family, spec):
    from .kinetics import KineticRecord

    cls = gene["molecule_class"]
    form = {"receptor": "ligand_gated", "gpcr": "gpcr", "transporter": "transporter"}
    form = form.get(cls, "HH")
    names, mean, scale = [], [], []
    if cls == "channel":
        names = ["gbar_nS", "v_half_mV", "slope_mV", "tau_s"]
        mean = [
            spec["gbar"],
            spec.get("v_half", spec.get("v_half_mV")),
            spec.get("slope", spec.get("slope_mV")),
            spec.get("tau", spec.get("tau_s")),
        ]
        scale = [0.5 * mean[0], 5.0, 2.0, 0.5 * mean[3]]
    elif cls in ("receptor", "gpcr"):
        names, mean, scale = ["tau_s"], [spec["tau"]], [0.5 * spec["tau"]]
    elif cls == "transporter":
        names, mean, scale = ["Cl_equilibrium_mM"], [spec["chloride"]], [2.0]
    inflate = 1.0 if spec["measured"] else UNMEASURED_COVARIANCE_SCALE
    if not names:  # innexins: no kinetic parameters; unnamed unit placeholder
        mean, scale = [1.0], [1.0]
    cov = np.diag(np.square(scale) * inflate)
    return KineticRecord(
        gene["gene_id"],
        form,
        tuple(mean),
        tuple(map(tuple, cov)),
        dict(spec.get("ion", {})) if cls in ("channel", "receptor") else {},
        f"{'measured' if spec['measured'] else 'UNMEASURED'}: {spec['source']}",
        family=family,
        embedding=tuple(np.asarray(gene["plm_embedding"], float)),
        gbar_nS=spec.get("gbar", 1.0 if cls == "receptor" else 0.0),
        reversal_mV=spec.get("reversal"),
        tau_s=mean[names.index("tau_s")] if "tau_s" in names else 0.01,
        v_half_mV=mean[1] if cls == "channel" else -35,
        slope_mV=mean[2] if cls == "channel" else 10,
        parameter_names=tuple(names),
        transport_equilibrium_mM={"Cl": spec["chloride"]} if "chloride" in spec else {},
        h_v_half_mV=spec.get("h_v_half"),
        h_slope_mV=spec.get("h_slope", 5.0),
        h_tau_s=spec.get("h_tau", 0.1),
        m_power=spec.get("m_power", 1),
    )


def build_worm_library(graph, gene_names, metadata):
    """One curated record per modeled gene, keyed by its gene ID (M5)."""
    from .kinetics import KineticsLibrary

    records, families = {}, {}
    for gene in graph.molecular.genes.to_pylist():
        if gene["molecule_class"] in ("peptide", "effector", "other"):
            continue
        family = kinetic_family(gene_names[gene["gene_id"]], gene["molecule_class"])
        if family not in CURATED_FAMILIES:
            family = gene["molecule_class"]
        families[gene["gene_id"]] = family
        records[gene["gene_id"]] = _record(gene, family, CURATED_FAMILIES[family])
    return KineticsLibrary(records, metadata=metadata, gene_families=families)


def library_summary(library):
    rows = {}
    for record in library.records.values():
        row = rows.setdefault(record.family, {"genes": 0, "measured": None})
        row["genes"] += 1
        row["measured"] = record.source.startswith("measured")
        row["source"] = record.source
    return rows
