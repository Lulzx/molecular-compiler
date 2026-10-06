import json

import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler import linear_response as lr
from molecular_compiler.phase0 import freeze_registration, load_registration
from molecular_compiler.phase0_worm import (
    amplitude_ceiling,
    cluster_bootstrap,
    jaxley_capability_audit,
    stage_a,
)
from molecular_compiler.worm_public import (
    canonical_class,
    cengen_unit,
    classify_gene,
    gene_family,
    load_atlas,
    normalize_name,
    read_beets_pairs,
    write_atlas,
)

CLASSES = [
    "ADA", "ASEL", "ASER", "AVA", "AWC_OFF", "AWC_ON", "CEP", "DA", "DA9", "DB",
    "DB01", "IL2_DV", "IL2_LR", "RMD_DV", "RMD_LR", "RME_DV", "RME_LR", "SAB",
    "VA", "VA12", "VB", "VB01", "VB02", "VC", "VC_4_5", "VD_DD", "AVL",
]  # fmt: skip


def test_M1_worm_neuron_names_map_to_cengen_units_and_canonical_classes():
    assert normalize_name("VA01") == "VA1" and normalize_name("AVAL") == "AVAL"
    cases = {
        "ADAL": ("ADA", "ADA"),
        "ASER": ("ASER", "ASE"),
        "AWCL": ("AWC_mean", "AWC"),
        "IL2VR": ("IL2_DV", "IL2"),
        "IL2L": ("IL2_LR", "IL2"),
        "RMDDL": ("RMD_DV", "RMD"),
        "RMEV": ("RME_DV", "RME"),
        "CEPDL": ("CEP", "CEP"),
        "SABD": ("SAB", "SAB"),
        "DA09": ("DA9", "DA"),
        "DB1": ("DB01", "DB"),
        "VB2": ("VB02", "VB"),
        "VC5": ("VC_4_5", "VC"),
        "VD3": ("VD_DD", "VD"),
        "DD1": ("VD_DD", "DD"),
        "AVL": ("AVL", "AVL"),
    }
    for neuron, (unit, canonical) in cases.items():
        assert cengen_unit(neuron, CLASSES) == unit
        assert canonical_class(neuron, CLASSES) == canonical
    with pytest.raises(ValueError):
        cengen_unit("XYZ", CLASSES)


def test_M1_R2_gene_classes_and_families():
    assert classify_gene("glc-3") == "receptor"
    assert classify_gene("twk-18") == "channel"
    assert classify_gene("flp-18") == "peptide"
    assert classify_gene("npr-1") == "gpcr"
    assert classify_gene("inx-1") == "innexin"
    assert classify_gene("kcc-2") == "transporter"
    assert classify_gene("act-1") is None
    assert gene_family("twk-18") == "twk" and gene_family("slo-1") == "slo"


def test_M1_beets_pairs_keep_most_potent_mature_peptide(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "GPCR ID,GPCR name,Peptide,log EC50 (M),95% Confidence Interval,EC50 (M)\n"
        "B0334.6-1,NPR-41-1,NLP-13-1,-6.9,x,1.0E-07\n"
        "B0334.6-1,NPR-41-1,pyroNLP-13-2,-7.9,x,1.0E-08\n"
        "C1.1-1,DMSR-1-1,FLP-1,-6,x,1.0E-06\n"
    )
    pairs = read_beets_pairs(path)
    assert [(p["peptide"], p["receptor"]) for p in pairs] == [
        ("flp-1", "dmsr-1"),
        ("nlp-13", "npr-41"),
    ]
    assert pairs[1]["ec50_nM"] == pytest.approx(10.0)
    assert pairs[1]["receptor_sequence_name"] == "B0334.6"


def test_M1_signal_propagation_atlas_round_trip(tmp_path):
    names = ["B", "A"]
    trials = [[np.array([1.0, 2.0]), np.array([])], [np.array([3.0]), np.array([4.0])]]
    strain = {
        "dff": np.array([[1.5, np.nan], [3.0, 4.0]]),
        "q": np.ones((2, 2)),
        "q_eq": np.ones((2, 2)),
        "occurrences": np.array([[2, 0], [1, 1]]),
        "trials": trials,
    }
    write_atlas(
        tmp_path / "a.npz", names, {"wt": strain, "unc31": strain}, {"A": 0, "B": 1}
    )
    loaded = load_atlas(tmp_path / "a.npz")
    assert loaded["neuron_ids"].tolist() == [1, 0]
    np.testing.assert_array_equal(loaded["wt"]["trials"][0][0], [1.0, 2.0])
    assert len(loaded["unc31"]["trials"][0][1]) == 0


def _inputs(n=6, genes=5, seed=0):
    rng = np.random.default_rng(seed)
    chemical = np.log1p(rng.integers(0, 3, (n, n)) * (rng.random((n, n)) < 0.4))
    gap = np.triu(rng.random((n, n)) < 0.3, 1).astype(float)
    expression = rng.random((n, genes)) * (rng.random((n, genes)) < 0.6)
    return lr.AtlasInputs(
        chemical=chemical,
        gap=gap + gap.T,
        expression=expression,
        tokens=rng.normal(size=(genes, 4)),
        releases=(rng.random((n, 3)) < 0.5).astype(float),
        receptor_ligand=np.array([0, 1, 2, -1, -1]),
        innexins=np.array([3]),
        pair_peptide=np.array([4]),
        pair_receptor=np.array([3]),
        pair_potency=np.array([1.0]),
        transmitter_sign=np.ones(n),
        identity=rng.normal(size=(n, 3)),
    )


def test_M4_R3_linear_response_masks_are_exact_and_peptides_are_optional():
    inputs = _inputs()
    model = lr.Model("c", "compositional", k=4, peptides=True)
    params = lr.init(model, inputs, seed=1)
    params = {**params, "chemical_bias": jnp.array(0.0), "gap_bias": jnp.array(-50.0)}
    silent = lr.AtlasInputs(
        **{**inputs.__dict__, "expression": np.zeros_like(inputs.expression)}
    )
    w = lr.coupling(model, params, silent)
    assert float(jnp.abs(w).max()) < 1e-12
    no_pep = lr.Model("b4", "compositional", k=4, peptides=False)
    off = {**params, "peptide_gain": jnp.array(-1e3)}
    np.testing.assert_allclose(
        lr.coupling(model, off, inputs), lr.coupling(no_pep, params, inputs), atol=1e-6
    )


def test_C_R2_linear_response_rule_parameters_independent_of_neuron_count():
    small, large = _inputs(n=6), _inputs(n=20)
    model = lr.Model("c", "compositional", k=4, peptides=True)
    assert lr.parameter_count(lr.init(model, small)) == lr.parameter_count(
        lr.init(model, large)
    )


def test_linear_response_scales_by_autoresponse_and_fits_synthetic_targets():
    inputs = _inputs(n=8)
    model = lr.Model("c", "compositional", k=4, peptides=True)
    truth = lr.init(model, inputs, seed=3)
    truth = {
        **truth,
        "receptor": truth["receptor"] * 50,
        "chemical_bias": jnp.array(0.3),
    }
    auto = jnp.linspace(0.5, 1.5, 8)
    target = np.asarray(lr.predict(model, truth, inputs, auto))
    np.testing.assert_allclose(np.diag(target), np.asarray(auto), rtol=1e-5)
    weights = 1 - np.eye(8)
    _, history = lr.fit(model, inputs, target, weights, auto, steps=300, seed=0)
    assert history[-1] < 0.5 * history[0]
    for family in ("connectome_only", "connectome_free", "dense_free", "black_box"):
        baseline = lr.Model(family, family, hidden=4)
        _, h = lr.fit(baseline, inputs, target, weights, auto, steps=50)
        assert np.isfinite(h[-1])


def test_phase0_stage_a_registration_keeps_classes_together_and_freezes(tmp_path):
    names = ["ADAL", "ADAR", "AVAL", "AVAR", "AVL", "DA1", "DA9", "VD1", "DD1"]
    manifest = {
        "neuron_names": names,
        "canonical_classes": {n: canonical_class(n, CLASSES) for n in names},
    }
    registration = stage_a(manifest)
    assert registration == stage_a(manifest)
    folds = registration["splits"]["leave_class_out"]["class_to_fold"]
    classes = registration["class_partition"]["neuron_to_class"]
    assert classes["DA1"] == classes["DA9"] == "DA"
    assert len({folds[classes[n]] for n in ("ADAL", "ADAR")}) == 1
    path = tmp_path / "stage-a.json"
    digest = freeze_registration(path, registration)
    assert load_registration(path)["processing_hash"] == digest
    with pytest.raises(FileExistsError):
        freeze_registration(path, registration)
    payload = json.loads(path.read_text())
    payload["registration"]["thresholds"]["K3_maximum_ambiguous_fraction"] = 0.9
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_registration(path)


def test_M12_cluster_bootstrap_and_split_half_ceiling():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 5)
    values = rng.normal(1.0, 0.1, 50)
    result = cluster_bootstrap(None, clusters, lambda i: values[i].mean())
    assert result["interval"][0] < 1.0 < result["interval"][1]
    assert cluster_bootstrap(None, np.zeros(5), lambda i: 1.0)["interval"] is None
    signal = rng.normal(size=(30, 30))
    trials = [
        [signal[i, j] + rng.normal(0, 0.1, 6) for j in range(30)] for i in range(30)
    ]
    ceiling, count = amplitude_ceiling(trials, np.ones((30, 30), dtype=bool))
    assert count == 900 and ceiling > 0.95


def test_S_R4_jaxley_interface_audit_selects_in_house_backend():
    audit = jaxley_capability_audit()
    assert audit["decision"] == "in_house"
    assert not any(c["passed"] for c in audit["conditions"].values())
