from dataclasses import replace

import numpy as np
import pytest

from molecular_compiler.data import (
    MeasurementModel,
    prepare,
    read_table,
    table,
    validate_table,
    write_table,
)
from molecular_compiler.provenance import DataRecord, DataRegister, require_publishable


def test_M1_R1_invalid_units_and_confidence(system):
    graph, _, _ = system
    bad = graph.connectome.neurons.replace_schema_metadata({b"units": b"{}"})
    with pytest.raises(ValueError, match="units"):
        validate_table("neurons", bad)
    rows = graph.connectome.neurons.to_pylist()
    rows[0]["segmentation_confidence"] = 2
    with pytest.raises(ValueError, match="outside"):
        table("neurons", rows)


def test_M1_R1_foreign_keys(system):
    graph, _, _ = system
    rows = graph.connectome.synapses.to_pylist()
    rows[0]["pre_id"] = 999
    connectome = replace(graph.connectome, synapses=table("synapses", rows))
    with pytest.raises(ValueError, match="unknown neuron"):
        prepare(connectome, graph.molecular)


def test_M1_R2_unmapped_genes_retained(system):
    graph, _, _ = system
    assert all(r["ortholog_group"] is None for r in graph.molecular.genes.to_pylist())
    assert len(graph.gene_ids) == graph.abundance.shape[1] == 7


def test_M1_R4_parquet_provenance_roundtrip(system, tmp_path):
    graph, _, _ = system
    path = tmp_path / "neurons.parquet"
    write_table("neurons", graph.connectome.neurons, path, graph.metadata)
    recovered = read_table("neurons", path)
    assert recovered.to_pylist() == graph.connectome.neurons.to_pylist()
    assert b"provenance" in recovered.schema.metadata


def test_M1_R5_unregistered_and_excluded(system):
    graph, _, _ = system
    with pytest.raises(ValueError, match="unregistered"):
        prepare(replace(graph.connectome, register=DataRegister()), graph.molecular)
    excluded = DataRegister(
        [
            DataRecord(
                "synthetic-connectome",
                "C. elegans",
                "fixture",
                "synthetic",
                "1",
                "unknown",
                "excluded",
            )
        ]
    )
    with pytest.raises(ValueError, match="excluded"):
        prepare(replace(graph.connectome, register=excluded), graph.molecular)


def test_M1_R5_restricted_inheritance(system):
    graph, _, _ = system
    record = replace(
        graph.connectome.register.require("synthetic-connectome"), tier="restricted"
    )
    updated = prepare(
        replace(graph.connectome, register=DataRegister([record])), graph.molecular
    )
    assert updated.metadata["tier"] == "restricted"
    with pytest.raises(ValueError, match="open-tier"):
        require_publishable(updated.metadata)


def test_M1_R6_c302_differences_require_explanation(system):
    graph, _, _ = system
    reference = {
        "neuron_ids": graph.neuron_ids.tolist(),
        "chemical_edges": 999,
        "gap_contacts": 1,
    }
    connectome = replace(graph.connectome, c302_reference=reference)
    with pytest.raises(ValueError, match="explanation"):
        prepare(connectome, graph.molecular)
    result = prepare(
        replace(connectome, c302_explanation="fixture differs deliberately"),
        graph.molecular,
    )
    assert result.metadata["c302_crosscheck"]["differences"]["chemical_edges"]


def test_M2_R1_R2_known_labels_and_identity_samples(system):
    graph, _, _ = system
    assert np.all(np.asarray(graph.assignments).max(axis=1) == 1)
    assert np.allclose(graph.metadata["identity_entropy"], 0)
    assert graph.identity_samples().shape == (8, 4, 7)
    np.testing.assert_array_equal(graph.sample_abundance(1), graph.sample_abundance(2))


def test_M2_unknown_labels_transport(system):
    graph, _, _ = system
    rows = graph.connectome.neurons.to_pylist()
    rows[0]["type_label"] = None
    prepared = prepare(
        replace(graph.connectome, neurons=table("neurons", rows)), graph.molecular
    )
    np.testing.assert_allclose(prepared.assignments.sum(axis=1), 1)
    assert np.all(np.asarray(prepared.assignments) >= 0)


def test_M3_R1_identity_with_inflated_uncertainty():
    expression = np.array([[0.0, 1.0], [2.0, 3.0]])
    model = MeasurementModel.fit(expression)
    abundance, variance = model.predict(expression)
    np.testing.assert_allclose(abundance, expression)
    assert np.all(variance >= 1)


def test_M3_paired_regression():
    x = np.arange(6.0).reshape(3, 2)
    model = MeasurementModel.fit(x, 2 * x + 1, classes=["channel", "channel"])
    np.testing.assert_allclose(model.predict(x)[0], 2 * x + 1, atol=1e-6)


def test_duplicate_ids_rejected(system):
    graph, _, _ = system
    rows = graph.connectome.neurons.to_pylist()
    rows[1]["neuron_id"] = rows[0]["neuron_id"]
    with pytest.raises(ValueError, match="duplicate"):
        table("neurons", rows)
