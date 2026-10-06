import numpy as np
import pytest

from molecular_compiler import ObservationModel, Stimulus, compile, observe, simulate
from molecular_compiler.config import load_config
from molecular_compiler.embeddings import embed_genes, pooled_embedding
from molecular_compiler.storage import (
    export_nwb,
    import_nwb,
    load_checkpoint,
    read_recording,
    save_checkpoint,
    write_recording,
)


def test_checkpoint_roundtrip_and_license(system, tmp_path):
    graph, rules, _ = system
    path = tmp_path / "model.safetensors"
    save_checkpoint(rules, path, graph.metadata)
    restored, metadata = load_checkpoint(path)
    assert restored.checkpoint_hash() == rules.checkpoint_hash()
    assert metadata["processing_hash"] == graph.metadata["processing_hash"]
    with pytest.raises(ValueError, match="open-tier"):
        save_checkpoint(
            rules, path, {**graph.metadata, "tier": "restricted"}, publish=True
        )


def test_Zarr_recording_roundtrip(system, tmp_path):
    recording = observe(
        simulate(compile(*system), Stimulus(), 0.002), ObservationModel()
    )
    write_recording(recording, tmp_path / "recording.zarr")
    restored = read_recording(tmp_path / "recording.zarr")
    np.testing.assert_array_equal(recording.y, restored.y)
    assert restored.metadata["tier"] == "open"


def test_NWB_recording_roundtrip(system, tmp_path):
    pytest.importorskip("pynwb")
    recording = observe(
        simulate(compile(*system), Stimulus(), 0.002), ObservationModel()
    )
    path = tmp_path / "recording.nwb"
    export_nwb(recording, path)
    restored = import_nwb(path)
    np.testing.assert_allclose(recording.y, restored.y)
    assert restored.indicator == recording.indicator


def test_ESM_overlapping_residue_pooling_and_isoforms():
    sequence = "A" * 1400 + "B" * 1000

    def fake_embedder(fragment):
        return np.broadcast_to(
            np.array([1 if c == "A" else 2 for c in fragment])[:, None],
            (len(fragment), 1280),
        )

    embedding = pooled_embedding(sequence, fake_embedder)
    expected = (1400 + 2 * 1000) / 2400
    np.testing.assert_allclose(embedding, expected, rtol=1e-3)
    assert embedding.dtype == np.float16
    genes, metadata = embed_genes(
        [{"gene_id": "g", "protein_seq": "AA", "isoforms": ["BB"]}],
        fake_embedder,
        "0" * 64,
    )
    assert len(genes["g"]["isoforms"]) == 1
    assert metadata["layer"] == 33


def test_yaml_schema_rejects_typos(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("species: C. elegans\nseed: 0\nduration_s: 1\nresoluton: {}\n")
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        load_config(path)


def test_canonical_project_roundtrip(system, tmp_path):
    from molecular_compiler.project import export_project, load_project

    graph, _, kinetics = system
    export_project(graph, kinetics, tmp_path)
    recovered, library = load_project(tmp_path)
    assert recovered.gene_ids == graph.gene_ids
    assert library.metadata["tier"] == "open"
    np.testing.assert_allclose(recovered.abundance, graph.abundance)


def test_compact_checkpoint_preserves_frozen_basis(system, tmp_path):
    from molecular_compiler import RuleNetwork

    rules = RuleNetwork.initialize_compact(rank=4, gauge_audited=True, d_z=8)
    path = tmp_path / "compact.safetensors"
    save_checkpoint(rules, path, system[0].metadata)
    recovered, _ = load_checkpoint(path)
    assert recovered.parameter_count == 2
    assert recovered.checkpoint_hash() == rules.checkpoint_hash()


def test_simgraph_export_is_standard_json_with_checked_arrays(system, tmp_path):
    import json

    from molecular_compiler.provenance import content_hash
    from molecular_compiler.storage import save_simgraph

    save_simgraph(compile(*system), tmp_path)

    def reject_constant(value):
        raise ValueError(value)

    metadata = json.loads(
        (tmp_path / "metadata.json").read_text(), parse_constant=reject_constant
    )
    with np.load(tmp_path / "arrays.npz") as arrays:
        assert metadata["array_hash"] == content_hash(dict(arrays))
    assert metadata["surrogates"][0]["lower"][0] == "-inf"
