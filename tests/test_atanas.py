import json

import numpy as np
import pytest

from molecular_compiler import atanas as A

h5py = pytest.importorskip("h5py")

CANON = {"AVAL", "AVAR", "RIML", "AIBL", "SMDDL"}


def _write(path, T=10, K=6, transpose=False, behavior=True, nan=False):
    rng = np.random.default_rng(0)
    original = rng.normal(100, 5, (T, K))
    if nan:
        original[3, 1] = np.nan
    with h5py.File(path, "w") as f:
        f["timing/timestamp_confocal"] = np.arange(T) * 0.6
        f["gcamp/trace_array_original"] = original.T if transpose else original
        f["gcamp/trace_array"] = np.zeros_like(original.T if transpose else original)
        if behavior:
            for i, name in enumerate(A.BEHAVIOR):
                f[f"behavior/{name}"] = np.full(T, float(i))
    return original


LABELS = {
    1: ("RIML", 4.0),
    2: ("AVAL", 5.0),
    3: ("AVAR", 2.0),  # low rating
    4: ("RIM?", 5.0),  # not canonical
    5: ("AIBL", 3.0),
    6: ("AIBL", 5.0),  # duplicate with ROI 5
}


def test_I2_loader_reads_original_trace_with_one_based_rois(tmp_path):
    path = tmp_path / "a.h5"
    original = _write(path)
    rec = A.load_recording(path, LABELS, "A1", CANON)
    assert rec.neurons == ["AVAL", "RIML"]  # sorted; ROI 2 and ROI 1
    np.testing.assert_array_equal(rec.roi, [2, 1])
    np.testing.assert_allclose(rec.traces, original[:, [1, 0]])  # not z-scored
    np.testing.assert_allclose(rec.ratings, [5.0, 4.0])
    assert set(rec.behavior) == set(A.BEHAVIOR) and rec.t.shape == (10,)
    reasons = {d[0]: d[4] for d in rec.decisions}
    assert reasons[3] == "below ordinal rating threshold"
    assert reasons[4] == "not an exact canonical neuron"
    assert reasons[5] == reasons[6] == "duplicate canonical identity"


def test_I2_loader_handles_transposed_traces_nan_and_missing_behavior(tmp_path):
    path = tmp_path / "b.h5"
    original = _write(path, transpose=True, behavior=False, nan=True)
    rec = A.load_recording(path, {1: ("RIML", 5.0), 2: ("AVAL", 5.0)}, canonical=CANON)
    assert rec.behavior == {}
    np.testing.assert_allclose(rec.traces, original[:, [1, 0]])
    assert np.isnan(rec.traces[3, 0])


def test_I2_loader_rejects_ambiguous_shape_and_bad_rois(tmp_path):
    path = tmp_path / "c.h5"
    _write(path, T=6, K=6)
    with pytest.raises(ValueError, match="unambiguous"):
        A.load_recording(path, LABELS)
    path = tmp_path / "d.h5"
    _write(path)
    with pytest.raises(ValueError, match="outside source range"):
        A.load_recording(path, {0: ("AVAL", 5.0)})
    with pytest.raises(ValueError, match="outside source range"):
        A.load_recording(path, {7: ("AVAL", 5.0)})


def test_I2_manifest_checksum_and_baseline(tmp_path):
    _write(tmp_path / "a.h5")
    digest = A.sha256_file(tmp_path / "a.h5")
    entry = {"label": "RIML", "confidence": 5}
    labels = {"data": {"A1": {"idx_neuron-label": {"1": entry}}}}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    manifest = {"animals": [{"file": "a.h5", "animal_id": "A1", "sha256": digest}]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    args = (tmp_path, tmp_path / "m.json", tmp_path / "labels.json", CANON)
    (rec,) = A.load_baseline(*args)
    assert rec.animal == "A1" and rec.neurons == ["RIML"] and rec.sha256 == digest
    manifest["animals"][0]["sha256"] = "0" * 64
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        A.load_baseline(*args)
    manifest["animals"][0].update(file="../a.h5", sha256=digest)
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonlocal"):
        A.read_manifest(tmp_path / "m.json")
