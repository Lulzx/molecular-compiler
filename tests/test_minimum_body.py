import json
import sys

import numpy as np
import pytest

from molecular_compiler import compile
from molecular_compiler import minimum_body as MB
from molecular_compiler.body_ladder import RUNGS
from molecular_compiler.encoder import Readout
from molecular_compiler.fixtures import synthetic_system

h5py = pytest.importorskip("h5py")

NAMES = ["DVA", "SMDDL", "DB1", "VB1"]
DT_REC = 0.001  # two simulator steps (dt_sim = 0.0005)
T = 24
OPTIONS = {"warmup_s": 0.002, "l1_iterations": 1, "l1_fit_s": 0.016}


def fluorescence(calcium):
    c = np.asarray(calcium)
    return 0.1 + c / (1e-4 + c)


@pytest.fixture(scope="module")
def sim():
    return compile(*synthetic_system())


def write_animals(directory, count=4, nan=True):
    """Atanas-format files: ROIs 1-4 are the simulator neurons, ROI 5 is not in it."""
    labels = {"data": {}}
    for a in range(count):
        rng = np.random.default_rng(a)
        traces = (
            100 + np.cumsum(rng.normal(0, 1, (T, 5)), axis=0) + rng.normal(0, 2, (T, 5))
        )
        if nan and a == 0:
            traces[5, 1] = np.nan
        with h5py.File(directory / f"worm{a}.h5", "w") as f:
            f["timing/timestamp_confocal"] = 10.0 + np.arange(T) * DT_REC
            f["gcamp/trace_array_original"] = traces
            for name in ("velocity", "head_angle", "angular_velocity", "pumping"):
                if name == "pumping" and a == 1:
                    continue  # one animal lacks a channel
                f[f"behavior/{name}"] = rng.normal(0, 1, T)
        names = [*NAMES, "AVAL"]
        labels["data"][f"worm{a}"] = {
            "idx_neuron-label": {
                str(i + 1): {"label": n, "confidence": 5.0} for i, n in enumerate(names)
            }
        }
    path = directory / "labels.json"
    path.write_text(json.dumps(labels))
    return path


@pytest.fixture
def recordings(tmp_path):
    labels = write_animals(tmp_path)
    return MB.load_directory(tmp_path, labels, canonical=set(NAMES) | {"AVAL"})


def run(sim, recordings, **kwargs):
    return MB.track_i2(sim, NAMES, recordings, fluorescence, **{**OPTIONS, **kwargs})


def test_I2_alignment_uses_common_neurons_and_one_grid(sim, recordings):
    common, index, animals = MB.align_recordings(recordings, NAMES, 2 * 0.0005)
    assert common == NAMES and list(index) == [0, 1, 2, 3]
    assert all(a.fluorescence.shape == (4, a.n_grid) for a in animals)
    assert np.isfinite(animals[0].fluorescence).all()  # NaN gap interpolated
    assert animals[1].missing_channels == ("pumping",)
    dff = MB.to_dff(animals[0].fluorescence, 10)
    assert dff.shape == (4, 10) and (dff.min(axis=1) < 0.02).all()  # F0 in window
    with pytest.raises(ValueError, match="canonical neurons"):
        MB.align_recordings(recordings, ["X1", "X2", "X3", "X4"], 0.001)


def test_I2_report_structure_and_M13_R2_completeness(sim, recordings):
    report = run(sim, recordings, horizons=(0.012, "full"))
    json.dumps(report)
    assert report["data_tier"] == "restricted" and report["stage"] == "I2"
    assert report["part_ii"]["status"] == "not_evaluable"
    assert "Revision 13" in report["part_ii"]["reason"]
    assert report["config"]["n_animals"] == 4 and report["l1_fit"]["iterations"] == 1
    assert report["declared_approximations"]
    assert set(report["animals"]) == {f"worm{i}" for i in range(4)}
    for entry in report["animals"].values():
        assert set(entry["horizons"]) == {"0.012", "full"}
        for h in entry["horizons"].values():
            assert h["status"] == "evaluated"
            records = h["records"]
            # M13-R4: every rung is reported, in ladder order, with R2 fields.
            assert [r["rung"] for r in records] == list(RUNGS)
            for r in records:
                if r["rung"] == "L0":
                    assert r["boundary_condition"] == "open_loop"
                if r["rung"] in ("L3", "L4", "L5"):
                    assert r["boundary_condition"] == "closed_loop"
                assert r["horizon_s"] == h["horizon_s"] > 0
                assert r["status_this_worm"] == "not_evaluable"
                if r["status"] != "unavailable":
                    assert r["outcome"] in MB.OUTCOMES
                    assert r["outcome"] != "sustains_this_worm"
                    assert r["part_ii"]["status"] == "not_evaluable"
                    assert "passed" in r["part_i"]
                    if r["status"] == "pass":
                        assert r["outcome"] == "sustains_a_worm_identity_not_evaluable"
                    else:
                        assert r["outcome"] == "fails"
            status = {r["rung"]: r["status"] for r in records}
            assert status["L4"] == status["L5"] == "unavailable"
            assert "modulator" in records[4]["note"]
            assert h["lowest_rung_sustaining_this_worm"] == "not_evaluable"
            assert h["body_model_defects_this_worm"] == "not_evaluable"
            passes = [r["rung"] for r in records if r["status"] == "pass"]
            assert h["lowest_rung_sustaining_a_worm"] == (passes[0] if passes else None)
    # 'full' covers the whole recording; the fixed horizon is shorter.
    h = report["animals"]["worm0"]["horizons"]
    assert h["full"]["horizon_s"] > h["0.012"]["horizon_s"]


def test_I2_horizon_outside_recording_is_not_evaluable(sim, recordings):
    report = run(sim, recordings, horizons=(0.002, 5.0), rungs=("L0",))
    for entry in report["animals"].values():
        for h in entry["horizons"].values():
            assert h["status"] == "not_evaluable" and h["reason"]


def test_I2_decoder_path_evaluates_part_ii(sim, recordings):
    latents = {f"worm{i}": np.array([float(i), 0.0]) for i in range(4)}
    report = run(
        sim,
        recordings,
        horizons=(0.012,),
        rungs=("L0", "L1", "L2"),
        decoder=lambda window: latents["worm0"],  # always decodes to worm0
        latents=latents,
        window_s=0.006,
    )
    assert report["part_ii"]["status"] == "evaluated"
    seen = 0
    for animal, entry in report["animals"].items():
        h = entry["horizons"]["0.012"]
        assert h["lowest_rung_sustaining_this_worm"] != "not_evaluable"
        for r in h["records"]:
            ii = r["part_ii"]
            assert ii["status"] == "evaluated" and len(ii["margins"]) >= 2
            assert ii["checkpoints_s"][-1] == pytest.approx(h["horizon_s"])
            assert ii["passed"] == (animal == "worm0")
            if not r["part_i"]["passed"]:
                expect = "fails"
            else:
                expect = "sustains_this_worm" if ii["passed"] else "sustains_a_worm"
            assert r["outcome"] == expect
            assert r["status_this_worm"] == (
                "pass" if expect == "sustains_this_worm" else "fail"
            )
            seen += 1
        # 'this worm' can never be a lower rung than 'a worm'.
        a, b = h["lowest_rung_sustaining_a_worm"], h["lowest_rung_sustaining_this_worm"]
        if b is not None:
            assert a is not None and RUNGS.index(a) <= RUNGS.index(b)
    assert seen == 12


def test_I2_missing_latent_makes_that_animal_not_evaluable(sim, recordings):
    latents = {f"worm{i}": np.array([float(i)]) for i in range(1, 4)}  # no worm0
    report = run(
        sim,
        recordings,
        horizons=(0.012,),
        rungs=("L0",),
        decoder=lambda w: np.zeros(1),
        latents=latents,
        window_s=0.006,
    )
    r0 = report["animals"]["worm0"]["horizons"]["0.012"]
    assert r0["records"][0]["part_ii"]["status"] == "not_evaluable"
    assert r0["lowest_rung_sustaining_this_worm"] == "not_evaluable"
    r1 = report["animals"]["worm1"]["horizons"]["0.012"]
    assert r1["records"][0]["part_ii"]["status"] == "evaluated"


def test_I2_L3_and_L4_run_with_modulators_driven_by_pumping(sim, recordings):
    k = np.asarray(sim.neuromod["concentrations"]).shape[0]
    l4 = {"gain": np.ones((k, 1))}  # every modulator <- pumping
    report = run(
        sim, recordings, horizons=(0.008,), rungs=("L0", "L3", "L4", "L5"), l4=l4
    )
    for entry in report["animals"].values():
        records = entry["horizons"]["0.008"]["records"]
        status = {r["rung"]: r["status"] for r in records}
        assert status["L3"] in ("pass", "fail") and status["L4"] in ("pass", "fail")
        assert status["L5"] == "unavailable"


def test_I2_L2_replay_drives_head_neurons_from_behavior(sim, recordings):
    body = MB.MinimumBody(sim, NAMES, recordings, fluorescence, **OPTIONS)
    assert body.gain[1, MB.CHANNELS.index("head_angle")] > 0  # SMDDL
    assert body.gain[0, MB.CHANNELS.index("abs_head_angle")] > 0  # DVA
    assert not body.gain[2:].any()
    a = body.animals[0]
    base = body.emulate("L0", a, 8, None)
    replay = body.emulate("L2", a, 8, None)
    assert replay.shape == base.shape == (4, 8)
    assert not np.allclose(replay, base)


def test_I2_load_directory_without_manifest_and_load_decoder(tmp_path):
    labels = write_animals(tmp_path)
    recs = MB.load_directory(tmp_path, labels)
    assert [r.animal for r in recs] == [f"worm{i}" for i in range(4)]
    assert recs[0].sha256
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .h5"):
        MB.load_directory(empty, labels)
    (tmp_path / "stray.h5").write_bytes(b"")
    with pytest.raises(ValueError, match="missing animal labels"):
        MB.load_directory(tmp_path, labels)
    path = tmp_path / "decoder.npz"
    np.savez(
        path,
        weights=np.eye(2),
        bias=np.zeros(2),
        mean=np.zeros(2),
        neurons=np.array(["SMDDL", "DVA"]),
        animals=np.array(["worm0", "worm1"]),
        latents=np.eye(2),
    )
    decoder, latents = MB.load_decoder(path)
    assert set(latents) == {"worm0", "worm1"}
    window = np.array([[1.0, 1.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]])  # NAMES order
    assert np.allclose(decoder.bind(NAMES)(window), [3.0, 1.0])  # SMDDL, DVA order
    bad = MB.ReadoutDecoder(Readout(np.eye(1), np.zeros(1), np.zeros(1)), ["ZZZ"])
    with pytest.raises(ValueError, match="absent"):
        bad.bind(NAMES)


def test_I2_cli_command_is_wired(capsys, monkeypatch):
    from molecular_compiler.__main__ import main

    monkeypatch.setattr(sys, "argv", ["molc", "track-i2", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--trained" in out and "--recordings" in out and "--output" in out


def test_I2_L4_pumping_changes_emulation_through_modulators(recordings):
    from molecular_compiler.compiler import ModulatoryState

    graph, rules, kinetics = synthetic_system()
    modulated = compile(graph, rules, kinetics, state=ModulatoryState((0.0, 0.0)))
    l4 = {"gain": np.full((2, 1), 50.0)}
    body = MB.MinimumBody(modulated, NAMES, recordings, fluorescence, l4=l4, **OPTIONS)
    a = body.animals[0]
    proprioceptive = body.emulate("L3", a, 8, None)
    interoceptive = body.emulate("L4", a, 8, None)
    assert np.all(np.isfinite(interoceptive))
    # Slow signaling relaxes over 60 s, so over 8 ms the effect is tiny but nonzero.
    assert np.max(np.abs(proprioceptive - interoceptive)) > 0
