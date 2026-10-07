import subprocess
import sys
import types

import numpy as np
import pytest

from molecular_compiler.baselines import external_predictions
from molecular_compiler.external_baselines import (
    run_flyvis,
    run_shiu_b0,
    write_external_predictions,
)


def test_B6_round_trip_through_external_predictions(tmp_path):
    values = np.arange(12.0).reshape(3, 4)
    path = write_external_predictions(
        tmp_path / "b6.npz", values, "B6", "flyvis", "1.0", "t", "s", model_id="m"
    )
    read, metadata = external_predictions(path, "B6")
    np.testing.assert_array_equal(read, values)
    assert metadata["model_id"] == "m" and metadata["artifact_hash"]


def test_B6_writer_rejects_bad_input(tmp_path):
    ok = np.zeros((1, 2))
    with pytest.raises(ValueError, match="provenance"):
        write_external_predictions(tmp_path / "a.npz", ok, "B6", "x", "1", "", "s")
    with pytest.raises(ValueError, match="finite"):
        write_external_predictions(
            tmp_path / "a.npz", np.full((1, 2), np.nan), "B6", "x", "1", "t", "s"
        )
    with pytest.raises(ValueError, match="npz"):
        write_external_predictions(tmp_path / "a.txt", ok, "B6", "x", "1", "t", "s")


def test_B6_flyvis_absent_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "flyvis", None)
    with pytest.raises(RuntimeError, match="flyvis"):
        run_flyvis(tmp_path / "b.npz", ["T4a"], np.zeros((1, 2, 1, 3)), "m", "t", "s")


def test_B6_flyvis_conversion_with_stub_package(tmp_path, monkeypatch):
    # The stub follows the interface run_flyvis assumes, not the real package.
    class Network:
        def fade_in_state(self, *_):
            return None

        def simulate(self, stimuli, dt, initial_state=None):
            shape = stimuli.shape[:2] + (3,)
            return np.broadcast_to(np.arange(3.0), shape) * 1.0

    class View:
        def __init__(self, model_id):
            types_ = np.array([b"T4a", b"T4a", b"Mi1"])
            self.connectome = types.SimpleNamespace(
                nodes=types.SimpleNamespace(type=types_)
            )

        def init_network(self):
            return Network()

    stub = types.ModuleType("flyvis")
    stub.NetworkView, stub.__version__ = View, "9.9"
    monkeypatch.setitem(sys.modules, "flyvis", stub)
    stimuli = np.zeros((2, 5, 1, 4))
    path = run_flyvis(tmp_path / "b6.npz", ["T4a", "Mi1"], stimuli, "flow/0", "t", "s")
    values, metadata = external_predictions(path, "B6")
    assert values.shape == (2, 10) and metadata["model_id"] == "flow/0"
    np.testing.assert_allclose(values[:, 0], [0.5, 2.0])
    assert metadata["version"] == "9.9"
    with pytest.raises(ValueError, match="not in flyvis"):
        run_flyvis(tmp_path / "x.npz", ["Tm9"], stimuli, "flow/0", "t", "s")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _checkout(tmp_path):
    repo = tmp_path / "Drosophila_brain_model"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "x")
    return repo


def test_B0_fly_round_trip_with_commit_provenance(tmp_path):
    repo = _checkout(tmp_path)
    script = (
        "open('rates.csv','w').write("
        "'flywire_id,stim_a,stim_b\\n11,1.0,2.0\\n22,3.0,4.0\\n33,5.0,6.0\\n')"
    )
    path = run_shiu_b0(
        tmp_path / "b0.npz",
        repo,
        [sys.executable, "-c", script],
        "rates.csv",
        [33, 11],
        "s",
    )
    values, metadata = external_predictions(path, "B0_fly")
    np.testing.assert_array_equal(values, [[5.0, 6.0], [1.0, 2.0]])
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert metadata["version"] == f"git:{head}"
    assert metadata["stimulations"] == ["stim_a", "stim_b"]


def test_B0_fly_missing_checkout_or_output_errors(tmp_path):
    out = tmp_path / "b.npz"
    with pytest.raises(FileNotFoundError, match="checkout"):
        run_shiu_b0(out, tmp_path / "nope", ["true"], "r.csv", [1], "s")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="git checkout"):
        run_shiu_b0(out, plain, ["true"], "r.csv", [1], "s")
    repo = _checkout(tmp_path)
    noop = [sys.executable, "-c", "pass"]
    with pytest.raises(FileNotFoundError, match="rates file"):
        run_shiu_b0(out, repo, noop, "r.csv", [1], "s")
    failing = [sys.executable, "-c", "raise SystemExit(2)"]
    with pytest.raises(RuntimeError, match="command failed"):
        run_shiu_b0(out, repo, failing, "r.csv", [1], "s")
