"""B6 (flyvis) and fly-scale B0 (Shiu et al.) runners writing `external_predictions` files.

Both jobs run outside the JAX gradient path (S-R2). Output is `<name>.npz` with a
`predictions` array plus `<name>.json` provenance, exactly what
`baselines.external_predictions` reads. The flyvis and Shiu adapters are written
against their documented interfaces but have not been run against the real
packages here; only the conversion and provenance code is tested.
"""

import importlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

from .provenance import content_hash


def write_external_predictions(
    path, predictions, baseline_id, source, version, training_hash, split_hash, **extra
):
    """Write predictions and provenance in the layout `external_predictions` reads."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("external prediction path must end in .npz")
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("predictions must be a finite [units, time] array")
    metadata = {
        "baseline_id": baseline_id,
        "source": source,
        "version": version,
        "training_hash": training_hash,
        "split_hash": split_hash,
        **extra,
    }
    required = ("source", "version", "training_hash", "split_hash")
    if not all(metadata.get(k) for k in required):
        raise ValueError("external baseline lacks provenance")
    np.savez(path, predictions=values)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def run_flyvis(
    path, cell_types, stimuli, model_id, training_hash, split_hash, dt_s=1 / 50
):
    """B6: simulate a pretrained flyvis network and write per-cell-type responses.

    `stimuli` is [samples, frames, 1, hexals] flyvis movie input, `model_id` a
    flyvis network-view name (e.g. "flow/0000/000"). Responses are averaged over
    the cells of each requested type and samples are concatenated along time,
    giving [cell_types, samples * frames].
    """
    try:
        flyvis = importlib.import_module("flyvis")
    except ImportError as error:
        raise RuntimeError(
            "B6 needs the `flyvis` package (PyTorch, separate evaluation job); "
            "install it in a separate environment"
        ) from error
    try:
        version = importlib.metadata.version("flyvis")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(flyvis, "__version__", None)
    if not version:
        raise RuntimeError("cannot determine flyvis version for provenance")
    cell_types = [str(t) for t in cell_types]
    if not cell_types:
        raise ValueError("B6 needs at least one cell type")
    stimuli = np.asarray(stimuli, dtype=float)
    if stimuli.ndim != 4:
        raise ValueError("flyvis stimuli require [samples, frames, 1, hexals]")
    view = flyvis.NetworkView(model_id)
    network = view.init_network()
    initial = network.fade_in_state(1.0, dt_s, stimuli[:, :1])
    # [samples, frames, nodes]
    activity = np.asarray(network.simulate(stimuli, dt_s, initial_state=initial))
    node_types = np.array(
        [
            t.decode() if isinstance(t, bytes) else str(t)
            for t in view.connectome.nodes.type[:]
        ]
    )
    rows = []
    for cell_type in cell_types:
        selected = node_types == cell_type
        if not selected.any():
            raise ValueError(f"cell type {cell_type!r} not in flyvis model {model_id}")
        rows.append(activity[:, :, selected].mean(axis=2).reshape(-1))
    return write_external_predictions(
        path,
        np.stack(rows),
        "B6",
        "flyvis",
        str(version),
        training_hash,
        split_hash,
        model_id=model_id,
        cell_types=cell_types,
        stimulus_hash=content_hash(stimuli),
        n_samples=int(stimuli.shape[0]),
        dt_s=dt_s,
    )


def _git(checkout, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"{checkout} is not a usable git checkout") from error


def read_shiu_rates(rates_file, neuron_ids):
    """CSV: `flywire_id` then one Hz column per stimulation -> [neurons, stimulations]."""
    rows = np.genfromtxt(
        rates_file, delimiter=",", names=True, dtype=None, encoding=None
    )
    names = rows.dtype.names
    if not names or names[0] != "flywire_id" or len(names) < 2:
        raise ValueError("rates file needs a flywire_id column and >= 1 rate column")
    ids = np.atleast_1d(rows["flywire_id"]).astype(np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate flywire_id in rates file")
    position = {int(v): k for k, v in enumerate(ids)}
    missing = [int(v) for v in neuron_ids if int(v) not in position]
    if missing:
        raise ValueError(f"rates file lacks {len(missing)} requested neurons")
    table = np.column_stack([np.atleast_1d(rows[k]).astype(float) for k in names[1:]])
    return table[[position[int(v)] for v in neuron_ids]], list(names[1:])


def run_shiu_b0(
    path,
    checkout,
    command,
    rates_file,
    neuron_ids,
    split_hash,
    training_hash=None,
    timeout_s=3600,
):
    """Fly-scale B0: run a local Shiu et al. checkout and convert its firing rates.

    `command` is an argv list executed with the checkout as cwd (it must write
    `rates_file`, relative to the checkout or absolute). Provenance records the
    checkout's commit and whether the tree was dirty. B0 is not trained, so the
    default training_hash identifies the commit with no training data.
    """
    checkout = Path(checkout)
    if not checkout.is_dir():
        raise FileNotFoundError(
            f"Drosophila_brain_model checkout not found: {checkout}"
        )
    commit = _git(checkout, "rev-parse", "HEAD")
    dirty = bool(_git(checkout, "status", "--porcelain"))
    try:
        subprocess.run(
            list(command),
            cwd=checkout,
            check=True,
            timeout=timeout_s,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Shiu B0 command failed: {error}") from error
    rates_file = Path(rates_file)
    rates_file = rates_file if rates_file.is_absolute() else checkout / rates_file
    if not rates_file.is_file():
        raise FileNotFoundError(f"Shiu B0 produced no rates file: {rates_file}")
    rates, stimulations = read_shiu_rates(rates_file, neuron_ids)
    return write_external_predictions(
        path,
        rates,
        "B0_fly",
        "philshiu/Drosophila_brain_model",
        f"git:{commit}",
        training_hash or content_hash("untrained", commit),
        split_hash,
        commit=commit,
        dirty_checkout=dirty,
        command=[str(c) for c in command],
        stimulations=stimulations,
        neuron_ids=[int(v) for v in neuron_ids],
    )
