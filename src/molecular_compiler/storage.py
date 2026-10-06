"""Parquet/Zarr/NWB interchange and checksummed safetensors checkpoints."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import zarr
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from .observation import Recording
from .provenance import content_hash, require_publishable
from .rules import RuleNetwork


def write_recording(recording, path):
    recording.validate()
    if (
        not recording.metadata
        or "processing_hash" not in recording.metadata
        or recording.metadata.get("tier") not in {"open", "restricted"}
    ):
        raise ValueError("recordings require inherited provenance")
    root = zarr.open_group(str(path), mode="w")
    for name in ("y", "t", "neuron_map", "match_confidence", "y_ref"):
        value = getattr(recording, name)
        if value is not None:
            array = np.asarray(value)
            chunks = tuple(
                min(s, 1024 if index else 64) for index, s in enumerate(array.shape)
            )
            root.create_array(name, data=array, chunks=chunks)
    root.attrs.update(
        indicator=recording.indicator,
        prep=recording.prep,
        animal_state=recording.animal_state,
        provenance=recording.metadata,
        units={
            "y": "relative_fluorescence",
            "t": "s",
            "y_ref": "relative_fluorescence",
        },
    )


def read_recording(path):
    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get("units", {}).get("t") != "s" or not root.attrs.get("provenance"):
        raise ValueError("recording missing units or provenance")
    recording = Recording(
        jnp.asarray(root["y"][:]),
        jnp.asarray(root["t"][:]),
        root["neuron_map"][:],
        root["match_confidence"][:],
        root.attrs["indicator"],
        root.attrs["prep"],
        root.attrs["animal_state"],
        root["y_ref"][:] if "y_ref" in root else None,
        dict(root.attrs["provenance"]),
    )
    recording.validate()
    return recording


def save_checkpoint(rules, path, metadata, publish=False):
    if publish:
        require_publishable(metadata)
    if metadata.get("tier") not in {"open", "restricted"} or not metadata.get(
        "processing_hash"
    ):
        raise ValueError("checkpoint requires inherited data provenance")
    payload = {
        "provenance": metadata,
        "threshold": rules.threshold,
        "budget": rules.budget,
        "checkpoint_hash": rules.checkpoint_hash(),
        "ortholog_groups": rules.ortholog_groups,
        "format_version": 1,
    }
    values = {key: np.ascontiguousarray(value) for key, value in rules.params.items()}
    if rules.frozen_params is not None:
        values.update(
            {
                f"frozen/{key}": np.ascontiguousarray(value)
                for key, value in rules.frozen_params.items()
            }
        )
    # Empty ortholog arrays are valid and preserve an unmapped-gene vocabulary.
    save_file(
        values,
        str(path),
        metadata={"molecular_compiler": json.dumps(payload, sort_keys=True)},
    )


def load_checkpoint(path):
    with safe_open(str(path), framework="numpy") as handle:
        payload = json.loads(handle.metadata()["molecular_compiler"])
    if payload["format_version"] != 1:
        raise ValueError("unknown checkpoint format")
    values = load_file(str(path))
    frozen = {
        key.removeprefix("frozen/"): jnp.asarray(value)
        for key, value in values.items()
        if key.startswith("frozen/")
    }
    rules = RuleNetwork(
        {
            key: jnp.asarray(value)
            for key, value in values.items()
            if not key.startswith("frozen/")
        },
        threshold=payload["threshold"],
        budget=payload["budget"],
        ortholog_groups=tuple(payload["ortholog_groups"]),
        frozen_params=frozen or None,
    )
    if rules.checkpoint_hash() != payload["checkpoint_hash"]:
        raise ValueError("checkpoint content hash differs")
    return rules, payload["provenance"]


def save_simgraph(sim, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{group}/{key}": np.asarray(value)
        for group in ("neuron_params", "syn_params", "gap")
        for key, value in getattr(sim, group).items()
    }
    for name in ("syn_post_ptr", "syn_pre_idx", "syn_post_idx"):
        arrays[name] = np.asarray(getattr(sim, name))
    for key, value in sim.neuromod.items():
        if hasattr(value, "shape"):
            arrays[f"neuromod/{key}"] = np.asarray(value)
    np.savez_compressed(path / "arrays.npz", **arrays)
    metadata = {
        "provenance": sim.metadata,
        "resolution": asdict(sim.resolution),
        "surrogates": [
            {
                **asdict(s),
                "lower": [float(v) if np.isfinite(v) else "-inf" for v in s.lower],
                "upper": [float(v) if np.isfinite(v) else "+inf" for v in s.upper],
            }
            for s in sim.surrogates
        ],
        "array_hash": content_hash(arrays),
        "kinetics_ids": [r.molecule_id for r in sim.channel_records],
        "receptor_ids": [r.molecule_id for r in sim.receptor_records],
        "neuromod_mode": sim.neuromod["mode"],
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))


def export_nwb(recording, path, session_description="Molecular compiler recording"):
    """NWB stays outside gradients; install the nwb extra."""
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries

    recording.validate()
    if not recording.metadata:
        raise ValueError("NWB export requires provenance")
    nwb = NWBFile(
        session_description, recording.metadata["processing_hash"], datetime.now(UTC)
    )
    for name in ("y", "y_ref"):
        value = getattr(recording, name)
        if value is not None:
            nwb.add_acquisition(
                TimeSeries(
                    name=name,
                    data=np.asarray(value).T,
                    timestamps=np.asarray(recording.t),
                    unit="relative_fluorescence",
                )
            )
    nwb.add_scratch(
        np.asarray(recording.neuron_map),
        name="neuron_map",
        description="connectome neuron IDs",
    )
    nwb.add_scratch(
        np.asarray(recording.match_confidence),
        name="match_confidence",
        description="assignment probabilities",
    )
    payload = {
        "provenance": recording.metadata,
        "indicator": recording.indicator,
        "prep": recording.prep,
        "animal_state": recording.animal_state,
    }
    nwb.add_scratch(
        json.dumps(payload),
        name="molecular_compiler",
        description="provenance and observation metadata",
    )
    with NWBHDF5IO(str(path), "w") as handle:
        handle.write(nwb)


def import_nwb(path):
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as handle:
        nwb = handle.read()
        metadata = json.loads(str(nwb.scratch["molecular_compiler"].data))
        y = np.asarray(nwb.acquisition["y"].data[:]).T
        t = np.asarray(nwb.acquisition["y"].timestamps[:])
        reference = (
            np.asarray(nwb.acquisition["y_ref"].data[:]).T
            if "y_ref" in nwb.acquisition
            else None
        )
        recording = Recording(
            jnp.asarray(y),
            jnp.asarray(t),
            np.asarray(nwb.scratch["neuron_map"].data[:]),
            np.asarray(nwb.scratch["match_confidence"].data[:]),
            metadata["indicator"],
            metadata["prep"],
            metadata["animal_state"],
            reference,
            metadata["provenance"],
        )
    recording.validate()
    return recording
