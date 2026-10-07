"""Raw loader for the Atanas et al. (2023) WormWideWeb/Flavell processed files (Stage I2).

Field names and label rules follow the WormSim importer (WORMWIDEWEB.md):
`timing/timestamp_confocal`, `behavior/{velocity,head_angle,angular_velocity,
pumping}` and NeuroPAL label JSON with ROI ids that are one-based. Unlike
that importer, this reads `gcamp/trace_array_original`: the published
`trace_array` is a whole-recording z-score, so each sample depends on future
samples (PREPROCESSING-AUDIT.md). Traces are not normalized, smoothed or
interpolated here. Nonfinite samples stay NaN.

Label rules: exact canonical identity that appears once in the animal and
rating at least `min_rating` (ordinal 0-5, not a calibrated probability).
Every decision is returned with its reason. Files are checked against a
manifest of {file, animal_id, sha256} when one is given. h5py is imported
lazily (extra `phase0`).
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIME_KEY = "timing/timestamp_confocal"
TRACE_KEY = "gcamp/trace_array_original"
BEHAVIOR = ("velocity", "head_angle", "angular_velocity", "pumping")
MIN_RATING = 3.0


@dataclass
class Recording:
    animal: str
    t: np.ndarray  # [T] seconds
    traces: np.ndarray  # [T, K] original fluorescence of the accepted ROIs
    neurons: list  # canonical names, sorted, length K
    ratings: np.ndarray  # [K] ordinal NeuroPAL rating
    roi: np.ndarray  # [K] one-based source ROI ids
    behavior: dict  # channel -> [T], channels present in the file
    sha256: str
    decisions: list = field(default_factory=list)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(path):
    """{animal_id: {roi (int, one-based): (label, rating)}} from NeuroPAL JSON."""
    with open(path) as f:
        data = json.load(f)["data"]
    return {
        animal: {
            int(roi): (entry["label"], float(entry["confidence"]))
            for roi, entry in record["idx_neuron-label"].items()
        }
        for animal, record in data.items()
    }


def read_manifest(path):
    """[{file, animal_id, sha256}] from a fetch receipt; filenames only, no paths."""
    with open(path) as f:
        animals = json.load(f)["animals"]
    for a in animals:
        if Path(a["file"]).name != a["file"]:
            raise ValueError("manifest contains a nonlocal filename")
    return animals


def select_labels(labels, n_rois, canonical=None, min_rating=MIN_RATING):
    """Decisions [(roi, label, rating, accepted, reason)], sorted by ROI id."""
    counts = {}
    for name, _ in labels.values():
        counts[name] = counts.get(name, 0) + 1
    out = []
    for roi, (name, rating) in sorted(labels.items()):
        if roi < 1 or roi > n_rois or not 0 <= rating <= 5:
            raise ValueError(f"label ROI {roi} or rating {rating} outside source range")
        if canonical is not None and name not in canonical:
            reason = "not an exact canonical neuron"
        elif counts[name] > 1:
            reason = "duplicate canonical identity"
        elif rating < min_rating:
            reason = "below ordinal rating threshold"
        else:
            reason = "accepted"
        out.append((roi, name, rating, reason == "accepted", reason))
    return out


def load_recording(
    path,
    labels,
    animal="",
    canonical=None,
    min_rating=MIN_RATING,
    expected_sha256=None,
):
    """One animal's file; `labels` is that animal's {roi: (label, rating)}."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "reading Atanas et al. HDF5 files needs h5py: install the 'phase0' extra"
        ) from e
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"checksum mismatch for {Path(path).name}")
    with h5py.File(path, "r") as h5:
        t = np.asarray(h5[TIME_KEY][()], dtype=float).ravel()
        raw = np.asarray(h5[TRACE_KEY][()], dtype=float)
        behavior = {
            name: np.asarray(h5[f"behavior/{name}"][()], dtype=float).ravel()
            for name in BEHAVIOR
            if f"behavior/{name}" in h5
        }
    if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if raw.ndim != 2 or (raw.shape[0] == len(t)) == (raw.shape[1] == len(t)):
        raise ValueError("trace shape must have one unambiguous time axis")
    traces = raw if raw.shape[0] == len(t) else raw.T
    for name, values in behavior.items():
        if len(values) != len(t):
            raise ValueError(f"behavior channel {name} does not match confocal grid")
    decisions = select_labels(labels, traces.shape[1], canonical, min_rating)
    accepted = sorted((d for d in decisions if d[3]), key=lambda d: d[1])
    rois = np.array([d[0] for d in accepted], dtype=int)
    return Recording(
        animal,
        t,
        traces[:, rois - 1],  # ROI ids are one-based
        [d[1] for d in accepted],
        np.array([d[2] for d in accepted]),
        rois,
        behavior,
        digest,
        decisions,
    )


def load_baseline(directory, manifest, labels_path, canonical=None, **kwargs):
    """All manifest animals under `directory`, each verified against its sha256."""
    labels = read_labels(labels_path)
    out = []
    for a in read_manifest(manifest):
        if a["animal_id"] not in labels:
            raise ValueError(f"missing animal labels: {a['animal_id']}")
        out.append(
            load_recording(
                Path(directory) / a["file"],
                labels[a["animal_id"]],
                a["animal_id"],
                canonical,
                expected_sha256=a["sha256"],
                **kwargs,
            )
        )
    return out
