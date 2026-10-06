"""M1 ingestion of Randi et al. (2023) whole-brain traces and stimulations.

The OSF export (10.17605/OSF.IO/E2SYT) holds, per animal, GCaMP6s traces of
tracked nuclei at 2 volumes/s (interpolated and photobleach-corrected by the
authors, not smoothed), NeuroPAL labels per trace, and the trace index and
volume of each two-photon stimulation.

Labels are matched to canonical neurons without correcting typos:
- an exact neuron name matches with confidence 1;
- a trailing "?" halves the confidence;
- a label naming a class or a side-less subset (e.g. "AVD", "IL1V", "DB")
  matches one of its k candidate neurons with confidence 1/k;
- AWCON/AWCOFF match AWCL/AWCR with confidence 1/2 (stochastic laterality);
- anything else (glia, numbers, typos) is unmatched (neuron -1, confidence 0).
M1-R3: matches below the training threshold stay in the recording for audit.
"""

import json
import re
from pathlib import Path

import numpy as np

from .worm_public import normalize_name

PULSE_S = {"wt": 0.5, "unc31": 0.3}
STIMULUS = {"wavelength_nm": 850.0, "power_mW": 1.2, "opsin": "GUR-3/PRDX-2"}
VOLUME_RATE_HZ = 2.0


def match_label(label, neurons):
    """Return (neuron name or None, confidence)."""
    raw = label.strip()
    if not raw:
        return None, 0.0
    uncertain = raw.endswith("?")
    name = normalize_name(raw.rstrip("?").strip())
    scale = 0.5 if uncertain else 1.0
    if name in neurons:
        return name, scale
    if name in {"AWCON", "AWCOFF"}:
        return "AWCL", 0.5 * scale
    if not re.fullmatch(r"[A-Z][A-Z0-9]*", name):
        return None, 0.0
    candidates = [
        n
        for n in neurons
        if n.startswith(name) and re.fullmatch(r"[LRDV]{1,2}|\d+", n[len(name) :])
    ]
    if candidates:
        return min(candidates), scale / len(candidates)
    return None, 0.0


def read_animal(directory, index, neurons):
    root = Path(directory)

    def text(name):
        return (root / f"{index}_{name}.txt").read_text()

    y = np.loadtxt(root / f"{index}_gcamp.txt", ndmin=2).T
    t = np.loadtxt(root / f"{index}_t.txt", ndmin=1)
    labels = text("labels").split("\n")[: y.shape[0]]
    labels += [""] * (y.shape[0] - len(labels))
    stim_rois = np.loadtxt(root / f"{index}_stim_neurons.txt", dtype=int, ndmin=1)
    stim_volumes = np.loadtxt(root / f"{index}_stim_volume_i.txt", dtype=int, ndmin=1)
    if y.shape[1] != len(t) or len(stim_rois) != len(stim_volumes):
        raise ValueError(f"animal {index}: inconsistent export shapes")
    matches = [match_label(label, neurons) for label in labels]
    # Two traces cannot be the same neuron: duplicate partial matches move to the
    # next free candidate so each canonical neuron appears at most once.
    used, neuron_map, confidence = set(), [], []
    for (name, conf), label in zip(matches, labels):
        if name is not None and name in used and conf < 1:
            base = normalize_name(label.strip().rstrip("?"))
            free = [
                n
                for n in sorted(neurons)
                if n.startswith("AWC" if base in {"AWCON", "AWCOFF"} else base)
                and n not in used
            ]
            name = free[0] if free else None
            conf = conf if free else 0.0
        if name is not None and name in used:
            name, conf = None, 0.0
        if name is not None:
            used.add(name)
        neuron_map.append(name)
        confidence.append(conf)
    return {
        "index": index,
        "dataset": text("ds_name").strip(),
        "y": y,
        "t": t,
        "labels": labels,
        "neurons": neuron_map,
        "match_confidence": np.asarray(confidence),
        "stim_rois": stim_rois,
        "stim_volumes": stim_volumes,
    }


def animals(directory):
    indices = sorted(
        int(p.name.split("_")[0]) for p in Path(directory).glob("*_gcamp.txt")
    )
    return indices


def response_windows(animal, pre_s=30.0, post_s=30.0):
    """dF/F0 windows around each stimulation (F0 = mean of the pre window)."""
    pre = round(pre_s * VOLUME_RATE_HZ)
    post = round(post_s * VOLUME_RATE_HZ)
    y = animal["y"]
    events = []
    for roi, volume in zip(animal["stim_rois"], animal["stim_volumes"]):
        if volume - pre < 0 or volume + post > y.shape[1]:
            continue
        window = y[:, volume - pre : volume + post]
        baseline = window[:, :pre].mean(axis=1, keepdims=True)
        valid = baseline[:, 0] > 0
        dff = np.full(window.shape, np.nan)
        dff[valid] = window[valid] / baseline[valid] - 1
        events.append({"roi": int(roi), "volume": int(volume), "dff": dff})
    return events


def build_response_dataset(directory, strain, neurons, ids, output):
    """Per-event, per-neuron responses with animal identity, for Phase 1.

    Saved arrays (one row per (event, responding trace) pair with both sides
    matched): animal, event, stimulated, responder, confidences, mean dF/F0
    over the 30 s post window, and the full 60 s window at 2 Hz.
    """
    rows = {
        k: []
        for k in (
            "animal",
            "event",
            "stimulated",
            "responder",
            "stim_confidence",
            "resp_confidence",
            "mean_dff",
            "window",
        )
    }
    summary = {"animals": 0, "events": 0, "events_matched": 0, "traces": 0}
    report = []
    event_id = 0
    for index in animals(directory):
        animal = read_animal(directory, index, neurons)
        summary["animals"] += 1
        summary["traces"] += len(animal["neurons"])
        report.append(
            {
                "animal": index,
                "dataset": animal["dataset"],
                "traces": len(animal["neurons"]),
                "matched": int(sum(n is not None for n in animal["neurons"])),
                "confident": int(np.sum(animal["match_confidence"] >= 0.95)),
                "duration_s": float(animal["t"][-1] - animal["t"][0]),
            }
        )
        for event in response_windows(animal):
            summary["events"] += 1
            roi = event["roi"]
            if (
                roi < 0
                or roi >= len(animal["neurons"])
                or animal["neurons"][roi] is None
            ):
                event_id += 1
                continue
            summary["events_matched"] += 1
            stimulated = ids[animal["neurons"][roi]]
            for trace, name in enumerate(animal["neurons"]):
                values = event["dff"][trace]
                if name is None or not np.all(np.isfinite(values)):
                    continue
                rows["animal"].append(index)
                rows["event"].append(event_id)
                rows["stimulated"].append(stimulated)
                rows["responder"].append(ids[name])
                rows["stim_confidence"].append(animal["match_confidence"][roi])
                rows["resp_confidence"].append(animal["match_confidence"][trace])
                rows["mean_dff"].append(values[60:].mean())
                rows["window"].append(values.astype(np.float32))
            event_id += 1
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    np.savez_compressed(output, **arrays)
    summary["pairs"] = len(arrays["animal"])
    summary["strain"] = strain
    summary["pulse_s"] = PULSE_S[strain]
    return summary, report


def ingest(cache_root, project, register):
    """Register-checked conversion of both strains into the Phase 1 project."""
    from dataclasses import asdict

    from .provenance import content_hash, inherit

    record = register.require("randi2023_traces")
    manifest = json.loads((Path(project) / "manifest.json").read_text())
    neurons = manifest["neuron_names"]
    ids = {n: i for i, n in enumerate(neurons)}
    out = Path(project) / "responses"
    out.mkdir(exist_ok=True)
    result = {"register": asdict(record)}
    for strain, folder in (("wt", "exported_data"), ("unc31", "exported_data_unc31")):
        directory = Path(cache_root) / "randi" / folder
        summary, animals_report = build_response_dataset(
            directory, strain, neurons, ids, out / f"{strain}.npz"
        )
        result[strain] = {"summary": summary, "animals": animals_report}
    from .phase0_worm import align_atlas
    from .worm_public import load_atlas

    atlas = align_atlas(
        load_atlas(Path(project) / "signal_propagation.npz"), len(neurons)
    )
    for strain in ("wt", "unc31"):
        path = out / f"{strain}.npz"
        arrays = dict(np.load(path))
        flags = flag_events(arrays, atlas[strain])
        np.savez_compressed(path, **arrays, **flags)
        result[strain]["summary"]["atlas_included_events"] = len(
            np.unique(arrays["event"][flags["atlas_included"]])
        )
        result[strain]["summary"]["outlier_trials"] = int(flags["outlier"].sum())
    result["provenance"] = inherit([record], content_hash(result))
    result["stimulus"] = STIMULUS | {"pulse_s": PULSE_S}
    (out / "ingestion-report.json").write_text(json.dumps(result, indent=2))
    return result


OUTLIER_DFF = 5.0


def flag_events(responses, atlas_strain, tolerance=1e-4):
    """Mark events Randi et al. kept (autoresponse criterion) and outlier trials.

    The atlas stores each included event's autoresponse; an event is included
    when its stimulated neuron's own response equals one of those values. This
    reproduces their inclusion without re-implementing unpublished thresholds.
    Trials with |dF/F0| >= 5 come from near-zero baselines and are flagged.
    """
    stimulated, responder = responses["stimulated"], responses["responder"]
    events, values = responses["event"], responses["mean_dff"]
    self_rows = stimulated == responder
    included = {}
    for event, neuron, value in zip(
        events[self_rows], stimulated[self_rows], values[self_rows]
    ):
        trials = atlas_strain["trials"][neuron][neuron]
        included[event] = bool(
            len(trials) and np.min(np.abs(trials - value)) < tolerance
        )
    return {
        "atlas_included": np.array([included.get(e, False) for e in events]),
        "outlier": np.abs(values) >= OUTLIER_DFF,
    }


def load_responses(project, strain, confidence=0.95, atlas_only=True):
    """Training-eligible off-diagonal responses (M1-R3 confidence threshold)."""
    raw = dict(np.load(Path(project) / "responses" / f"{strain}.npz"))
    keep = (
        (raw["stim_confidence"] >= confidence)
        & (raw["resp_confidence"] >= confidence)
        & ~raw["outlier"]
    )
    if atlas_only:
        keep &= raw["atlas_included"]
    return {k: v[keep] for k, v in raw.items()}
