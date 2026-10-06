"""Pre-registered feasibility analyses. Synthetic results cannot clear real gates."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr

from .provenance import content_hash


def rank_analysis(design, epsilon=1e-5):
    x = np.asarray(design, dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)) or epsilon <= 0:
        raise ValueError(
            "rank analysis requires a finite matrix and positive threshold"
        )
    singular = np.linalg.svd(x, compute_uv=False)
    cutoff = epsilon * singular[0] if len(singular) else epsilon
    rank = int(np.sum(singular > cutoff))
    eigen = singular**2
    participation = float(eigen.sum() ** 2 / np.sum(eigen**2)) if np.any(eigen) else 0.0
    return {
        "rank_epsilon": rank,
        "epsilon_relative": epsilon,
        "participation_ratio": participation,
        "singular_values": singular.tolist(),
        "rows": x.shape[0],
        "columns": x.shape[1],
    }


def molecular_design(graph, threshold=0.01):
    """Chemical/contact rows plus implicit peptide-pair Gram matrix.

    The peptide design's source/target Cartesian products are accumulated
    algebraically in O(N*P*d^2), without enumerating N^2 pairs.
    """
    z = np.asarray(graph.abundance)
    _, d = z.shape
    ids = {int(neuron): i for i, neuron in enumerate(graph.neuron_ids)}
    rows = []
    for row in graph.connectome.synapses.to_pylist():
        rows.append(
            np.r_[
                z[ids[row["pre_id"]]],
                z[ids[row["post_id"]]],
                row["size"],
                row["path_dist_post"],
                row["local_diameter_post"],
            ]
        )
    for row in graph.connectome.contacts.to_pylist():
        rows.append(np.r_[z[ids[row["i_id"]]], z[ids[row["j_id"]]], row["area"], 0, 0])
    design = np.asarray(rows).reshape(-1, 2 * d + 3)
    gram = design.T @ design
    gi = {gene: i for i, gene in enumerate(graph.gene_ids)}
    pair_count = 0
    for pair in graph.molecular.peptide_receptor_pairs.to_pylist():
        sources = z[z[:, gi[pair["peptide_gene_id"]]] > threshold]
        targets = z[z[:, gi[pair["receptor_gene_id"]]] > threshold]
        ns, nt = len(sources), len(targets)
        gram[:d, :d] += nt * (sources.T @ sources)
        gram[d : 2 * d, d : 2 * d] += ns * (targets.T @ targets)
        cross = np.outer(sources.sum(axis=0), targets.sum(axis=0))
        gram[:d, d : 2 * d] += cross
        gram[d : 2 * d, :d] += cross.T
        pair_count += ns * nt
    eigen = np.maximum(np.linalg.eigvalsh(gram), 0)
    singular = np.sqrt(eigen[::-1])
    cutoff = 1e-5 * (singular[0] if len(singular) else 1)
    return {
        "rank_epsilon": int(np.sum(singular > cutoff)),
        "participation_ratio": float(eigen.sum() ** 2 / max(np.sum(eigen**2), 1e-30)),
        "chemical_contact_rows": len(rows),
        "implicit_peptide_rows": pair_count,
        "singular_values": singular.tolist(),
        "gram": gram,
    }


def split_diagnostics(training_expression, heldout_expression, threshold=0.01):
    train, held = (
        np.asarray(training_expression, dtype=float),
        np.asarray(heldout_expression, dtype=float),
    )
    if (
        train.ndim != 2
        or held.ndim != 2
        or train.shape[1] != held.shape[1]
        or not len(train)
    ):
        raise ValueError("split diagnostic expression dimensions differ")
    training_genes = np.any(train > threshold, axis=0)
    heldout_genes = np.any(held > threshold, axis=0)
    novel = float(np.sum(heldout_genes & ~training_genes) / max(heldout_genes.sum(), 1))
    profiles = []
    for row in held:
        constraints = np.vstack([train.T, np.ones(len(train))])
        fit = linprog(
            np.zeros(len(train)),
            A_eq=constraints,
            b_eq=np.r_[row, 1],
            bounds=(0, None),
            method="highs",
        )
        inside = bool(fit.success)
        profiles.append(
            {
                "inside_training_hull": inside,
                "nearest_training_distance": float(
                    np.linalg.norm(train - row, axis=1).min()
                ),
                "novel_gene_fraction": float(
                    np.sum((row > threshold) & ~training_genes)
                    / max(np.sum(row > threshold), 1)
                ),
                "interpretation": "interpolation"
                if inside and not np.any((row > threshold) & ~training_genes)
                else "extrapolation",
            }
        )
    return {
        "novel_gene_fraction": novel,
        "heldout_profiles": profiles,
        "training_expression_rank": rank_analysis(train),
    }


def gauge_audit(expression, gain=None, opsin=None, threshold=0.2):
    expression = np.asarray(expression)
    report = {}
    for name, values in (("indicator", gain), ("opsin", opsin)):
        if values is None:
            report[name] = {"status": "unavailable", "confounded": None}
            continue
        values = np.asarray(values)
        if (
            values.shape != (len(expression),)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                "gauge audit requires positive measured gains aligned to identity"
            )
        correlations = []
        for g in expression.T:
            if np.ptp(g) and np.ptp(values):
                correlations.append(float(spearmanr(g, values).statistic))
        strongest = max(map(abs, correlations), default=0.0)
        report[name] = {
            "status": "complete",
            "max_abs_identity_correlation": strongest,
            "confounded": strongest > threshold,
        }
    report["complete"] = all(r["status"] == "complete" for r in report.values())
    return report


@dataclass(frozen=True)
class Phase0Thresholds:
    minimum_rank: int
    maximum_ambiguous_fraction: float
    maximum_interpolation_fraction: float
    minimum_independent_contrasts: float
    maximum_parameters: int
    peptide_effect_minimum: float
    metric_targets: dict

    def __post_init__(self):
        if self.minimum_rank < 1 or self.minimum_independent_contrasts <= 0:
            raise ValueError("rank thresholds must be positive")
        if not isinstance(self.maximum_parameters, int) or self.maximum_parameters < 1:
            raise ValueError("absolute parameter budget must be a positive integer")
        if (
            not 0 <= self.maximum_ambiguous_fraction <= 1
            or not 0 <= self.maximum_interpolation_fraction <= 1
        ):
            raise ValueError("fractions must be in [0,1]")
        if not self.metric_targets:
            raise ValueError("metric targets must be pre-registered")


def freeze_registration(path, registration):
    """Create once with exclusive open. Existing split registration cannot drift."""
    path = Path(path)
    payload = {
        "registration": registration,
        "processing_hash": content_hash(registration),
    }
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return payload["processing_hash"]


def load_registration(path):
    payload = json.loads(Path(path).read_text())
    if content_hash(payload["registration"]) != payload["processing_hash"]:
        raise ValueError("pre-registration hash mismatch")
    return payload


def feasibility_report(
    rank,
    diagnostics,
    gauge,
    ambiguous_fraction,
    peptide_interval,
    thresholds,
    real_data=False,
    backend_trial=None,
    capacity_ablation=None,
):
    interpolation = np.mean(
        [
            r["interpretation"] == "interpolation"
            for r in diagnostics["heldout_profiles"]
        ]
    )
    k1 = rank["rank_epsilon"] >= thresholds.minimum_rank
    k3 = ambiguous_fraction <= thresholds.maximum_ambiguous_fraction
    k4 = (
        interpolation <= thresholds.maximum_interpolation_fraction
        and diagnostics["training_expression_rank"]["participation_ratio"]
        >= thresholds.minimum_independent_contrasts
    )
    k2 = (
        None
        if peptide_interval is None
        else peptide_interval[0] > thresholds.peptide_effect_minimum
    )
    passed = bool(
        real_data
        and k1
        and k3
        and k4
        and gauge["complete"]
        and peptide_interval is not None
        and backend_trial is not None
        and capacity_ablation is not None
    )
    return {
        "status": "passed" if passed else "incomplete_or_design_change_required",
        "real_data": real_data,
        "K1": {
            "passed": bool(k1),
            "rank": rank["rank_epsilon"],
            "parameter_budget": thresholds.maximum_parameters,
            "capacity_ablation": capacity_ablation,
        },
        "K2": {"peptides_required": k2, "paired_interval": peptide_interval},
        "K3": {"passed": bool(k3), "ambiguous_fraction": ambiguous_fraction},
        "K4": {"passed": bool(k4), "interpolation_fraction": float(interpolation)},
        "gauge_audit": gauge,
        "backend_trial": backend_trial,
        "metric_targets": thresholds.metric_targets,
    }
