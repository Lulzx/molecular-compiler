"""M11: prioritized experiment recommendations and calibrated score labels."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RuleEnsemble:
    members: tuple
    calibrated: bool = False
    calibration: dict = field(default_factory=dict)
    training_provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.members) < 2:
            raise ValueError(
                "uncertainty requires at least two independently trained members"
            )
        if self.calibrated and not self.calibration.get("passed", False):
            raise ValueError("calibrated label requires a passing coverage check")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    cost: float
    criterion: str
    rule_directions: tuple
    predictions: object  # [ensemble, observations]
    observation_variance: float = 0.01
    available: bool = True
    availability_source: str = ""
    description: str = ""

    def __post_init__(self):
        if self.kind not in {"stimulation", "mutant", "drug", "ExM_panel"}:
            raise ValueError("unknown candidate type")
        if (
            self.cost <= 0
            or self.observation_variance <= 0
            or self.criterion not in {"K1", "K2", "K3", "K4", "M3"}
        ):
            raise ValueError("invalid candidate cost, noise or criterion")
        if not self.rule_directions:
            raise ValueError("each proposal must identify constrained rule directions")
        if self.kind == "mutant" and self.available and not self.availability_source:
            raise ValueError("available mutants require a public strain source")


@dataclass(frozen=True)
class CandidateCatalog:
    candidates: tuple


@dataclass(frozen=True)
class Proposal:
    candidate_id: str
    criterion: str
    rule_directions: tuple
    cost: float
    score: float
    score_per_cost: float
    score_label: str
    description: str


def propose_experiments(posterior, candidates, budget):
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be nonnegative and finite")
    priority = {"K2": 0, "K3": 1, "K1": 2, "K4": 2, "M3": 3}
    ranked = []
    for c in candidates.candidates:
        if c.kind == "mutant" and not c.available:
            continue
        x = np.asarray(c.predictions)
        if x.ndim < 2 or len(x) != len(posterior.members) or not np.all(np.isfinite(x)):
            raise ValueError("candidate predictions must match finite ensemble outputs")
        variance = x.reshape(len(x), -1).var(axis=0, ddof=1)
        # Gaussian entropy approximation to mixture information.
        score = float(0.5 * np.log1p(variance / c.observation_variance).sum())
        proposal = Proposal(
            c.candidate_id,
            c.criterion,
            c.rule_directions,
            c.cost,
            score,
            score / c.cost,
            "approximate_information_gain"
            if posterior.calibrated
            else "sensitivity_proxy",
            c.description,
        )
        ranked.append(
            (priority[c.criterion], -proposal.score_per_cost, c.candidate_id, proposal)
        )
    result, remaining = [], budget
    for _, _, _, p in sorted(ranked):
        if p.cost <= remaining:
            result.append(p)
            remaining -= p.cost
    return result


def calibration_check(
    predictions, observations, classes, tolerance=0.05, includes_observation_noise=False
):
    x, y, classes = (
        np.asarray(predictions),
        np.asarray(observations),
        np.asarray(classes),
    )
    if x.ndim < 2 or len(x) < 2 or x.shape[1:] != y.shape or len(classes) != y.shape[0]:
        raise ValueError("calibration dimensions differ")
    if not includes_observation_noise:
        raise ValueError("coverage test requires M9 observation noise in predictions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("calibration requires finite predictions and observations")
    covered = (y >= x.min(axis=0)) & (y <= x.max(axis=0))
    target = (len(x) - 1) / (len(x) + 1)
    overall = float(covered.mean())
    by_class = {str(c): float(covered[classes == c].mean()) for c in np.unique(classes)}
    passed = abs(overall - target) <= tolerance and all(
        abs(v - target) <= tolerance for v in by_class.values()
    )
    return {
        "passed": passed,
        "target": target,
        "overall": overall,
        "by_class": by_class,
        "interval_label": "ensemble_range",
        "tolerance": tolerance,
        "split": "leave_neuron_out",
    }
