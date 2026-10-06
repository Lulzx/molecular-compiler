"""M12: held-out, residual-free reports with paired cluster bootstrap gates."""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import rankdata

from .baselines import synapse_only
from .compiler import compile
from .observation import ObservationModel, observe
from .provenance import content_hash, require_publishable
from .simulation import simulate
from .training import energy_distance, statistic_features

METRICS = {
    "spontaneous_correlation",
    "spontaneous_psd",
    "spontaneous_occupancy",
    "stimulus_explained_variance",
    "perturbation_detection",
    "perturbation_sign",
    "perturbation_amplitude",
    "perturbation_latency",
    "perturbation_state",
}


def paired_bootstrap(differences, seed=0, samples=2000):
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "mean": float(values.mean()) if len(values) else None,
            "interval": None,
            "n_neurons": len(values),
        }
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "n_neurons": len(values),
    }


def paired_statistic_bootstrap(
    predicted,
    baseline,
    observed,
    metric,
    ceiling,
    seed=0,
    samples=2000,
    response_threshold=0.01,
    clusters=None,
):
    predicted, baseline, observed = map(np.asarray, (predicted, baseline, observed))
    if (
        predicted.shape != baseline.shape
        or predicted.shape != observed.shape
        or predicted.ndim != 1
    ):
        raise ValueError("paired statistic amplitudes must be aligned vectors")

    def statistic(values):
        if metric == "perturbation_detection":
            return auroc(
                np.abs(observed[values]) > response_threshold, np.abs(predicted[values])
            ), auroc(
                np.abs(observed[values]) > response_threshold, np.abs(baseline[values])
            )
        if metric == "perturbation_amplitude":
            return _correlation(predicted[values], observed[values]), _correlation(
                baseline[values], observed[values]
            )
        raise ValueError("unknown global perturbation statistic")

    clusters = np.arange(len(predicted)) if clusters is None else np.asarray(clusters)
    if clusters.shape != predicted.shape:
        raise ValueError("bootstrap clusters must align with amplitudes")
    unique = np.unique(clusters)
    if len(unique) < 2 or ceiling is None or not np.isfinite(ceiling) or ceiling <= 0:
        return {"mean": None, "interval": None, "n_clusters": len(unique)}
    a, b = statistic(np.arange(len(predicted)))
    point = None if a is None or b is None else (a - b) / ceiling
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == label) for label in drawn])
        a, b = statistic(indices)
        if a is not None and b is not None:
            draws.append((a - b) / ceiling)
    return {
        "mean": point,
        "interval": np.quantile(draws, [0.025, 0.975]).tolist()
        if len(draws) >= samples // 2
        else None,
        "n_clusters": len(unique),
        "valid_bootstrap_samples": len(draws),
    }


def auroc(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    positive, negative = labels.sum(), (~labels).sum()
    if not positive or not negative:
        return None
    rank = rankdata(scores)
    return float(
        (rank[labels].sum() - positive * (positive + 1) / 2) / (positive * negative)
    )


def split_half_ceiling(trials, seed=0):
    trials = np.asarray(trials, dtype=float)
    if len(trials) < 4:
        raise ValueError("noise ceiling requires at least four independent repeats")
    order = np.random.default_rng(seed).permutation(len(trials))
    a, b = (
        trials[order[::2]].mean(axis=0).ravel(),
        trials[order[1::2]].mean(axis=0).ravel(),
    )
    if np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("noise ceiling undefined for constant recordings")
    r = np.corrcoef(a, b)[0, 1]
    return float(2 * r / (1 + r))


def wired_reachability(graph, target_id):
    adjacency = {int(neuron): [] for neuron in graph.neuron_ids}
    for edge in graph.connectome.synapses.to_pylist():
        adjacency[edge["pre_id"]].append(edge["post_id"])
    for edge in graph.connectome.contacts.to_pylist():
        if edge["gap_junction_observed"] is True:
            adjacency[edge["i_id"]].append(edge["j_id"])
            adjacency[edge["j_id"]].append(edge["i_id"])
    reached, pending = set(), [target_id]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(adjacency[node])
    return np.array([int(i) in reached for i in graph.neuron_ids])


def _correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def score_recordings(
    predictions,
    observations,
    noise_ceilings,
    dt_s,
    response_threshold=0.01,
    sign_ambiguous=None,
    wired=None,
):
    p, o = np.asarray(predictions), np.asarray(observations)
    if p.ndim == 2:
        p = p[None]
    if o.ndim == 2:
        o = o[None]
    if (
        p.shape[1:] != o.shape[1:]
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(o))
    ):
        raise ValueError("evaluation recording shape or values invalid")
    pm, om = p.mean(axis=0), o.mean(axis=0)
    n, length = pm.shape
    ps, os = np.asarray(statistic_features(pm)), np.asarray(statistic_features(om))
    correlation_distance = np.mean((ps[:, :n] - os[:, :n]) ** 2, axis=1)
    psd_end = n + length // 2 + 1
    psd_distance = np.mean((ps[:, n:psd_end] - os[:, n:psd_end]) ** 2, axis=1)
    p_occ, o_occ = ps[:, psd_end:], os[:, psd_end:]
    midpoint = (p_occ + o_occ) / 2
    occupancy_distance = 0.5 * np.sum(
        p_occ * np.log(np.maximum(p_occ, 1e-12) / np.maximum(midpoint, 1e-12))
        + o_occ * np.log(np.maximum(o_occ, 1e-12) / np.maximum(midpoint, 1e-12)),
        axis=1,
    )
    ev = 1 - np.mean((pm - om) ** 2, axis=1) / np.maximum(np.var(om, axis=1), 1e-12)
    pa, oa = pm.mean(axis=1), om.mean(axis=1)
    responsive = np.abs(oa) > response_threshold
    correct_sign = (np.sign(pa) == np.sign(oa)).astype(float)

    def latency(y):
        amplitudes = np.max(np.abs(y), axis=1)
        crossing = np.abs(y) >= 0.1 * amplitudes[:, None]
        return np.argmax(crossing, axis=1) * dt_s

    latency_score = -np.abs(latency(pm) - latency(om))
    state_score = -np.asarray(
        [float(energy_distance(p[:, i], o[:, i])) for i in range(n)]
    )
    raw = {
        "spontaneous_correlation": -correlation_distance,
        "spontaneous_psd": -psd_distance,
        "spontaneous_occupancy": -occupancy_distance,
        "stimulus_explained_variance": ev,
        "perturbation_sign": np.where(responsive, correct_sign, np.nan),
        "perturbation_latency": np.where(responsive, latency_score, np.nan),
        "perturbation_state": state_score,
    }
    metrics, per_neuron, raw_metrics = {}, {}, {}
    errors = {
        "spontaneous_correlation",
        "spontaneous_psd",
        "spontaneous_occupancy",
        "perturbation_latency",
        "perturbation_state",
    }
    for name, values in raw.items():
        raw_metrics[name] = (
            float(np.nanmean(values)) * (-1 if name in errors else 1)
            if np.any(np.isfinite(values))
            else None
        )
    for name, values in raw.items():
        ceiling = noise_ceilings.get(name)
        if ceiling is None or not np.isfinite(ceiling) or ceiling <= 0:
            metrics[name] = None
            continue
        normalized = values / ceiling
        per_neuron[name] = normalized
        metrics[name] = (
            float(np.nanmean(normalized)) if np.any(np.isfinite(normalized)) else None
        )
    for name, value in (
        ("perturbation_detection", auroc(responsive, np.abs(pa))),
        ("perturbation_amplitude", _correlation(pa, oa)),
    ):
        raw_metrics[name] = value
        ceiling = noise_ceilings.get(name)
        metrics[name] = (
            value / ceiling
            if value is not None
            and ceiling is not None
            and np.isfinite(ceiling)
            and ceiling > 0
            else None
        )
    # AUROC and amplitude correlation are ensemble statistics; keep their
    # paired bootstrap separate, rather than inventing per-neuron AUROCs.
    groups = {}
    for group_name, assignment in (
        ("sign_ambiguous", sign_ambiguous),
        ("wired_path", wired),
    ):
        if assignment is None:
            continue
        assignment = np.asarray(assignment, dtype=bool)
        groups[group_name] = {}
        for value in (False, True):
            mask = (assignment == value) & responsive
            ceiling = noise_ceilings.get("perturbation_sign")
            groups[group_name][str(value)] = (
                float(correct_sign[mask].mean() / ceiling)
                if mask.any() and ceiling and ceiling > 0
                else None
            )
    return {
        "raw_metrics": raw_metrics,
        "metric_units": {
            name: (
                "s"
                if name == "perturbation_latency"
                else "relative_fluorescence"
                if name == "perturbation_state"
                else "nats"
                if name == "spontaneous_occupancy"
                else "dimensionless"
            )
            for name in METRICS
        },
        "normalization": {
            "scales": dict(noise_ceilings),
            "score_direction": "higher_is_better",
            "negated_error_metrics": sorted(errors),
        },
        "metrics": metrics,
        "per_neuron": per_neuron,
        "sign_groups": groups,
        "amplitudes": (pa, oa),
        "responsive": responsive,
    }


@dataclass
class EvaluationCase:
    case_id: str
    graph: object
    stimulus: object
    duration_s: float
    observations: object
    noise_ceilings: dict
    observation_model: ObservationModel = field(default_factory=ObservationModel)
    target_id: int | None = None
    state: object | None = None
    resolution: object | None = None
    baselines: dict = field(default_factory=dict)  # id -> [trial,N,L]
    baseline_provenance: dict = field(default_factory=dict)
    heldout_neuron_ids: tuple = ()
    response_start: int = 0
    animal_state: str = "default"


@dataclass
class Split:
    name: str
    cases: tuple
    training_entities: tuple
    heldout_entities: tuple
    class_partition: dict
    registration_hash: str
    diagnostics: dict
    metric_targets: dict
    real_data: bool = False
    gauge_audit: dict = field(default_factory=dict)

    def definition(self):
        return {
            "name": self.name,
            "training_entities": list(self.training_entities),
            "heldout_entities": list(self.heldout_entities),
            "class_partition": self.class_partition,
            "metric_targets": self.metric_targets,
            "cases": [
                {
                    "case_id": c.case_id,
                    "data_hash": c.graph.metadata["processing_hash"],
                    "heldout_neuron_ids": list(c.heldout_neuron_ids),
                    "target_id": c.target_id,
                    "animal_state": c.animal_state,
                    "response_start": c.response_start,
                }
                for c in self.cases
            ],
        }

    def freeze(self, path):
        from .phase0 import freeze_registration

        self.registration_hash = freeze_registration(path, self.definition())
        return self.registration_hash

    def validate(self):
        if self.name not in {
            "leave_animal_out",
            "leave_neuron_out",
            "leave_class_out",
            "leave_state_out",
            "leave_species_out",
        }:
            raise ValueError("unknown split")
        if set(self.training_entities) & set(self.heldout_entities):
            raise ValueError("training/held-out leakage")
        if (
            not self.registration_hash
            or not self.class_partition
            or not self.diagnostics
        ):
            raise ValueError(
                "evaluation requires frozen splits, classes and diagnostics"
            )
        if not self.cases or not self.heldout_entities:
            raise ValueError("empty held-out evaluation")
        if self.real_data and content_hash(self.definition()) != self.registration_hash:
            raise ValueError(
                "real-data split must match its immutable pre-registration hash"
            )
        for case in self.cases:
            neurons = case.graph.connectome.neurons.to_pylist()
            if self.name == "leave_class_out":
                held = set(case.heldout_neuron_ids)
                class_for = {
                    r["neuron_id"]: self.class_partition.get(
                        str(r["neuron_id"]), r["type_label"]
                    )
                    for r in neurons
                }
                for label in self.heldout_entities:
                    members = {
                        neuron for neuron, cls in class_for.items() if cls == label
                    }
                    if not members or not members <= held:
                        raise ValueError(
                            "entire classes, including bilateral homologs, must be held out"
                        )
                if any(class_for.get(i) not in self.heldout_entities for i in held):
                    raise ValueError("held-out scoring contains a training class")
            elif self.name == "leave_neuron_out":
                if (
                    not set(case.heldout_neuron_ids) <= set(self.heldout_entities)
                    or not case.heldout_neuron_ids
                ):
                    raise ValueError("held-out neuron mask differs from split")
            elif (
                self.name == "leave_animal_out"
                and case.graph.metadata["animal_id"] not in self.heldout_entities
            ):
                raise ValueError("case animal is not held out")
            elif (
                self.name == "leave_species_out"
                and case.graph.metadata["species"] not in self.heldout_entities
            ):
                raise ValueError("case species is not held out")
            elif (
                self.name == "leave_state_out"
                and case.animal_state not in self.heldout_entities
            ):
                raise ValueError("case state is not held out")


@dataclass
class EvalReport:
    metadata: dict
    cases: list
    comparisons: dict
    accepted: bool
    missing_gates: list

    def to_dict(self):
        return {
            "metadata": self.metadata,
            "cases": self.cases,
            "comparisons": self.comparisons,
            "accepted": self.accepted,
            "missing_gates": self.missing_gates,
        }

    def require_publishable(self):
        require_publishable(self.metadata)


def evaluate(rules, kinetics, split, benchmarks):
    split.validate()
    if not benchmarks or not set(benchmarks) <= METRICS:
        raise ValueError("unknown or empty benchmark metrics")
    case_reports, differences, global_comparisons = [], {}, {}
    missing = []
    if not rules.training_provenance.get("training_hash"):
        missing.append("ensemble training provenance unavailable")
    if not split.gauge_audit.get("complete", False):
        missing.append("Phase 0 gauge audit incomplete")
    if any(member.budget is None for member in rules.members):
        missing.append("absolute parameter budget not frozen")
    tier = "open"
    hashes, attributions = [], set()
    # Evaluation takes fresh compile results, so attached per-animal residuals
    # and adapters can never leak into headline predictions.
    for case in split.cases:
        predictions = []
        b4_predictions = []
        samples = 8
        for member in rules.members:
            member_predictions, member_b4 = [], []
            for identity in range(samples):
                sim = compile(
                    case.graph,
                    replace(member, adapter=None),
                    kinetics,
                    state=case.state,
                    resolution=case.resolution,
                    identity_sample=identity,
                )
                traj = simulate(sim, case.stimulus, case.duration_s)
                member_predictions.append(
                    np.asarray(observe(traj, case.observation_model).y)
                )
                if "B4" not in case.baselines:
                    member_b4.append(
                        np.asarray(
                            observe(
                                simulate(
                                    synapse_only(sim), case.stimulus, case.duration_s
                                ),
                                case.observation_model,
                            ).y
                        )
                    )
            predictions.append(np.mean(member_predictions, axis=0))
            if member_b4:
                b4_predictions.append(np.mean(member_b4, axis=0))
        prediction = np.asarray(predictions)
        observed = np.asarray(case.observations)
        if observed.ndim == 2:
            observed = observed[None]
        indices = (
            [list(case.graph.neuron_ids).index(i) for i in case.heldout_neuron_ids]
            if case.heldout_neuron_ids
            else list(range(len(case.graph.neuron_ids)))
        )
        if not 0 <= case.response_start < prediction.shape[-1]:
            raise ValueError("invalid response analysis window")

        def select(values, observed=observed, indices=indices, case=case):
            values = np.asarray(values)
            if values.ndim == 2:
                values = values[None]
            if values.shape[1:] != observed.shape[1:]:
                raise ValueError(
                    "baseline recordings are not aligned to evaluation case"
                )
            selected = values[:, indices]
            if case.response_start:
                selected = selected - selected[:, :, : case.response_start].mean(
                    axis=2, keepdims=True
                )
            return selected[:, :, case.response_start :]

        ambiguity = np.zeros(len(case.graph.neuron_ids), dtype=bool)
        if case.target_id is not None:
            for e, row in enumerate(case.graph.connectome.synapses.to_pylist()):
                if row["pre_id"] == case.target_id:
                    # Match by synapse_id because compilation changes row order.
                    linked = np.flatnonzero(
                        np.asarray(sim.syn_params["synapse_ids"]) == row["synapse_id"]
                    )[0]
                    ambiguity[list(case.graph.neuron_ids).index(row["post_id"])] |= (
                        bool(sim.syn_params["sign_ambiguous"][linked])
                    )
        wired = (
            wired_reachability(case.graph, case.target_id)[indices]
            if case.target_id is not None
            else None
        )
        scored = score_recordings(
            select(prediction),
            select(observed),
            case.noise_ceilings,
            sim.resolution.dt_s,
            sign_ambiguous=ambiguity[indices],
            wired=wired,
        )
        baseline_values = {**case.baselines}
        if b4_predictions:
            baseline_values["B4"] = np.asarray(b4_predictions)
        for baseline in ("B0", "B1", "B2", "B4", "B5"):
            if baseline not in baseline_values:
                missing.append(f"{case.case_id}: {baseline} comparison missing")
        baseline_scores = {}
        neuron_ids = [int(case.graph.neuron_ids[i]) for i in indices]
        neuron_classes = {
            r["neuron_id"]: r["type_label"]
            for r in case.graph.connectome.neurons.to_pylist()
        }
        clusters = [
            split.class_partition.get(str(i), neuron_classes[i])
            if split.name == "leave_class_out"
            else str(i)
            for i in neuron_ids
        ]
        for baseline, values in baseline_values.items():
            if baseline in {"B1", "B2", "B5"} and not case.baseline_provenance.get(
                baseline, {}
            ).get("training_hash"):
                missing.append(
                    f"{case.case_id}: {baseline} training provenance missing"
                )
            score = score_recordings(
                select(values),
                select(observed),
                case.noise_ceilings,
                sim.resolution.dt_s,
                sign_ambiguous=ambiguity[indices],
                wired=wired,
            )
            baseline_scores[baseline] = score["metrics"]
            for metric in benchmarks:
                if metric in {"perturbation_detection", "perturbation_amplitude"}:
                    result = paired_statistic_bootstrap(
                        scored["amplitudes"][0],
                        score["amplitudes"][0],
                        scored["amplitudes"][1],
                        metric,
                        case.noise_ceilings.get(metric),
                        clusters=clusters,
                    )
                    global_comparisons.setdefault((baseline, metric), []).append(result)
                    continue
                if (
                    metric not in scored["per_neuron"]
                    or metric not in score["per_neuron"]
                ):
                    missing.append(
                        f"{case.case_id}: paired {metric} comparison unavailable"
                    )
                    continue
                values = scored["per_neuron"][metric] - score["per_neuron"][metric]
                differences.setdefault((baseline, metric), []).extend(
                    zip(clusters, neuron_ids, values)
                )
        case_reports.append(
            {
                "case_id": case.case_id,
                "metrics": scored["metrics"],
                "raw_metrics": scored["raw_metrics"],
                "metric_units": scored["metric_units"],
                "normalization": scored["normalization"],
                "baselines": baseline_scores,
                "sign_groups": scored["sign_groups"],
                "sign_ambiguous_fraction": float(np.mean(ambiguity)),
                "boundary_condition": traj.metadata["boundary_condition"],
                "sigma2": rules.training_provenance.get("sigma2"),
                "gauge": observe(traj, case.observation_model).metadata["gauge"],
                "data_processing_hash": case.graph.metadata["processing_hash"],
                "max_solve_residual": traj.metadata["max_solve_residual"],
                "scored_neuron_ids": [int(case.graph.neuron_ids[i]) for i in indices],
            }
        )
        hashes.append(case.graph.metadata["processing_hash"])
        attributions.update(sim.metadata["attributions"])
        if sim.metadata["tier"] == "restricted":
            tier = "restricted"
    comparisons = {}
    for (baseline, metric), pairs in differences.items():
        # Average repeats within a neuron, then neurons within a class.
        by_cluster = {}
        for cluster, neuron, value in pairs:
            by_cluster.setdefault(cluster, {}).setdefault(neuron, []).append(value)
        contrasts = [
            np.mean([np.mean(v) for v in neurons.values()])
            for neurons in by_cluster.values()
        ]
        result = paired_bootstrap(contrasts)
        result["n_clusters"] = result.pop("n_neurons")
        result["cluster_unit"] = (
            "class" if split.name == "leave_class_out" else "neuron"
        )
        comparisons[f"{baseline}:{metric}"] = result
    for (baseline, metric), results in global_comparisons.items():
        if len(results) == 1:
            comparisons[f"{baseline}:{metric}"] = results[0]
        else:
            # A pooled neuron/state bootstrap needs the repeated-trial catalog.
            comparisons[f"{baseline}:{metric}"] = {
                "mean": None,
                "interval": None,
                "case_intervals": results,
                "reason": "multiple cases require a joint clustered statistic bootstrap",
            }
    for baseline in ("B0", "B1", "B2"):
        for metric in benchmarks:
            result = comparisons.get(f"{baseline}:{metric}")
            if not result or result["interval"] is None or result["interval"][0] <= 0:
                missing.append(
                    f"{baseline}:{metric} does not clear paired 95% interval"
                )
    for metric in benchmarks:
        target = split.metric_targets.get(metric)
        scores = [r["metrics"].get(metric) for r in case_reports]
        if target is None or any(score is None or score < target for score in scores):
            missing.append(f"{metric}: frozen numerical target unavailable or unmet")
    if not split.real_data:
        missing.append("synthetic data cannot establish scientific acceptance")
    if split.name not in {"leave_class_out", "leave_neuron_out"}:
        missing.append("Phase 1 acceptance requires held-out neuron and class results")
    if tier != "open":
        missing.append("headline results require open-tier data")
    metadata = {
        "split": split.name,
        "registration_hash": split.registration_hash,
        "residuals_enabled": False,
        "identity_samples": 8,
        "ensemble_size": len(rules.members),
        "uncertainty_label": "calibrated_ensemble"
        if rules.calibrated
        else "sensitivity_probe",
        "interval_label": "ensemble_range",
        "split_diagnostics": split.diagnostics,
        "gauge_audit": split.gauge_audit,
        "tier": tier,
        "processing_hash": content_hash(hashes),
        "data_processing_hashes": hashes,
        "attributions": sorted(attributions),
        "kinetics_deviations": kinetics.deviation_report(),
        "training_provenance": rules.training_provenance,
    }
    return EvalReport(
        metadata, case_reports, comparisons, not missing, sorted(set(missing))
    )
