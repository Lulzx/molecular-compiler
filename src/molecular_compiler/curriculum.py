"""M10 workflow: the 6.4 curriculum, the 6.5 deep ensemble and the 12.3 Laplace fallback."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .training import (
    LossWeights,
    energy_distance,
    infer_initial_state,
    laplace_samples,
    mmd,
    statistic_features,
    train,
    trajectory_nll,
)

SENSITIVITY_PROBE = "sensitivity probe"
PROBE_NOTE = "ensemble range, not a credible interval"


@dataclass(frozen=True)
class StageSpec:
    index: int
    data: str
    trained: tuple  # parameter groups added at this stage (cumulative)
    purpose: str


# Section 6.4. Data and trained groups accumulate ("+" rows).
STAGES = (
    StageSpec(0, "kinetics", ("m5",), "kinetic models reproduce source data"),
    StageSpec(1, "windows", ("m4", "m9", "encoder"), "basic dynamics, short windows"),
    StageSpec(2, "perturbations", (), "causal structure (L_pert)"),
    StageSpec(3, "long", (), "long-horizon statistics (L_stat)"),
    StageSpec(4, "states", ("neuromodulation",), "state dependence"),
    StageSpec(5, "second_species", ("species_adapters",), "transfer"),
)
# Reported ablation only (Section 10.2): not in the stage-2 trained set.
PEPTIDERGIC_GROUP = "peptidergic_heads"


@dataclass
class CurriculumData:
    """Data sources by stage; each callable maps the full params to model output.

    kinetics(params) -> scalar loss against in vitro data (stage 0)
    windows: list[ShootingWindow]; predict_window(params, state0, window) -> [N,W]
    perturbations(params) -> list of (simulated, observed) response samples
    long, second_species: (observed [N,L], simulate(params) -> [N,L])
    states: {name: (observed, simulate)} for multiple internal states
    prior(params) -> scalar L_prior (optional)
    """

    kinetics: object | None = None
    windows: object | None = None
    predict_window: object | None = None
    perturbations: object | None = None
    long: object | None = None
    states: dict | None = None
    second_species: object | None = None
    prior: object | None = None

    def missing(self, stage):
        """Name of the data a stage lacks, or None."""
        present = {
            0: self.kinetics is not None,
            1: self.windows is not None and self.predict_window is not None,
            2: self.perturbations is not None,
            3: self.long is not None,
            4: bool(self.states),
            5: self.second_species is not None,
        }
        return None if present[stage] else STAGES[stage].data


@dataclass(frozen=True)
class ExitCondition:
    """The spec gives no numeric exit thresholds; these are explicit, recorded knobs."""

    min_relative_decrease: float = 0.0
    max_final_loss: float | None = None

    def check(self, initial, final):
        if not (np.isfinite(initial) and np.isfinite(final)):
            return False, "non-finite loss"
        decrease = (initial - final) / max(abs(initial), 1e-12)
        if decrease < self.min_relative_decrease:
            return False, f"relative loss decrease {decrease:.3g} below required"
        if self.max_final_loss is not None and final > self.max_final_loss:
            return False, f"final loss {final:.3g} above {self.max_final_loss:.3g}"
        return True, "met"


@dataclass
class CurriculumResult:
    params: dict
    stages: list = field(default_factory=list)
    completed: bool = False
    stopped_at: int | None = None
    stop_reason: str | None = None


def _stat_term(simulated, observed):
    return mmd(statistic_features(simulated), statistic_features(observed))


def stage_loss(params, stage, data, weights, noise_sd):
    """Cumulative objective through `stage`, built from the existing loss terms."""
    zero = jnp.array(0.0)
    terms = {"kinetics": zero, "trajectory": zero, "statistics": zero}
    terms.update(perturbation=zero, prior=zero)
    if stage == 0:
        terms["kinetics"] = jnp.asarray(data.kinetics(params))
        return terms["kinetics"], terms
    terms["trajectory"] = jnp.mean(
        jnp.stack(
            [
                trajectory_nll(
                    data.predict_window(
                        params, infer_initial_state(params["encoder"], w.preceding), w
                    ),
                    w.observed,
                    noise_sd,
                )
                for w in data.windows
            ]
        )
    )
    if stage >= 2:
        pairs = data.perturbations(params)
        terms["perturbation"] = jnp.mean(
            jnp.stack([energy_distance(a, b) for a, b in pairs])
        )
    stats = []
    if stage >= 3:
        observed, simulate = data.long
        stats.append(_stat_term(simulate(params), observed))
    if stage >= 4:
        stats += [_stat_term(sim(params), obs) for obs, sim in data.states.values()]
    if stage >= 5:
        observed, simulate = data.second_species
        stats.append(_stat_term(simulate(params), observed))
    if stats:
        terms["statistics"] = jnp.mean(jnp.stack(stats))
    if data.prior is not None:
        terms["prior"] = jnp.asarray(data.prior(params))
    total = (
        sum(getattr(weights, k) * terms[k] for k in weights.__dict__)
        + terms["kinetics"]
    )
    return total, terms


def _trained_groups(stage, peptidergic=False):
    groups = [g for spec in STAGES[: stage + 1] for g in spec.trained]
    return groups + [PEPTIDERGIC_GROUP] if peptidergic else groups


def _fit_stage(params, stage, data, groups, weights, noise_sd, steps, lr, seed):
    frozen = {k: v for k, v in params.items() if k not in groups}

    def loss(sub, _):
        return stage_loss({**frozen, **sub}, stage, data, weights, noise_sd)[0]

    sub = {g: params[g] for g in groups}
    sub, history = train(
        sub, loss, steps=steps, learning_rate=lr, weight_decay=0, seed=seed
    )
    return {**frozen, **sub}, history


def run_curriculum(
    params,
    data,
    last_stage=5,
    steps=100,
    learning_rate=1e-3,
    weights=None,
    noise_sd=0.1,
    exits=None,
    peptidergic_ablation=False,
    seed=0,
):
    """Run stages 0..last_stage in order; stop at the first stage that fails to exit.

    `params` is a dict of parameter groups (m4, m5, m9, encoder, neuromodulation,
    species_adapters, optionally peptidergic_heads). A stage whose data or
    parameter groups are absent, or whose exit condition is unmet, stops the
    curriculum with the reason recorded; later stages are never run.
    """
    weights = weights or LossWeights()
    exits = exits or {}
    result = CurriculumResult(dict(params))

    def stop(stage, record, reason):
        record.update(exit_met=False, reason=reason)
        result.stages.append(record)
        result.stopped_at, result.stop_reason = stage, reason

    for stage in range(last_stage + 1):
        spec = STAGES[stage]
        groups = _trained_groups(stage)
        record = {
            "stage": stage,
            "data": spec.data,
            "trained": groups,
            "purpose": spec.purpose,
        }
        if (missing := data.missing(stage)) is not None:
            stop(stage, record, f"missing data: {missing}")
            return result
        if absent := [g for g in groups if g not in result.params]:
            stop(stage, record, f"missing parameter groups: {absent}")
            return result
        initial = float(stage_loss(result.params, stage, data, weights, noise_sd)[0])
        updated, history = _fit_stage(
            result.params,
            stage,
            data,
            groups,
            weights,
            noise_sd,
            steps,
            learning_rate,
            seed + stage,
        )
        final, terms = stage_loss(updated, stage, data, weights, noise_sd)
        record.update(
            initial_loss=initial,
            final_loss=float(final),
            terms={k: float(v) for k, v in terms.items()},
            history=history,
        )
        if stage == 2 and peptidergic_ablation:
            record["ablation"] = _ablation(
                result.params,
                stage,
                data,
                weights,
                noise_sd,
                steps,
                learning_rate,
                seed,
            )
        met, reason = exits.get(stage, ExitCondition()).check(initial, float(final))
        record.update(exit_met=met, reason=reason)
        result.stages.append(record)
        if not met:
            result.stopped_at, result.stop_reason = stage, reason
            return result
        result.params = updated
    result.completed = True
    return result


def _ablation(params, stage, data, weights, noise_sd, steps, lr, seed):
    """Reported-only stage-2 variant with peptidergic heads trained as well."""
    if PEPTIDERGIC_GROUP not in params:
        return {"name": PEPTIDERGIC_GROUP, "run": False, "reason": "no such group"}
    groups = _trained_groups(stage, peptidergic=True)
    updated, _ = _fit_stage(
        params, stage, data, groups, weights, noise_sd, steps, lr, seed + stage
    )
    return {
        "name": PEPTIDERGIC_GROUP,
        "run": True,
        "final_loss": float(stage_loss(updated, stage, data, weights, noise_sd)[0]),
    }


def ensemble_range(predictions):
    """Per-element min/max/mean/std across members; predictions are [K,...]."""
    p = jnp.asarray(predictions)
    return {
        "min": p.min(axis=0),
        "max": p.max(axis=0),
        "mean": p.mean(axis=0),
        "std": p.std(axis=0),
    }


def nominal_coverage(k):
    """P(held-out value inside min-max of k exchangeable members) = (k-1)/(k+1)."""
    return (k - 1) / (k + 1)


def coverage(predictions, observed, classes=None, mask=None):
    """Fraction of held-out observations inside the members' range, overall/per class.

    predictions [K,N,L]; observed [N,L]; classes [N] labels; mask [N] bool.
    """
    p, y = np.asarray(predictions), np.asarray(observed)
    inside = (y >= p.min(axis=0)) & (y <= p.max(axis=0))
    keep = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, bool)
    per_class = {}
    if classes is not None:
        labels = np.asarray(classes)
        for c in np.unique(labels[keep]):
            per_class[c.item()] = float(inside[keep & (labels == c)].mean())
    return float(inside[keep].mean()), per_class


def coverage_check(predictions, observed, classes=None, mask=None, tolerance=0.05):
    """Section 6.5: pass if coverage is within 5 points of nominal, overall and per class."""
    nominal = nominal_coverage(len(predictions))
    overall, per_class = coverage(predictions, observed, classes, mask)
    deviations = [abs(overall - nominal)] + [
        abs(v - nominal) for v in per_class.values()
    ]
    return {
        "k": len(predictions),
        "nominal": nominal,
        "overall": overall,
        "per_class": per_class,
        "tolerance": tolerance,
        "passed": bool(max(deviations) <= tolerance),
    }


def train_ensemble(init_params, data, k=5, seed=0, **curriculum):
    """K independently initialised, independently data-ordered curriculum runs.

    `init_params(seed) -> params`; member i uses seed + i for both. Only members
    that complete the requested curriculum are usable for prediction.
    """
    if k < 2:
        raise ValueError("an ensemble needs at least two members")
    members = []
    for i in range(k):
        run = run_curriculum(init_params(seed + i), data, seed=seed + i, **curriculum)
        members.append({"seed": seed + i, "result": run, "usable": run.completed})
    return members


def _with_noise(predictions, noise_sd, seed):
    """Independent M9 observation noise per member (6.5 calibration predictions)."""
    if noise_sd <= 0:
        return predictions
    noise = noise_sd * jax.random.normal(jax.random.key(seed), predictions.shape)
    return predictions + noise


def calibrate_ensemble(
    members,
    predict,
    observed,
    classes=None,
    noise_sd=0.0,
    head_key="m4",
    residual_fn=None,
    laplace_draws=1,
    laplace_prior_precision=1.0,
    seed=0,
):
    """Coverage check on held-out data, then the 12.3 Laplace fallback if it fails.

    `predict(params) -> [N,L]` on held-out neurons (leave_neuron_out);
    `residual_fn(params) -> residuals` is differentiable in params[head_key]. The
    fallback replaces each member by `laplace_draws` GGN-Laplace draws over its head
    weights and repeats the check on the mixture; nominal coverage is computed from
    the number of mixture components (spec K=5 value when `laplace_draws` is 1).
    """
    usable = [m for m in members if m["usable"]]
    if len(usable) < 2:
        raise ValueError("fewer than two usable ensemble members")

    def stack(param_sets):
        preds = jnp.stack([predict(p) for p in param_sets])
        return _with_noise(preds, noise_sd, seed)

    member_params = [m["result"].params for m in usable]
    base = stack(member_params)
    report = {
        "label": SENSITIVITY_PROBE,
        "interval": PROBE_NOTE,
        "k_members": len(members),
        "k_usable": len(usable),
        "check": coverage_check(base, observed, classes),
        "range": ensemble_range(base),
        "fallback": None,
        "calibrated": False,
    }
    if report["check"]["passed"]:
        report["calibrated"] = True
    elif residual_fn is not None:
        mixture = []
        for i, params in enumerate(member_params):
            draws = laplace_samples(
                params[head_key],
                lambda head, params=params: residual_fn({**params, head_key: head}),
                count=laplace_draws,
                seed=seed + i,
                prior_precision=laplace_prior_precision,
            )
            mixture += [{**params, head_key: head} for head in draws]
        mixed = stack(mixture)
        report["fallback"] = {
            "method": "per-member GGN Laplace over head weights",
            "head": head_key,
            "components": len(mixture),
            "check": coverage_check(mixed, observed, classes),
            "range": ensemble_range(mixed),
        }
        report["calibrated"] = report["fallback"]["check"]["passed"]
    # Until a method passes the check the ensemble stays a sensitivity probe.
    if report["calibrated"]:
        report["label"] = "calibrated ensemble (passed Section 6.5 check)"
    return report
