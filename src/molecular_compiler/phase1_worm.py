"""Phase 1: the full M4-M9 compiler on C. elegans perturbation responses.

Targets, folds, labels, ceilings and baselines are the frozen Phase 0 ones
(Stage A), so the comparison with B0, B1 and B2 is like for like. The compiler
prediction comes from simulating the compiled network instead of the linear
response.

Declared approximations (Section 10.3):
- one compartment per neuron at dt = 10 ms in float64, chosen by the
  Section 12.6 convergence test on this graph;
- the curated kinetics library of `worm_kinetics` (one m and one h gate);
- connectome sizes are EM section counts, converted with a fixed conductance
  per section;
- the pre-stimulus state is a 20 s relaxation from rest, treated as fixed
  for gradients (stop_gradient);
- fluorescence F = F_basal + Hill(Ca) with F_basal = 0.1 and no indicator
  kernel (GCaMP6s filtering changes a 30 s mean by under 3%); calcium units
  are arbitrary, so the half-saturation is the median resting calcium of the
  initial model (computed once, from the model alone, never from data);
- basal [Cl-]i = 5 mM, the lower bound of the policy prior (unmeasured; with
  10 mM, E_Cl lies above rest and anion synapses depolarize, see K3);
- Section 6.3 drive gauge as in Phase 0: each simulated column is divided by
  the simulated self-response and multiplied by the observed autoresponse,
  with one global gain; simulated self-responses are floored at 0.01;
- stimulation is a 0.5 s, 200 pA current step;
- residuals are off.
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .phase0_worm import (
    FOLDS,
    SEED,
    _dump,
    align_atlas,
    autoresponse,
    fold_masks,
    observed_mask,
)

DT_S = 0.01
WARMUP_S = 20.0
PULSE_S = 0.5
POST_S = 30.0
DRIVE_PA = 200.0
F_BASAL = 0.1
SYNAPSE_UNIT_NS = 0.01
CHLORIDE_MM = 5.0
SELF_FLOOR = 0.01
SEGMENT_STEPS = 50
GAP_UNIT_NS = 0.05
D_Z = 2
TRAINABLE = ("density", "context", "gap", "bias")


def policy():
    from .compiler import ResolutionPolicy
    from .worm_kinetics import PASSIVE

    return replace(
        ResolutionPolicy.default(),
        n_comp=1,
        dt_s=DT_S,
        solver="pcg",
        synapse_unit_nS=SYNAPSE_UNIT_NS,
        gap_unit_nS=GAP_UNIT_NS,
        intracellular_mM=(15.0, 140.0, CHLORIDE_MM, 0.0001),
        **PASSIVE,
    )


def frozen_basis(graph, seed=SEED):
    """PLM PCA projection (k = 2, Stage B) and identity encoder, frozen."""
    from .rules import RuleNetwork

    embeddings = np.asarray(
        [g["plm_embedding"] for g in graph.molecular.genes.to_pylist()], float
    )
    centered = embeddings - embeddings.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    projection = vt[:D_Z].T / (s[:D_Z] / np.sqrt(len(embeddings)))
    base = RuleNetwork.initialize(seed=seed, d_z=D_Z)
    frozen = {
        **base.params,
        "projection": jnp.asarray(projection),
        "encoder": jnp.eye(D_Z),
        "stp": jnp.zeros_like(base.params["stp"]),
        "receptor_features": jnp.zeros_like(base.params["receptor_features"]),
    }
    return frozen


def setup(project, budget):
    from .provenance import DataRegister, inherit
    from .rules import RuleNetwork
    from .worm_kinetics import build_worm_library
    from .worm_public import load_atlas, load_worm_project

    graph, manifest = load_worm_project(project)
    n = len(manifest["neuron_names"])
    atlas = align_atlas(load_atlas(Path(project) / "signal_propagation.npz"), n)
    curation = DataRegister.load("data-register.json").require(
        "phase1_kinetics_curation"
    )
    library = build_worm_library(
        graph, manifest["gene_names"], inherit([curation], "phase1")
    )
    frozen = frozen_basis(graph)
    rules = RuleNetwork.initialize_partial(
        frozen, {k: frozen[k] for k in TRAINABLE}, budget
    )
    return graph, manifest, atlas, library, rules


def make_simulator(graph, rules, library, pol):
    """Return f(params, nuisance, columns) -> predicted dF/F0 [N, len(columns)]."""
    from .compiler import compile
    from .simulation import _sequential, initial_state

    n = len(graph.neuron_ids)
    warm_steps = round(WARMUP_S / DT_S)
    pulse_steps = round(PULSE_S / DT_S)
    post_steps = round(POST_S / DT_S)

    def relax(sim):
        return _sequential(sim, initial_state(sim), jnp.zeros((warm_steps, n)))[0]

    initial = relax(compile(graph, rules, library, resolution=pol))
    half_saturation = float(np.median(np.maximum(np.asarray(initial.calcium), 0)))

    def fluorescence(calcium):
        power = jnp.maximum(calcium, 0) ** 2
        return F_BASAL + power / (half_saturation**2 + power)

    total_steps = pulse_steps + post_steps
    if total_steps % SEGMENT_STEPS:
        raise ValueError("response window must be a whole number of segments")

    def responses(params, columns):
        # Rest is a fixed point: relax with parameters cut from the tape so no
        # tangent or residual flows through the warmup.
        frozen = rules.with_params(jax.lax.stop_gradient(params))
        rest = relax(compile(graph, frozen, library, resolution=pol))
        sim = compile(graph, rules.with_params(params), library, resolution=pol)
        baseline = fluorescence(rest.calcium.reshape(n, -1)[:, 0])

        def one_step(state, current):
            from .simulation import step

            new, out = step(sim, state, current)
            return new, fluorescence(out[1])

        # Two-level checkpointing: only segment boundaries are stored, and each
        # segment is recomputed step by step during the backward pass.
        @jax.checkpoint
        def segment(state, currents):
            return jax.lax.scan(jax.checkpoint(one_step), state, currents)

        def column(target):
            drive = jnp.zeros((total_steps, n))
            drive = drive.at[:pulse_steps, target].set(DRIVE_PA)
            drive = drive.reshape(-1, SEGMENT_STEPS, n)
            _, values = jax.lax.scan(segment, rest, drive)
            values = values.reshape(total_steps, n)
            return values[pulse_steps:].mean(axis=0) / baseline - 1

        return jax.vmap(column, out_axes=1)(columns)

    def predict(params, nuisance, columns, auto):
        raw = responses(params, columns)
        self_response = raw[columns, jnp.arange(len(columns))]
        scale = auto[columns] / jnp.maximum(self_response, SELF_FLOOR)
        return jnp.exp(nuisance["log_gain"]) * raw * scale[None, :]

    predict.half_saturation = half_saturation
    return predict, responses


def train_fold(
    predict, params, target, weights, auto, steps, batch, seed=SEED, lr=0.05
):
    """AdamW on rule parameters and the global gain; random column batches."""
    trainable = {"rules": params, "nuisance": {"log_gain": jnp.array(0.0)}}
    columns = np.flatnonzero(weights.sum(axis=0) > 0)
    target = jnp.asarray(np.nan_to_num(target))
    weights = jnp.asarray(weights)
    auto = jnp.asarray(auto)
    schedule = optax.cosine_decay_schedule(lr, steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, 0.0))
    state = optimizer.init(trainable)

    def loss(p, chosen):
        prediction = predict(p["rules"], p["nuisance"], chosen, auto)
        w = weights[:, chosen]
        error = prediction - target[:, chosen]
        return jnp.sum(w * error**2) / jnp.maximum(w.sum(), 1.0)

    @jax.jit
    def update(p, s, chosen):
        value, grad = jax.value_and_grad(loss)(p, chosen)
        changes, s = optimizer.update(grad, s, p)
        return optax.apply_updates(p, changes), s, value, optax.tree.norm(grad)

    rng = np.random.default_rng(seed)
    history = []
    for iteration in range(steps):
        chosen = jnp.asarray(rng.choice(columns, size=batch, replace=False))
        start = time.perf_counter()
        trainable, state, value, norm = update(trainable, state, chosen)
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)):
            raise FloatingPointError("non-finite Phase 1 loss or gradient")
        history.append(
            {
                "step": iteration,
                "loss": float(value),
                "gradient_norm": float(norm),
                "seconds": time.perf_counter() - start,
            }
        )
        print(json.dumps(history[-1]), flush=True)
    return trainable, history


def predict_all(predict, trainable, columns, auto, chunk=8):
    blocks = []
    run = jax.jit(lambda c: predict(trainable["rules"], trainable["nuisance"], c, auto))
    for start in range(0, len(columns), chunk):
        part = columns[start : start + chunk]
        padded = np.pad(part, (0, chunk - len(part)), mode="edge")
        blocks.append(np.asarray(run(jnp.asarray(padded)))[:, : len(part)])
    return np.concatenate(blocks, axis=1)


def run_fold(project, registration, split, fold, output, steps, batch):
    """Train on one frozen fold and write held-out predictions (restricted)."""
    reg = registration["registration"]
    budget = 13
    stage_b = Path("configs/phase0-stage-b.json")
    if stage_b.exists():
        budget = json.loads(stage_b.read_text())["registration"][
            "absolute_parameter_budget"
        ]
    graph, manifest, atlas, library, rules = setup(project, budget)
    with jax.enable_x64(True):
        pol = policy()
        predict, _ = make_simulator(graph, rules, library, pol)
        wt = atlas["wt"]
        observed = observed_mask(wt)
        auto, _ = autoresponse(wt["dff"])
        weights_all = np.minimum(wt["occurrences"], 20).astype(float)
        for f, test, train, _cluster in fold_masks(split, reg, manifest, observed):
            if f != fold:
                continue
            start = time.perf_counter()
            trainable, history = train_fold(
                predict,
                rules.params,
                wt["dff"],
                weights_all * train,
                auto,
                steps,
                batch,
            )
            columns = np.flatnonzero(test.any(axis=0))
            prediction = predict_all(predict, trainable, columns, jnp.asarray(auto))
            full = np.full(wt["dff"].shape, np.nan)
            full[:, columns] = prediction
            index = np.nonzero(test)
            result = {
                "split": split,
                "fold": fold,
                "registration_hash": registration["processing_hash"],
                "parameter_budget": budget,
                "trainable_parameters": int(
                    sum(np.size(v) for v in jax.tree.leaves(trainable["rules"]))
                ),
                "nuisance_parameters": 1,
                "steps": steps,
                "batch": batch,
                "history": history,
                "params": jax.tree.map(lambda x: np.asarray(x).tolist(), trainable),
                "test_pairs": np.stack(index, axis=1).tolist(),
                "test_prediction": full[index].tolist(),
                "wall_seconds": time.perf_counter() - start,
                "policy": {
                    "dt_s": DT_S,
                    "n_comp": 1,
                    "drive_pA": DRIVE_PA,
                    "synapse_unit_nS": SYNAPSE_UNIT_NS,
                    "chloride_mM": CHLORIDE_MM,
                    "half_saturation": predict.half_saturation,
                    "self_floor": SELF_FLOOR,
                    "gap_unit_nS": GAP_UNIT_NS,
                },
                "data_tier": "restricted",
                "residuals_enabled": False,
            }
            out = Path(output)
            out.mkdir(parents=True, exist_ok=True)
            _dump(out / f"{split}-fold{fold}.json", result)
            return result
    raise ValueError(f"fold {fold} not in 0..{FOLDS - 1}")


def report(project, registration, output, steps=1500):
    """Pool fold predictions, refit Phase 0 baselines and summarize."""
    from .phase0_worm import (
        amplitude_ceiling,
        atlas_inputs,
        cross_validate,
        sign_ceiling,
        standard_models,
        summarize,
    )
    from .worm_public import load_worm_project

    reg = registration["registration"]
    graph, manifest = load_worm_project(project)
    n = len(manifest["neuron_names"])
    from .worm_public import load_atlas

    atlas = align_atlas(load_atlas(Path(project) / "signal_propagation.npz"), n)
    wt = atlas["wt"]
    inputs = atlas_inputs(graph, manifest)
    out = Path(output)
    result = {"registration_hash": registration["processing_hash"], "splits": {}}
    for split in ("leave_class_out", "leave_neuron_out"):
        files = [out / f"{split}-fold{f}.json" for f in range(FOLDS)]
        if not all(p.exists() for p in files):
            result["splits"][split] = {"status": "not_run"}
            continue
        models = [m for m in standard_models() if m.name in ("B0", "B1", "B2")]
        predictions, values, labels, clusters, pairs, _ = cross_validate(
            models, inputs, wt, reg, manifest, split, steps
        )
        lookup = {}
        for path in files:
            fold = json.loads(path.read_text())
            for (i, j), v in zip(fold["test_pairs"], fold["test_prediction"]):
                lookup[(i, j)] = v
        predictions["compiler"] = np.array([lookup[tuple(p)] for p in pairs.tolist()])
        observed = observed_mask(wt)
        test = np.zeros_like(observed)
        test[tuple(pairs.T)] = True
        ceilings = {
            "amplitude": amplitude_ceiling(wt["trials"], test)[0],
            "sign": sign_ceiling(wt["trials"], test & (wt["q"] < 0.05))[0],
        }
        summary = summarize(predictions, values, labels, clusters, ceilings)
        summary["ceilings"] = ceilings
        summary["acceptance"] = acceptance(summary)
        result["splits"][split] = summary
    result["data_tier"] = "restricted"
    _dump(out / "phase1-report.json", result)
    return result


def acceptance(summary):
    """Section 9.3: compiler beats B0, B1 and B2 (bootstrap interval above 0)."""
    verdict = {}
    for baseline in ("B0", "B1", "B2"):
        entry = summary["comparisons"].get(f"compiler-{baseline}", {})
        verdict[baseline] = {
            metric: (
                None
                if not entry.get(metric) or entry[metric]["interval"] is None
                else entry[metric]["interval"][0] > 0
            )
            for metric in (
                "perturbation_detection",
                "perturbation_sign",
                "perturbation_amplitude",
            )
        }
    verdict["passed"] = all(
        value is True
        for baseline in ("B0", "B1", "B2")
        for value in verdict[baseline].values()
    )
    return verdict
