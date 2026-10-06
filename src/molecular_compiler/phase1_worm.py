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
import pickle
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
        graph,
        manifest["gene_names"],
        inherit([curation], "phase1"),
        basal_chloride_mM=CHLORIDE_MM,
    )
    frozen = frozen_basis(graph)
    rules = RuleNetwork.initialize_partial(
        frozen, {k: frozen[k] for k in TRAINABLE}, budget
    )
    return graph, manifest, atlas, library, rules


def make_simulator(graph, rules, library, pol, half_saturation=None):
    """Return f(params, nuisance, columns) -> predicted dF/F0 [N, len(columns)]."""
    from .compiler import compile
    from .simulation import _sequential, initial_state

    n = len(graph.neuron_ids)
    warm_steps = round(WARMUP_S / pol.dt_s)
    pulse_steps = round(PULSE_S / pol.dt_s)
    post_steps = round(POST_S / pol.dt_s)

    def relax(sim):
        return _sequential(sim, initial_state(sim), jnp.zeros((warm_steps, n)))[0]

    if half_saturation is None:
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


def _save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(jax.device_get(payload), handle)
    temporary.replace(path)


def _load_checkpoint(path, config):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload["config"] != config:
        raise ValueError(
            f"{path}: checkpoint was written with {payload['config']}, not {config}"
        )
    return payload


def train_fold(
    predict,
    params,
    target,
    weights,
    auto,
    steps,
    batch,
    seed=SEED,
    lr=0.05,
    checkpoint=None,
):
    """AdamW on rule parameters and the global gain; random column batches.

    With `checkpoint`, parameters, optimizer state, the batch RNG and the
    history are saved after every step, and a matching checkpoint is resumed.
    A resumed run draws the same batches as an uninterrupted one.
    """
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
    config = {"steps": steps, "batch": batch, "seed": seed, "lr": lr}
    if checkpoint is not None:
        payload = _load_checkpoint(checkpoint, config)
        if payload is not None:
            trainable = jax.tree.map(jnp.asarray, payload["trainable"])
            state = jax.tree.map(jnp.asarray, payload["state"])
            rng.bit_generator.state = payload["rng"]
            history = payload["history"]
            print(json.dumps({"resumed_at_step": len(history)}), flush=True)
    for iteration in range(len(history), steps):
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
        if checkpoint is not None:
            _save_checkpoint(
                checkpoint,
                {
                    "config": config,
                    "trainable": trainable,
                    "state": state,
                    "rng": rng.bit_generator.state,
                    "history": history,
                },
            )
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
            Path(output).mkdir(parents=True, exist_ok=True)
            trainable, history = train_fold(
                predict,
                rules.params,
                wt["dff"],
                weights_all * train,
                auto,
                steps,
                batch,
                checkpoint=Path(output) / f"{split}-fold{fold}.ckpt",
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
        auto, _ = autoresponse(wt["dff"])
        summary["residual_analysis"] = residual_report(
            inputs, auto, pairs, values, predictions["compiler"]
        )
        result["splits"][split] = summary
    result["data_tier"] = "restricted"
    _dump(out / "phase1-report.json", result)
    return result


def residual_features(inputs, auto, pairs):
    """Per-pair features the Phase 1 model does not use, plus two diagnostics.

    - peptide_coupling: released peptide (stimulated) x matched receptor
      (responder) x potency; peptides are off in Phase 1 (Section 10.2, K2);
    - no_direct_connection: neither a chemical synapse nor a gap junction
      joins the pair (diagnostic for polysynaptic or extrasynaptic paths);
    - log_autoresponse: the gauge input (Section 6.3); a trend here means the
      drive gauge is inadequate.
    """
    responder, stimulated = pairs[:, 0], pairs[:, 1]
    e = inputs.expression
    release = e[:, inputs.pair_peptide] * inputs.pair_potency
    receive = e[:, inputs.pair_receptor]
    peptide = (receive[responder] * release[stimulated]).sum(axis=1)
    direct = (inputs.chemical[responder, stimulated] > 0) | (
        inputs.gap[responder, stimulated] > 0
    )
    features = np.column_stack(
        [
            np.log1p(peptide),
            (~direct).astype(float),
            np.log(np.maximum(auto[stimulated], 1e-6)),
        ]
    )
    return features, ["peptide_coupling", "no_direct_connection", "log_autoresponse"]


def residual_report(inputs, auto, pairs, observed, predicted):
    """M10 residual analysis on held-out errors (residuals are off in Phase 1)."""
    from .training import residual_analysis

    features, names = residual_features(inputs, auto, pairs)
    error = np.asarray(observed) - np.asarray(predicted)
    finite = np.isfinite(error)
    rows = residual_analysis(error[finite], features[finite], names)
    return [
        {
            k: (
                bool(v)
                if isinstance(v, np.bool_)
                else float(v)
                if k != "feature"
                else v
            )
            for k, v in row.items()
        }
        for row in rows
    ]


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


def net_reversal(sim):
    """Density-weighted reversal per synapse edge, and the receptor mass."""
    density = np.asarray(sim.syn_params["density"])
    gbar = np.asarray(sim.syn_params["gbar"])
    weight = density * gbar[None, :]
    mass = weight.sum(axis=1)
    reversal = np.asarray(sim.syn_params["reversal"])
    net = np.where(
        mass > 0, (weight * reversal).sum(axis=1) / np.maximum(mass, 1e-30), np.nan
    )
    return net, mass


def audit(project, output):
    """Pre-training audit of the Phase 1 model (restricted; no targets read)."""
    from .compiler import compile
    from .simulation import _sequential, initial_state
    from .worm_kinetics import family_recovery, library_summary

    graph, _manifest, _atlas, library, rules = setup(project, 13)
    with jax.enable_x64(True):
        pol = policy()
        sim = compile(graph, rules, library, resolution=pol)
        n = len(graph.neuron_ids)
        steps = round(WARMUP_S / DT_S)
        first, _ = _sequential(sim, initial_state(sim), jnp.zeros((steps, n)))
        second, _ = _sequential(sim, first, jnp.zeros((steps // 2, n)))
        voltage = np.asarray(second.voltage)[:, 0]
        drift = float(
            np.max(np.abs(np.asarray(second.voltage) - np.asarray(first.voltage)))
        )
        net, _mass = net_reversal(sim)
        post = np.asarray(sim.syn_post_idx)
        valid = np.isfinite(net)
        inhibitory = valid & (net < voltage[post])
        result = {
            "library": library_summary(library),
            "family_recovery": family_recovery(library),
            "sign_audit": sim.metadata["sign_audit"],
            "rest": {
                "voltage_percentiles_mV": dict(
                    zip(
                        ("10", "50", "90"),
                        np.percentile(voltage, [10, 50, 90]).tolist(),
                    )
                ),
                "max_drift_mV_over_10s": drift,
                "median_calcium": float(
                    np.median(np.maximum(np.asarray(second.calcium), 0))
                ),
            },
            "synapses": {
                "edges": len(net),
                "with_receptors": int(valid.sum()),
                "net_inhibitory_at_rest": int(inhibitory.sum()),
                "net_inhibitory_fraction": float(
                    inhibitory.sum() / max(valid.sum(), 1)
                ),
                "chloride_mM": CHLORIDE_MM,
            },
            "policy": {
                "dt_s": DT_S,
                "synapse_unit_nS": SYNAPSE_UNIT_NS,
                "gap_unit_nS": GAP_UNIT_NS,
            },
            "data_tier": "restricted",
        }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "audit.json", result)
    return result


def pair_matrix(responses, n, rows=None):
    """Trial means, counts and per-pair trial lists [responder, stimulated]."""
    rows = np.arange(len(responses["animal"])) if rows is None else rows
    responder = responses["responder"][rows]
    stimulated = responses["stimulated"][rows]
    values = responses["mean_dff"][rows]
    total, count = np.zeros((n, n)), np.zeros((n, n), dtype=int)
    np.add.at(total, (responder, stimulated), values)
    np.add.at(count, (responder, stimulated), 1)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    order = np.lexsort((stimulated, responder))
    trials = [[np.empty(0)] * n for _ in range(n)]
    keys = responder[order] * n + stimulated[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], len(keys)]):
        index = order[start]
        trials[responder[index]][stimulated[index]] = values[order[start:stop]]
    return mean, count, trials


def held_out_animals(project, output, steps=1500, folds=FOLDS, seed=SEED):
    """Exploratory: generalization to unseen animals (pre_registered: false).

    Uses the ingested Randi traces (atlas-included, non-outlier, label
    confidence >= 0.95). Animals are split into folds; models are fit on the
    pair means of training animals and scored on the pair means of held-out
    animals. The drive gauge uses the held-out animals' own autoresponses.
    Labels are the atlas q-values, computed on all animals (declared leak:
    labels only define the detection and sign subsets). This is the split of
    Creamer et al. that the pooled atlas could not reproduce in Phase 0.
    """
    from . import linear_response as lr
    from .phase0_worm import (
        amplitude_ceiling,
        atlas_inputs,
        sign_ceiling,
        standard_models,
        summarize,
    )
    from .randi_traces import load_responses
    from .worm_public import load_atlas, load_worm_project

    graph, manifest = load_worm_project(project)
    names = manifest["neuron_names"]
    n = len(names)
    atlas = align_atlas(load_atlas(Path(project) / "signal_propagation.npz"), n)
    inputs = atlas_inputs(graph, manifest)
    responses = load_responses(project, "wt")
    animals = np.unique(responses["animal"])
    rng = np.random.default_rng(seed)
    assignment = dict(zip(rng.permutation(animals), np.arange(len(animals)) % folds))
    fold_of = np.array([assignment[a] for a in responses["animal"]])
    models = standard_models()
    pooled = {m.name: [] for m in models}
    observed_values, labels, clusters, test_pairs = [], [], [], []
    test_trials = [[np.empty(0)] * n for _ in range(n)]
    for fold in range(folds):
        train, count, _ = pair_matrix(responses, n, np.flatnonzero(fold_of != fold))
        test, test_count, trials = pair_matrix(
            responses, n, np.flatnonzero(fold_of == fold)
        )
        off = ~np.eye(n, dtype=bool)
        weights = np.minimum(count, 20).astype(float) * off
        train_auto, _ = autoresponse(train)
        test_auto, _ = autoresponse(test)
        index = np.nonzero((test_count > 0) & off)
        observed_values.append(test[index])
        labels.append(atlas["wt"]["q"][index] < 0.05)
        clusters.append(np.array(names, dtype=object)[index[1]])
        test_pairs.append(np.stack(index, axis=1))
        for i, j in zip(*index):
            test_trials[i][j] = np.concatenate([test_trials[i][j], trials[i][j]])
        for model in models:
            params, _ = lr.fit(model, inputs, train, weights, train_auto, steps=steps)
            prediction = np.asarray(
                lr.predict(model, params, inputs, jnp.asarray(test_auto))
            )
            pooled[model.name].append(prediction[index])
    predictions = {k: np.concatenate(v) for k, v in pooled.items()}
    values = np.concatenate(observed_values)
    labels = np.concatenate(labels)
    pairs = np.concatenate(test_pairs)
    mask = np.zeros((n, n), dtype=bool)
    mask[tuple(pairs.T)] = True
    q = atlas["wt"]["q"]
    ceilings = {
        "amplitude": amplitude_ceiling(test_trials, mask)[0],
        "sign": sign_ceiling(test_trials, mask & (q < 0.05))[0],
    }
    summary = summarize(predictions, values, labels, np.concatenate(clusters), ceilings)
    result = {
        "pre_registered": False,
        "split": "held_out_animals",
        "folds": folds,
        "animals": len(animals),
        "trials": len(responses["animal"]),
        "ceilings": ceilings,
        **summary,
        "data_tier": "restricted",
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "held-out-animals.json", result)
    return result


def convergence_test(project, output, columns=24, seed=SEED, tolerance=0.02):
    """Section 12.6 on the Phase 1 model: halve dt; separately add two compartments.

    The initial (untrained) model is compared at the chosen resolution and at
    each refinement on the same observed pairs of a seeded subset of
    stimulated columns. A refinement passes when every Section 9.1 metric
    changes by less than `tolerance` x its noise ceiling (detection has no
    ceiling and uses `tolerance` in raw AUROC, declared). The half-saturation
    is fixed from the base resolution so only the numerics change.
    """
    from .phase0_worm import _metrics, amplitude_ceiling, sign_ceiling

    graph, _manifest, atlas, library, rules = setup(project, 13)
    wt = atlas["wt"]
    observed = observed_mask(wt)
    auto, _ = autoresponse(wt["dff"])
    candidates = np.flatnonzero(observed.any(axis=0))
    chosen = np.sort(
        np.random.default_rng(seed).choice(candidates, columns, replace=False)
    )
    mask = np.zeros_like(observed)
    mask[:, chosen] = observed[:, chosen]
    index = np.nonzero(mask)
    labels = wt["q"][index] < 0.05
    ceilings = {
        "perturbation_amplitude": amplitude_ceiling(wt["trials"], mask)[0],
        "perturbation_sign": sign_ceiling(wt["trials"], mask & (wt["q"] < 0.05))[0],
        "perturbation_detection": 1.0,
    }
    base = policy()
    variants = {
        "base": base,
        "half_dt": replace(base, dt_s=base.dt_s / 2),
        "plus_two_compartments": replace(base, n_comp=base.n_comp + 2),
    }
    results, half_saturation = {}, None
    with jax.enable_x64(True):
        for name, pol in variants.items():
            start = time.perf_counter()
            predict, _ = make_simulator(graph, rules, library, pol, half_saturation)
            half_saturation = predict.half_saturation
            trainable = {
                "rules": rules.params,
                "nuisance": {"log_gain": jnp.array(0.0)},
            }
            full = np.full(observed.shape, np.nan)
            full[:, chosen] = predict_all(predict, trainable, chosen, jnp.asarray(auto))
            results[name] = {
                "metrics": _metrics(full[index], wt["dff"][index], labels),
                "prediction": full[index],
                "seconds": time.perf_counter() - start,
            }
            print(json.dumps({name: results[name]["metrics"]}), flush=True)
    report = {"columns": chosen.tolist(), "pairs": len(index[0]), "ceilings": ceilings}
    for name in ("half_dt", "plus_two_compartments"):
        checks = {}
        for metric, ceiling in ceilings.items():
            a = results["base"]["metrics"][metric]
            b = results[name]["metrics"][metric]
            change = None if a is None or b is None else abs(b - a)
            limit = None if ceiling is None else tolerance * ceiling
            checks[metric] = {
                "base": a,
                "refined": b,
                "change": change,
                "limit": limit,
                "passed": None if change is None or limit is None else change < limit,
            }
        difference = results[name]["prediction"] - results["base"]["prediction"]
        report[name] = {
            "checks": checks,
            "passed": all(c["passed"] is True for c in checks.values()),
            "max_abs_prediction_change": float(np.nanmax(np.abs(difference))),
            "seconds": results[name]["seconds"],
        }
    report["policy"] = {"dt_s": base.dt_s, "n_comp": base.n_comp}
    report["data_tier"] = "restricted"
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "convergence.json", report)
    return report
