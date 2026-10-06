from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler import (
    ModulatoryState,
    ObservationModel,
    ResolutionPolicy,
    Stimulus,
    attach_residuals,
    compile,
    observe,
    simulate,
)
from molecular_compiler.baselines import synapse_only
from molecular_compiler.solver import voltage_solve


def test_M4_R3_exact_masks_and_M4_R5_unwired_peptides(system):
    graph, rules, kinetics = system
    sim = compile(graph, rules, kinetics)
    assert np.all(np.asarray(sim.neuromod["release"])[[1, 3]] == 0)
    assert np.all(np.asarray(sim.neuromod["sensitivity"])[[0, 2]] == 0)
    assert np.all(np.asarray(sim.neuromod["sensitivity"])[[1, 3]] > 0)
    assert 3 not in sim.syn_post_idx
    assert sim.neuromod["release"].shape == (4, 1)


def test_M4_R2_neuron_permutation(system):
    graph, rules, kinetics = system
    order = np.array([2, 0, 3, 1])
    permuted = replace(
        graph,
        neuron_ids=graph.neuron_ids[order],
        abundance=graph.abundance[order],
        abundance_variance=graph.abundance_variance[order],
        assignments=graph.assignments[order],
        connectome=replace(
            graph.connectome, neurons=graph.connectome.neurons.take(order)
        ),
    )
    a, b = compile(graph, rules, kinetics), compile(permuted, rules, kinetics)
    np.testing.assert_allclose(
        b.neuron_params["channel_density"], a.neuron_params["channel_density"][order]
    )
    np.testing.assert_allclose(
        b.neuromod["sensitivity"], a.neuromod["sensitivity"][order]
    )
    stimulus = Stimulus(pulses=((0, 0, 0.003, 1.0),))
    ta, tb = simulate(a, stimulus, 0.005), simulate(b, stimulus, 0.005)
    np.testing.assert_allclose(tb.voltage, ta.voltage[:, order], rtol=1e-9, atol=1e-9)


def test_M6_R1_no_stored_synaptic_sign(system):
    sim = compile(*system)
    assert "sign" not in [f.name for f in fields(sim)]
    assert "sign" not in sim.syn_params
    assert "reversal" in sim.syn_params


def test_M6_R2_csr_and_geometric_gates(system):
    sim = compile(*system)
    assert list(sim.syn_post_ptr) == [0, 0, 1, 2, 2]
    assert np.all(np.diff(sim.syn_post_idx) >= 0)
    assert np.all(sim.syn_params["gate"] > 0)
    assert np.all(sim.syn_params["gate"] < 1)


@pytest.mark.parametrize("mode", ["event", "parallel"])
def test_M8_R1_matching_modes(system, mode):
    sim = compile(*system, resolution=replace(ResolutionPolicy.default(), n_comp=1))
    pulse = Stimulus(pulses=((0, 0, 0.002, 1.0),))
    reference = simulate(sim, pulse, 0.005)
    alternative = simulate(sim, pulse, 0.005, mode=mode)
    np.testing.assert_allclose(reference.voltage, alternative.voltage, atol=1e-6)
    if mode == "parallel":
        assert alternative.metadata["parallel_fallback"]
    else:
        assert not alternative.metadata["event_accelerated"]


def test_M8_R4_determinism_and_observation_seed(system):
    sim = compile(*system)
    a, b = (simulate(sim, Stimulus(), 0.005, seed=42) for _ in range(2))
    np.testing.assert_array_equal(a.voltage, b.voltage)
    oa, ob = (observe(a, ObservationModel(noise_sd=0.1, seed=7)) for _ in range(2))
    np.testing.assert_array_equal(oa.y, ob.y)


def test_M8_R5_pcg_matches_dense(system):
    graph, rules, kinetics = system
    dense = compile(
        graph,
        rules,
        kinetics,
        resolution=replace(
            ResolutionPolicy.default(), solver="dense", solve_tolerance=1e-10
        ),
    )
    sparse = replace(dense, resolution=replace(dense.resolution, solver="pcg"))
    rng = np.random.default_rng(3)
    diagonal = jnp.asarray(rng.uniform(1, 2, (4, 3)))
    rhs = jnp.asarray(rng.normal(size=(4, 3)))
    direct, _, _ = voltage_solve(dense, diagonal, rhs, jnp.zeros_like(rhs))
    iterative, residual, iterations = voltage_solve(
        sparse, diagonal, rhs, jnp.zeros_like(rhs)
    )
    np.testing.assert_allclose(iterative, direct, atol=1e-9)
    assert residual < 1e-10
    assert 1 <= iterations <= sparse.resolution.max_cg_iterations


@pytest.mark.parametrize("solver", ["dense", "pcg"])
def test_M8_R5_implicit_gradient_finite_difference(system, solver):
    sim = compile(
        *system,
        resolution=replace(
            ResolutionPolicy.default(), solver=solver, solve_tolerance=1e-10
        ),
    )
    diagonal = jnp.full((4, 3), 1.0)
    rhs = jnp.arange(12.0).reshape(4, 3)

    def loss(value):
        updated = replace(sim, gap={**sim.gap, "conductance": jnp.array([value])})
        v, _, _ = voltage_solve(updated, diagonal, rhs, jnp.zeros_like(rhs))
        return (v**2).sum()

    value = 0.7
    gradient = jax.grad(loss)(value)
    h = 1e-5
    finite = (loss(value + h) - loss(value - h)) / (2 * h)
    np.testing.assert_allclose(gradient, finite, rtol=1e-6)


def test_S_R1_end_to_end_gradient(system):
    graph, rules, kinetics = system
    resolution = replace(ResolutionPolicy.default(), n_comp=1)

    def loss(params):
        sim = compile(graph, rules.with_params(params), kinetics, resolution=resolution)
        return observe(
            simulate(sim, Stimulus(), 0.002), ObservationModel(half_saturation=1e-6)
        ).y.sum()

    gradients = jax.jit(jax.grad(loss))(rules.params)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(gradients))
    assert sum(np.linalg.norm(g) for g in jax.tree.leaves(gradients)) > 0


def test_M8_R5_float32_reference(system):
    sim = compile(*system)
    d64, b64 = jnp.ones((4, 3)), jnp.arange(12.0).reshape(4, 3)
    a, _, _ = voltage_solve(sim, d64, b64, jnp.zeros_like(b64))
    b, residual, _ = voltage_solve(
        sim,
        d64.astype(jnp.float32),
        b64.astype(jnp.float32),
        jnp.zeros((4, 3), dtype=jnp.float32),
    )
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    assert residual < 1e-5


def test_M4_R5_peptide_ablation_changes_unwired_target(system):
    sim = compile(*system, resolution=replace(ResolutionPolicy.default(), n_comp=1))
    a = simulate(sim, Stimulus(pulses=((0, 0, 0.01, 10.0),)), 0.02)
    b = simulate(synapse_only(sim), Stimulus(pulses=((0, 0, 0.01, 10.0),)), 0.02)
    assert a.voltage[-1, 3, 0] > b.voltage[-1, 3, 0]


def test_headline_compilation_disables_residuals(system):
    sim = compile(*system)
    adjusted = attach_residuals(sim, jnp.ones(len(sim.syn_pre_idx)))
    assert adjusted.metadata["residuals_enabled"]
    assert not compile(*system).metadata["residuals_enabled"]
    np.testing.assert_allclose(
        adjusted.syn_params["gate"], sim.syn_params["gate"] * np.e
    )


def test_M7_R4_nondefault_state_full_fallback(system):
    sim = compile(*system, state=ModulatoryState(concentrations=(1.0,)))
    assert all(s.family == "full" for s in sim.surrogates)
    assert sim.neuromod["nondefault"]
    assert np.all(np.isfinite(simulate(sim, Stimulus(), 0.002).voltage))


def test_invalid_requests_fail_loudly(system):
    sim = compile(*system)
    for duration in (-1, float("nan"), 0.0007):
        with pytest.raises(ValueError):
            simulate(sim, Stimulus(), duration)
    with pytest.raises(ValueError):
        simulate(sim, Stimulus(pulses=((99, 0, 1, 1),)), 0.002)
    with pytest.raises(ValueError):
        simulate(sim, Stimulus(), 0.002, mode="unknown")


def test_M8_R1_sparse_spike_events_match_reference(system):
    policy = replace(
        ResolutionPolicy.default(),
        n_comp=1,
        release_mode="spiking",
        spike_threshold_mV=-50.0,
    )
    sim = compile(*system, resolution=policy)
    currents = jnp.zeros((40, 4)).at[:8, 0].set(8.0).at[20:28, 0].set(8.0)
    reference = simulate(sim, Stimulus(currents=currents), 0.02)
    events = simulate(sim, Stimulus(currents=currents), 0.02, mode="event")
    np.testing.assert_allclose(events.voltage, reference.voltage, atol=1e-8)
    np.testing.assert_allclose(events.calcium, reference.calcium, atol=1e-10)
    assert events.metadata["event_accelerated"]
    assert np.max(np.asarray(events.final_state.last_release_time)) > 0


def test_M5_markov_channel_preserves_probability(system):
    graph, rules, kinetics = system
    record = replace(
        kinetics.records["ca"],
        model_form="markov",
        transition_generator=((-1.0, 1.0), (2.0, -2.0)),
    )
    library = replace(kinetics, records={**kinetics.records, "ca": record})
    result = simulate(compile(graph, rules, library), Stimulus(), 0.005)
    np.testing.assert_allclose(result.final_state.gates.sum(axis=-1), 1.0, atol=1e-9)
    assert np.all(np.asarray(result.final_state.gates) >= 0)


def test_S_R1_kinetic_parameter_gradient(system):
    graph, rules, kinetics = system

    def loss(tau):
        record = replace(
            kinetics.records["ca"], parameter_names=("tau_s",), params=(tau,)
        )
        library = replace(kinetics, records={**kinetics.records, "ca": record})
        return simulate(
            compile(graph, rules, library),
            Stimulus(pulses=((0, 0, 0.001, 1.0),)),
            0.002,
        ).voltage.sum()

    derivative = jax.jit(jax.grad(loss))(0.01)
    assert np.isfinite(derivative) and abs(derivative) > 0


def test_M6_R3_chloride_sign_audit(system):
    graph, rules, kinetics = system
    receptor = replace(
        kinetics.records["glr"], reversal_mV=None, ion_selectivity={"Cl": 1.0}
    )
    library = replace(kinetics, records={**kinetics.records, "glr": receptor})
    sim = compile(graph, rules, library)
    assert np.all(sim.syn_params["sign_ambiguous"])
    assert sim.metadata["sign_audit"]["ambiguous_fraction"] == 1
    assert np.all(sim.neuron_params["chloride"] < 10.0)


def test_M4_R6_ligand_tokens_do_not_transfer(system):
    from molecular_compiler.data import table

    graph, rules, kinetics = system
    gene_rows = graph.molecular.genes.to_pylist()
    for row in gene_rows:
        if row["molecule_class"] == "peptide":
            row["plm_embedding"] = [10.0] * 1280
    changed = replace(
        graph, molecular=replace(graph.molecular, genes=table("genes", gene_rows))
    )
    a, b = compile(graph, rules, kinetics), compile(changed, rules, kinetics)
    np.testing.assert_allclose(a.neuromod["release"], b.neuromod["release"])
    np.testing.assert_allclose(a.neuromod["sensitivity"], b.neuromod["sensitivity"])


def test_identity_deviations_recompile_before_linking(system):
    sim = compile(*system)
    corrected = attach_residuals(
        sim,
        jnp.zeros(len(sim.syn_pre_idx)),
        delta=jnp.ones_like(sim.neuron_params["z"]),
    )
    assert corrected.metadata["residuals_enabled"]
    assert not np.allclose(
        corrected.neuron_params["channel_density"], sim.neuron_params["channel_density"]
    )


def test_grid_mode_is_finite_and_factorized():
    from molecular_compiler.fixtures import synthetic_system

    sim = compile(*synthetic_system(species="Drosophila"))
    trajectory = simulate(sim, Stimulus(), 0.001)
    assert sim.neuromod["mode"] == "grid"
    assert trajectory.metadata["grid_boundary"] == "periodic"
    assert np.all(np.isfinite(trajectory.voltage))


def test_M4_R1_compact_rules_obey_rank_budget_and_are_differentiable(system):
    from molecular_compiler import RuleNetwork

    graph, _, kinetics = system
    rules = RuleNetwork.initialize_compact(
        rank=4, budget_fraction=0.5, gauge_audited=True, d_z=8
    )
    assert rules.parameter_count == rules.budget == 2

    def loss(params):
        return simulate(
            compile(graph, rules.with_params(params), kinetics), Stimulus(), 0.002
        ).voltage.sum()

    gradients = jax.jit(jax.grad(loss))(rules.params)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(gradients))


def test_screened_grid_decay_length_gradient_matches_finite_difference():
    from dataclasses import replace

    from molecular_compiler.fixtures import synthetic_system
    from molecular_compiler.simulation import grid_release

    sim = compile(*synthetic_system(species="Drosophila"))
    release = jnp.zeros((sim.n_neurons, 1)).at[0, 0].set(1.0)

    def response(length):
        configured = replace(sim, neuromod={**sim.neuromod, "decay_length_um": length})
        return grid_release(configured, release)[0, sim.neuromod["grid_index"][1]]

    gradient = jax.grad(response)(50.0)
    finite_difference = (response(50.01) - response(49.99)) / 0.02
    assert abs(float(gradient)) > 1e-10
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-4, atol=1e-10)
