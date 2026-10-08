import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler import ModulatoryState, Stimulus, compile, simulate
from molecular_compiler.benchmark import surrogate_execution_benchmark
from molecular_compiler.fixtures import synthetic_surrogates
from molecular_compiler.simulation import initial_state, step
from molecular_compiler.surrogates import plan_fast_execution

PULSE = Stimulus(pulses=((0, 0, 0.005, 20.0),))


def test_M7_default_path_bit_identical(system):
    # Golden values recorded from the code before the fast path existed, on
    # macOS arm64. Other CPUs round differently by about one ulp, so the check
    # is to rtol 1e-12 rather than exact equality.
    sim = compile(*system)
    assert sim.fast_plan is None
    t = simulate(sim, PULSE, 0.01)
    assert t.metadata["surrogate_execution"] == "hybrid"
    np.testing.assert_allclose(
        np.asarray(t.voltage[-1]),
        np.array(
            [
                [-22.90065058473155, -39.30400812688014, -51.06073017904351],
                [-43.81259609379187, -54.29916983060784, -57.916568341120716],
                [-59.3825929570413, -59.45383458180269, -59.46672424349865],
                [-59.45484738751824, -59.46710378552946, -59.469295938659464],
            ]
        ),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(t.calcium[-1]),
        [
            2.8448999490018844e-06,
            1.8929011583642937e-06,
            1.745154455977926e-06,
            1.7450222043075624e-06,
        ],
        rtol=1e-12,
        atol=0,
    )
    assert float(np.asarray(t.voltage).sum()) == pytest.approx(
        -12894.401735204023, rel=1e-12, abs=0
    )


def test_M7_R2_fast_matches_hybrid_when_all_surrogates_valid(system):
    hybrid_sim = synthetic_surrogates(compile(*system))
    fast_sim = plan_fast_execution(hybrid_sim)
    hybrid, fast = (simulate(s, PULSE, 0.01) for s in (hybrid_sim, fast_sim))
    assert fast.metadata["surrogate_execution"] == "fast"
    assert fast.metadata["fast_surrogate_neurons"] == 4
    # Surrogate soma output is identical; only non-soma compartments differ.
    np.testing.assert_allclose(
        fast.voltage[:, :, 0], hybrid.voltage[:, :, 0], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(fast.calcium, hybrid.calcium, rtol=1e-12, atol=1e-15)
    assert not np.any(fast.surrogate_fallback) and not np.any(hybrid.surrogate_fallback)
    # Full kinetics were skipped: gates never left their initial values.
    np.testing.assert_array_equal(fast.final_state.gates, initial_state(fast_sim).gates)
    assert not np.array_equal(hybrid.final_state.gates, initial_state(fast_sim).gates)


def test_M7_fast_mixed_partition_close_to_hybrid_and_differentiable(system):
    # Type 0 (neurons 0, 2) uses a surrogate; type 1 (neurons 1, 3) keeps full kinetics.
    hybrid_sim = synthetic_surrogates(compile(*system), types={0})
    fast_sim = plan_fast_execution(hybrid_sim)
    assert list(fast_sim.fast_plan.full_idx) == [1, 3]
    hybrid, fast = (simulate(s, PULSE, 0.01) for s in (hybrid_sim, fast_sim))
    np.testing.assert_allclose(
        fast.voltage[:, [0, 2], 0], hybrid.voltage[:, [0, 2], 0], rtol=1e-12
    )
    assert np.all(np.isfinite(fast.voltage))
    # Full-kinetics neurons feel the surrogate soma instead of a full-model soma.
    assert np.max(np.abs(fast.voltage[:, [1, 3]] - hybrid.voltage[:, [1, 3]])) < 5.0

    def loss(currents):
        return simulate(fast_sim, Stimulus(currents=currents), 0.005).voltage.sum()

    gradient = jax.grad(loss)(jnp.zeros((10, 4)).at[:, 0].set(1.0))
    assert np.all(np.isfinite(gradient)) and np.any(gradient != 0)


def test_M7_R2_envelope_exit_runs_full_kinetics_and_is_logged(system):
    hybrid_sim = synthetic_surrogates(compile(*system), upper_drive=5.0)
    fast_sim = plan_fast_execution(hybrid_sim)
    hybrid, fast = (simulate(s, PULSE, 0.01) for s in (hybrid_sim, fast_sim))
    flags = np.asarray(fast.surrogate_fallback)
    np.testing.assert_array_equal(flags, np.asarray(hybrid.surrogate_fallback))
    assert fast.metadata["surrogate_fallback_count"] == 10 and flags[:10, 0].all()
    assert not flags[10:].any() and not flags[:, 1:].any()
    # While neuron 0 is outside its envelope the whole step is the hybrid step.
    np.testing.assert_array_equal(fast.voltage[:10], hybrid.voltage[:10])
    assert not np.array_equal(
        fast.final_state.gates, initial_state(fast_sim).gates
    )  # full kinetics ran on the exit steps
    # Single step: exit branch equals the hybrid step, in-envelope branch does not.
    state0 = initial_state(fast_sim)
    out_exit, _ = step(fast_sim, state0, jnp.array([20.0, 0, 0, 0]))
    ref_exit, _ = step(hybrid_sim, state0, jnp.array([20.0, 0, 0, 0]))
    np.testing.assert_array_equal(out_exit.gates, ref_exit.gates)
    out_ok, _ = step(fast_sim, state0, jnp.zeros(4))
    np.testing.assert_array_equal(out_ok.gates, state0.gates)


def test_M7_R4_nondefault_modulatory_state_uses_full_dynamics(system):
    state = ModulatoryState(concentrations=(1.0,))
    unconditioned = synthetic_surrogates(
        compile(*system, state=state), conditioned=False
    )
    plan = plan_fast_execution(unconditioned)
    assert not plan.fast_plan.eligible.any()
    hybrid, fast = (simulate(s, PULSE, 0.01) for s in (unconditioned, plan))
    np.testing.assert_array_equal(fast.voltage, hybrid.voltage)
    assert np.asarray(fast.surrogate_fallback).all()
    conditioned = plan_fast_execution(
        synthetic_surrogates(compile(*system, state=state), conditioned=True)
    )
    assert conditioned.fast_plan.eligible.all()
    result = simulate(conditioned, PULSE, 0.01)
    assert not np.asarray(result.surrogate_fallback).any()


def test_M7_fast_path_rejects_event_mode(system):
    sim = plan_fast_execution(synthetic_surrogates(compile(*system)))
    with pytest.raises(ValueError, match="event"):
        simulate(sim, Stimulus(), 0.001, mode="event")


def test_M7_benchmark_reports_both_backends_without_timing_assertions():
    result = surrogate_execution_benchmark(sizes=(8,), repeats=1)
    row = result["rows"][0]
    assert row["neurons"] == 8 and row["surrogate_neurons"] == 8
    assert set(row) >= {"hybrid", "fast", "soma_max_abs_difference_mV"}
    assert row["soma_max_abs_difference_mV"] < 1e-9
