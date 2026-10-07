from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler import compile
from molecular_compiler import restore as R
from molecular_compiler.fixtures import synthetic_system


class Toy(NamedTuple):
    x: jax.Array  # fast
    s: jax.Array  # slow


class ToySim:
    """x relaxes to s + input; s integrates the input at `rate` (0 = constant)."""

    def __init__(self, rate, n=3):
        self.rate, self.n = rate, n

    def rest(self):
        return Toy(jnp.zeros(self.n), jnp.zeros(self.n))

    def run(self, state, currents):
        def step(st, i):
            x = st.x + 0.1 * (-st.x + st.s + i)
            return Toy(x, st.s + self.rate * i), x

        return jax.lax.scan(step, state, currents)


def _drive(n=3, length=80, quiet=100):
    """Input, then a quiet tail so the fast state has settled to the slow one."""
    rng = np.random.default_rng(0)
    drive = np.abs(rng.normal(1.0, 0.3, (length + quiet, n)))
    drive[length:] = 0.0
    return jnp.asarray(drive)


def test_I1_trivial_pass_when_slow_variable_is_constant():
    report = R.restore_test(
        ToySim(rate=0.0),
        [R.Candidate("u"), R.Candidate("u+s", ("s",))],
        _drive(),
        jnp.zeros((60, 3)),
        tau=0.5,
        noise_ceiling=0.05,
        relax_steps=100,
    )
    assert report.smallest == "u"


def test_I1_dropping_an_evolving_slow_variable_fails_restoration():
    report = R.restore_test(
        ToySim(rate=0.01),
        [R.Candidate("u"), R.Candidate("u+s", ("s",))],
        _drive(),
        jnp.zeros((60, 3)),
        tau=0.5,
        noise_ceiling=0.05,
        relax_steps=100,
    )
    assert [r.sufficient for r in report.results] == [False, True]
    assert report.smallest == "u+s"
    assert report.results[0].error > 10 * report.results[1].error


def test_I1_none_sufficient_reports_none():
    report = R.restore_test(
        ToySim(rate=0.01),
        [R.Candidate("u")],
        _drive(),
        jnp.zeros((60, 3)),
        tau=0.5,
        noise_ceiling=0.05,
    )
    assert report.smallest is None


def test_I1_history_inference_regenerates_fast_state_and_reduces_loss():
    sim = ToySim(rate=0.01)
    drive = _drive()
    state, response = sim.run(sim.rest(), drive)
    history = (drive[-60:], response[-60:])
    candidate = R.Candidate("u+s", ("s",))
    restored, trace = R.restore(
        sim,
        state,
        candidate,
        sim.rest(),
        "infer",
        history=history,
        steps=150,
        learning_rate=0.05,
        prior_weight=1e-6,
    )
    assert trace[-1] < 0.1 * trace[0]
    np.testing.assert_allclose(restored.s, state.s)
    np.testing.assert_allclose(restored.x, state.x, atol=0.05)
    report = R.restore_test(
        sim,
        [candidate],
        drive,
        jnp.zeros((60, 3)),
        0.5,
        0.05,
        mode="infer",
        history_steps=60,
        steps=150,
        learning_rate=0.05,
        prior_weight=1e-6,
    )
    assert report.smallest == "u+s"


def test_I1_nested_candidates_order_and_unknown_mode():
    names = [c.name for c in R.nested_candidates()]
    assert names == ["u", "u+c", "u+c+m", "u+c+m+synapse"]
    sizes = [len(c.keep) for c in R.nested_candidates()]
    assert sizes == sorted(sizes) and sizes[0] == 0
    sim = ToySim(0.0)
    with pytest.raises(ValueError):
        R.restore(sim, sim.rest(), R.Candidate("u"), sim.rest(), "bogus")


def test_I1_compiled_simulator_trivial_pass_for_constant_slow_state():
    sim = compile(*synthetic_system())
    adapter = R.CompiledSimulator(sim)
    drive = np.zeros((40, sim.n_neurons))
    probe = np.zeros((40, sim.n_neurons))
    probe[5:15, 0] = 5.0
    report = R.restore_test(
        adapter,
        R.nested_candidates()[:1],
        drive,
        probe,
        tau=1.0,
        noise_ceiling=1e-5,
        relax_steps=10,
    )
    assert report.smallest == "u"
    # a pulse that leaves synaptic state behind makes d0 differ from the full state
    drive[5:15, 0] = 5.0
    report = R.restore_test(
        adapter,
        R.nested_candidates(),
        drive,
        probe,
        tau=1.0,
        noise_ceiling=1e-5,
        relax_steps=10,
    )
    full = report.results[-1]
    assert full.sufficient and full.error < report.results[0].error
