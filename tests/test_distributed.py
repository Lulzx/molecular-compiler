"""Run in an isolated process with virtual CPU devices; no GPU claim."""

import os
import subprocess
import sys


def test_C_R3_two_device_halo_and_voltage_solve():
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from molecular_compiler import compile, ResolutionPolicy
from molecular_compiler.fixtures import synthetic_system
from molecular_compiler.distributed import partition_graph, halo_exchange, distributed_voltage_solve
from molecular_compiler.solver import voltage_solve
sim = compile(*synthetic_system(5), resolution=replace(ResolutionPolicy.default(), solver="dense", solve_tolerance=1e-10))
# Include a gap across device boundary as well as the local fixture contact.
sim = replace(sim, gap={"i":jnp.array([0,0]), "j":jnp.array([1,4]), "conductance":jnp.array([0.2,0.3])})
plan = partition_graph(sim)
assert plan.partitions == 2 and plan.halo_neurons > 0
x = jnp.arange(6.).reshape(2,3,1)
halo = halo_exchange(plan,x)
assert halo.shape[0] == 2
assert 0. in np.asarray(halo[1,0])
diagonal = jnp.full((5,3),2.)
rhs = jnp.arange(15.).reshape(5,3)
reference,_,_ = voltage_solve(sim,diagonal,rhs,jnp.zeros_like(rhs))
actual,residual,iterations = distributed_voltage_solve(sim,plan,diagonal,rhs,jnp.zeros_like(rhs))
np.testing.assert_allclose(actual,reference,atol=1e-8)
assert residual < 1e-10
print("two virtual CPU devices: halo and PCG match dense reference")
"""
    env = dict(
        os.environ,
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    process = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_C_R3_two_device_partitioned_synapses_match_single_step():
    script = r"""
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from molecular_compiler import compile, ResolutionPolicy
from molecular_compiler.fixtures import synthetic_system
from molecular_compiler.distributed import (
    partition_graph, distributed_step, synaptic_state_bytes)
import jax
from molecular_compiler.simulation import initial_state, step
for mode in ("spiking", "graded"):
    policy = replace(ResolutionPolicy.default(), solver="dense", solve_tolerance=1e-10,
                     release_mode=mode)
    sim = compile(*synthetic_system(6), resolution=policy)
    plan = partition_graph(sim)
    assert plan.partitions == 2
    pre, post = np.asarray(sim.syn_pre_idx), np.asarray(sim.syn_post_idx)
    assert np.any(pre // plan.local_size != post // plan.local_size), "no cross-device synapse"
    assert int(plan.syn_counts.sum()) == len(pre)
    assert sum(synaptic_state_bytes(sim, plan)) > 0
    ref = dist = initial_state(sim)
    run_ref = jax.jit(lambda s, c: step(sim, s, c))
    run_dist = jax.jit(lambda s, c: distributed_step(sim, plan, s, c))
    for k in range(4):
        # Drive hard enough that spiking mode actually releases.
        current = jnp.full(6, 3000.0 if k < 2 else 0.0)
        ref, _ = run_ref(ref, current)
        dist, _ = run_dist(dist, current)
        for a, b in zip(ref, dist):
            np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-8, rtol=1e-8)
    assert float(jnp.abs(ref.receptors).max()) > 0, mode
print("partitioned synapses match", synaptic_state_bytes(sim, plan))
"""
    env = dict(
        os.environ,
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    process = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=400,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
