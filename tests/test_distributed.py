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
