"""M13: reduced differentiable 2D worm body (overdamped chain, resistive-force drag)."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BodyParams:
    """Reduced-body constants. All values are illustrative, not fitted to data.

    Drag is resistive-force theory with anisotropic coefficients. Only the ratio
    drag_normal / drag_tangential enters the force balance (the body is
    overdamped), so the absolute drag scale is not a parameter.
    """

    n_segments: int = 50
    length_mm: float = 1.0
    drag_tangential: float = 1.0  # illustrative; relative units
    drag_normal: float = 40.0  # illustrative; agar-crawling-like ratio
    tau_bend_s: float = 0.2  # viscoelastic relaxation time c/k of the bending joints
    max_curvature_per_mm: float = 6.0  # illustrative bound on muscle-set curvature

    @property
    def n_joints(self):
        return self.n_segments - 1

    @property
    def ds_mm(self):
        return self.length_mm / self.n_segments


DEFAULT_PARAMS = BodyParams()


class WormState(NamedTuple):
    position: jnp.ndarray  # (2,) mm; centre of segment 0 (the head)
    heading: jnp.ndarray  # () rad; tangent angle of segment 0, pointing head to tail
    kappa: jnp.ndarray  # (n_joints,) joint angles rad; positive bends dorsally


def initial_state(params=DEFAULT_PARAMS, position=(0.0, 0.0), heading=0.0):
    return WormState(
        jnp.asarray(position, dtype=float),
        jnp.asarray(heading, dtype=float),
        jnp.zeros(params.n_joints),
    )


def curvature(state, params=DEFAULT_PARAMS):
    """Curvature per joint in 1/mm (positive = dorsal)."""
    return state.kappa / params.ds_mm


def _geometry(state, params):
    angles = state.heading + jnp.concatenate([jnp.zeros(1), jnp.cumsum(state.kappa)])
    tangent = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    steps = 0.5 * params.ds_mm * (tangent[:-1] + tangent[1:])
    centres = state.position + jnp.concatenate(
        [jnp.zeros((1, 2)), jnp.cumsum(steps, axis=0)]
    )
    return tangent, centres


def forward_axis(state):
    """Unit vector the worm moves along when going forward (tail to head)."""
    return -jnp.stack([jnp.cos(state.heading), jnp.sin(state.heading)])


def center_of_mass(state, params=DEFAULT_PARAMS):
    return _geometry(state, params)[1].mean(axis=0)


def step(state, activation, dt_s, params=DEFAULT_PARAMS):
    """Advance one step; pure and differentiable.

    activation is (2, n_joints) in [0, 1]: dorsal then ventral muscle drive. The
    preferred joint angle is set by the dorsal-ventral difference; joints relax to
    it with time constant tau_bend_s (exact exponential, stable at any dt). The
    rigid motion then follows from zero net force and torque under RFT drag.
    """
    activation = jnp.asarray(activation)
    target_kappa = (
        params.max_curvature_per_mm * params.ds_mm * (activation[0] - activation[1])
    )
    decay = jnp.exp(-dt_s / params.tau_bend_s)
    new_kappa = target_kappa + (state.kappa - target_kappa) * decay
    kappa_rate = (new_kappa - state.kappa) / dt_s
    tangent, centres = _geometry(state, params)
    normal = jnp.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    # Deformation velocity of each segment centre with zero rigid motion.
    angle_rate = jnp.concatenate([jnp.zeros(1), jnp.cumsum(kappa_rate)])
    dstep = (
        0.5
        * params.ds_mm
        * (normal[:-1] * angle_rate[:-1, None] + normal[1:] * angle_rate[1:, None])
    )
    deform = jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(dstep, axis=0)])
    rel = centres - centres[0]
    # v_i = v0 + omega * z x rel_i + deform_i ; solve sum A_i^T D_i v_i = 0.
    drag = params.ds_mm * (
        params.drag_tangential * tangent[:, :, None] * tangent[:, None, :]
        + params.drag_normal * normal[:, :, None] * normal[:, None, :]
    )
    zeros = jnp.zeros(len(rel))
    jac = jnp.stack(
        [
            jnp.stack([jnp.ones_like(zeros), zeros, -rel[:, 1]], axis=-1),
            jnp.stack([zeros, jnp.ones_like(zeros), rel[:, 0]], axis=-1),
        ],
        axis=1,
    )  # (n, 2, 3)
    matrix = jnp.einsum("nai,nab,nbj->ij", jac, drag, jac)
    rhs = -jnp.einsum("nai,nab,nb->i", jac, drag, deform)
    velocity = jnp.linalg.solve(matrix, rhs)
    return WormState(
        state.position + dt_s * velocity[:2],
        state.heading + dt_s * velocity[2],
        new_kappa,
    )


class WormBody:
    """BodyModel-style wrapper of the functional step, for the non-differentiable loop.

    reset(state0) returns curvature (1/mm); step(motor, dt_s) takes a flat
    (2 * n_joints,) activation (dorsal then ventral) or a (2, n_joints) array.
    """

    def __init__(self, params=DEFAULT_PARAMS):
        self.params = params
        self.state = initial_state(params)
        self._step = jax.jit(lambda s, a, dt: step(s, a, dt, params))

    def reset(self, state0):
        self.state = initial_state(
            self.params,
            state0.get("position", (0.0, 0.0)),
            state0.get("heading", 0.0),
        )
        return np.asarray(curvature(self.state, self.params))

    def step(self, motor, dt_s):
        motor = np.asarray(motor, dtype=float)
        if motor.size != 2 * self.params.n_joints or dt_s <= 0:
            raise ValueError("invalid muscle activation or timestep")
        motor = motor.reshape(2, self.params.n_joints)
        self.state = self._step(self.state, motor, dt_s)
        return np.asarray(curvature(self.state, self.params))
