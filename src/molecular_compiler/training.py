"""M10: differentiable distribution losses, multiple shooting and AdamW."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import linregress


def pairwise_distance(x, y):
    # Smooth at zero for finite derivatives; exact zero still subtracts out.
    return jnp.sqrt(jnp.sum((x[:, None] - y[None]) ** 2, axis=-1) + 1e-12)


def energy_distance(x, y):
    x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
    return (
        2 * pairwise_distance(x, y).mean()
        - pairwise_distance(x, x).mean()
        - pairwise_distance(y, y).mean()
    )


def mmd(x, y, bandwidth=1.0, weights=None):
    def kernel(a, b):
        return jnp.exp(
            -jnp.sum((a[:, None] - b[None]) ** 2, axis=-1) / (2 * bandwidth**2)
        )

    if weights is None:
        return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
    weight = jnp.asarray(weights) / jnp.maximum(jnp.sum(weights), 1.0)
    return (
        weight @ kernel(x, x) @ weight
        + weight @ kernel(y, y) @ weight
        - 2 * weight @ kernel(x, y) @ weight
    )


def statistic_features(y):
    """Correlation rows, normalized PSD, and amplitude-bin occupancy."""
    y = jnp.asarray(y)
    centered = y - y.mean(axis=1, keepdims=True)
    variance = jnp.mean(centered**2, axis=1)
    correlation = (
        centered
        @ centered.T
        / (
            y.shape[1]
            * jnp.sqrt(jnp.maximum(variance[:, None] * variance[None], 1e-12))
        )
    )
    psd = jnp.abs(jnp.fft.rfft(centered, axis=1)) ** 2
    psd /= jnp.maximum(psd.sum(axis=1, keepdims=True), 1e-12)
    standardized = centered / jnp.sqrt(jnp.maximum(variance[:, None], 1e-12))
    # Smooth bin memberships retain a useful gradient.
    centers = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    occupancy = jax.nn.softmax(
        -((standardized[:, :, None] - centers) ** 2), axis=-1
    ).mean(axis=1)
    return jnp.concatenate([correlation, psd, occupancy], axis=1)


def trajectory_nll(predicted, observed, noise_sd=0.1, mask=None):
    if predicted.shape != observed.shape or noise_sd <= 0:
        raise ValueError("NLL shapes or observation noise invalid")
    mask = jnp.ones(predicted.shape[0]) if mask is None else jnp.asarray(mask)
    per_neuron = jnp.mean(
        0.5 * ((predicted - observed) / noise_sd) ** 2
        + jnp.log(noise_sd * jnp.sqrt(2 * jnp.pi)),
        axis=1,
    )
    return (per_neuron * mask).sum() / jnp.maximum(mask.sum(), 1)


@dataclass(frozen=True)
class LossWeights:
    trajectory: float = 1.0
    statistics: float = 1.0
    perturbation: float = 1.0
    prior: float = 0.01


def objective(
    predicted,
    observed,
    perturbations=None,
    prior=0.0,
    mask=None,
    weights=None,
    noise_sd=0.1,
):
    weights = weights or LossWeights()
    mask = jnp.ones(predicted.shape[0]) if mask is None else jnp.asarray(mask)
    terms = {
        "trajectory": trajectory_nll(predicted, observed, noise_sd, mask),
        "statistics": mmd(
            statistic_features(predicted * mask[:, None]),
            statistic_features(observed * mask[:, None]),
            weights=mask,
        ),
        "perturbation": jnp.array(0.0),
        "prior": prior,
    }
    if perturbations:
        terms["perturbation"] = jnp.mean(
            jnp.stack([energy_distance(a, b) for a, b in perturbations])
        )
    return sum(getattr(weights, key) * value for key, value in terms.items()), terms


def residual_penalty(epsilon, delta, log_sigma2):
    sigma2 = jnp.exp(log_sigma2)
    # Log variance term prevents minimizing the prior by inflating sigma forever.
    return 0.5 * jnp.sum(epsilon**2 / sigma2 + log_sigma2) + 0.5 * jnp.sum(delta**2)


def residual_analysis(epsilon, unused_features, names, significance=0.05):
    epsilon, x = np.asarray(epsilon), np.asarray(unused_features)
    if x.ndim != 2 or len(x) != len(epsilon) or x.shape[1] != len(names):
        raise ValueError("residual feature dimensions differ")
    result = []
    for i, name in enumerate(names):
        if len(epsilon) < 3 or np.ptp(x[:, i]) == 0:
            continue
        fit = linregress(x[:, i], epsilon)
        result.append(
            {
                "feature": name,
                "r_squared": fit.rvalue**2,
                "p_value": fit.pvalue,
                "candidate_addition": fit.pvalue < significance / max(len(names), 1),
            }
        )
    return result


@dataclass(frozen=True)
class ShootingWindow:
    preceding: object
    observed: object
    stimulus: object


def make_windows(recording, stimulus, window_steps, preceding_steps, lyapunov_time_s):
    dt = float(np.median(np.diff(recording.t)))
    if window_steps < 1 or preceding_steps < 1 or window_steps * dt >= lyapunov_time_s:
        raise ValueError(
            "shooting windows must be shorter than Lyapunov time with preceding context"
        )
    result = []
    for start in range(
        preceding_steps, recording.y.shape[1] - window_steps + 1, window_steps
    ):
        result.append(
            ShootingWindow(
                recording.y[:, start - preceding_steps : start],
                recording.y[:, start : start + window_steps],
                stimulus[start : start + window_steps],
            )
        )
    if not result:
        raise ValueError("recording too short for shooting windows")
    return result


def initialize_state_encoder(seed, n_observed, n_state):
    return {
        "weight": jax.random.normal(jax.random.key(seed), (2 * n_observed, n_state))
        * 0.01,
        "bias": jnp.zeros(n_state),
    }


def infer_initial_state(params, preceding):
    features = jnp.concatenate([preceding.mean(axis=1), preceding[:, -1]])
    return features @ params["weight"] + params["bias"]


def train(
    params,
    loss_fn,
    steps=100,
    learning_rate=1e-3,
    clip_norm=1.0,
    weight_decay=1e-4,
    seed=0,
    batch_count=1,
):
    if steps < 1 or learning_rate <= 0 or batch_count < 1:
        raise ValueError("invalid training configuration")
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    state = optimizer.init(params)

    @jax.jit
    def update(params, optimizer_state, batch):
        loss, gradient = jax.value_and_grad(loss_fn)(params, batch)
        changes, new_state = optimizer.update(gradient, optimizer_state, params)
        return (
            optax.apply_updates(params, changes),
            new_state,
            loss,
            optax.tree.norm(gradient),
        )

    rng = np.random.default_rng(seed)
    history = []
    for iteration in range(steps):
        params, state, loss, norm = update(
            params, state, jnp.asarray(rng.integers(batch_count))
        )
        if not np.isfinite(float(loss)) or not np.isfinite(float(norm)):
            raise FloatingPointError("non-finite training loss or gradient")
        history.append(
            {"step": iteration, "loss": float(loss), "gradient_norm": float(norm)}
        )
    return params, history


def laplace_samples(params, residual_fn, count=8, seed=0, prior_precision=1.0):
    """Generalized Gauss-Newton Laplace fallback for head-weight vectors."""
    from jax.flatten_util import ravel_pytree

    if count < 1 or prior_precision <= 0:
        raise ValueError("invalid Laplace approximation configuration")
    flat, unravel = ravel_pytree(params)
    jacobian = jax.jacrev(lambda vector: residual_fn(unravel(vector)).ravel())(flat)
    precision = jacobian.T @ jacobian + prior_precision * jnp.eye(flat.size)
    chol = jnp.linalg.cholesky(precision)
    noise = jax.random.normal(jax.random.key(seed), (count, flat.size))
    perturbations = jax.vmap(lambda e: jnp.linalg.solve(chol.T, e))(noise)
    return [unravel(flat + draw) for draw in perturbations]
