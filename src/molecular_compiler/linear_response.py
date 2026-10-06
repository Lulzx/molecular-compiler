"""Phase 0 linear-response models of the signal propagation atlas.

The atlas reports trial-averaged response amplitudes, not traces, so Phase 0
compares models through their steady-state linear response. Each model defines
an effective coupling W (row = responder, column = source). The network state
under unit drive of neuron j is the j-th column of (D - W)^-1, with
D_ii = 1 + sum_k |W_ik|, which is a conductance network whose leak grows with
its coupling and is therefore always nonsingular. The prediction for responder
i is scaled by the stimulated neuron's own observed response (Section 6.3: the
drive is estimated from the stimulated neuron) and one global gain (uniform
gauge).

This is a declared approximation of the M4-M9 pipeline: no kinetics, no
voltage dependence and no time course. It tests whether the molecular rule,
connectome gate and peptidergic pathway carry predictive information about
causal responses, which is the question K1, K2 and K4 ask.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass
class AtlasInputs:
    """Dense worm-scale inputs, all indexed by canonical neuron order."""

    chemical: np.ndarray  # [post, pre] log1p section counts
    gap: np.ndarray  # symmetric log1p section counts
    expression: np.ndarray  # [neuron, gene] log1p TPM / global max
    tokens: np.ndarray  # [gene, k_max] frozen PCA of PLM embeddings
    releases: np.ndarray  # [neuron, transmitter] 0/1
    receptor_ligand: np.ndarray  # [gene] transmitter index or -1
    innexins: np.ndarray  # gene indices
    pair_peptide: np.ndarray  # gene index per peptide-receptor pair
    pair_receptor: np.ndarray  # gene index per pair
    pair_potency: np.ndarray  # -log10(EC50 / 1 uM), clipped at 0
    transmitter_sign: np.ndarray  # [neuron] +1 / -1 / 0 for B0
    identity: np.ndarray  # [neuron, d] PCA of expression for B1
    names: list = field(default_factory=list)


def _effective_response(w, autoresponse, gain):
    n = w.shape[0]
    w = w * (1 - jnp.eye(n))
    leak = 1 + jnp.sum(jnp.abs(w), axis=1)
    x = jnp.linalg.solve(jnp.diag(leak) - w, jnp.eye(n))
    scale = autoresponse / jnp.diag(x)
    return gain * x * scale[None, :]


def _softplus(x):
    return jax.nn.softplus(x)


def compositional_coupling(params, inputs, k, peptides=True, per_gene=False):
    """M4-form rule: PLM-token heads, structural masks, factorized peptides."""
    tokens = inputs.tokens[:, :k]
    e = inputs.expression
    theta = tokens @ params["receptor"] + params["receptor_bias"]
    if per_gene:
        theta = theta + params["receptor_offset"]
    ligand = inputs.receptor_ligand
    # Receptor drive per (responder, transmitter); masks are exact zeros.
    receptor = jnp.stack(
        [(e * jnp.where(ligand == t, theta, 0.0)).sum(axis=1) for t in range(3)],
        axis=1,
    )
    chemical_rule = receptor @ inputs.releases.T + params["chemical_bias"]
    w = inputs.chemical * chemical_rule
    inx = e[:, inputs.innexins] @ tokens[inputs.innexins]
    h = params["gap"] + params["gap"].T
    w = w + inputs.gap * _softplus(params["gap_bias"] + inx @ h @ inx.T)
    if peptides and len(inputs.pair_peptide):
        sensitivity = (
            tokens[inputs.pair_receptor] @ params["peptide"] + params["peptide_bias"]
        )
        if per_gene:
            sensitivity = sensitivity + params["peptide_offset"][inputs.pair_receptor]
        release = e[:, inputs.pair_peptide] * inputs.pair_potency
        receive = e[:, inputs.pair_receptor] * sensitivity
        w = w + _softplus(params["peptide_gain"]) * (receive @ release.T)
    return w


def init_compositional(k, n_genes, seed=0, per_gene=False):
    keys = jax.random.split(jax.random.key(seed), 3)
    params = {
        "receptor": 0.01 * jax.random.normal(keys[0], (k,)),
        "receptor_bias": jnp.array(0.0),
        "chemical_bias": jnp.array(0.05),
        "gap": 0.01 * jax.random.normal(keys[1], (k, k)),
        "gap_bias": jnp.array(-2.0),
        "peptide": 0.01 * jax.random.normal(keys[2], (k,)),
        "peptide_bias": jnp.array(0.0),
        "peptide_gain": jnp.array(-3.0),
        "log_gain": jnp.array(0.0),
    }
    if per_gene:
        params["receptor_offset"] = jnp.zeros(n_genes)
        params["peptide_offset"] = jnp.zeros(n_genes)
    return params


def parameter_count(params, k=None):
    """Trainable count, with the symmetric gap head counted once."""
    total = sum(int(np.size(v)) for v in jax.tree_util.tree_leaves(params))
    if "gap" in params and np.ndim(params["gap"]) == 2:
        k = params["gap"].shape[0]
        total -= k * (k - 1) // 2
    return total


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    k: int = 0
    peptides: bool = False
    per_gene: bool = False
    hidden: int = 32
    ridge: float = 1e-3


def init(model, inputs, seed=0):
    n, g = inputs.expression.shape
    if model.family == "compositional":
        return init_compositional(model.k, g, seed, model.per_gene)
    if model.family == "connectome_only":
        return {"chemical": jnp.array(0.05), "gap": jnp.array(-2.0), "log_gain": 0.0}
    if model.family == "connectome_free":
        chem, gap = inputs.chemical > 0, inputs.gap > 0
        return {
            "chemical": jnp.zeros(int(chem.sum())),
            "gap": jnp.full(int(np.triu(gap, 1).sum()), -2.0),
            "log_gain": jnp.array(0.0),
        }
    if model.family == "dense_free":
        return {"w": jnp.zeros((n, n)), "log_gain": jnp.array(0.0)}
    if model.family == "black_box":
        d = inputs.identity.shape[1]
        keys = jax.random.split(jax.random.key(seed), 3)

        def mlp(key):
            a, b = jax.random.split(key)
            return {
                "w1": jax.random.normal(a, (2 * d, model.hidden)) / np.sqrt(2 * d),
                "b1": jnp.zeros(model.hidden),
                "w2": 0.01 * jax.random.normal(b, (model.hidden,)),
                "b2": jnp.array(0.0),
            }

        return {
            "chemical": mlp(keys[0]),
            "gap": mlp(keys[1]),
            "dense": mlp(keys[2]),
            "log_gain": jnp.array(0.0),
        }
    raise ValueError(model.family)


def coupling(model, params, inputs):
    if model.family == "compositional":
        return compositional_coupling(
            params, inputs, model.k, model.peptides, model.per_gene
        )
    if model.family == "connectome_only":
        signed = inputs.chemical * inputs.transmitter_sign[None, :]
        return params["chemical"] * signed + _softplus(params["gap"]) * inputs.gap
    if model.family == "connectome_free":
        chem = (
            jnp.zeros(inputs.chemical.shape)
            .at[inputs.chemical > 0]
            .set(params["chemical"])
        )
        upper = np.triu(inputs.gap > 0, 1)
        g = jnp.zeros(inputs.gap.shape).at[upper].set(_softplus(params["gap"]))
        return inputs.chemical * chem + inputs.gap * (g + g.T)
    if model.family == "dense_free":
        return params["w"]
    if model.family == "black_box":
        z = inputs.identity
        n, d = z.shape
        pairs = jnp.concatenate(
            [
                jnp.broadcast_to(z[:, None], (n, n, d)),
                jnp.broadcast_to(z[None], (n, n, d)),
            ],
            axis=2,
        )

        def apply(p):
            hidden = jnp.tanh(pairs @ p["w1"] + p["b1"])
            return hidden @ p["w2"] + p["b2"]

        gap = apply(params["gap"])
        return (
            inputs.chemical * apply(params["chemical"])
            + inputs.gap * _softplus(0.5 * (gap + gap.T))
            + 0.1 * apply(params["dense"])
        )
    raise ValueError(model.family)


def predict(model, params, inputs, autoresponse):
    w = coupling(model, params, inputs)
    return _effective_response(w, autoresponse, jnp.exp(params["log_gain"]))


def fit(model, inputs, target, weights, autoresponse, steps=1500, seed=0, lr=0.02):
    """Weighted least squares on observed off-diagonal training pairs."""
    params = init(model, inputs, seed)
    target = jnp.asarray(np.nan_to_num(target))
    weights = jnp.asarray(weights)
    autoresponse = jnp.asarray(autoresponse)
    total = jnp.maximum(weights.sum(), 1.0)
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(lr, 0.0))
    state = optimizer.init(params)

    def loss(p):
        error = predict(model, p, inputs, autoresponse) - target
        fit_term = jnp.sum(weights * error**2) / total
        penalty = sum(
            jnp.sum(v**2)
            for key, v in _flatten(p)
            if not key.endswith(("bias", "log_gain", "b1", "b2"))
        )
        return fit_term + model.ridge * penalty

    @jax.jit
    def update(p, s):
        value, grad = jax.value_and_grad(loss)(p)
        changes, s = optimizer.update(grad, s, p)
        return optax.apply_updates(p, changes), s, value

    history = []
    for step in range(steps):
        params, state, value = update(params, state)
        if step % 250 == 0 or step == steps - 1:
            history.append(float(value))
    if not np.all(np.isfinite(history)):
        raise FloatingPointError(f"{model.name}: non-finite training loss")
    return params, history


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _flatten(value, f"{prefix}/{key}" if prefix else key)
    else:
        yield prefix, tree
