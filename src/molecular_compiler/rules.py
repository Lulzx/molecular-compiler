"""M4: molecule-token rules with exact expression and ligand masks."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .provenance import content_hash


@dataclass(frozen=True)
class RuleNetwork:
    params: dict
    threshold: float = 0.01
    budget: int | None = None
    adapter: tuple | None = None
    ortholog_groups: tuple = ()
    frozen_params: dict | None = None

    @classmethod
    def initialize(
        cls,
        seed=0,
        d_z=8,
        plm_dim=1280,
        ortholog_count=0,
        rank=None,
        budget_fraction=0.5,
        gauge_audited=False,
        ortholog_groups=(),
        max_parameters=None,
    ):
        if ortholog_groups:
            if len(set(ortholog_groups)) != len(ortholog_groups):
                raise ValueError("duplicate ortholog vocabulary entries")
            ortholog_count = len(ortholog_groups)
        elif ortholog_count:
            raise ValueError("ortholog offsets require a fixed named vocabulary")
        if d_z < 1 or plm_dim < 1 or ortholog_count < 0:
            raise ValueError("invalid rule dimensions")
        shapes = {
            "projection": (plm_dim, d_z),
            "ortholog": (ortholog_count, d_z),
            "encoder": (d_z, d_z),
            "density": (d_z,),
            "context": (d_z,),
            "gap": (d_z, d_z),
            "release": (d_z,),
            "sensitivity": (d_z,),
            "stp": (2 * d_z + 3, 3),
            "receptor_features": (3,),
            "bias": (1,),
        }
        keys = jax.random.split(jax.random.key(seed), len(shapes))
        params = {
            name: jax.random.normal(key, shape) * (0.1 / max(shape[0], 1) ** 0.5)
            for (name, shape), key in zip(shapes.items(), keys)
        }
        count = sum(x.size for x in params.values())
        budget = max_parameters
        if budget is not None and (not isinstance(budget, int) or budget < 1):
            raise ValueError("absolute parameter budget must be a positive integer")
        if budget is not None or rank is not None:
            if not gauge_audited:
                raise ValueError("gauge audit must precede parameter-budget freeze")
            if budget is None:
                if not 0 < budget_fraction <= 1:
                    raise ValueError("parameter budget fraction must be in (0,1]")
                # An explicit legacy heuristic; rank is not a capacity theorem.
                budget = int(rank * budget_fraction)
            if count > budget:
                raise ValueError(
                    f"{count} parameters exceed frozen budget {budget}; reduce capacity"
                )
        return cls(params, budget=budget, ortholog_groups=tuple(ortholog_groups))

    @classmethod
    def initialize_compact(
        cls,
        rank=None,
        budget_fraction=0.5,
        gauge_audited=False,
        seed=0,
        d_z=64,
        ortholog_groups=(),
        max_parameters=None,
    ):
        """Two tied head coefficients on a frozen PLM/set-encoder basis.

        This conservative architecture is an option for small absolute budgets.
        Input rank is a diagnostic; held-out capacity ablations must establish
        whether this frozen basis is sufficient for real data.
        """
        if not gauge_audited:
            raise ValueError("gauge audit must precede parameter-budget freeze")
        budget = max_parameters
        if budget is None:
            if rank is None or not 0 < budget_fraction <= 1:
                raise ValueError("supply an absolute budget or explicit rank heuristic")
            budget = int(rank * budget_fraction)
        if not isinstance(budget, int) or budget < 2:
            raise ValueError("budget cannot support two learned coefficients")
        base = cls.initialize(seed=seed, d_z=d_z, ortholog_groups=ortholog_groups)
        return cls(
            {"scale": jnp.ones(1), "bias": jnp.zeros(1)},
            budget=budget,
            ortholog_groups=tuple(ortholog_groups),
            frozen_params=base.params,
        )

    @classmethod
    def initialize_partial(cls, frozen, trainable, budget, ortholog_groups=()):
        """Frozen basis with a named trainable subset under an absolute budget."""
        count = sum(int(np.size(v)) for v in trainable.values())
        if not isinstance(budget, int) or count > budget:
            raise ValueError(f"{count} parameters exceed frozen budget {budget}")
        if not set(trainable) <= set(frozen):
            raise ValueError("trainable entries must name frozen-basis entries")
        frozen = {k: jnp.asarray(v) for k, v in frozen.items()}
        for name, value in trainable.items():
            if np.shape(value) != frozen[name].shape:
                raise ValueError(f"{name}: trainable shape differs from basis")
        return cls(
            {k: jnp.asarray(v) for k, v in trainable.items()},
            budget=budget,
            ortholog_groups=tuple(ortholog_groups),
            frozen_params=frozen,
        )

    @property
    def weights(self):
        if self.frozen_params is None:
            return self.params
        if "scale" not in self.params:
            # Partially frozen network: trainable entries override frozen ones.
            return {**self.frozen_params, **self.params}
        return {
            name: (
                self.params["bias"]
                if name == "bias"
                else value
                if name in {"projection", "encoder", "ortholog"}
                else value * self.params["scale"]
            )
            for name, value in self.frozen_params.items()
        }

    @property
    def parameter_count(self):
        return sum(v.size for v in self.params.values())

    def with_params(self, params):
        if jax.tree.structure(params) != jax.tree.structure(self.params):
            raise ValueError("parameter structure changed")
        if any(
            a.shape != b.shape
            for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(self.params))
        ):
            raise ValueError("parameter shapes changed")
        return replace(self, params=params)

    def checkpoint_hash(self):
        return content_hash(
            {k: np.asarray(v) for k, v in self.params.items()},
            self.threshold,
            self.ortholog_groups,
            None
            if self.frozen_params is None
            else {k: np.asarray(v) for k, v in self.frozen_params.items()},
        )

    def encode(
        self, abundance, embeddings, ortholog_indices, nontransferable_indices=()
    ):
        tokens = embeddings @ self.weights["projection"]
        if self.weights["ortholog"].shape[0]:
            tokens += jnp.where(
                ortholog_indices[:, None] >= 0,
                self.weights["ortholog"][jnp.maximum(ortholog_indices, 0)],
                0,
            )
        expressed = abundance > self.threshold
        weights = jnp.where(expressed, abundance, 0)
        if nontransferable_indices:
            weights = weights.at[:, jnp.asarray(nontransferable_indices)].set(0)
        encoded = (weights @ jnp.tanh(tokens)) / jnp.maximum(
            weights.sum(axis=1, keepdims=True), 1
        )
        return jnp.tanh(encoded @ self.weights["encoder"]), tokens, expressed

    def densities(self, z, tokens, mask, zero_shot=True):
        context = self.weights["context"]
        if not zero_shot and self.adapter is not None:
            a, b = self.adapter
            context = context + (a @ b).reshape(context.shape)
        logits = (
            (z @ context)[:, None]
            + (tokens @ self.weights["density"])[None]
            + self.weights["bias"]
        )
        return jax.nn.softplus(logits) * mask

    def receptor_densities(
        self, z, tokens, expressed, post, features, indices, ligand_mask
    ):
        logits = (
            (z[post] @ self.weights["context"])[:, None]
            + (tokens[indices] @ self.weights["density"])[None]
            + self.weights["bias"]
        )
        logits += (features @ self.weights["receptor_features"])[:, None]
        return jax.nn.softplus(logits) * expressed[post][:, indices] * ligand_mask

    def gap_conductance(self, z, i, j, area, innexin_mask):
        h = (self.weights["gap"] + self.weights["gap"].T) / 2
        logits = jnp.einsum("ed,df,ef->e", z[i], h, z[j])
        return area * jax.nn.softplus(logits) * innexin_mask[i] * innexin_mask[j]

    def plasticity(self, z, pre, post, features):
        raw = jnp.concatenate([z[pre], z[post], features], axis=1) @ self.weights["stp"]
        unit = jax.nn.sigmoid(raw)
        return unit[:, 0], 0.01 + 2.99 * unit[:, 1], 0.001 + 1.999 * unit[:, 2]

    def peptide_profiles(self, z, tokens, expressed, peptide_indices, pairs):
        # Ligand tokens do not parameterize coupling; species-native pairing
        # selects receptor tokens. This preserves receptor-based transfer.
        p = (
            jax.nn.softplus(z @ self.weights["release"])[:, None]
            * expressed[:, peptide_indices]
        )
        q = jnp.zeros_like(p)
        for pi, receptor in pairs:
            gain = jax.nn.softplus(
                z @ self.weights["sensitivity"]
                + tokens[receptor] @ self.weights["density"]
            )
            q = q.at[:, pi].add(gain * expressed[:, receptor])
        return p, q

    def ortholog_penalty(self):
        return jnp.sum(self.weights["ortholog"] ** 2)
