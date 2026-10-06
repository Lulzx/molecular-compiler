"""M5: shared kinetics, biophysical reversals and Jaxley mechanisms."""

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np


def rush_larsen(x, steady, tau_s, dt_s):
    return steady + (x - steady) * jnp.exp(-dt_s / jnp.maximum(tau_s, 1e-12))


def nernst(outside, inside, charge=1, temperature_C=20):
    return (
        1000
        * 8.314462618
        * (temperature_C + 273.15)
        / (charge * 96485.33212)
        * jnp.log(outside / inside)
    )


def ghk(inside, outside, permeability, temperature_C=20):
    """Monovalent GHK; chloride concentration is inverted in numerator."""
    numerator = sum(
        permeability.get(ion, 0) * (inside[ion] if ion == "Cl" else outside[ion])
        for ion in permeability
    )
    denominator = sum(
        permeability.get(ion, 0) * (outside[ion] if ion == "Cl" else inside[ion])
        for ion in permeability
    )
    return (
        1000
        * 8.314462618
        * (temperature_C + 273.15)
        / 96485.33212
        * jnp.log(numerator / denominator)
    )


@dataclass(frozen=True)
class KineticRecord:
    molecule_id: str
    model_form: str
    params_prior_mean: tuple
    params_prior_cov: tuple
    ion_selectivity: dict
    source: str
    temperature_C: float = 20
    family: str = "generic"
    embedding: tuple = ()
    gbar_nS: float = 0.1
    reversal_mV: float | None = None
    tau_s: float = 0.01
    v_half_mV: float = -35
    slope_mV: float = 10
    q10: float = 2
    params: tuple | None = None
    mechanism: object | None = field(default=None, compare=False, repr=False)
    mechanism_gate: str | None = None
    fallback_from: str | None = None
    parameter_names: tuple = ()
    transition_generator: tuple = ()
    open_states: tuple = (1,)
    gate_powers: tuple = ()
    transport_equilibrium_mM: dict = field(default_factory=dict)
    transporter_current_pA: float = 0.0
    # Optional first-order inactivation gate for the generic HH form:
    # h_inf = 1 / (1 + exp((V - h_v_half) / h_slope)), open = m^m_power * h.
    h_v_half_mV: float | None = None
    h_slope_mV: float = 5.0
    h_tau_s: float = 0.1
    m_power: int = 1

    def __post_init__(self):
        if self.model_form not in {
            "HH",
            "markov",
            "ligand_gated",
            "gpcr",
            "transporter",
        }:
            raise ValueError("unknown kinetic form")
        mean, cov = (
            np.asarray(self.params_prior_mean),
            np.asarray(self.params_prior_cov),
        )
        if (
            cov.shape != (mean.size, mean.size)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(cov))
        ):
            raise ValueError("invalid kinetic prior dimensions")
        if mean.size and np.min(np.linalg.eigvalsh(cov)) <= 0:
            raise ValueError("kinetic covariance must be positive definite")
        if (
            self.gbar_nS < 0
            or self.tau_s <= 0
            or self.q10 <= 0
            or self.slope_mV == 0
            or not self.source
        ):
            raise ValueError("invalid kinetics or missing source")
        if any(
            ion not in {"Na", "K", "Cl", "Ca"} or concentration <= 0
            for ion, concentration in self.transport_equilibrium_mM.items()
        ):
            raise ValueError(
                "transport equilibria require known ions and positive concentrations"
            )
        if self.parameter_names and len(self.parameter_names) != mean.size:
            raise ValueError("kinetic parameter names must match prior dimensions")
        if self.model_form == "markov":
            generator = np.asarray(self.transition_generator)
            if (
                generator.ndim != 2
                or generator.shape[0] != generator.shape[1]
                or not len(generator)
            ):
                raise ValueError("Markov models require a square transition generator")
            off_diagonal = generator - np.diag(np.diag(generator))
            if not np.allclose(generator.sum(axis=1), 0) or np.any(off_diagonal < 0):
                raise ValueError(
                    "Markov transition generator must conserve probability"
                )
            if any(index < 0 or index >= len(generator) for index in self.open_states):
                raise ValueError("invalid Markov open-state index")
        if self.gate_powers and len(self.gate_powers) != self.state_size:
            raise ValueError("gate powers must match channel state size")
        if self.h_v_half_mV is not None and (
            self.model_form != "HH"
            or self.mechanism is not None
            or self.h_slope_mV <= 0
            or self.h_tau_s <= 0
            or self.m_power < 1
        ):
            raise ValueError("inactivation gate requires generic HH kinetics")
        if self.params is not None and len(self.params) != mean.size:
            raise ValueError("kinetic parameter dimensions differ from prior")

    def parameter(self, name, default):
        if name not in self.parameter_names:
            return default
        values = self.params if self.params is not None else self.params_prior_mean
        return values[self.parameter_names.index(name)]

    @property
    def conductance(self):
        return self.parameter("gbar_nS", self.gbar_nS)

    def time_constant(self, temperature_C):
        return self.parameter("tau_s", self.tau_s) / self.parameter(
            "q10", self.q10
        ) ** ((temperature_C - self.temperature_C) / 10)

    @property
    def state_size(self):
        if self.model_form == "markov":
            return len(self.transition_generator)
        if self.mechanism is not None:
            return len(self.mechanism.channel_states)
        return 1 if self.h_v_half_mV is None else 2

    def _h_steady(self, voltage):
        return jax.nn.sigmoid(-(voltage - self.h_v_half_mV) / self.h_slope_mV)

    def initial_gates(self, voltage, width):
        if self.state_size > width:
            raise ValueError("gating state width too small")
        result = jnp.zeros(voltage.shape + (width,), dtype=voltage.dtype)
        if self.model_form == "markov":
            if not self.state_size:
                raise ValueError("Markov models require a transition generator")
            return result.at[..., 0].set(1.0)
        if self.mechanism is not None:
            states = self.mechanism.init_state(
                {}, voltage, self.mechanism.channel_params, 0.0
            )
            for index, key in enumerate(sorted(self.mechanism.channel_states)):
                result = result.at[..., index].set(states[key])
            return result
        steady = jax.nn.sigmoid(
            (voltage - self.parameter("v_half_mV", self.v_half_mV))
            / self.parameter("slope_mV", self.slope_mV)
        )
        result = result.at[..., 0].set(steady)
        if self.h_v_half_mV is not None:
            result = result.at[..., 1].set(self._h_steady(voltage))
        return result

    def step_gates(self, voltage, old, dt_s, temperature_C):
        if self.model_form == "markov":
            from jax.scipy.linalg import expm

            generator = jnp.asarray(self.transition_generator) * self.parameter(
                "rate_scale", 1.0
            )
            probability = old[..., : self.state_size] @ expm(generator * dt_s)
            return old.at[..., : self.state_size].set(probability)
        if self.mechanism is not None:
            keys = sorted(self.mechanism.channel_states)
            states = {key: old[..., index] for index, key in enumerate(keys)}
            parameters = dict(self.mechanism.channel_params)
            for key in self.parameter_names:
                if key in parameters:
                    parameters[key] = self.parameter(key, parameters[key])
            updated = self.mechanism.update_states(
                states,
                dt_s * 1000 * self.q10 ** ((temperature_C - self.temperature_C) / 10),
                voltage,
                parameters,
            )
            for index, key in enumerate(keys):
                old = old.at[..., index].set(updated[key])
            return old
        new = old.at[..., 0].set(self.gate(voltage, old[..., 0], dt_s, temperature_C))
        if self.h_v_half_mV is not None:
            tau = self.h_tau_s / self.parameter("q10", self.q10) ** (
                (temperature_C - self.temperature_C) / 10
            )
            new = new.at[..., 1].set(
                rush_larsen(old[..., 1], self._h_steady(voltage), tau, dt_s)
            )
        return new

    def open_probability(self, gates):
        if self.model_form == "markov":
            return gates[..., jnp.array(self.open_states)].sum(axis=-1)
        default = (1.0,) * self.state_size
        if self.mechanism is None and self.h_v_half_mV is not None:
            default = (float(self.m_power), 1.0)
        powers = jnp.asarray(self.gate_powers or default)
        return jnp.prod(gates[..., : self.state_size] ** powers, axis=-1)

    def gate(self, voltage, old, dt_s, temperature_C):
        if self.mechanism is not None:
            if self.state_size != 1:
                raise ValueError("multi-gate mechanisms require step_gates")
            return self.step_gates(voltage, old[..., None], dt_s, temperature_C)[..., 0]
        from .jaxley_backend import gate_step

        return gate_step(self, voltage, old, dt_s, temperature_C)

    def prior_penalty(self):
        actual = jnp.asarray(
            self.params if self.params is not None else self.params_prior_mean
        )
        diff = actual - jnp.asarray(self.params_prior_mean)
        return diff @ jnp.linalg.solve(jnp.asarray(self.params_prior_cov), diff)


@dataclass
class KineticsLibrary:
    records: dict
    temperature_C: float = 20
    metadata: dict | None = None
    gene_families: dict = field(default_factory=dict)

    def family_for(self, gene_id, default):
        """Curated family label when assigned (spec 10.2), else the default."""
        return self.gene_families.get(gene_id, default)

    def ensure(self, gene_id, embedding, family=None):
        if gene_id in self.records:
            return self.records[gene_id]
        candidates = [
            r
            for r in self.records.values()
            if r.embedding
            and len(r.embedding) == len(embedding)
            and (family is None or r.family == family)
        ]
        if not candidates:
            raise ValueError(
                f"no kinetic prior or embedded family neighbor for {gene_id}"
            )
        reference = min(
            candidates,
            key=lambda r: np.linalg.norm(np.asarray(r.embedding) - embedding),
        )
        return replace(
            reference,
            molecule_id=gene_id,
            embedding=tuple(embedding),
            params_prior_cov=tuple(
                map(tuple, np.asarray(reference.params_prior_cov) * 4)
            ),
            fallback_from=reference.molecule_id,
        )

    def penalty(self):
        return sum(r.prior_penalty() for r in self.records.values())

    def deviation_report(self, threshold=3):
        return [
            {
                "molecule_id": r.molecule_id,
                "mahalanobis_squared": float(r.prior_penalty()),
                "flagged": float(r.prior_penalty()) > threshold**2,
                "interpretation": "consistent_with_data",
            }
            for r in self.records.values()
        ]

    def family_accuracy(self):
        records = [r for r in self.records.values() if r.embedding]
        outcomes = []
        for r in records:
            candidates = [s for s in records if s is not r]
            if candidates:
                nearest = min(
                    candidates,
                    key=lambda s: np.linalg.norm(np.asarray(r.embedding) - s.embedding),
                )
                outcomes.append(nearest.family == r.family)
        return float(np.mean(outcomes)) if outcomes else None


def markov_step(probabilities, generator, dt_s):
    from jax.scipy.linalg import expm

    return probabilities @ expm(generator * dt_s)


def binding_step(open_state, ligand_nM, ec50_nM, tau_s, dt_s):
    target = ligand_nM / (ec50_nM + ligand_nM)
    return rush_larsen(open_state, target, tau_s, dt_s)


def transporter_concentration(density, external, basal=10, maximum=80):
    return jnp.minimum(external, basal + (maximum - basal) * density / (1 + density))
