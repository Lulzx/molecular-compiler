"""M6 linker: geometry times molecular rules, no stored synaptic sign."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .kinetics import ghk, nernst
from .surrogates import Surrogate


@dataclass(frozen=True)
class ModulatoryState:
    concentrations: tuple = ()
    signaling: tuple = (0.0, 0.0, 0.0, 0.0)
    mode: str = "global"
    decay_length_um: object | None = None


@dataclass(frozen=True)
class ResolutionPolicy:
    release_mode: str = "graded"
    spike_threshold_mV: float = -20.0
    dt_s: float = 0.0005
    n_comp: int = 3
    capacitance_pF: float = 1.0
    leak_nS: float = 0.02
    leak_reversal_mV: float = -60.0
    axial_nS: float = 0.1
    # Conductance per unit of connectome size (synapse size, gap contact area).
    # 1.0 keeps sizes as given; EM section counts need an explicit conversion.
    synapse_unit_nS: float = 1.0
    gap_unit_nS: float = 1.0
    length_constant_um: float = 100.0
    chloride_prior_mM: tuple = (5.0, 40.0)
    rest_voltage_range_mV: tuple = (-65.0, -35.0)
    solver: str = "auto"
    dense_limit: int = 1024
    solve_tolerance: float = 1e-5
    max_cg_iterations: int = 100
    parallel_iterations: int = 8
    temperature_C: float = 20.0
    grid_shape: tuple = (16, 16, 16)
    grid_spacing_um: float = 10.0
    decay_length_um: float = 50.0
    extracellular_mM: tuple = (145.0, 5.0, 120.0, 2.0)
    intracellular_mM: tuple = (15.0, 140.0, 10.0, 0.0001)

    def __post_init__(self):
        if self.release_mode not in {"graded", "spiking"}:
            raise ValueError("release mode must be graded or spiking")
        if (
            self.dt_s <= 0
            or self.n_comp < 1
            or self.capacitance_pF <= 0
            or self.leak_nS <= 0
        ):
            raise ValueError("resolution must yield a positive definite voltage system")
        if (
            self.axial_nS < 0
            or self.synapse_unit_nS <= 0
            or self.gap_unit_nS <= 0
            or self.length_constant_um <= 0
            or self.solve_tolerance <= 0
            or self.max_cg_iterations < 1
        ):
            raise ValueError("invalid coupling or solver configuration")
        if self.solver not in {"auto", "dense", "pcg"}:
            raise ValueError("unknown voltage solver")
        if any(
            x <= 0
            for x in self.chloride_prior_mM
            + self.extracellular_mM
            + self.intracellular_mM
        ):
            raise ValueError("ion concentrations must be positive")
        if self.chloride_prior_mM[0] > self.chloride_prior_mM[1]:
            raise ValueError("chloride prior endpoints out of order")
        if (
            self.grid_spacing_um <= 0
            or self.decay_length_um <= 0
            or any(x < 2 for x in self.grid_shape)
        ):
            raise ValueError("invalid spatial grid")

    @classmethod
    def default(cls, species="C. elegans"):
        if species in {"C. elegans", "Caenorhabditis elegans"}:
            return cls()
        if species in {"Drosophila", "Drosophila melanogaster"}:
            return cls(dt_s=0.0001, n_comp=4, release_mode="spiking")
        if species in {"zebrafish", "Danio rerio"}:
            return cls(dt_s=0.00005, n_comp=4, release_mode="spiking")
        raise ValueError("species needs an explicit resolution policy")


@dataclass
class SimGraph:
    neuron_params: dict
    surrogates: tuple
    syn_post_ptr: jax.Array
    syn_pre_idx: jax.Array
    syn_post_idx: jax.Array
    syn_params: dict
    gap: dict
    neuromod: dict
    metadata: dict
    resolution: ResolutionPolicy
    channel_records: tuple = ()
    receptor_records: tuple = ()
    peptide_records: tuple = ()
    residuals: dict | None = None
    syn_pre_ptr: object | None = None
    syn_pre_edges: object | None = None
    partition_plan: object | None = None
    schwarz_layout: object | None = None
    recompile: object | None = None
    fast_plan: object | None = None

    @property
    def n_neurons(self):
        return self.neuron_params["capacitance"].shape[0]


def _indices(graph):
    genes = graph.molecular.genes.to_pylist()
    classes = {
        name: [i for i, g in enumerate(genes) if g["molecule_class"] == name]
        for name in {g["molecule_class"] for g in genes}
    }
    groups = sorted(
        {g["ortholog_group"] for g in genes if g["ortholog_group"] is not None}
    )
    ortholog = jnp.array(
        [
            groups.index(g["ortholog_group"]) if g["ortholog_group"] else -1
            for g in genes
        ]
    )
    return genes, classes, ortholog


def _skeleton_geometry(skeletons, syn, id_index, policy, resting, path, diam):
    """Opt-in M6 morphology: per-synapse path, diameter and compartment from trees."""
    path, diam = np.array(path), np.array(diam)
    compartment = np.zeros(len(syn), dtype=np.int32)
    has = np.zeros(len(syn), dtype=bool)
    if isinstance(resting, jax.core.Tracer):
        # Reduction bisects on concrete conductances; it cannot run under grad.
        raise TypeError(
            "skeleton compartment reduction needs concrete channel densities; "
            "compile with skeletons outside jax transformations"
        )
    resting = np.asarray(resting)
    reductions = {}
    for s, row in enumerate(syn):
        tree = skeletons.get(row["post_id"])
        if tree is None:
            continue
        i = id_index[row["post_id"]]
        if i not in reductions:
            # Same lambda0 as the compiler's per-synapse length constant.
            lam0 = policy.length_constant_um * np.sqrt(policy.leak_nS / resting[i])
            reductions[i] = tree.reduce(policy.n_comp, lam0)
        node = tree.nearest_node(row["xyz"])
        path[s], diam[s] = tree.path_um[node], 2 * tree.radius[node]
        compartment[s] = reductions[i].node_compartment[node]
        has[s] = True
    return (
        jnp.asarray(path),
        jnp.asarray(diam),
        jnp.asarray(compartment),
        jnp.asarray(has),
    )


def _compile(
    graph,
    rules,
    kinetics,
    state=None,
    resolution=None,
    identity_sample=None,
    identity_deviation=None,
    skeletons=None,
):
    policy = resolution or ResolutionPolicy.default(graph.metadata["species"])
    state = state or ModulatoryState()
    if state.mode not in {"global", "grid"}:
        raise ValueError("modulatory mode must be global or grid")
    genes, classes, _ = _indices(graph)
    ortholog = np.array(
        [
            rules.ortholog_groups.index(g["ortholog_group"])
            if g["ortholog_group"] in rules.ortholog_groups
            else -1
            for g in genes
        ],
        dtype=np.int32,
    )
    if rules.weights["ortholog"].shape[0] and rules.weights["ortholog"].shape[0] <= int(
        ortholog.max(initial=-1)
    ):
        raise ValueError("ortholog vocabulary too small")
    embeddings = jnp.asarray(
        np.asarray([g["plm_embedding"] for g in genes], dtype=np.float32)
    )
    if embeddings.shape[1] != rules.weights["projection"].shape[0]:
        raise ValueError("PLM dimension does not match rule network")
    abundance = (
        graph.abundance
        if identity_sample is None
        else graph.sample_abundance(identity_sample)
    )
    z, tokens, expressed = rules.encode(
        abundance,
        embeddings,
        ortholog,
        nontransferable_indices=classes.get("peptide", []),
    )
    if identity_deviation is not None:
        if np.shape(identity_deviation) != z.shape:
            raise ValueError("identity deviation dimensions invalid")
        z = z + jnp.asarray(identity_deviation)
    density = rules.densities(z, tokens, expressed, zero_shot=rules.adapter is None)
    n, comp = len(graph.neuron_ids), policy.n_comp
    id_index = {int(v): i for i, v in enumerate(graph.neuron_ids)}
    syn = graph.connectome.synapses.to_pylist()
    # Counting sort gives O(nnz + N), including CSR construction.
    buckets = [[] for _ in range(n)]
    for row in syn:
        buckets[id_index[row["post_id"]]].append(row)
    syn = [row for bucket in buckets for row in bucket]
    post = jnp.array([id_index[r["post_id"]] for r in syn], dtype=jnp.int32)
    pre = jnp.array([id_index[r["pre_id"]] for r in syn], dtype=jnp.int32)
    counts = np.array([len(bucket) for bucket in buckets])
    ptr = jnp.asarray(np.r_[0, np.cumsum(counts)], dtype=jnp.int32)
    channel_indices = classes.get("channel", [])
    receptor_indices = classes.get("receptor", [])
    peptide_indices = classes.get("peptide", [])

    def record(i):
        return kinetics.ensure(
            genes[i]["gene_id"],
            np.asarray(genes[i]["plm_embedding"]),
            kinetics.family_for(genes[i]["gene_id"], genes[i]["molecule_class"]),
        )

    # Structural presence is known before entering the differentiable program.
    active = np.any(np.asarray(abundance) > rules.threshold, axis=0)
    for i, g in enumerate(genes):
        if active[i] and g["molecule_class"] in {
            "channel",
            "receptor",
            "transporter",
            "innexin",
            "gpcr",
        }:
            record(i)
    ch_records, rec_records = (
        tuple(map(record, channel_indices)),
        tuple(map(record, receptor_indices)),
    )
    outside = dict(zip(("Na", "K", "Cl", "Ca"), policy.extracellular_mM))
    inside = {
        ion: jnp.full(n, concentration)
        for ion, concentration in zip(("Na", "K", "Cl", "Ca"), policy.intracellular_mM)
    }
    transporters = classes.get("transporter", [])
    transporter_current = jnp.zeros(n)
    for ion, basal_concentration in list(inside.items()):
        total_density = jnp.zeros(n)
        target_sum = jnp.zeros(n)
        for transporter in transporters:
            model = record(transporter)
            if not model.transport_equilibrium_mM:
                raise ValueError(
                    f"{model.molecule_id}: transporter concentration model required"
                )
            if ion in model.transport_equilibrium_mM:
                rho = density[:, transporter]
                total_density += rho
                equilibrium = model.parameter(
                    f"{ion}_equilibrium_mM", model.transport_equilibrium_mM[ion]
                )
                target_sum += rho * equilibrium
        inside[ion] = (basal_concentration + target_sum) / (1 + total_density)
    for transporter in transporters:
        transporter_current += (
            density[:, transporter] * record(transporter).transporter_current_pA
        )

    def reversal(r, concentrations=inside):
        if r.reversal_mV is not None:
            return jnp.full(n, r.reversal_mV)
        permeability = r.ion_selectivity
        if not permeability:
            raise ValueError(f"{r.molecule_id}: missing reversal or ion selectivity")
        if len(permeability) == 1:
            ion = next(iter(permeability))
            return nernst(
                outside[ion],
                concentrations[ion],
                -1 if ion == "Cl" else (2 if ion == "Ca" else 1),
                policy.temperature_C,
            )
        if "Ca" in permeability:
            raise ValueError(
                "mixed divalent GHK requires an explicit kinetic reversal model"
            )
        return ghk(concentrations, outside, permeability, policy.temperature_C)

    channel_e = (
        jnp.stack([reversal(r) for r in ch_records], axis=1)
        if ch_records
        else jnp.zeros((n, 0))
    )
    receptor_e = (
        jnp.stack([reversal(r)[post] for r in rec_records], axis=1)
        if rec_records
        else jnp.zeros((len(syn), 0))
    )
    ligand_mask = np.zeros((len(syn), len(receptor_indices)))
    nr = graph.connectome.neurons.to_pylist()
    for s, row in enumerate(syn):
        source = nr[id_index[row["pre_id"]]]
        released = graph.molecular.released_ligands.get(
            str(row["pre_id"]),
            graph.molecular.released_ligands.get(source["type_label"], []),
        )
        for r, idx in enumerate(receptor_indices):
            ligand_mask[s, r] = (
                graph.molecular.receptor_ligands.get(genes[idx]["gene_id"]) in released
            )
    path = jnp.asarray([r["path_dist_post"] for r in syn])
    diam = jnp.asarray([r["local_diameter_post"] for r in syn])
    # Lambda scales with diameter and inferred resting membrane resistance.
    resting = policy.leak_nS + (
        density[:, channel_indices] * jnp.asarray([r.conductance for r in ch_records])
    ).sum(axis=1)
    skeleton_comp = None
    if skeletons:
        path, diam, skeleton_comp, has_skeleton = _skeleton_geometry(
            skeletons, syn, id_index, policy, resting, path, diam
        )
    length = policy.length_constant_um * jnp.sqrt(
        jnp.maximum(diam, 1e-6) * policy.leak_nS / resting[post]
    )
    attenuation = jnp.exp(-path / length)
    compartment = jnp.minimum(
        jnp.floor(path / jnp.maximum(length, 1e-6)).astype(jnp.int32), comp - 1
    )
    if skeleton_comp is not None:
        compartment = jnp.where(has_skeleton, skeleton_comp, compartment)
    features = jnp.column_stack(
        [path / 100, diam, jnp.asarray([r["size"] for r in syn])]
    )
    u, rec_tau, fac_tau = rules.plasticity(z, pre, post, features)
    syn_density = rules.receptor_densities(
        z,
        tokens,
        expressed,
        post,
        features,
        jnp.array(receptor_indices, dtype=jnp.int32),
        jnp.asarray(ligand_mask),
    )
    confidence = jnp.asarray([r["detection_confidence"] for r in syn])
    gate = (
        jnp.asarray([r["size"] for r in syn])
        * policy.synapse_unit_nS
        * confidence
        * attenuation
    )
    contacts = [
        r
        for r in graph.connectome.contacts.to_pylist()
        if r["gap_junction_observed"] is not False
    ]
    seen = set()
    for r in contacts:
        pair = tuple(sorted((r["i_id"], r["j_id"])))
        if pair[0] == pair[1] or pair in seen:
            raise ValueError("gap contacts must be unique unordered pairs")
        seen.add(pair)
    gap_i = jnp.array([id_index[r["i_id"]] for r in contacts], dtype=jnp.int32)
    gap_j = jnp.array([id_index[r["j_id"]] for r in contacts], dtype=jnp.int32)
    innexin_mask = jnp.any(expressed[:, classes.get("innexin", [])], axis=1)
    gap_g = rules.gap_conductance(
        z,
        gap_i,
        gap_j,
        jnp.array([r["area"] for r in contacts]) * policy.gap_unit_nS,
        innexin_mask,
    )
    # M6-R3: audit driving-force intervals; no sign enters simulator parameters.
    audit_low, audit_high = [], []
    for r in rec_records:
        reversals = []
        for chloride in policy.chloride_prior_mM:
            concentrations = {**inside, "Cl": jnp.full(n, chloride)}
            reversals.append(reversal(r, concentrations)[post])
        audit_low.append(jnp.minimum(*reversals) - policy.rest_voltage_range_mV[1])
        audit_high.append(jnp.maximum(*reversals) - policy.rest_voltage_range_mV[0])
    if rec_records:
        ambiguous = jnp.any(
            (jnp.stack(audit_low, axis=1) <= 0)
            & (jnp.stack(audit_high, axis=1) >= 0)
            & (syn_density > 0),
            axis=1,
        )
    else:
        ambiguous = jnp.zeros(len(syn), dtype=bool)
    gi = {g["gene_id"]: i for i, g in enumerate(genes)}
    pairs = [
        (peptide_indices.index(gi[r["peptide_gene_id"]]), gi[r["receptor_gene_id"]])
        for r in graph.molecular.peptide_receptor_pairs.to_pylist()
    ]
    p, q = rules.peptide_profiles(
        z, tokens, expressed, jnp.array(peptide_indices, dtype=jnp.int32), pairs
    )
    pair_rows = graph.molecular.peptide_receptor_pairs.to_pylist()
    pep_records = []
    peptide_release, peptide_sensitivity, pep_tau, pep_ec50 = [], [], [], []
    for (pi, receptor), row in zip(pairs, pair_rows):
        model = record(receptor)
        if model.model_form != "gpcr":
            raise ValueError("peptide receptors require GPCR kinetics")
        pep_records.append(model)
        peptide_release.append(p[:, pi])
        peptide_sensitivity.append(
            jax.nn.softplus(
                z @ rules.weights["sensitivity"]
                + tokens[receptor] @ rules.weights["density"]
            )
            * expressed[:, receptor]
        )
        pep_tau.append(model.time_constant(policy.temperature_C))
        pep_ec50.append(row["ec50_nM"] if row["ec50_nM"] is not None else 1.0)
    p = jnp.stack(peptide_release, axis=1) if peptide_release else jnp.zeros((n, 0))
    q = (
        jnp.stack(peptide_sensitivity, axis=1)
        if peptide_sensitivity
        else jnp.zeros((n, 0))
    )
    effector = jnp.any(expressed[:, classes.get("effector", [])], axis=1)
    gpcr = jnp.any(expressed[:, classes.get("gpcr", [])], axis=1)
    mod_sensitivity = (
        jax.nn.softplus(z @ rules.weights["sensitivity"]) * effector * gpcr
    )
    species_grid = graph.metadata["species"] not in {
        "C. elegans",
        "Caenorhabditis elegans",
    }
    mode = "grid" if species_grid or state.mode == "grid" else "global"
    xyz = np.asarray([r["soma_xyz"] for r in nr])
    grid_indices = np.floor((xyz - xyz.min(axis=0)) / policy.grid_spacing_um).astype(
        int
    )
    effective_grid_shape = tuple(
        map(
            int, np.maximum(np.asarray(policy.grid_shape), grid_indices.max(axis=0) + 1)
        )
    )
    policy = replace(policy, grid_shape=effective_grid_shape)
    grid_index = np.ravel_multi_index(grid_indices.T, policy.grid_shape)
    metadata = dict(graph.metadata)
    if kinetics.metadata is None:
        raise ValueError("compilation requires registered kinetics provenance")
    metadata["tier"] = (
        "restricted"
        if "restricted" in {metadata["tier"], kinetics.metadata["tier"]}
        else "open"
    )
    if kinetics.metadata["tier"] == "excluded":
        raise ValueError("excluded kinetics cannot be compiled")
    metadata["kinetics_processing_hash"] = kinetics.metadata["processing_hash"]
    metadata["inputs"] = metadata["inputs"] + kinetics.metadata["inputs"]
    metadata["attributions"] = sorted(
        set(metadata["attributions"]) | set(kinetics.metadata["attributions"])
    )
    traced = isinstance(density, jax.core.Tracer)
    metadata.update(
        rule_checkpoint_hash="traced" if traced else rules.checkpoint_hash(),
        identity_sample=identity_sample,
        residuals_enabled=identity_deviation is not None,
        species_adapter_enabled=rules.adapter is not None,
        parameter_count=rules.parameter_count,
        parameter_budget=rules.budget,
        parameter_budget_frozen=rules.budget is not None,
        sign_audit={"chloride_prior_mM": policy.chloride_prior_mM},
        backend="in_house",
        surrogate_families={"full": len(graph.type_names)},
    )
    if not traced:
        metadata["sign_audit"]["ambiguous_fraction"] = (
            float(jnp.mean(ambiguous)) if len(syn) else 0.0
        )
    inputs_dim = 1 + len(state.concentrations) + len(state.signaling) + 1
    surrogates = tuple(
        Surrogate("full", tuple([-np.inf] * inputs_dim), tuple([np.inf] * inputs_dim))
        for _ in graph.type_names
    )
    result = SimGraph(
        {
            "channel_density": density[:, channel_indices],
            "transporter_current": transporter_current,
            "channel_gbar": jnp.array([r.conductance for r in ch_records]),
            "channel_reversal": channel_e,
            "receptor_reversal": jnp.stack([reversal(r) for r in rec_records], axis=1)
            if rec_records
            else jnp.zeros((n, 0)),
            "capacitance": jnp.full((n, comp), policy.capacitance_pF),
            "leak": jnp.full((n, comp), policy.leak_nS),
            "type_index": jnp.argmax(graph.assignments, axis=1),
            "z": z,
            "chloride": inside["Cl"],
            "neuron_ids": graph.neuron_ids,
        },
        surrogates,
        ptr,
        pre,
        post,
        {
            "density": syn_density,
            "reversal": receptor_e,
            "gbar": jnp.asarray([r.conductance for r in rec_records]),
            "gate": gate,
            "attenuation": attenuation,
            "compartment": compartment,
            "U": u,
            "tau_rec": rec_tau,
            "tau_fac": fac_tau,
            "sign_ambiguous": ambiguous,
            "synapse_ids": jnp.array([r["synapse_id"] for r in syn]),
        },
        {"i": gap_i, "j": gap_j, "conductance": gap_g},
        {
            "mode": mode,
            "decay_length_um": jnp.asarray(
                policy.decay_length_um
                if state.decay_length_um is None
                else state.decay_length_um
            ),
            "release": p,
            "sensitivity": q,
            "tau_s": jnp.asarray(pep_tau),
            "ec50_nM": jnp.asarray(pep_ec50),
            "mod_sensitivity": mod_sensitivity,
            "concentrations": jnp.asarray(state.concentrations),
            "signaling0": jnp.asarray(state.signaling),
            "nondefault": bool(any(state.concentrations) or any(state.signaling)),
            "grid_index": jnp.asarray(grid_index),
            "effector": effector,
        },
        metadata,
        policy,
        ch_records,
        rec_records,
        tuple(pep_records),
    )
    pre_buckets = [[] for _ in range(n)]
    for edge_index, row in enumerate(syn):
        pre_buckets[id_index[row["pre_id"]]].append(edge_index)
    result.syn_pre_ptr = jnp.asarray(
        np.r_[0, np.cumsum([len(b) for b in pre_buckets])], dtype=jnp.int32
    )
    result.syn_pre_edges = jnp.asarray(
        [edge for bucket in pre_buckets for edge in bucket], dtype=jnp.int32
    )
    result.recompile = lambda delta: _compile(
        graph, rules, kinetics, state, policy, identity_sample, delta, skeletons
    )
    return result


def compile(
    graph,
    rules,
    kinetics,
    state=None,
    resolution=None,
    identity_sample=None,
    skeletons=None,
):
    """M4-M7. Compile with residuals disabled; adapters are explicit on rules.

    `skeletons` maps neuron_id to a `SkeletonTree`. Synapses onto those neurons
    take path distance, diameter and compartment from the nearest skeleton node
    (electrotonic binning); all others keep the supplied values.
    """
    return _compile(
        graph, rules, kinetics, state, resolution, identity_sample, skeletons=skeletons
    )


def attach_residuals(sim, epsilon, delta=None, sigma2=1.0):
    if np.shape(epsilon) != np.shape(sim.syn_params["gate"]) or sigma2 <= 0:
        raise ValueError("residual dimensions or variance invalid")
    if delta is not None and np.shape(delta) != np.shape(sim.neuron_params["z"]):
        raise ValueError("identity deviation dimensions invalid")
    if sim.residuals is not None:
        raise ValueError(
            "attach residuals to a fresh compile to avoid stacking corrections"
        )
    if delta is not None:
        if sim.recompile is None:
            raise ValueError("identity deviations require source-graph recompilation")
        sim = sim.recompile(delta)
    return replace(
        sim,
        syn_params={
            **sim.syn_params,
            "gate": sim.syn_params["gate"] * jnp.exp(jnp.asarray(epsilon)),
        },
        residuals={"epsilon": jnp.asarray(epsilon), "delta": delta, "sigma2": sigma2},
        metadata={**sim.metadata, "residuals_enabled": True, "sigma2": sigma2},
    )
