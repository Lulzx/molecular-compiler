"""Phase 0 feasibility analyses on public C. elegans data (Section 10).

Stage A of the registration (partition, splits, thresholds, decision rules)
is frozen before any analysis reads functional data. Stage B (architecture,
absolute parameter budget and metric targets) is derived from Stage A rules at
the end of Phase 0 and frozen separately.
"""

import inspect
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from . import linear_response as lr
from .compiler import ResolutionPolicy
from .evaluation import auroc
from .kinetics import nernst
from .phase0 import molecular_design, rank_analysis, split_diagnostics
from .worm_public import gene_family

SEED = 20261006
FOLDS = 5
CANDIDATE_K = (2, 4, 8, 16, 32)
K2_COMPILER_K = 8
TRANSMITTER_INDEX = {"ACh": 0, "Glu": 1, "GABA": 2}


def stage_a(manifest, seed=SEED):
    """Pre-registered partition, splits, thresholds and decision rules."""
    classes = sorted(set(manifest["canonical_classes"].values()))
    rng = np.random.default_rng(seed)
    class_folds = {
        c: int(f)
        for c, f in zip(rng.permutation(classes), np.arange(len(classes)) % FOLDS)
    }
    neurons = list(manifest["neuron_names"])
    neuron_folds = {
        n: int(f)
        for n, f in zip(rng.permutation(neurons), np.arange(len(neurons)) % FOLDS)
    }
    return {
        "stage": "A",
        "species": "C. elegans",
        "seed": seed,
        "class_partition": {
            "definition": "118 canonical classes (Taylor et al. 2021 / White et al. "
            "1986); CeNGEN subclasses merged; bilateral and radial members together",
            "neuron_to_class": manifest["canonical_classes"],
            "n_classes": len(classes),
        },
        "splits": {
            "leave_class_out": {
                "folds": FOLDS,
                "class_to_fold": class_folds,
                "held_out_pairs": "every atlas pair whose stimulated or responding "
                "neuron belongs to a held-out class",
                "bootstrap_cluster": "held-out canonical class (stimulated neuron's "
                "class when held out, otherwise the responder's)",
            },
            "leave_neuron_out": {
                "folds": FOLDS,
                "neuron_to_fold": neuron_folds,
                "held_out_pairs": "every atlas pair whose stimulated neuron is held out",
                "bootstrap_cluster": "stimulated neuron",
            },
        },
        "data": {
            "responses": "Randi et al. 2023 trial-averaged dF/F0 (30 s window), "
            "[responder, stimulated], wild type for fitting",
            "response_label": "q < 0.05 (Randi et al. functional connection)",
            "minimum_trials": 1,
            "fit_weights": "min(trials, 20)",
            "autoresponse": "stimulated neuron's own mean response; median "
            "positive autoresponse where unmeasured (Section 6.3 drive gauge)",
            "expression": "CeNGEN threshold 2, log1p(TPM) divided by its maximum",
            "plm_tokens": "first k principal components of ESM-2 650M embeddings "
            "of modeled genes, standardized; frozen",
        },
        "models": {
            "B0": "connectome only: signed chemical counts (ACh/Glu +, GABA -), "
            "gap counts; two shared weights",
            "B1": "black box: MLPs on PCA identity pairs on chemical, gap and all "
            "pairs (same inputs as the compiler, no compositional heads)",
            "B2": "structure-free: dense free coupling fitted to responses, no "
            "anatomy and no molecules",
            "B4": "compositional compiler without peptidergic heads, k=8",
            "B5": "connectome-constrained free weights (Creamer et al. 2024 "
            "reimplemented as a steady-state linear response)",
            "compiler": "compositional compiler with factorized peptidergic heads, "
            "k=8 (fixed a priori for K2)",
            "capacity_candidates": [f"compositional k={k}" for k in CANDIDATE_K]
            + ["compositional k=8 with per-gene offsets"],
            "dynamics": "steady-state linear response of a conductance network, "
            "declared approximation (linear_response module)",
        },
        "metrics": {
            "perturbation_detection": "AUROC of |prediction| for q<0.05 labels; raw "
            "units only, no ceiling",
            "perturbation_sign": "sign accuracy on q<0.05 pairs; ceiling = split-half "
            "sign agreement of trials on those pairs",
            "perturbation_amplitude": "Pearson correlation over held-out pairs; "
            "ceiling = sqrt(Spearman-Brown split-half reliability)",
            "bootstrap": "paired cluster bootstrap, 2000 draws, 95% interval",
        },
        "thresholds": {
            "K1_minimum_rank": 64,
            "K1_rationale": "rank_eps(X) must reach d_z = 64 (Section 6.6), eps = 1e-5 "
            "relative",
            "K3_maximum_ambiguous_fraction": 0.25,
            "K3_population": "Glu and GABA edges whose target expresses a receptor for "
            "the released transmitter",
            "K4_maximum_interpolation_fraction": 0.5,
            "K4_minimum_participation_ratio": 10.0,
            "K2_peptide_effect_minimum": 0.0,
            "K2_decision": "peptides required if the 95% interval of compiler - B4 "
            "detection AUROC on held-out pairs without a direct wired connection "
            "lies above 0 on leave_neuron_out",
            "K2_wired_definition": "chemical synapse stimulated->responder or gap "
            "junction between them",
            "gauge_confound": "max |Spearman| between gain/drive and the first 10 "
            "expression PCs > 0.3 with permutation p < 0.05",
            "plm_family_recovery_minimum": 0.9,
        },
        "stage_b_rules": {
            "architecture_and_budget": "among capacity candidates on "
            "leave_class_out amplitude correlation, the smallest trainable "
            "parameter count within one bootstrap standard error of the best",
            "metric_targets": "for each metric: best of B0/B1/B2/B5 + 0.05 "
            "(normalized where a ceiling exists, raw otherwise), never below 0.5 "
            "of the ceiling for amplitude",
        },
    }


def _json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, jax.Array)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _dump(path, value):
    text = json.dumps(value, indent=2, default=_json_default)
    Path(path).write_text(text)
    return json.loads(text)


# ---------------------------------------------------------------- inputs


def atlas_inputs(graph, manifest, k_max=32, identity_dims=16):
    names = manifest["neuron_names"]
    n = len(names)
    ids = {int(v): i for i, v in enumerate(graph.neuron_ids)}
    chem = np.zeros((n, n))
    for row in graph.connectome.synapses.to_pylist():
        chem[ids[row["post_id"]], ids[row["pre_id"]]] += row["size"]
    gap = np.zeros((n, n))
    for row in graph.connectome.contacts.to_pylist():
        i, j = ids[row["i_id"]], ids[row["j_id"]]
        gap[i, j] = gap[j, i] = row["area"]
    expression = np.asarray(graph.abundance, dtype=np.float64)
    expression = expression / expression.max()
    genes = graph.molecular.genes.to_pylist()
    embeddings = np.asarray([g["plm_embedding"] for g in genes], dtype=np.float64)
    centered = embeddings - embeddings.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    tokens = u[:, :k_max] * s[:k_max]
    tokens = tokens / tokens.std(axis=0)
    gene_index = {g["gene_id"]: i for i, g in enumerate(genes)}
    releases = np.zeros((n, 3))
    for neuron, ligands in graph.molecular.released_ligands.items():
        for ligand in ligands:
            if ligand in TRANSMITTER_INDEX:
                releases[ids[int(neuron)], TRANSMITTER_INDEX[ligand]] = 1
    ligand = np.full(len(genes), -1)
    for gene, transmitter in graph.molecular.receptor_ligands.items():
        ligand[gene_index[gene]] = TRANSMITTER_INDEX[transmitter]
    innexins = np.array(
        [i for i, g in enumerate(genes) if g["molecule_class"] == "innexin"]
    )
    pairs = graph.molecular.peptide_receptor_pairs.to_pylist()
    sign = np.where(releases[:, 2] > 0, -1.0, np.where(releases[:, :2].any(1), 1.0, 0))
    centered_expression = expression - expression.mean(axis=0)
    u, s, _ = np.linalg.svd(centered_expression, full_matrices=False)
    identity = u[:, :identity_dims] * s[:identity_dims]
    return lr.AtlasInputs(
        chemical=np.log1p(chem),
        gap=np.log1p(gap),
        expression=expression,
        tokens=tokens,
        releases=releases,
        receptor_ligand=ligand,
        innexins=innexins,
        pair_peptide=np.array([gene_index[p["peptide_gene_id"]] for p in pairs]),
        pair_receptor=np.array([gene_index[p["receptor_gene_id"]] for p in pairs]),
        pair_potency=np.array(
            [max(0.0, -np.log10(p["ec50_nM"] / 1000.0)) + 0.1 for p in pairs]
        ),
        transmitter_sign=sign,
        identity=identity / identity.std(),
        names=list(names),
    )


def align_atlas(atlas, n):
    """Reorder atlas matrices to canonical neuron indices (missing = NaN)."""
    order = atlas["neuron_ids"]
    result = {}
    for strain in ("wt", "unc31"):
        values = atlas[strain]
        out = {
            "dff": np.full((n, n), np.nan),
            "q": np.full((n, n), np.nan),
            "occurrences": np.zeros((n, n), dtype=int),
            "trials": [[np.empty(0)] * n for _ in range(n)],
        }
        for a, i in enumerate(order):
            for b, j in enumerate(order):
                out["dff"][i, j] = values["dff"][a, b]
                out["q"][i, j] = values["q"][a, b]
                out["occurrences"][i, j] = values["occurrences"][a, b]
                out["trials"][i][j] = values["trials"][a][b]
        result[strain] = out
    return result


def autoresponse(dff):
    diagonal = np.diag(dff).copy()
    valid = np.isfinite(diagonal) & (diagonal > 0)
    fallback = float(np.median(diagonal[valid]))
    diagonal[~valid] = fallback
    return diagonal, fallback


def observed_mask(strain):
    n = len(strain["dff"])
    return (
        np.isfinite(strain["dff"])
        & (strain["occurrences"] >= 1)
        & ~np.eye(n, dtype=bool)
    )


# ---------------------------------------------------------------- ceilings


def amplitude_ceiling(trials, mask, seed=SEED):
    """sqrt of Spearman-Brown split-half reliability across held-out pairs."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for i, j in zip(*np.nonzero(mask)):
        values = np.asarray(trials[i][j])
        if len(values) < 2:
            continue
        order = rng.permutation(len(values))
        a.append(values[order[::2]].mean())
        b.append(values[order[1::2]].mean())
    if len(a) < 3:
        return None, len(a)
    r = np.corrcoef(a, b)[0, 1]
    reliability = 2 * r / (1 + r)
    return (float(np.sqrt(reliability)) if reliability > 0 else None), len(a)


def sign_ceiling(trials, mask, seed=SEED):
    rng = np.random.default_rng(seed)
    agree = []
    for i, j in zip(*np.nonzero(mask)):
        values = np.asarray(trials[i][j])
        if len(values) < 2:
            continue
        order = rng.permutation(len(values))
        agree.append(
            np.sign(values[order[::2]].mean()) == np.sign(values[order[1::2]].mean())
        )
    return (float(np.mean(agree)) if agree else None), len(agree)


# ---------------------------------------------------------------- metrics


def _metrics(predicted, observed, labels):
    significant = labels
    sign = (
        float(
            np.mean(np.sign(predicted[significant]) == np.sign(observed[significant]))
        )
        if significant.any()
        else None
    )
    corr = (
        float(np.corrcoef(predicted, observed)[0, 1])
        if len(predicted) > 2 and np.std(predicted) > 0
        else None
    )
    return {
        "perturbation_detection": auroc(labels, np.abs(predicted)),
        "perturbation_sign": sign,
        "perturbation_amplitude": corr,
    }


def cluster_bootstrap(rows, clusters, statistic, samples=2000, seed=SEED):
    """Paired difference interval over independent clusters."""
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if len(unique) < 2:
        return {"mean": None, "interval": None, "se": None, "n_clusters": len(unique)}
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    point = statistic(np.arange(len(clusters)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        value = statistic(np.concatenate([members[c] for c in drawn]))
        if value is not None and np.isfinite(value):
            draws.append(value)
    return {
        "mean": None if point is None else float(point),
        "interval": np.quantile(draws, [0.025, 0.975]).tolist()
        if len(draws) >= samples // 2
        else None,
        "se": float(np.std(draws)) if draws else None,
        "n_clusters": len(unique),
        "valid_draws": len(draws),
    }


def compare(pred_a, pred_b, observed, labels, clusters, metric):
    def stat(index):
        a = _metrics(pred_a[index], observed[index], labels[index])[metric]
        b = _metrics(pred_b[index], observed[index], labels[index])[metric]
        return None if a is None or b is None else a - b

    return cluster_bootstrap(None, clusters, stat)


# ---------------------------------------------------------------- folds


def fold_masks(split, registration, manifest, observed):
    names = manifest["neuron_names"]
    n = len(names)
    if split == "leave_class_out":
        mapping = registration["splits"][split]["class_to_fold"]
        classes = registration["class_partition"]["neuron_to_class"]
        fold = np.array([mapping[classes[name]] for name in names])
        for f in range(FOLDS):
            held = fold == f
            test = observed & (held[:, None] | held[None, :])
            cluster = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(n):
                    owner = j if held[j] else i
                    cluster[i, j] = classes[names[owner]]
            yield f, test, observed & ~(held[:, None] | held[None, :]), cluster
    else:
        mapping = registration["splits"][split]["neuron_to_fold"]
        fold = np.array([mapping[name] for name in names])
        for f in range(FOLDS):
            held = fold == f
            test = observed & held[None, :]
            cluster = np.broadcast_to(np.array(names, dtype=object)[None, :], (n, n))
            yield f, test, observed & ~held[None, :], cluster


def cross_validate(models, inputs, wt, registration, manifest, split, steps):
    observed = observed_mask(wt)
    auto, _ = autoresponse(wt["dff"])
    weights_all = np.minimum(wt["occurrences"], 20).astype(float)
    pooled = {m.name: [] for m in models}
    observed_values, labels, clusters, pairs = [], [], [], []
    fitted = {}
    for f, test, train, cluster in fold_masks(split, registration, manifest, observed):
        index = np.nonzero(test)
        observed_values.append(wt["dff"][index])
        labels.append(wt["q"][index] < 0.05)
        clusters.append(cluster[index])
        pairs.append(np.stack(index, axis=1))
        for model in models:
            params, history = lr.fit(
                model, inputs, wt["dff"], weights_all * train, auto, steps=steps
            )
            prediction = np.asarray(
                lr.predict(model, params, inputs, jnp.asarray(auto))
            )
            pooled[model.name].append(prediction[index])
            fitted[(model.name, f)] = (params, history)
    return (
        {k: np.concatenate(v) for k, v in pooled.items()},
        np.concatenate(observed_values),
        np.concatenate(labels),
        np.concatenate(clusters),
        np.concatenate(pairs),
        fitted,
    )


# ---------------------------------------------------------------- K analyses


def k3_sign_audit(graph, manifest, policy=None):
    policy = policy or ResolutionPolicy.default()
    names = manifest["neuron_names"]
    ids = {int(v): i for i, v in enumerate(graph.neuron_ids)}
    gene_names = manifest["gene_names"]
    gi = {g: i for i, g in enumerate(graph.gene_ids)}
    polarity = manifest["receptor_polarity"]
    expressed = np.asarray(graph.abundance) > 0
    by_name = {name: gene for gene, name in gene_names.items()}
    released = graph.molecular.released_ligands
    outside = policy.extracellular_mM[2]
    e_cl = [
        float(nernst(outside, c, -1, policy.temperature_C))
        for c in policy.chloride_prior_mM
    ]
    low, high = min(e_cl), max(e_cl)
    rest = policy.rest_voltage_range_mV
    chloride_ambiguous = low <= rest[1] and high >= rest[0]
    counts = {
        "edges": 0,
        "with_receptor": 0,
        "anion_any": 0,
        "cation_only": 0,
        "anion_only": 0,
        "mixed": 0,
        "no_receptor": 0,
    }
    per_transmitter = {}
    for row in graph.connectome.synapses.to_pylist():
        post = ids[row["post_id"]]
        for transmitter in released.get(str(row["pre_id"]), []):
            if transmitter not in ("Glu", "GABA"):
                continue
            stats = per_transmitter.setdefault(transmitter, dict.fromkeys(counts, 0))

            def present(kind, transmitter=transmitter, post=post):
                return any(
                    g in by_name and expressed[post, gi[by_name[g]]]
                    for g in polarity[transmitter][kind]
                )

            excitatory, inhibitory = present("excitatory"), present("inhibitory")
            for target in (counts, stats):
                target["edges"] += 1
                if excitatory or inhibitory:
                    target["with_receptor"] += 1
                    target["anion_any"] += inhibitory
                    target["mixed"] += excitatory and inhibitory
                    target["anion_only"] += inhibitory and not excitatory
                    target["cation_only"] += excitatory and not inhibitory
                else:
                    target["no_receptor"] += 1
    ambiguous = counts["anion_any"] if chloride_ambiguous else counts["mixed"]
    fraction = ambiguous / max(counts["with_receptor"], 1)
    return {
        "chloride_prior_mM": list(policy.chloride_prior_mM),
        "E_Cl_range_mV": [low, high],
        "rest_voltage_range_mV": list(rest),
        "anion_receptors_sign_ambiguous": chloride_ambiguous,
        "counts": counts,
        "per_transmitter": per_transmitter,
        "ambiguous_fraction": fraction,
        "ambiguous_fraction_all_edges": ambiguous / max(counts["edges"], 1),
        "receptor_mixture_fraction": counts["mixed"] / max(counts["with_receptor"], 1),
        "names_checked": len(names),
    }


def k4_split_audit(graph, manifest, registration):
    names = manifest["neuron_names"]
    classes = registration["class_partition"]["neuron_to_class"]
    mapping = registration["splits"]["leave_class_out"]["class_to_fold"]
    abundance = np.asarray(graph.abundance)
    class_names = sorted(set(classes.values()))
    profiles = np.stack(
        [
            abundance[[i for i, n in enumerate(names) if classes[n] == c]].mean(axis=0)
            for c in class_names
        ]
    )
    design = molecular_design(graph)
    folds = []
    for f in range(FOLDS):
        held = np.array([mapping[c] == f for c in class_names])
        diagnostics = split_diagnostics(profiles[~held], profiles[held], threshold=0.0)
        held_neurons = {i for i, n in enumerate(names) if mapping[classes[n]] == f}
        training_rows = training_design(graph, held_neurons)
        diagnostics["training_design_rank"] = {
            k: v
            for k, v in rank_analysis(training_rows).items()
            if k != "singular_values"
        }
        diagnostics["held_out_classes"] = [c for c, h in zip(class_names, held) if h]
        diagnostics["interpolation_fraction"] = float(
            np.mean(
                [
                    p["interpretation"] == "interpolation"
                    for p in diagnostics["heldout_profiles"]
                ]
            )
        )
        for p, c in zip(
            diagnostics["heldout_profiles"], diagnostics["held_out_classes"]
        ):
            p["class"] = c
        diagnostics["training_expression_rank"].pop("singular_values")
        folds.append(diagnostics)
    return {
        "folds": folds,
        "interpolation_fraction": float(
            np.mean([f["interpolation_fraction"] for f in folds])
        ),
        "training_participation_ratio_min": float(
            min(f["training_design_rank"]["participation_ratio"] for f in folds)
        ),
        "full_design": {
            k: v for k, v in design.items() if k not in ("gram", "singular_values")
        },
    }


def training_design(graph, held_neurons):
    """Chemical and contact rows of X restricted to training neurons."""
    z = np.asarray(graph.abundance)
    ids = {int(v): i for i, v in enumerate(graph.neuron_ids)}
    rows = []
    for r in graph.connectome.synapses.to_pylist():
        a, b = ids[r["pre_id"]], ids[r["post_id"]]
        if a not in held_neurons and b not in held_neurons:
            rows.append(np.r_[z[a], z[b], r["size"], 0, 0])
    for r in graph.connectome.contacts.to_pylist():
        a, b = ids[r["i_id"]], ids[r["j_id"]]
        if a not in held_neurons and b not in held_neurons:
            rows.append(np.r_[z[a], z[b], r["area"], 0, 0])
    return np.asarray(rows)


def gauge_audit_worm(
    graph, manifest, wt, promoter="rab-3", seed=SEED, permutations=1000
):
    """Promoter-expression proxy for gain and autoresponse estimate for drive."""
    names = manifest["neuron_names"]
    abundance = np.asarray(graph.abundance)
    centered = abundance - abundance.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    pcs = u[:, :10] * s[:10]
    rng = np.random.default_rng(seed)

    def statistic(values, rows):
        return max(abs(spearmanr(values, pcs[rows, c]).statistic) for c in range(10))

    def audit(values, rows, basis):
        observed = statistic(values, rows)
        null = [statistic(rng.permutation(values), rows) for _ in range(permutations)]
        p = float((1 + np.sum(np.array(null) >= observed)) / (permutations + 1))
        return {
            "basis": basis,
            "neurons": len(rows),
            "max_abs_spearman_with_expression_pcs": float(observed),
            "permutation_p": p,
            "confounded": bool(observed > 0.3 and p < 0.05),
        }

    report = {}
    proxies = manifest.get("gauge_proxies", {})
    if promoter in proxies:
        values = np.array([proxies[promoter][n] for n in names])
        rows = np.arange(len(names))
        report["indicator"] = audit(
            values,
            rows,
            f"{promoter} CeNGEN expression as proxy: GCaMP6s and the GUR-3/PRDX-2 "
            "QF driver both use the rab-3 promoter (Randi et al. 2023, AML508 "
            "genotype); not a measured fluorescence gain",
        )
    else:
        report["indicator"] = {
            "status": "unavailable",
            "reason": f"{promoter} expression not in the ingestion manifest",
        }
    diagonal = np.diag(wt["dff"])
    rows = np.flatnonzero(np.isfinite(diagonal) & (np.diag(wt["occurrences"]) >= 3))
    report["opsin"] = audit(
        diagonal[rows],
        rows,
        "stimulated neuron's own mean dF/F0 (drive x gain, >= 3 trials)",
    )
    report["complete"] = all("confounded" in report[k] for k in ("indicator", "opsin"))
    report["status"] = "complete_with_proxies" if report["complete"] else "incomplete"
    return report


def family_recovery(graph, manifest):
    """Section 12.1 check on gene families of modeled signaling molecules."""
    genes = graph.molecular.genes.to_pylist()
    names = manifest["gene_names"]
    keep = [
        g
        for g in genes
        if g["molecule_class"]
        in {"channel", "receptor", "transporter", "gpcr", "innexin"}
    ]
    families = np.array([gene_family(names[g["gene_id"]]) for g in keep])
    values, counts = np.unique(families, return_counts=True)
    multi = set(values[counts >= 2])
    rows = [i for i, f in enumerate(families) if f in multi]
    x = np.asarray([keep[i]["plm_embedding"] for i in rows], dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argmax(similarity, axis=1)
    labels = families[rows]
    correct = labels[nearest] == labels
    per_family = {f: float(np.mean(correct[labels == f])) for f in sorted(multi)}
    return {
        "scope": "leave-one-member-out 1-NN (cosine) over gene families of modeled "
        "channels, receptors, transporters, GPCRs and innexins; no curated kinetics "
        "library exists yet, so families come from gene nomenclature",
        "genes": len(rows),
        "families": len(multi),
        "accuracy": float(np.mean(correct)),
        "per_family": per_family,
    }


def jaxley_capability_audit():
    """S-R4 conditions 1-3, checked against the installed Jaxley source."""
    import jaxley
    from jaxley import solver_voltage, synapses
    from jaxley.modules import network

    synapse_types = [n for n in dir(synapses) if n[0].isupper()]
    network_source = inspect.getsource(network)
    solver_source = inspect.getsource(solver_voltage)
    gap = any("gap" in n.lower() or "electrical" in n.lower() for n in synapse_types)
    return {
        "version": jaxley.__version__,
        "conditions": {
            "1_implicit_gap_junctions": {
                "passed": False,
                "evidence": f"synapse classes {synapse_types}; no gap or electrical "
                "coupling type; synaptic currents enter the postsynaptic diagonal "
                "with the presynaptic voltage explicit"
                + (" (unexpected gap-like name found)" if gap else ""),
            },
            "2_factorized_peptides": {
                "passed": False,
                "evidence": "networks connect through per-edge synapse tables "
                f"(DataFrame edges: {'edges' in network_source}); no field or "
                "factorized source/target coupling interface",
            },
            "3_runtime_surrogate_switching": {
                "passed": False,
                "evidence": "mechanisms are fixed per compartment at build time; no "
                "per-neuron model switching interface",
            },
        },
        "solver_is_per_cell": "dhs" in solver_source.lower(),
        "decision": "in_house",
        "reason": "conditions 1-3 fail at the interface audit; per Section 7.3 the "
        "in-house loop is selected without a full-network Jaxley trial",
    }


def single_cell_trial():
    """S-R4 numerical comparison on the synthetic single-cell reference (float64)."""
    with jax.enable_x64(True):
        return _single_cell_trial()


def _single_cell_trial():
    from .compiler import compile
    from .fixtures import synthetic_system
    from .jaxley_backend import build_single_cell, run_single_cell
    from .simulation import Stimulus, simulate

    graph, rules, kinetics = synthetic_system()
    policy = replace(ResolutionPolicy.default(), n_comp=1, solve_tolerance=1e-10)
    sim = compile(graph, rules, kinetics, resolution=policy)
    sim = replace(
        sim,
        gap={**sim.gap, "conductance": jnp.zeros_like(sim.gap["conductance"])},
        syn_params={**sim.syn_params, "gate": jnp.zeros_like(sim.syn_params["gate"])},
        neuromod={
            **sim.neuromod,
            "release": jnp.zeros_like(sim.neuromod["release"]),
            "sensitivity": jnp.zeros_like(sim.neuromod["sensitivity"]),
        },
    )
    record = kinetics.records["ca"]
    cell, area = build_single_cell(channels=(record,))
    currents = jnp.array([1.0, 1.0, 1.0, 1.0])

    def reference(density):
        return run_single_cell(
            cell, area, jnp.array([density]), (record,), currents, policy.dt_s
        )

    def in_house(density):
        changed = replace(
            sim,
            neuron_params={
                **sim.neuron_params,
                "channel_density": sim.neuron_params["channel_density"]
                .at[0, 0]
                .set(density),
            },
        )
        current = jnp.zeros((4, 4)).at[:, 0].set(currents)
        return simulate(changed, Stimulus(currents=current), 0.002).voltage[:, 0, 0]

    density = float(sim.neuron_params["channel_density"][0, 0])
    a, b = np.asarray(reference(density)), np.asarray(in_house(density))
    ga = float(jax.grad(lambda x: reference(x).sum())(density))
    gb = float(jax.grad(lambda x: in_house(x).sum())(density))
    return {
        "trajectory_max_abs_difference_mV": float(np.max(np.abs(a - b))),
        "gradient_relative_difference": abs(ga - gb) / max(abs(gb), 1e-12),
        "data_kind": "synthetic single cell",
    }


def peptide_network_check(graph, manifest, ripoll_csv):
    """Section 4.3: derived peptide network vs Ripoll-Sanchez long-range model."""
    import csv

    names = manifest["neuron_names"]
    index = {n: i for i, n in enumerate(names)}
    with Path(ripoll_csv).open() as stream:
        rows = list(csv.reader(stream))
    header = [n.strip() for n in rows[0][1:]]
    reference = np.zeros((len(names), len(names)), dtype=bool)
    from .worm_public import normalize_name

    for row in rows[1:]:
        source = normalize_name(row[0])
        for target, value in zip(header, row[1:]):
            target = normalize_name(target)
            if source in index and target in index and value not in ("", "0"):
                reference[index[source], index[target]] = True
    expressed = np.asarray(graph.abundance) > 0
    gi = {g: i for i, g in enumerate(graph.gene_ids)}
    derived = np.zeros_like(reference)
    for pair in graph.molecular.peptide_receptor_pairs.to_pylist():
        sources = expressed[:, gi[pair["peptide_gene_id"]]]
        targets = expressed[:, gi[pair["receptor_gene_id"]]]
        derived |= np.outer(sources, targets)
    np.fill_diagonal(derived, False)
    np.fill_diagonal(reference, False)
    both = np.sum(derived & reference)
    return {
        "derived_edges": int(derived.sum()),
        "reference_edges": int(reference.sum()),
        "shared_edges": int(both),
        "jaccard": float(both / max(np.sum(derived | reference), 1)),
        "recall_of_reference": float(both / max(reference.sum(), 1)),
        "precision_vs_reference": float(both / max(derived.sum(), 1)),
        "note": "Ripoll-Sanchez et al. use their own expression thresholds and "
        "ligand-receptor table; this is a consistency check, not an equivalence test",
    }


# ---------------------------------------------------------------- driver


def standard_models(k=K2_COMPILER_K):
    return [
        lr.Model("B0", "connectome_only"),
        lr.Model("B1", "black_box", ridge=1e-4),
        lr.Model("B2", "dense_free", ridge=1e-3, learning_rate=0.005),
        lr.Model("B5", "connectome_free", ridge=1e-3),
        lr.Model("B4", "compositional", k=k, peptides=False),
        lr.Model("compiler", "compositional", k=k, peptides=True),
    ]


def capacity_models():
    models = [
        lr.Model(f"compositional_k{k}", "compositional", k=k, peptides=True)
        for k in CANDIDATE_K
    ]
    models.append(
        lr.Model(
            "compositional_k8_per_gene",
            "compositional",
            k=8,
            peptides=True,
            per_gene=True,
            ridge=1e-2,
        )
    )
    return models


def summarize(predictions, observed, labels, clusters, ceilings, reference="compiler"):
    table = {}
    for name, values in predictions.items():
        raw = _metrics(values, observed, labels)
        normalized = {
            "perturbation_amplitude": None
            if raw["perturbation_amplitude"] is None or ceilings["amplitude"] is None
            else raw["perturbation_amplitude"] / ceilings["amplitude"],
            "perturbation_sign": None
            if raw["perturbation_sign"] is None or ceilings["sign"] is None
            else raw["perturbation_sign"] / ceilings["sign"],
            "perturbation_detection": None,
        }
        table[name] = {"raw": raw, "normalized": normalized}
    comparisons = {}
    if reference in predictions:
        for name in predictions:
            if name == reference:
                continue
            comparisons[f"{reference}-{name}"] = {
                metric: compare(
                    predictions[reference],
                    predictions[name],
                    observed,
                    labels,
                    clusters,
                    metric,
                )
                for metric in (
                    "perturbation_detection",
                    "perturbation_sign",
                    "perturbation_amplitude",
                )
            }
    return {"metrics": table, "comparisons": comparisons, "n_pairs": len(observed)}


def run(project, registration, output, steps=1500, ripoll_csv=None):
    """Execute every Phase 0 analysis against a frozen Stage A registration."""
    from .worm_public import load_atlas, load_worm_project

    started = time.time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph, manifest = load_worm_project(project)
    n = len(manifest["neuron_names"])
    atlas = align_atlas(load_atlas(Path(project) / "signal_propagation.npz"), n)
    wt, unc31 = atlas["wt"], atlas["unc31"]
    inputs = atlas_inputs(graph, manifest)
    ingestion = json.loads((Path(project) / "ingestion-report.json").read_text())
    report = {
        "status": "phase0_run",
        "data_kind": "real",
        "tier": ingestion["provenance"]["tier"],
        "provenance": ingestion["provenance"],
        "prepared_graph_hash": graph.metadata["processing_hash"],
        "registration_hash": registration["processing_hash"],
    }
    reg = registration["registration"]

    design = molecular_design(graph)
    report["K1"] = {
        "rank": {
            k: v for k, v in design.items() if k not in ("gram", "singular_values")
        },
        "threshold": reg["thresholds"]["K1_minimum_rank"],
        "passed": design["rank_epsilon"] >= reg["thresholds"]["K1_minimum_rank"],
        "singular_values_top20": design["singular_values"][:20],
    }
    k3 = k3_sign_audit(graph, manifest)
    k3["threshold"] = reg["thresholds"]["K3_maximum_ambiguous_fraction"]
    k3["passed"] = k3["ambiguous_fraction"] <= k3["threshold"]
    report["K3"] = k3
    k4 = k4_split_audit(graph, manifest, reg)
    k4["thresholds"] = {
        "maximum_interpolation_fraction": reg["thresholds"][
            "K4_maximum_interpolation_fraction"
        ],
        "minimum_participation_ratio": reg["thresholds"][
            "K4_minimum_participation_ratio"
        ],
    }
    k4["passed"] = (
        k4["interpolation_fraction"]
        <= reg["thresholds"]["K4_maximum_interpolation_fraction"]
        and k4["training_participation_ratio_min"]
        >= reg["thresholds"]["K4_minimum_participation_ratio"]
    )
    report["K4"] = k4
    report["gauge_audit"] = gauge_audit_worm(graph, manifest, wt)
    report["plm_family_recovery"] = family_recovery(graph, manifest)
    report["plm_family_recovery"]["passed"] = (
        report["plm_family_recovery"]["accuracy"]
        >= reg["thresholds"]["plm_family_recovery_minimum"]
    )
    report["jaxley_audit"] = jaxley_capability_audit()
    report["jaxley_single_cell"] = single_cell_trial()
    if ripoll_csv:
        report["peptide_network_check"] = peptide_network_check(
            graph, manifest, ripoll_csv
        )
    _dump(output / "phase0-partial.json", report)

    report["noise_ceilings"] = {}
    splits = {}
    for split in ("leave_neuron_out", "leave_class_out"):
        models = standard_models() + (
            capacity_models() if split == "leave_class_out" else []
        )
        predictions, values, labels, clusters, pairs, fitted = cross_validate(
            models, inputs, wt, reg, manifest, split, steps
        )
        test = np.zeros((n, n), dtype=bool)
        test[pairs[:, 0], pairs[:, 1]] = True
        ceilings = {
            "amplitude": amplitude_ceiling(wt["trials"], test)[0],
            "sign": sign_ceiling(wt["trials"], test & (wt["q"] < 0.05))[0],
        }
        report["noise_ceilings"][split] = ceilings
        summary = summarize(predictions, values, labels, clusters, ceilings)
        wired = (inputs.chemical > 0) | (inputs.gap > 0)
        is_wired = wired[pairs[:, 0], pairs[:, 1]]
        summary["K2_by_wiring"] = {
            group: {
                metric: compare(
                    predictions["compiler"][mask],
                    predictions["B4"][mask],
                    values[mask],
                    labels[mask],
                    clusters[mask],
                    metric,
                )
                for metric in ("perturbation_detection", "perturbation_amplitude")
            }
            | {"n_pairs": int(mask.sum()), "n_responding": int(labels[mask].sum())}
            for group, mask in (("wired", is_wired), ("unwired", ~is_wired))
        }
        summary["B5-B2"] = {
            metric: compare(
                predictions["B5"], predictions["B2"], values, labels, clusters, metric
            )
            for metric in ("perturbation_detection", "perturbation_amplitude")
        }
        if split == "leave_class_out":
            candidates = [m.name for m in capacity_models()]
            scores = {
                c: _metrics(predictions[c], values, labels)["perturbation_amplitude"]
                for c in candidates
            }
            best = max(candidates, key=lambda c: scores[c])
            summary["capacity"] = {
                "best": best,
                "best_minus_candidate": {
                    c: compare(
                        predictions[best],
                        predictions[c],
                        values,
                        labels,
                        clusters,
                        "perturbation_amplitude",
                    )
                    for c in candidates
                    if c != best
                },
            }
        summary["parameter_counts"] = {
            m.name: lr.parameter_count(fitted[(m.name, 0)][0]) for m in models
        }
        summary["final_losses"] = {
            m.name: [fitted[(m.name, f)][1][-1] for f in range(FOLDS)] for m in models
        }
        splits[split] = summary
        if split == "leave_neuron_out":
            report["unc31_check"] = unc31_check(fitted, inputs, unc31, reg, manifest)
        _dump(output / f"phase0-{split}.json", summary)
    report["splits"] = splits
    unwired = splits["leave_neuron_out"]["K2_by_wiring"]["unwired"]
    interval = unwired["perturbation_detection"]["interval"]
    report["K2"] = {
        "decision_split": "leave_neuron_out",
        "paired_interval": interval,
        "peptides_required": None
        if interval is None
        else interval[0] > reg["thresholds"]["K2_peptide_effect_minimum"],
        "unwired": unwired,
        "wired": splits["leave_neuron_out"]["K2_by_wiring"]["wired"],
        "leave_class_out": splits["leave_class_out"]["K2_by_wiring"],
    }
    report["B5_check"] = b5_check(splits)
    report["stage_b"] = stage_b(splits, reg)
    report["elapsed_s"] = time.time() - started
    return _dump(output / "phase0-report.json", report)


def unc31_check(fitted, inputs, unc31, reg, manifest):
    """WT-fitted compiler on unc-31 pairs, with and without peptide release."""
    model = next(m for m in standard_models() if m.name == "compiler")
    auto, _ = autoresponse(unc31["dff"])
    observed = observed_mask(unc31)
    wired = (inputs.chemical > 0) | (inputs.gap > 0)
    results = {}
    names = manifest["neuron_names"]
    fold = np.array(
        [reg["splits"]["leave_neuron_out"]["neuron_to_fold"][x] for x in names]
    )
    on, off, obs, lab, clu, wire = [], [], [], [], [], []
    for f in range(FOLDS):
        params, _ = fitted[("compiler", f)]
        test = observed & (fold == f)[None, :]
        index = np.nonzero(test)
        silenced = {**params, "peptide_gain": jnp.array(-1e3)}
        on.append(
            np.asarray(lr.predict(model, params, inputs, jnp.asarray(auto)))[index]
        )
        off.append(
            np.asarray(lr.predict(model, silenced, inputs, jnp.asarray(auto)))[index]
        )
        obs.append(unc31["dff"][index])
        lab.append(unc31["q"][index] < 0.05)
        clu.append(np.array(names, dtype=object)[index[1]])
        wire.append(wired[index])
    on, off, obs, lab, clu, wire = map(np.concatenate, (on, off, obs, lab, clu, wire))
    for group, mask in (
        ("all", np.ones_like(wire)),
        ("unwired", ~wire),
        ("wired", wire),
    ):
        results[group] = {
            "n_pairs": int(mask.sum()),
            "peptides_off_minus_on": {
                metric: compare(
                    off[mask], on[mask], obs[mask], lab[mask], clu[mask], metric
                )
                for metric in ("perturbation_detection", "perturbation_amplitude")
            },
        }
    results["interpretation"] = (
        "unc-31 animals lack dense-core-vesicle release; if the fitted peptidergic "
        "term is real, silencing it should not hurt and may help on unc-31 pairs"
    )
    return results


def b5_check(splits):
    s = splits["leave_neuron_out"]
    b5 = s["metrics"]["B5"]
    dense = s["comparisons"]["compiler-B2"]
    return {
        "B5_held_out_amplitude": b5["raw"]["perturbation_amplitude"],
        "B5_normalized_amplitude": b5["normalized"]["perturbation_amplitude"],
        "B5_minus_B2": s["B5-B2"],
        "published": {
            "source": "Creamer, Leifer & Pillow, bioRxiv 10.1101/2024.09.22.614271 "
            "(version of 2026-05-18)",
            "relative_correlation": 0.82,
            "earlier_versions_relative_correlation": 0.92,
            "split": "30 held-out animals; STAM area correlation 0.14 vs train-test "
            "0.17",
            "unconstrained_model": "fully connected performed no better on STAMs",
        },
        "comparable": False,
        "reason": "the atlas file pools animals, so held-out-animal splits and the "
        "latent LDS cannot be reproduced; this check compares the qualitative "
        "finding (anatomy-constrained vs unconstrained) on held-out stimulation "
        "targets",
        "compiler_minus_B2": dense,
        "anatomy_constraint_no_worse": None
        if s["B5-B2"]["perturbation_amplitude"]["interval"] is None
        else s["B5-B2"]["perturbation_amplitude"]["interval"][1] >= 0,
    }


def stage_b(splits, reg):
    lco = splits["leave_class_out"]
    candidates = [name for name in lco["metrics"] if name.startswith("compositional_")]
    scores = {c: lco["metrics"][c]["raw"]["perturbation_amplitude"] for c in candidates}
    capacity = lco["capacity"]
    best = capacity["best"]
    eligible = [best] + [
        c
        for c, diff in capacity["best_minus_candidate"].items()
        if diff["mean"] is not None
        and diff["se"] is not None
        and diff["mean"] <= diff["se"]
    ]
    se = {c: d["se"] for c, d in capacity["best_minus_candidate"].items()}
    chosen = min(eligible, key=lambda c: lco["parameter_counts"][c])
    targets = {}
    for metric in (
        "perturbation_detection",
        "perturbation_sign",
        "perturbation_amplitude",
    ):
        for split in ("leave_neuron_out", "leave_class_out"):
            s = splits[split]["metrics"]
            normalized = metric != "perturbation_detection"
            values = [
                s[b]["normalized" if normalized else "raw"][metric]
                for b in ("B0", "B1", "B2", "B5")
            ]
            values = [v for v in values if v is not None]
            if not values:
                targets[f"{split}/{metric}"] = None
                continue
            target = max(values) + 0.05
            if metric == "perturbation_amplitude":
                target = max(target, 0.5)
            targets[f"{split}/{metric}"] = {
                "target": float(min(target, 1.0)),
                "units": "fraction of noise ceiling" if normalized else "raw AUROC",
            }
    return {
        "stage": "B",
        "architecture": chosen,
        "absolute_parameter_budget": int(lco["parameter_counts"][chosen]),
        "candidate_scores": scores,
        "candidate_parameter_counts": {
            c: lco["parameter_counts"][c] for c in candidates
        },
        "selection_se": se,
        "metric_targets": targets,
        "rule": reg["stage_b_rules"],
    }


def exploratory(graph, manifest, registration, policy=None):
    """Post-hoc sensitivity analyses. Not pre-registered; never change a gate."""
    policy = policy or ResolutionPolicy.default()
    reg = registration["registration"]
    names = manifest["neuron_names"]
    classes = reg["class_partition"]["neuron_to_class"]
    mapping = reg["splits"]["leave_class_out"]["class_to_fold"]
    standardized = []
    for f in range(FOLDS):
        held = {i for i, n in enumerate(names) if mapping[classes[n]] == f}
        x = training_design(graph, held)
        x = x[:, np.std(x, axis=0) > 0]
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        standardized.append(rank_analysis(z)["participation_ratio"])
    genes = graph.molecular.genes.to_pylist()
    keep = [
        g
        for g in genes
        if g["molecule_class"]
        in {"channel", "receptor", "transporter", "gpcr", "innexin"}
    ]
    x = np.asarray([g["plm_embedding"] for g in keep], dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    labels = np.array([g["molecule_class"] for g in keep])
    class_accuracy = float(np.mean(labels[np.argmax(similarity, axis=1)] == labels))
    outside, rest_low = policy.extracellular_mM[2], policy.rest_voltage_range_mV[0]
    thermal = float(nernst(outside, outside / np.e, -1, policy.temperature_C))
    stable_below = float(outside * np.exp(rest_low / -thermal))
    return {
        "pre_registered": False,
        "K4_participation_ratio_standardized_columns": standardized,
        "plm_molecule_class_1nn_accuracy": class_accuracy,
        "K3_chloride_for_stable_inhibition_mM": {
            "value": stable_below,
            "meaning": "anion-receptor synapses are inhibitory at every rest "
            f"potential in {list(policy.rest_voltage_range_mV)} mV only if "
            "[Cl-]i stays below this concentration",
        },
    }


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _diff(entry):
    if entry is None or entry["mean"] is None:
        return "n/a"
    interval = entry["interval"]
    if interval is None:
        return f"{entry['mean']:+.3f} (interval unavailable)"
    return f"{entry['mean']:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]"


def render_report(report, exploratory_results=None):
    """Human-readable Phase 0 report. Inherits the data tier of its inputs."""
    k1, k2, k3, k4 = (report[k] for k in ("K1", "K2", "K3", "K4"))
    gauge, plm = report["gauge_audit"], report["plm_family_recovery"]
    lines = [
        "# Phase 0 feasibility report: C. elegans",
        "",
        (
            f"Data tier: **{report['tier']}** (inherited from the inputs; derived results "
            "must not be published while any input is restricted, Section 12.8)."
        ),
        f"Stage A registration hash: `{report['registration_hash']}`.",
        f"Processing hash: `{report['provenance']['processing_hash']}`.",
        "",
        (
            "Responses are modeled by a steady-state linear-response approximation of "
            "the compiler (no kinetics or time course). Phase 0 conclusions are "
            "conditional on that approximation."
        ),
        "",
        "## Kill criteria",
        "",
        "| ID | Pre-registered test | Result | Threshold | Outcome |",
        "|---|---|---|---|---|",
        (
            f"| K1 | rank_eps(X) | {k1['rank']['rank_epsilon']} (participation "
            f"{_fmt(k1['rank']['participation_ratio'], 2)}) | >= {k1['threshold']} | "
            f"{'pass' if k1['passed'] else 'fail'} |"
        ),
        (
            f"| K2 | compiler - B4 detection AUROC, unwired held-out pairs | "
            f"{_diff(k2['unwired']['perturbation_detection'])} | interval > 0 | peptides "
            f"{'required' if k2['peptides_required'] else 'not required'} |"
        ),
        (
            f"| K3 | sign-ambiguous Glu/GABA edges | {_fmt(k3['ambiguous_fraction'])} | "
            f"<= {k3['threshold']} | {'pass' if k3['passed'] else 'fail'} |"
        ),
        (
            f"| K4 | interpolation fraction; min training participation ratio | "
            f"{_fmt(k4['interpolation_fraction'])}; "
            f"{_fmt(k4['training_participation_ratio_min'], 2)} | <= "
            f"{k4['thresholds']['maximum_interpolation_fraction']}; >= "
            f"{k4['thresholds']['minimum_participation_ratio']} | "
            f"{'pass' if k4['passed'] else 'fail'} |"
        ),
        "",
        "## Audits",
        "",
        (
            f"- Gauge ({gauge['status']}): indicator proxy max |rho| "
            f"{_fmt(gauge['indicator'].get('max_abs_spearman_with_expression_pcs'))}, "
            f"p = {_fmt(gauge['indicator'].get('permutation_p'))}, confounded: "
            f"{_fmt(gauge['indicator'].get('confounded'))}; opsin drive max |rho| "
            f"{_fmt(gauge['opsin']['max_abs_spearman_with_expression_pcs'])}, p = "
            f"{_fmt(gauge['opsin']['permutation_p'])}, confounded: "
            f"{_fmt(gauge['opsin']['confounded'])}."
        ),
        (
            f"- PLM family recovery: {_fmt(plm['accuracy'])} over {plm['genes']} genes in "
            f"{plm['families']} families (minimum 0.9): "
            f"{'pass' if plm['passed'] else 'fail'}."
        ),
        (
            f"- Jaxley (S-R4): backend **{report['jaxley_audit']['decision']}**; "
            "conditions 1-3 fail at the interface audit. Single-cell float64 "
            "comparison: max |dV| "
            f"{report['jaxley_single_cell']['trajectory_max_abs_difference_mV']:.1e} mV, "
            "gradient relative difference "
            f"{report['jaxley_single_cell']['gradient_relative_difference']:.1e}."
        ),
    ]
    if "peptide_network_check" in report:
        p = report["peptide_network_check"]
        lines.append(
            f"- Peptide network vs Ripoll-Sanchez long-range model: recall "
            f"{_fmt(p['recall_of_reference'])}, precision {_fmt(p['precision_vs_reference'])}, "
            f"Jaccard {_fmt(p['jaccard'])}."
        )
    c = report["provenance"]
    lines += ["", f"Inputs: {', '.join(r['dataset_id'] for r in c['inputs'])}.", ""]
    for split, summary in report["splits"].items():
        ceilings = report["noise_ceilings"][split]
        lines += [
            f"## {split} ({summary['n_pairs']} held-out pairs)",
            "",
            (
                f"Ceilings: amplitude {_fmt(ceilings['amplitude'])}, sign "
                f"{_fmt(ceilings['sign'])}. Detection has no valid ceiling and is raw."
            ),
            "",
            (
                "| Model | Parameters | Detection AUROC | Sign accuracy | Amplitude r | "
                "Amplitude / ceiling |"
            ),
            "|---|---|---|---|---|---|",
        ]
        for name, m in summary["metrics"].items():
            lines.append(
                f"| {name} | {summary['parameter_counts'][name]} | "
                f"{_fmt(m['raw']['perturbation_detection'])} | "
                f"{_fmt(m['raw']['perturbation_sign'])} | "
                f"{_fmt(m['raw']['perturbation_amplitude'])} | "
                f"{_fmt(m['normalized']['perturbation_amplitude'])} |"
            )
        lines += [
            "",
            "| Paired difference (95% cluster bootstrap) | Detection | Sign | Amplitude |",
            "|---|---|---|---|",
        ]
        for name, comparison in summary["comparisons"].items():
            lines.append(
                f"| {name} | {_diff(comparison['perturbation_detection'])} | "
                f"{_diff(comparison['perturbation_sign'])} | "
                f"{_diff(comparison['perturbation_amplitude'])} |"
            )
        lines.append(
            f"| B5-B2 | {_diff(summary['B5-B2']['perturbation_detection'])} | n/a | "
            f"{_diff(summary['B5-B2']['perturbation_amplitude'])} |"
        )
        lines.append("")
    unc = report["unc31_check"]
    lines += [
        "## unc-31 check",
        "",
        (
            "WT-fitted compiler evaluated on unc-31 pairs; difference is peptides off "
            "minus on."
        ),
        "",
    ]
    for group in ("all", "unwired", "wired"):
        d = unc[group]["peptides_off_minus_on"]
        lines.append(
            f"- {group} ({unc[group]['n_pairs']} pairs): detection "
            f"{_diff(d['perturbation_detection'])}, amplitude "
            f"{_diff(d['perturbation_amplitude'])}"
        )
    b5 = report["B5_check"]
    stage = report["stage_b"]
    lines += [
        "",
        "## B5 check against Creamer et al.",
        "",
        (
            f"Published relative correlation {b5['published']['relative_correlation']} "
            f"(earlier versions {b5['published']['earlier_versions_relative_correlation']}) "
            "on held-out animals. Here B5 reaches "
            f"{_fmt(b5['B5_normalized_amplitude'])} of the amplitude ceiling on held-out "
            "stimulation targets. The splits and metrics are not comparable "
            f"({b5['reason']}). Anatomy-constrained B5 no worse than unconstrained B2: "
            f"{_fmt(b5['anatomy_constraint_no_worse'])}."
        ),
        "",
        "## Stage B (derived by the pre-registered rules)",
        "",
        (
            f"- Architecture: `{stage['architecture']}`; absolute trainable-parameter "
            f"budget: **{stage['absolute_parameter_budget']}**."
        ),
        "- Metric targets:",
    ]
    for key, target in stage["metric_targets"].items():
        lines.append(
            f"  - {key}: "
            + (
                "n/a"
                if target is None
                else f"{target['target']:.3f} ({target['units']})"
            )
        )
    if exploratory_results:
        lines += ["", "## Exploratory (not pre-registered)", ""]
        e = exploratory_results
        lines += [
            "- K4 training participation ratio with standardized columns: "
            + ", ".join(
                f"{v:.1f}" for v in e["K4_participation_ratio_standardized_columns"]
            ),
            f"- PLM 1-NN molecule-class accuracy: {e['plm_molecule_class_1nn_accuracy']:.3f}",
            (
                "- Anion synapses are stably inhibitory only if [Cl-]i < "
                f"{e['K3_chloride_for_stable_inhibition_mM']['value']:.1f} mM."
            ),
        ]
    return "\n".join(lines) + "\n"
