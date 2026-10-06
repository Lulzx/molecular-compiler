"""Typed Parquet schemas, validation, and M1-M3 preparation."""

import json
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist

from .provenance import DataRegister, content_hash, inherit


def _schema(fields, units):
    return pa.schema(
        [pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in fields],
        metadata={b"units": json.dumps(units, sort_keys=True).encode()},
    )


f32, f64, i64, text = pa.float32(), pa.float64(), pa.int64(), pa.string()
xyz = pa.list_(f32, 3)
SCHEMAS = {
    "neurons": _schema(
        [
            ("neuron_id", i64, False),
            ("animal_id", text, False),
            ("type_label", text, True),
            ("soma_xyz", xyz, False),
            ("morphology_ref", text, True),
            ("segmentation_confidence", f32, False),
        ],
        {"soma_xyz": "um", "segmentation_confidence": "probability"},
    ),
    "synapses": _schema(
        [
            ("synapse_id", i64, False),
            ("pre_id", i64, False),
            ("post_id", i64, False),
            ("size", f32, False),
            ("vesicle_count", f32, True),
            ("xyz", xyz, False),
            ("path_dist_post", f32, False),
            ("local_diameter_post", f32, False),
            ("compartment_post", pa.int16(), False),
            ("nt_pred", pa.list_(f32), True),
            ("detection_confidence", f32, False),
        ],
        {
            "size": "count",
            "xyz": "um",
            "path_dist_post": "um",
            "local_diameter_post": "um",
            "detection_confidence": "probability",
        },
    ),
    "contacts": _schema(
        [
            ("i_id", i64, False),
            ("j_id", i64, False),
            ("area", f32, False),
            ("gap_junction_observed", pa.bool_(), True),
        ],
        {"area": "um2"},
    ),
    "expression": _schema(
        [
            ("unit_id", text, False),
            ("unit_level", text, False),
            ("gene_id", text, False),
            ("value", f32, False),
            ("assay", text, False),
            ("subcellular", text, False),
        ],
        {"value": "normalized_expression"},
    ),
    "genes": _schema(
        [
            ("gene_id", text, False),
            ("species", text, False),
            ("protein_seq", text, False),
            ("isoforms", pa.list_(text), False),
            ("ortholog_group", text, True),
            ("molecule_class", text, False),
            ("plm_embedding", pa.list_(pa.float16(), 1280), False),
        ],
        {"plm_embedding": "dimensionless"},
    ),
    "peptide_receptor_pairs": _schema(
        [
            ("peptide_gene_id", text, False),
            ("receptor_gene_id", text, False),
            ("ec50_nM", f32, True),
            ("evidence", text, False),
            ("source", text, False),
        ],
        {"ec50_nM": "nM"},
    ),
    "stimulations": _schema(
        [
            ("t_on", f64, False),
            ("t_off", f64, False),
            ("target_id", i64, False),
            ("wavelength_nm", f32, False),
            ("power", f32, False),
            ("opsin", text, False),
            ("opsin_expression", f32, True),
        ],
        {
            "t_on": "s",
            "t_off": "s",
            "wavelength_nm": "nm",
            "power": "mW",
            "opsin_expression": "relative",
        },
    ),
}
ENUMS = {
    "unit_level": {"cell", "type"},
    "assay": {"scRNA", "bulk", "ExSeq", "ExM_protein"},
    "subcellular": {"soma", "synapse", "neurite", "whole"},
    "molecule_class": {
        "channel",
        "receptor",
        "transporter",
        "innexin",
        "peptide",
        "gpcr",
        "effector",
        "other",
    },
    "evidence": {"in_vitro_screen", "in_vivo", "predicted"},
}


def table(name, rows, units=None):
    schema = SCHEMAS[name]
    if units is not None:
        schema = schema.with_metadata(
            {b"units": json.dumps(units, sort_keys=True).encode()}
        )
    result = pa.Table.from_pylist(rows, schema=schema)
    validate_table(name, result)
    return result


def validate_table(name, value):
    expected = SCHEMAS[name]
    if value.column_names != expected.names:
        raise ValueError(f"{name}: columns must match schema")
    for f in expected:
        column = value[f.name]
        if column.type != f.type or (not f.nullable and column.null_count):
            raise ValueError(f"{name}.{f.name}: invalid type or null")
    units = json.loads((value.schema.metadata or {}).get(b"units", b"{}"))
    required_units = json.loads(expected.metadata[b"units"])
    if name == "synapses" and units.get("size") == "um2":
        required_units["size"] = "um2"
    if any(units.get(k) != v for k, v in required_units.items()):
        raise ValueError(f"{name}: missing or invalid units")
    rows = value.to_pylist()
    for row in rows:
        for key, val in row.items():
            if key in ENUMS and val not in ENUMS[key]:
                raise ValueError(f"{name}.{key}: unknown enum")
            if isinstance(val, (float, list)) and val is not None:
                try:
                    if not np.all(np.isfinite(np.asarray(val, dtype=float))):
                        raise ValueError(f"{name}.{key}: non-finite value")
                except (TypeError, ValueError):
                    if key not in {"isoforms"}:
                        raise
            if key.endswith("confidence") and not 0 <= val <= 1:
                raise ValueError(f"{name}.{key}: outside [0,1]")
            if (
                key
                in {
                    "size",
                    "area",
                    "path_dist_post",
                    "value",
                    "vesicle_count",
                    "ec50_nM",
                    "power",
                }
                and val is not None
                and val < 0
            ):
                raise ValueError(f"{name}.{key}: negative quantity")
        if name == "synapses" and (
            row["local_diameter_post"] <= 0 or row["compartment_post"] < 0
        ):
            raise ValueError("invalid synapse geometry")
        if (
            name == "peptide_receptor_pairs"
            and row["ec50_nM"] is not None
            and row["ec50_nM"] <= 0
        ):
            raise ValueError("measured EC50 must be positive")
        if name == "synapses" and row["nt_pred"] is not None:
            probabilities = np.asarray(row["nt_pred"])
            if (
                np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
            ):
                raise ValueError(
                    "transmitter predictions must be probabilities summing to one"
                )
        if name == "stimulations" and row["t_off"] <= row["t_on"]:
            raise ValueError("stimulus duration must be positive")
    unique = {"neurons": "neuron_id", "synapses": "synapse_id", "genes": "gene_id"}.get(
        name
    )
    if unique and len({r[unique] for r in rows}) != len(rows):
        raise ValueError(f"{name}: duplicate {unique}")


def write_table(name, value, path, metadata):
    validate_table(name, value)
    schema_meta = dict(value.schema.metadata or {})
    schema_meta[b"provenance"] = json.dumps(metadata, sort_keys=True).encode()
    pq.write_table(value.replace_schema_metadata(schema_meta), path)


def read_table(name, path):
    result = pq.read_table(path)
    validate_table(name, result)
    if b"provenance" not in (result.schema.metadata or {}):
        raise ValueError("Parquet input lacks provenance")
    return result


@dataclass
class ConnectomeDataset:
    dataset_id: str
    neurons: pa.Table
    synapses: pa.Table
    contacts: pa.Table
    register: DataRegister
    c302_reference: dict | None = None
    c302_explanation: str | None = None


@dataclass
class MolecularDataset:
    dataset_id: str
    genes: pa.Table
    expression: pa.Table
    peptide_receptor_pairs: pa.Table
    register: DataRegister
    released_ligands: dict = field(default_factory=dict)
    receptor_ligands: dict = field(default_factory=dict)


@dataclass
class ExMDataset:
    dataset_id: str
    assignments: dict
    register: DataRegister
    protein_pairs: dict = field(default_factory=dict)


@dataclass
class MeasurementModel:
    slopes: jax.Array
    offsets: jax.Array
    variance: jax.Array

    @classmethod
    def fit(cls, expression, protein=None, classes=None):
        x = np.asarray(expression)
        slopes, offsets, variance = (
            np.ones(x.shape[1]),
            np.zeros(x.shape[1]),
            np.ones(x.shape[1]),
        )
        if protein is not None:
            y = np.asarray(protein)
            if y.shape != x.shape:
                raise ValueError("paired protein measurements must match expression")
            classes = np.asarray(
                classes if classes is not None else np.arange(x.shape[1])
            )
            for group in np.unique(classes):
                mask = classes == group
                xx, yy = x[:, mask].ravel(), y[:, mask].ravel()
                valid = np.isfinite(yy)
                if valid.sum() >= 3:
                    design = np.column_stack([xx[valid], np.ones(valid.sum())])
                    slope, offset = np.linalg.lstsq(design, yy[valid], rcond=None)[0]
                    slopes[mask], offsets[mask] = max(slope, 0), offset
                    variance[mask] = max(
                        np.mean((yy[valid] - design @ [slope, offset]) ** 2), 1e-6
                    )
        return cls(jnp.asarray(slopes), jnp.asarray(offsets), jnp.asarray(variance))

    def predict(self, expression):
        abundance = jnp.maximum(expression * self.slopes + self.offsets, 0)
        return abundance, self.variance * (1 + abundance**2)


@dataclass
class AnnotatedGraph:
    connectome: ConnectomeDataset
    molecular: MolecularDataset
    neuron_ids: np.ndarray
    type_names: tuple
    type_expression: jax.Array
    assignments: jax.Array
    abundance: jax.Array
    abundance_variance: jax.Array
    gene_ids: tuple
    metadata: dict
    measurement_model: MeasurementModel | None = None

    def sample_abundance(self, seed):
        indices = jax.random.categorical(
            jax.random.key(seed), jnp.log(self.assignments), axis=1
        )
        sampled = self.type_expression[indices]
        return (
            self.measurement_model.predict(sampled)[0]
            if self.measurement_model
            else sampled
        )

    def identity_samples(self, seed=0, count=8):
        return jnp.stack([self.sample_abundance(seed + i) for i in range(count)])


def prepare(connectome, molecular, anchors=None):
    records = [
        connectome.register.require(connectome.dataset_id),
        molecular.register.require(molecular.dataset_id),
    ]
    for name in ("neurons", "synapses", "contacts"):
        validate_table(name, getattr(connectome, name))
    for name in ("genes", "expression", "peptide_receptor_pairs"):
        validate_table(name, getattr(molecular, name))
    nr, sr, cr = (
        getattr(connectome, n).to_pylist() for n in ("neurons", "synapses", "contacts")
    )
    if not nr:
        raise ValueError("empty neuron set")
    ids = np.array([r["neuron_id"] for r in nr], dtype=np.int64)
    id_index = {int(v): i for i, v in enumerate(ids)}
    for row in sr + cr:
        keys = ("pre_id", "post_id") if "pre_id" in row else ("i_id", "j_id")
        if any(row[k] not in id_index for k in keys):
            raise ValueError("edge references unknown neuron")
    if any(r["animal_id"] != records[0].animal_id for r in nr):
        raise ValueError("animal_id does not match register")
    if records[0].species != records[1].species:
        raise ValueError("connectome and expression species differ")
    genes = molecular.genes.to_pylist()
    if any(g["species"] != records[1].species for g in genes):
        raise ValueError("gene species does not match register")
    gene_ids = tuple(g["gene_id"] for g in genes)
    gi = {g: i for i, g in enumerate(gene_ids)}
    er = molecular.expression.to_pylist()
    if any(r["gene_id"] not in gi for r in er):
        raise ValueError("expression references unknown gene")
    for pair in molecular.peptide_receptor_pairs.to_pylist():
        if pair["peptide_gene_id"] not in gi or pair["receptor_gene_id"] not in gi:
            raise ValueError("pair references unknown gene")
    type_names = tuple(sorted({r["unit_id"] for r in er if r["unit_level"] == "type"}))
    if not type_names:
        raise ValueError(
            "type-level reference expression is required for identity inference"
        )
    ti = {t: i for i, t in enumerate(type_names)}
    expression = np.zeros((len(ti), len(gi)))
    cells = {}
    for r in er:
        if r["assay"] == "ExM_protein":
            continue
        if r["unit_level"] == "type":
            expression[ti[r["unit_id"]], gi[r["gene_id"]]] = r["value"]
        else:
            cells.setdefault(r["unit_id"], np.zeros(len(gi)))[gi[r["gene_id"]]] = r[
                "value"
            ]
    fixed = {
        i: ti[r["type_label"]]
        for i, r in enumerate(nr)
        if r["type_label"] in ti and r["segmentation_confidence"] >= 0.95
    }
    if anchors:
        records.append(anchors.register.require(anchors.dataset_id))
        for neuron, label in anchors.assignments.items():
            if neuron not in id_index or label not in ti:
                raise ValueError("invalid identity anchor")
            fixed[id_index[neuron]] = ti[label]
    unknown = [i for i in range(len(ids)) if i not in fixed]
    plan = np.zeros((len(ids), len(ti)))
    if unknown:
        import ot

        # Connectivity fingerprints; do not allocate an N x N adjacency matrix.
        fingerprints = np.zeros((len(ids), 2 + 2 * len(ti)))
        for r in sr:
            pre, post = id_index[r["pre_id"]], id_index[r["post_id"]]
            fingerprints[pre, 0] += r["size"]
            fingerprints[post, 1] += r["size"]
            if post in fixed:
                fingerprints[pre, 2 + fixed[post]] += r["size"]
            if pre in fixed:
                fingerprints[post, 2 + len(ti) + fixed[pre]] += r["size"]
        c1 = cdist(fingerprints[unknown], fingerprints[unknown])
        c2 = cdist(expression, expression)
        c1 /= max(c1.max(), 1)
        c2 /= max(c2.max(), 1)
        markers = np.zeros((len(unknown), len(ti)))
        for a, i in enumerate(unknown):
            label = nr[i]["type_label"]
            if label in ti:
                markers[a] = 1
                markers[a, ti[label]] = 0
            elif str(ids[i]) in cells:
                markers[a] = cdist(cells[str(ids[i])][None], expression)[0]
        transport = ot.gromov.fused_gromov_wasserstein(
            markers,
            c1,
            c2,
            np.ones(len(unknown)) / len(unknown),
            np.ones(len(ti)) / len(ti),
            alpha=0.5,
        )
        plan[unknown] = transport / transport.sum(axis=1, keepdims=True)
    for i, t in fixed.items():
        plan[i, t] = 1
    neuron_expression = plan @ expression
    for neuron, measured in cells.items():
        if neuron.isdigit() and int(neuron) in id_index:
            neuron_expression[id_index[int(neuron)]] = measured
    protein_units = {}
    for row in er:
        if row["assay"] == "ExM_protein":
            protein_units.setdefault(row["unit_id"], {})[row["gene_id"]] = row["value"]
    if anchors:
        for unit, measurements in anchors.protein_pairs.items():
            protein_units.setdefault(str(unit), {}).update(measurements)
    paired_x, paired_y = [], []
    for unit, measurements in protein_units.items():
        if unit in ti:
            x = expression[ti[unit]]
        elif unit.isdigit() and int(unit) in id_index:
            x = neuron_expression[id_index[int(unit)]]
        else:
            raise ValueError("protein measurement references unknown cell or type")
        y = np.full(len(gi), np.nan)
        for gene, value in measurements.items():
            if gene not in gi or not np.isfinite(value) or value < 0:
                raise ValueError("invalid paired protein measurement")
            y[gi[gene]] = value
        paired_x.append(x)
        paired_y.append(y)
    molecule_classes = [g["molecule_class"] for g in genes]
    model = MeasurementModel.fit(
        np.asarray(paired_x) if paired_x else expression,
        np.asarray(paired_y) if paired_y else None,
        classes=molecule_classes,
    )
    abundance, variance = model.predict(jnp.asarray(neuron_expression))
    crosscheck = {"status": "unavailable", "reason": "c302 reference not supplied"}
    if connectome.c302_reference is not None:
        ref = connectome.c302_reference
        differences = {
            "neuron_set": set(ref["neuron_ids"]) != set(ids.tolist()),
            "chemical_edges": ref["chemical_edges"] != len(sr),
            "gap_contacts": ref["gap_contacts"]
            != sum(r["gap_junction_observed"] is True for r in cr),
        }
        if any(differences.values()) and not connectome.c302_explanation:
            raise ValueError("c302 differences require an explanation")
        crosscheck = {
            "status": "checked",
            "differences": differences,
            "explanation": connectome.c302_explanation,
        }
    digest = content_hash(
        connectome.neurons,
        connectome.synapses,
        connectome.contacts,
        molecular.genes,
        molecular.expression,
        molecular.peptide_receptor_pairs,
        molecular.released_ligands,
        molecular.receptor_ligands,
        {}
        if anchors is None
        else {
            "assignments": anchors.assignments,
            "protein_pairs": anchors.protein_pairs,
        },
        connectome.c302_reference,
        connectome.c302_explanation,
    )
    metadata = inherit(records, digest)
    metadata.update(
        species=records[0].species,
        animal_id=records[0].animal_id,
        c302_crosscheck=crosscheck,
        identity_entropy=(
            -np.sum(plan * np.log(np.maximum(plan, 1e-30)), axis=1)
        ).tolist(),
        molecular_measurement="paired_class_regression"
        if paired_x
        else "identity_with_inflated_uncertainty",
    )
    return AnnotatedGraph(
        connectome,
        molecular,
        ids,
        type_names,
        jnp.asarray(expression),
        jnp.asarray(plan),
        abundance,
        variance,
        gene_ids,
        metadata,
        model,
    )


def anchor_accuracy(assignments, heldout):
    """Held-out anchors must be excluded from the alignment passed to this helper."""
    if not heldout:
        raise ValueError("anchor accuracy requires held-out anchors")
    return float(np.mean([np.argmax(assignments[i]) == t for i, t in heldout.items()]))
