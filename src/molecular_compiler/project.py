"""Load registered, canonical dataset exports without source-specific guessing."""

import json
from pathlib import Path

from .data import ConnectomeDataset, MolecularDataset, prepare, read_table
from .kinetics import KineticRecord, KineticsLibrary
from .provenance import DataRegister, content_hash, inherit


def load_project(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    register = DataRegister.load(root / "data-register.json")
    connectome = ConnectomeDataset(
        manifest["connectome_dataset_id"],
        *[
            read_table(name, root / f"{name}.parquet")
            for name in ("neurons", "synapses", "contacts")
        ],
        register,
        c302_reference=manifest.get("c302_reference"),
        c302_explanation=manifest.get("c302_explanation"),
    )
    molecular = MolecularDataset(
        manifest["molecular_dataset_id"],
        *[
            read_table(name, root / f"{name}.parquet")
            for name in ("genes", "expression", "peptide_receptor_pairs")
        ],
        register,
        released_ligands=manifest.get("released_ligands", {}),
        receptor_ligands=manifest.get("receptor_ligands", {}),
    )
    rows = json.loads((root / "kinetics.json").read_text())
    records = [KineticRecord(**row) for row in rows]
    if len({r.molecule_id for r in records}) != len(records):
        raise ValueError("duplicate kinetic molecule IDs")
    kinetic_record = register.require(manifest["kinetics_dataset_id"])
    library = KineticsLibrary(
        {r.molecule_id: r for r in records},
        metadata=inherit([kinetic_record], content_hash(rows)),
    )
    return prepare(connectome, molecular), library


def export_project(graph, kinetics, directory):
    from dataclasses import asdict

    from .data import write_table

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record_values = dict(graph.connectome.register.records)
    record_values.update(graph.molecular.register.records)
    if not kinetics.metadata or not kinetics.metadata.get("inputs"):
        raise ValueError("project export requires registered kinetics")
    for record in kinetics.metadata["inputs"]:
        from .provenance import DataRecord

        record_values[record["dataset_id"]] = DataRecord(**record)
    (root / "data-register.json").write_text(
        json.dumps([asdict(r) for r in record_values.values()], indent=2)
    )
    manifest = {
        "connectome_dataset_id": graph.connectome.dataset_id,
        "molecular_dataset_id": graph.molecular.dataset_id,
        "kinetics_dataset_id": kinetics.metadata["inputs"][0]["dataset_id"],
        "released_ligands": graph.molecular.released_ligands,
        "receptor_ligands": graph.molecular.receptor_ligands,
        "c302_reference": graph.connectome.c302_reference,
        "c302_explanation": graph.connectome.c302_explanation,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in ("neurons", "synapses", "contacts"):
        write_table(
            name,
            getattr(graph.connectome, name),
            root / f"{name}.parquet",
            graph.metadata,
        )
    for name in ("genes", "expression", "peptide_receptor_pairs"):
        write_table(
            name,
            getattr(graph.molecular, name),
            root / f"{name}.parquet",
            graph.metadata,
        )
    rows = []
    for record in kinetics.records.values():
        if record.mechanism is not None:
            raise ValueError(
                "custom Jaxley mechanisms need a separately versioned mechanism plugin"
            )
        row = asdict(record)
        row.pop("mechanism")
        row["embedding"] = list(map(float, row["embedding"]))
        rows.append(row)
    (root / "kinetics.json").write_text(json.dumps(rows, indent=2))
