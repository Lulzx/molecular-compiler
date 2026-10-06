"""Explicitly synthetic fixtures for software tests; no animal-data claims."""

import numpy as np

from .data import ConnectomeDataset, MolecularDataset, prepare, table
from .kinetics import KineticRecord, KineticsLibrary
from .provenance import DataRecord, DataRegister, content_hash, inherit
from .rules import RuleNetwork


def synthetic_system(n=4, seed=0, species="C. elegans"):
    if n < 2:
        raise ValueError("synthetic network needs at least two neurons")
    rng = np.random.default_rng(seed)
    register = DataRegister(
        [
            DataRecord(
                "synthetic-connectome",
                species,
                "fixture",
                "generated fixture",
                "1",
                "CC0-1.0",
                "open",
            ),
            DataRecord(
                "synthetic-molecular",
                species,
                "fixture",
                "generated fixture",
                "1",
                "CC0-1.0",
                "open",
            ),
        ]
    )
    neurons = table(
        "neurons",
        [
            {
                "neuron_id": i,
                "animal_id": "fixture",
                "type_label": "A" if i % 2 == 0 else "B",
                "soma_xyz": [float(i * 10), 0.0, 0.0],
                "morphology_ref": None,
                "segmentation_confidence": 1.0,
            }
            for i in range(n)
        ],
    )
    synapses = table(
        "synapses",
        [
            {
                "synapse_id": i,
                "pre_id": i,
                "post_id": i + 1,
                "size": 1.0,
                "vesicle_count": None,
                "xyz": [float(i * 10), 0.0, 0.0],
                "path_dist_post": 5.0,
                "local_diameter_post": 1.0,
                "compartment_post": 0,
                "nt_pred": [1.0, 0.0, 0.0],
                "detection_confidence": 1.0,
            }
            for i in range(n - 2)
        ],
    )
    contacts = table(
        "contacts", [{"i_id": 0, "j_id": 1, "area": 0.1, "gap_junction_observed": True}]
    )
    specs = [
        ("ca", "channel"),
        ("glr", "receptor"),
        ("kcc", "transporter"),
        ("inx", "innexin"),
        ("flp", "peptide"),
        ("gpcr", "gpcr"),
        ("eff", "effector"),
    ]
    embeddings = rng.normal(0, 0.1, (len(specs), 1280)).astype(np.float16)
    genes = table(
        "genes",
        [
            {
                "gene_id": gene,
                "species": species,
                "protein_seq": "MAAA",
                "isoforms": [],
                "ortholog_group": None,
                "molecule_class": kind,
                "plm_embedding": embeddings[i].tolist(),
            }
            for i, (gene, kind) in enumerate(specs)
        ],
    )
    er = []
    for label in ("A", "B"):
        for gene, _ in specs:
            value = 1.0
            if gene == "flp" and label == "B":
                value = 0.0
            if gene == "gpcr" and label == "A":
                value = 0.0
            er.append(
                {
                    "unit_id": label,
                    "unit_level": "type",
                    "gene_id": gene,
                    "value": value,
                    "assay": "scRNA",
                    "subcellular": "whole",
                }
            )
    expression = table("expression", er)
    pairs = table(
        "peptide_receptor_pairs",
        [
            {
                "peptide_gene_id": "flp",
                "receptor_gene_id": "gpcr",
                "ec50_nM": 1.0,
                "evidence": "predicted",
                "source": "synthetic fixture",
            }
        ],
    )
    connectome = ConnectomeDataset(
        "synthetic-connectome", neurons, synapses, contacts, register
    )
    molecular = MolecularDataset(
        "synthetic-molecular",
        genes,
        expression,
        pairs,
        register,
        {"A": ["glutamate"], "B": ["glutamate"]},
        {"glr": "glutamate"},
    )
    graph = prepare(connectome, molecular)
    records = {}
    for i, (gene, kind) in enumerate(specs):
        if kind not in {"channel", "receptor", "transporter", "innexin", "gpcr"}:
            continue
        form = {
            "channel": "HH",
            "receptor": "ligand_gated",
            "transporter": "transporter",
            "innexin": "HH",
            "gpcr": "gpcr",
        }[kind]
        records[gene] = KineticRecord(
            gene,
            form,
            (1.0,),
            ((1.0,),),
            {"Ca": 1.0} if gene == "ca" else {},
            "synthetic fixture",
            family=kind,
            embedding=tuple(embeddings[i]),
            reversal_mV=60.0 if gene == "ca" else 0.0,
            gbar_nS=0.01,
            tau_s=0.01 if gene != "gpcr" else 0.05,
            transport_equilibrium_mM={"Cl": 2.0} if gene == "kcc" else {},
        )
    return (
        graph,
        RuleNetwork.initialize(seed=seed),
        KineticsLibrary(
            records,
            metadata=inherit(
                [register.require("synthetic-molecular")],
                content_hash("synthetic-kinetics-v1"),
            ),
        ),
    )
