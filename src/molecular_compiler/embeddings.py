"""Section 12.1 offline protein embedding, including overlapping long proteins."""

from hashlib import sha256
from pathlib import Path

import numpy as np

MODEL_NAME = "esm2_t33_650M_UR50D"


def pooled_embedding(sequence, residue_embedder, window=1022, stride=511):
    """Each residue is averaged over its windows before sequence pooling.

    residue_embedder accepts a protein string and returns [residues,1280]
    with BOS/EOS already removed. It must use ESM-2's final layer (33).
    """
    if not sequence or window < 1 or not 0 < stride <= window:
        raise ValueError("nonempty sequence and valid overlapping windows required")
    accumulation = np.zeros((len(sequence), 1280), dtype=np.float32)
    counts = np.zeros(len(sequence), dtype=np.int32)
    for start in range(0, len(sequence), stride):
        fragment = sequence[start : start + window]
        embedding = np.asarray(residue_embedder(fragment), dtype=np.float32)
        if embedding.shape != (len(fragment), 1280) or not np.all(
            np.isfinite(embedding)
        ):
            raise ValueError("ESM-2 residue embedding shape or values invalid")
        accumulation[start : start + len(fragment)] += embedding
        counts[start : start + len(fragment)] += 1
        if start + window >= len(sequence):
            break
    return (accumulation / counts[:, None]).mean(axis=0).astype(np.float16)


def esm2_embedder(checkpoint=None):
    """Load real ESM-2 outside training. Download occurs only when invoked."""
    import esm
    import torch

    model, alphabet = (
        esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
        if checkpoint
        else esm.pretrained.esm2_t33_650M_UR50D()
    )
    model.eval()
    converter = alphabet.get_batch_converter()

    def embed(fragment):
        _, _, tokens = converter([("protein", fragment)])
        with torch.no_grad():
            result = model(tokens, repr_layers=[33], return_contacts=False)
        return result["representations"][33][0, 1 : len(fragment) + 1].cpu().numpy()

    return embed


def checkpoint_file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def embed_genes(genes, embedder, checkpoint_hash):
    if len(checkpoint_hash) != 64:
        raise ValueError("embedding provenance requires checkpoint SHA-256")
    results = {}
    for gene in genes:
        results[gene["gene_id"]] = {
            "canonical": pooled_embedding(gene["protein_seq"], embedder),
            "isoforms": [pooled_embedding(seq, embedder) for seq in gene["isoforms"]],
        }
    return results, {
        "model": MODEL_NAME,
        "layer": 33,
        "pooling": "per_residue_window_average_then_mean",
        "checkpoint_hash": checkpoint_hash,
        "window": 1022,
        "stride": 511,
    }
