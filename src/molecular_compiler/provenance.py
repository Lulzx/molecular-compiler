"""M1-R4/R5: registered inputs and inherited licensing."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class DataRecord:
    dataset_id: str
    species: str
    animal_id: str
    source: str
    version: str
    license: str
    tier: str
    attribution: str = ""

    def __post_init__(self):
        if self.tier not in {"open", "restricted", "excluded"}:
            raise ValueError("unknown data tier")
        if not all(
            (
                self.dataset_id,
                self.species,
                self.animal_id,
                self.source,
                self.version,
                self.license,
            )
        ):
            raise ValueError("data register entries require complete provenance")


class DataRegister:
    def __init__(self, records=()):
        self.records = {r.dataset_id: r for r in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate dataset_id")

    @classmethod
    def load(cls, path):
        return cls([DataRecord(**r) for r in json.loads(Path(path).read_text())])

    def require(self, dataset_id):
        if dataset_id not in self.records:
            raise ValueError(f"unregistered dataset: {dataset_id}")
        record = self.records[dataset_id]
        if record.tier == "excluded":
            raise ValueError(f"excluded dataset: {dataset_id}")
        return record


def content_hash(*values):
    def convert(value):
        if hasattr(value, "to_pylist"):
            return value.to_pylist()
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        raise TypeError(type(value).__name__)

    return sha256(
        json.dumps(
            values,
            default=convert,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def inherit(records, processing_hash):
    if not records or any(r.tier == "excluded" for r in records):
        raise ValueError("artifacts require eligible registered inputs")
    return {
        "tier": "restricted"
        if any(r.tier == "restricted" for r in records)
        else "open",
        "inputs": [asdict(r) for r in records],
        "processing_hash": processing_hash,
        "attributions": sorted({r.attribution for r in records if r.attribution}),
    }


def require_publishable(metadata):
    if metadata.get("tier") != "open":
        raise ValueError("publication requires open-tier inputs")
