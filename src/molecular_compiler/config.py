"""YAML configuration with strict JSON-schema validation."""

from pathlib import Path

import jsonschema
import yaml

CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["species", "seed", "duration_s"],
    "properties": {
        "species": {"type": "string", "minLength": 1},
        "seed": {"type": "integer", "minimum": 0},
        "duration_s": {"type": "number", "exclusiveMinimum": 0},
        "mode": {"enum": ["sequential", "parallel", "event"]},
        "resolution": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "dt_s": {"type": "number", "exclusiveMinimum": 0},
                "n_comp": {"type": "integer", "minimum": 1},
                "solver": {"enum": ["auto", "dense", "pcg"]},
                "solve_tolerance": {"type": "number", "exclusiveMinimum": 0},
                "max_cg_iterations": {"type": "integer", "minimum": 1},
            },
        },
        "training": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "steps": {"type": "integer", "minimum": 1},
                "learning_rate": {"type": "number", "exclusiveMinimum": 0},
                "stage": {"type": "integer", "minimum": 0, "maximum": 5},
                "window_steps": {"type": "integer", "minimum": 1},
            },
        },
        "observation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "indicator": {"enum": ["GCaMP6s", "GCaMP6f", "GCaMP7f"]},
                "noise_sd": {"type": "number", "minimum": 0},
                "half_saturation": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "input_directory": {"type": "string"},
        "output_directory": {"type": "string"},
    },
}


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    jsonschema.validate(config, CONFIG_SCHEMA)
    return config
