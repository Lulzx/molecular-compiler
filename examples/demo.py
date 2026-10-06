"""Synthetic end-to-end compiler with differentiable rule training."""

import json

from molecular_compiler.demo import run

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
