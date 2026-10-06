"""Command line entrypoints for reproducible local workflows."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .benchmark import scaling_benchmark
from .compiler import ResolutionPolicy, compile
from .config import load_config
from .demo import run
from .fixtures import synthetic_system
from .observation import ObservationModel, observe
from .phase0 import molecular_design
from .project import export_project, load_project
from .rules import RuleNetwork
from .simulation import Stimulus, simulate
from .storage import load_checkpoint, save_simgraph, write_recording


def main():
    parser = argparse.ArgumentParser(prog="molecular-compiler")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="synthetic pipeline and gradient training")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--training-steps", type=int, default=5)
    benchmark = sub.add_parser("benchmark", help="measured synthetic scaling costs")
    benchmark.add_argument("--sizes", nargs="+", type=int, default=[8, 32, 128])
    benchmark.add_argument("--repeats", type=int, default=10)
    benchmark.add_argument("--output", default="artifacts/benchmark.json")
    fixture = sub.add_parser(
        "export-fixture", help="write a canonical synthetic project"
    )
    fixture.add_argument("directory")
    execute = sub.add_parser(
        "run", help="run registered Parquet inputs from YAML config"
    )
    execute.add_argument("config")
    execute.add_argument("--checkpoint")
    analysis = sub.add_parser(
        "rank", help="chemical/contact/implicit-peptide design rank"
    )
    analysis.add_argument("directory")
    args = parser.parse_args()
    if args.command == "demo":
        result = run(args.output, args.training_steps)
    elif args.command == "benchmark":
        result = scaling_benchmark(tuple(args.sizes), args.repeats)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    elif args.command == "export-fixture":
        graph, _, kinetics = synthetic_system()
        export_project(graph, kinetics, args.directory)
        result = {"directory": args.directory, "data_kind": "synthetic"}
    elif args.command == "rank":
        graph, _ = load_project(args.directory)
        result = molecular_design(graph)
        result.pop("gram")
    else:
        config = load_config(args.config)
        if config.get("training"):
            raise ValueError(
                "run executes inference; use the train API or demo command for optimization"
            )
        graph, kinetics = load_project(config["input_directory"])
        if config["species"] != graph.metadata["species"]:
            raise ValueError("configuration species differs from registered data")
        if args.checkpoint:
            rules, _ = load_checkpoint(args.checkpoint)
        else:
            rules = RuleNetwork.initialize(config["seed"])
        policy = replace(
            ResolutionPolicy.default(config["species"]), **config.get("resolution", {})
        )
        sim = compile(graph, rules, kinetics, resolution=policy)
        trajectory = simulate(
            sim,
            Stimulus(),
            config["duration_s"],
            mode=config.get("mode", "sequential"),
            seed=config["seed"],
        )
        recording = observe(
            trajectory,
            ObservationModel(**config.get("observation", {}), seed=config["seed"]),
        )
        path = Path(config.get("output_directory", "artifacts/run"))
        path.mkdir(parents=True, exist_ok=True)
        save_simgraph(sim, path / "simgraph")
        write_recording(recording, path / "recording.zarr")
        result = trajectory.metadata
        (path / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
