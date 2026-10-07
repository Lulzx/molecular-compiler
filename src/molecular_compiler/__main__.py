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
    parser = argparse.ArgumentParser(prog="molc")
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
    ingest = sub.add_parser(
        "worm-ingest", help="convert registered public worm sources (Phase 0)"
    )
    ingest.add_argument("--cache", default="data/cache")
    ingest.add_argument("--output", default="data/worm")
    ingest.add_argument("--register", default="data-register.json")
    ingest.add_argument("--checkpoint-hash")
    ingest.add_argument("--device", default="cpu")
    register = sub.add_parser(
        "phase0-register", help="freeze the Stage A pre-registration"
    )
    register.add_argument("--project", default="data/worm")
    register.add_argument("--output", default="configs/phase0-stage-a.json")
    phase0 = sub.add_parser("phase0-run", help="run every Phase 0 analysis")
    phase0.add_argument("--project", default="data/worm")
    phase0.add_argument("--registration", default="configs/phase0-stage-a.json")
    phase0.add_argument("--output", default="artifacts/phase0")
    phase0.add_argument("--steps", type=int, default=1500)
    phase0.add_argument("--stage-b", default="configs/phase0-stage-b.json")
    traces = sub.add_parser("traces-ingest", help="ingest Randi et al. 2023 traces")
    traces.add_argument("--cache", default="data/cache")
    traces.add_argument("--project", default="data/worm")
    traces.add_argument("--register", default="data-register.json")
    fold = sub.add_parser("phase1-fold", help="train the full compiler on one fold")
    fold.add_argument("--project", default="data/worm")
    fold.add_argument("--registration", default="configs/phase0-stage-a.json")
    fold.add_argument("--output", default="artifacts/phase1")
    fold.add_argument("--split", default="leave_class_out")
    fold.add_argument("--fold", type=int, required=True)
    fold.add_argument("--steps", type=int, default=60)
    fold.add_argument("--batch", type=int, default=4)
    fold.add_argument(
        "--variant", default="base", choices=["base", "cl3", "cl8", "hill1"]
    )
    check = sub.add_parser("phase1-audit", help="pre-training audit of the model")
    check.add_argument("--project", default="data/worm")
    check.add_argument("--output", default="artifacts/phase1")
    animals = sub.add_parser(
        "heldout-animals", help="exploratory held-out-animal comparison"
    )
    animals.add_argument("--project", default="data/worm")
    animals.add_argument("--output", default="artifacts/phase1")
    animals.add_argument("--steps", type=int, default=1500)
    individual = sub.add_parser(
        "individual-state", help="Track I Stage I0 per-animal latent (spec 10.4)"
    )
    individual.add_argument("--project", default="data/worm")
    individual.add_argument("--output", default="artifacts/individual")
    converge = sub.add_parser("phase1-convergence", help="Section 12.6 test")
    converge.add_argument("--project", default="data/worm")
    converge.add_argument("--output", default="artifacts/phase1")
    converge.add_argument("--columns", type=int, default=24)
    converge.add_argument(
        "--trained", default=None, help="base-variant fold result to test"
    )
    phase1 = sub.add_parser("phase1-report", help="pool Phase 1 folds vs baselines")
    phase1.add_argument("--project", default="data/worm")
    phase1.add_argument("--registration", default="configs/phase0-stage-a.json")
    phase1.add_argument("--output", default="artifacts/phase1")
    phase0.add_argument(
        "--ripoll",
        default="data/cache/cect/cect/data/"
        "01022024_neuropeptide_connectome_long_range_model.csv",
    )
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
    elif args.command == "worm-ingest":
        from .provenance import DataRegister
        from .worm_public import build_worm_project

        _, report = build_worm_project(
            args.cache,
            args.output,
            DataRegister.load(args.register),
            checkpoint_hash=args.checkpoint_hash,
            device=args.device,
        )
        result = {
            k: v
            for k, v in report.items()
            if k not in ("register", "provenance", "declared_approximations")
        }
    elif args.command == "phase0-register":
        from .phase0 import freeze_registration
        from .phase0_worm import stage_a

        manifest = json.loads((Path(args.project) / "manifest.json").read_text())
        result = {
            "path": args.output,
            "processing_hash": freeze_registration(args.output, stage_a(manifest)),
        }
    elif args.command == "traces-ingest":
        from .provenance import DataRegister
        from .randi_traces import ingest as ingest_traces

        report = ingest_traces(
            args.cache, args.project, DataRegister.load(args.register)
        )
        result = {s: report[s]["summary"] for s in ("wt", "unc31")}
    elif args.command == "phase1-fold":
        from .phase0 import load_registration
        from .phase1_worm import run_fold

        report = run_fold(
            args.project,
            load_registration(args.registration),
            args.split,
            args.fold,
            args.output,
            args.steps,
            args.batch,
            args.variant,
        )
        result = {
            k: report[k]
            for k in (
                "split",
                "fold",
                "variant",
                "wall_seconds",
                "training_convergence",
            )
        }
    elif args.command == "phase1-audit":
        from .phase1_worm import audit

        report = audit(args.project, args.output)
        result = {k: report[k] for k in ("sign_audit", "rest", "synapses")}
        result["family_recovery"] = {
            k: report["family_recovery"][k] for k in ("rate", "passed", "per_form")
        }
    elif args.command == "heldout-animals":
        from .phase1_worm import held_out_animals

        report = held_out_animals(args.project, args.output, steps=args.steps)
        result = {
            "metrics": report["metrics"],
            "ceilings": report["ceilings"],
            "n_pairs": report["n_pairs"],
        }
    elif args.command == "individual-state":
        from .individual import individual_state

        report = individual_state(
            args.project, args.output, log=lambda m: print(m, flush=True)
        )
        result = {
            "r_star": report["r_star"],
            "gates": {
                "I0-1": report["gates"]["I0-1"]["pass"],
                "I0-2": report["gates"]["I0-2"]["pass"],
                "I0-3": report["gates"]["I0-3"]["verdict"],
            },
        }
    elif args.command == "phase1-convergence":
        from .phase1_worm import convergence_test

        report = convergence_test(
            args.project, args.output, columns=args.columns, trained=args.trained
        )
        result = {
            name: {"passed": report[name]["passed"], "checks": report[name]["checks"]}
            for name in ("half_dt", "plus_two_compartments")
        }
    elif args.command == "phase1-report":
        from .phase0 import load_registration
        from .phase1_worm import report as phase1_report

        report = phase1_report(
            args.project, load_registration(args.registration), args.output
        )
        result = {
            split: value.get("acceptance", value)
            for split, value in report["splits"].items()
        }
    elif args.command == "phase0-run":
        from .phase0 import freeze_registration, load_registration
        from .phase0_worm import exploratory, render_report
        from .phase0_worm import run as run_phase0
        from .worm_public import load_worm_project

        registration = load_registration(args.registration)
        report = run_phase0(
            args.project,
            registration,
            args.output,
            steps=args.steps,
            ripoll_csv=args.ripoll,
        )
        extra = exploratory(*load_worm_project(args.project), registration)
        (Path(args.output) / "exploratory.json").write_text(json.dumps(extra, indent=2))
        (Path(args.output) / "phase0-report.md").write_text(
            render_report(report, extra)
        )
        freeze_registration(
            args.stage_b,
            {
                **report["stage_b"],
                "stage_a_hash": registration["processing_hash"],
                "phase0_report_processing_hash": report["provenance"][
                    "processing_hash"
                ],
            },
        )
        result = {
            k: report[k]
            for k in ("K1", "K2", "K3", "stage_b", "elapsed_s")
            if k in report
        }
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
