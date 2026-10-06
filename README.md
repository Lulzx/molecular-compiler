# molecular-compiler

A Python/JAX reference implementation of [spec.md](spec.md), revision 9.
It prepares registered connectome and molecular data, compiles compositional
molecular rules into a simulation, and differentiates through simulation and
calcium observation. It includes held-out evaluation, feasibility analyses,
experiment ranking, and body and distributed-solver interfaces.

The software is runnable. Scientific acceptance on animal data is **not yet
established**. The bundled demonstrations are synthetic. Phase 0 has been run
on public worm data (spec Section 10.2; procedure in [docs/phase0.md](docs/phase0.md)),
but its numerical report stays local because some inputs are restricted.
Phase 1 (the full compiled model on the same frozen comparison) is in
progress: spec Section 10.3, procedure and run state in
[docs/phase1.md](docs/phase1.md). See
[implementation status](docs/implementation.md) for the requirement map,
numerical evidence, and remaining delivery gates.

## Run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run molc demo --training-steps 10
uv run molc benchmark --sizes 8 32 128 --repeats 10
uv run pytest -q
```

The demo writes `artifacts/demo/report.json` and a trajectory archive. It runs
prepare → compile → simulate → observe, then optimizes molecular rule weights
through that full JAX program. The benchmark separates total step, backward
step, synaptic accumulation, and voltage-solve costs, including solver residuals
and iteration counts.

Install optional interchange or preprocessing dependencies when needed:

```sh
uv sync --locked --extra nwb
uv sync --locked --extra embeddings
```

Phase 0 on public worm data (sources cached under the gitignored `data/cache/`;
see [docs/phase0.md](docs/phase0.md)):

```sh
uv sync --locked --extra phase0 --extra embeddings
uv run molc worm-ingest --checkpoint-hash <esm2 sha256> --device mps
uv run molc phase0-run   # Stage A is already frozen in configs/
```

Phase 1 (full compiled model; see [docs/phase1.md](docs/phase1.md)):

```sh
uv run molc phase1-fold --split leave_class_out --fold 0 --steps 30 --batch 4
uv run molc phase1-report   # after all five folds are written
```

ESM-2 weights are loaded only when `esm2_embedder` is explicitly invoked.
Embeddings stay outside the gradient path. Long proteins use 1,022-residue
windows with stride 511; overlapping residue representations are averaged
before final pooling. Canonical sequences and isoforms are embedded separately.

## Python API

```python
from molecular_compiler import (
    ObservationModel,
    Stimulus,
    compile,
    observe,
    simulate,
)
from molecular_compiler.fixtures import synthetic_system

# Synthetic fixture; replace with registered animal inputs for research.
graph, rules, kinetics = synthetic_system()
sim = compile(graph, rules, kinetics)
trajectory = simulate(
    sim,
    Stimulus(pulses=((0, 0.001, 0.006, 5.0),)),  # pA, seconds
    duration_s=0.01,
)
recording = observe(trajectory, ObservationModel())
```

The six public interfaces are `prepare`, `compile`, `simulate`, `observe`,
`evaluate`, and `propose_experiments`. `attach_residuals` explicitly enables
within-animal corrections and recompiles identity deviations before linking.
Headline evaluation recompiles without those corrections or species adapters.

`RuleNetwork.initialize` provides independently trainable dense heads.
`RuleNetwork.initialize_compact` provides a frozen protein/set-encoder basis
with two tied learned head coefficients for small budgets. Both architectures
have parameter counts independent of neuron count. A supplied absolute
`max_parameters` budget is enforced and requires a completed gauge audit; the
synthetic demo's unrestricted network is not a frozen Phase 1 architecture.

## Registered inputs

Canonical projects contain:

- `manifest.json`, with connectome, molecular and kinetics dataset IDs,
  transmitter release/receptor mappings, and optional c302 cross-check evidence.
- `data-register.json`, with source, version, species, animal, license,
  attribution and `open` / `restricted` / `excluded` tier for every input.
- `neurons.parquet`, `synapses.parquet`, `contacts.parquet`, `genes.parquet`,
  `expression.parquet`, and `peptide_receptor_pairs.parquet`.
- `kinetics.json`, with molecule priors, covariance, ion selectivity, kinetics,
  sources and optional transporter equilibria.

Parquet fields and units are validated against the specification. Unregistered
or excluded inputs are rejected. Restricted inputs propagate their tier into
compiled graphs, checkpoints and reports. Publication helpers require open data.
The repository's [data register](data-register.json) starts empty; generated
fixtures create their own clearly labeled register.

A complete file-based workflow can be exercised without animal data:

```sh
uv run molc export-fixture artifacts/input
uv run molc rank artifacts/input
```

Copy `configs/worm.yaml`, set `input_directory: artifacts/input` and
`output_directory: artifacts/run`, then run:

```sh
uv run molc run your-config.yaml
```

YAML is validated against a strict JSON Schema. Recordings use Zarr with units
and provenance; NWB interchange is available through the `nwb` extra. Rules use
checksummed safetensors checkpoints, including any frozen basis.

## Numerics and evaluation

Voltages use mV, capacitance pF, conductance nS, current pA, and integration time
ms internally. The public API and kinetic time constants use seconds. Gating,
binding and STP use exponential updates. Gap junctions and axial compartments
participate in the same implicit voltage solve. Small graphs use dense Cholesky;
large graphs use matrix-free PCG with per-neuron block-Jacobi preconditioning.
Gradients use implicit adjoint solves. Two-level overlapping additive Schwarz
is available when profiling calls for stronger preconditioning.

Worm defaults are graded release, 0.5 ms and three compartments. Fly and
zebrafish defaults select spike release. `event` mode traverses only the
outgoing synapses of neurons that cross the spike threshold, while decaying
aggregated receptor conductances analytically. For graded release it uses the
reference update. Experimental `parallel` mode uses time-parallel waveform
iteration and falls back to the sequential reference when it does not converge.
It is not an implementation of DEER.

Peptides do not require anatomical edges. Coupling is factorized by ligand and
cognate receptor, with each pair's own GPCR kinetics. Worm coupling is global;
other species use a periodic screened-diffusion grid. Neuropeptide ligand
embeddings are excluded from the transferable neuron encoder; species-native
ligand/receptor pairing and receptor tokens determine the coupling.

Evaluation checks whole-class holdouts, scores only selected held-out neurons,
reports raw values, units, sign ambiguity and wired-path groups, normalizes
available metrics by pre-registered reliability scales, and gates acceptance against B0, B1 and B2 with paired
class/neuron bootstrap intervals. Missing baselines, ceilings, frozen budgets, gauge audits
or real data keep the report unaccepted. B4 is executable; B0/B1/B2/B5 reference
implementations and separate-job B6/fly-B0 artifact adapters are provided.

## License

Apache 2.0 for code. Input data retain their own licenses. See [LICENSE](LICENSE).
