# Implementation and acceptance status

This repository implements an executable reference system for revision 13 of
`spec.md`. Software tests and synthetic demonstrations establish numerical and
API behavior and satisfy the initial reference-software scope in Section 1.6.
They do not establish the scientific exits in Section 10.

## Module map

| Module | Implementation | Evidence / limits |
|---|---|---|
| M1 | `data.py`, `provenance.py`, `project.py`, `storage.py` | Typed/unit-validated Parquet; FK/ID validation; licensing register and tier inheritance; confidence masks; source hashes; Zarr/NWB; c302 reference comparison with required discrepancy explanations. Raw source exports must be converted to canonical tables. |
| M2 | `data.prepare`, `AnnotatedGraph` | Known high-confidence labels pass through; unknown identities use fused Gromov–Wasserstein through POT; anchors, posterior entropy, held-out-anchor scoring and eight assignment samples. Dense modality distances are an offline preparation cost. |
| M3 | `MeasurementModel` | Paired class regressions and abundance-dependent uncertainty; identity map with inflated variance where pairs are unavailable. Paired protein data can come from expression tables or anchors. |
| M4 | `RuleNetwork` | Protein token/set encoder; fixed ortholog vocabulary; exact masks; edge-only receptor and STP heads; symmetric gap head; receptor-based factorized peptide heads; adapter support; explicit absolute capacity budget. Dense and compact tied-head architectures are distinguished. |
| M5 | `KineticsLibrary`, `jaxley_backend.py` | Jaxley first-order channel mechanisms; externally supplied multi-gate mechanisms; Markov channels; ligand/GPCR binding; transporter equilibria and electrogenic current; Q10; nearest family priors; Mahalanobis/deviation reports. Actual molecule records need measured source parameters. Mixed divalent GHK requires an explicit reversal model and is rejected otherwise. |
| M6 | `compiler.py`, `morphology.py` | Linear-time CSR counting sort, geometric attenuation/compartment assignment, ion reversals, chloride audit, expected uncertain gaps and existence-weighted synapses. No fixed synaptic-sign field. SWC skeleton trees are validated and reduced to `n_comp` compartments by electrotonic binning. Total area is conserved exactly, and soma input resistance is matched by an axial scale. `compile(..., skeletons=...)` is opt-in; without skeletons, output is unchanged. With skeletons, neurons that have one also get per-compartment capacitance and leak scaled by reduced compartment area (neuron totals preserved) and per-link axial conductances in `neuron_params["axial"]`, used by the dense and PCG solves; other neurons keep the policy values, and the distributed solve raises for them. Limits: the Schwarz preconditioner uses the per-link values only through `coupling_edges`. Skeleton reduction cannot run under `jax.grad` and raises a clear error. No real SWC data has been used. |
| M7 | `surrogates.py`, `simulation.step` | Fixed ODE → neural ODE → full cascade, derivative and rollout validation, conditioned envelopes and runtime fallback. Full dynamics remain the default (bit-identical to the previous step). An opt-in fast path (`plan_fast_execution`) skips gates, channel conductance, calcium kinetics and the voltage solve for eligible neurons. Any envelope exit sends that whole step through the hybrid branch, and the exit is logged. Synthetic timings with all neurons eligible: about 3× faster at 512 neurons with PCG. The dense-solver sizes show larger, noisy gains. The tests use hand-built surrogates only; no fitted worm surrogates. |
| M8 | `simulation.py`, `solver.py` | Exponential gating/binding/STP; implicit coupled voltage; dense Cholesky and block-Jacobi PCG; adjoint gradients; scan/checkpointing; spike-driven event traversal; waveform-parallel fallback; global/grid peptide coupling; deterministic runs and residual checks. Two-level additive Schwarz with profile-triggered escalation is implemented and tested on synthetic graphs. |
| M9 | `observation.py`, `Stimulus`, `workflow.py` | Calcium current, fixed indicator filters, Hill saturation, reference-informed gains, within-type gain gauge and seeded noise. `fit_animal_nuisance` jointly fits per-animal log-drives and log-gains under the §6.3 gauge. Measured values are held fixed. On planted synthetic data, drives and gains are recovered up to the gauge. |
| M10 | `training.py`, `curriculum.py`, demo | Gaussian trajectory NLL, correlation/PSD/occupancy MMD, response energy distance, prior utilities, short-window multiple shooting and initial-state encoder, clipped AdamW, residual-feature regression, GGN Laplace samples. `run_curriculum` runs the §6.4 stages in order and stops at the first unmet exit condition. The default exit condition is a placeholder ("finite and not worse"), because §6.4 gives no numeric thresholds. `train_ensemble` and `calibrate_ensemble` cover K seeded members, coverage against the nominal `(K-1)/(K+1)`, and the §12.3 per-member Laplace fallback, and label output a sensitivity probe. Tested only on toy models; the curriculum has not run on animal data. |
| M11 | `experiments.py` | K2 → K3 → K1/K4 → ExM priorities, public availability requirement for mutants, cost/budget ranking, constrained directions, sensitivity-proxy labels until calibration. Candidate predictions must be generated by the caller's experimental forward model. |
| M12 | `evaluation.py`, `baselines.py`, `external_baselines.py`, `phase0.py`, `phase1_worm.py` | Class/animal/neuron/state/species splits; immutable real-data split definitions; noise ceilings; raw metrics/units, pre-registered normalization and paired class/neuron bootstrap gates; B0/B1/B2/B5 references, B4 ablation; K1/K4/gauge/sign utilities; calibration test. Phase 1 reports add training-convergence interpretation, gauge-invariant within-column metrics and assumption variants (Revision 12). `run_flyvis` (B6) and `run_shiu_b0` (fly-scale B0) write the `external_predictions` format with provenance. They have run only against stubs, because neither tool is installed. No animal-data acceptance result is claimed. |
| M13 | `body.py`, `worm_body.py`, `body_ladder.py`, `sustain.py` | Body protocol, explicit motor/sensor mappings, stopped body gradients and boundary labels. A reduced differentiable 2D worm body (50 segments, resistive-force drag; illustrative parameters) moves opposite to its travelling wave, and its rollout gradients match finite differences. Rungs L0–L5 are implemented; L5 talks to an external process and raises unless configured. `ClosedLoopRecord` enforces `M13-R2`. `sustain` covers both parts of `M13-R3`, and `climb_ladder` implements `M13-R4`. In closed-loop `simulate`, a body returning `BodyOutput` drives the global modulator concentrations c(t), which relax to its targets with `modulator_tau_s` (default 1 s) and enter the slow signaling; other bodies leave c constant. Limits: targets must match the compiled `ModulatoryState.concentrations` length; no BAAIWorm/Sibernetic backend; body parameters are not fitted. |
| Scale | `distributed.py`, `benchmark.py` | Boundary-only all-to-all halo exchange and distributed PCG via shard_map. Synapses are owned by the postsynaptic device, with per-device STP/receptor update and accumulation, and `distributed_step` matches single-device `step` on two virtual CPU devices. Synaptic state is gathered and scattered each step (no persistent sharded layout). Event mode is not distributed. Multi-host runs, large-connectome throughput and mouse-scale acceptance remain unverified. |

| Track I | `individual.py`, `restore.py`, `encoder.py`, `atanas.py` | I0 per-animal latent analysis (run; outcome in spec §10.4). I1 restore test with nested durable-state candidates, regeneration by relaxation or by history-based initial-state inference, and smallest-sufficient selection. Ridge decoders to the I0 latent and the `M13-R3` (ii) identity check. Atanas et al. raw loader reading `trace_array_original` with checksum verification. The loader has run only on synthetic HDF5. An I2 pipeline that assembles these pieces is not yet built. |

## Requirement-linked tests

Tests name the specification requirement IDs they exercise. The suite covers:

- M1-R1–R6: units/schema errors, foreign keys, unmapped genes, licensing,
  processing metadata, confidence masks and c302 discrepancy explanations.
- M2-R1/R2: exact known assignments and posterior sampling; unknown-label
  transport is exercised. M2-R3 has an explicit held-out-anchor helper, but no
  real held-out anchor dataset is included.
- M3-R1/R2: uncertain defaults and paired regression; uncertainty is available
  to experiment-design callers.
- M4-R1/R2/R3/R5/R6 and C-R2: budget rejection and compact fitting, permutation
  invariance, exact masks, unwired peptide effect and ligand-token independence.
- M5-R1/R2: family fallback, inflated covariance, prior deviations and Q10;
  channel probability conservation and differentiable kinetic parameters.
- M6-R1/R2/R3: absent fixed sign, CSR layout, geometric gates and chloride audit.
- M7-R1/R4: held-out surrogate fit and rollout rejection; non-default states
  use full dynamics. Envelope checking is implemented for runtime selection.
- M8-R1/R2/R4/R5 and S-R1/S-R4: matching sequential/event/parallel trajectories,
  repeatable seeds, dense/PCG agreement, finite-difference adjoint gradients,
  float32/float64 agreement, rule/kinetic-to-observation gradients, and Jaxley
  single-cell trajectory/gradient comparison.
- M9-R2: independent type gauges and preservation of measured nuisance gains.
- M11-R1/R2: priority/budget constraints, availability and sensitivity labels.
- M13-R1: closed-loop mapping and boundary-condition logging.
- C-R3 infrastructure: two virtual CPU devices, actual boundary exchange and
  distributed PCG equivalence. This is not evidence of multi-host scale.

Zarr, NWB, safetensors, canonical project and long-protein window round-trips
are tested. NWB tests skip only when the optional dependency is absent.

## Phase 0 and scientific gates

Phase 1 is in progress: see [phase1.md](phase1.md) and spec Section 10.3 (Revisions 9 and 12). The full compiled model is trained per fold on the frozen Phase 0 comparison, with assumption variants queued after the base folds. No Phase 1 result exists yet. Track I Stage I0 has run; its outcome is in spec Section 10.4 (Revision 13).

Phase 0 has been run on public worm data. The procedure is in [phase0.md](phase0.md), and the resulting design changes are in spec Section 10.2 (Revision 8). The numerical report inherits the `restricted` tier of its CeNGEN, Cook, Beets and WormBase inputs, so it stays local in `artifacts/phase0/`.

Status of the eight Phase 0 items:

1. **Sources.** Done. The data register holds CeNGEN, Cook 2019 (with the c302 cross-check), Randi wild-type and unc-31, Beets, Fenyves, Ripoll-Sánchez, WormBase, UniProt and ESM-2, each with a measured hash and license.
2. **Pre-registration.** Done. Stage A (class partition, splits, K1–K4 thresholds and rules) was frozen and committed before the analysis ran. Stage B (architecture, budget and metric targets) was derived from the Stage A rules. The gauge audit uses proxies: rab-3 promoter expression for indicator gain, and autoresponses for opsin drive.
3. **ESM-2 family recovery.** Run; it fails on nomenclature families. No curated kinetics library exists yet, and molecule priors still need measured sources.
4. **Jaxley.** Capability audit done; the in-house backend is selected. The single-cell trajectory and gradient comparison passes.
5. **B5.** Reimplemented as an anatomy-constrained linear response and compared qualitatively with Creamer et al. The published held-out-animal split cannot be reproduced from the pooled atlas.
6. **Partly done in Phase 1.** Raw Randi traces are ingested, and a curated kinetics library exists. A timestep check on the worm graph selected dt = 10 ms with one compartment; this deviates from the Section 12.6 test as written (spec 10.3). Ensemble training through the full curriculum is not run, because it costs too much on CPU.
7. **Partly done.** B0/B1/B2/B4/B5 were evaluated on held-out neurons and classes for detection, sign and amplitude, using the linear-response approximation. State conditions, latency and residual analysis need traces.
8. **Not done.** Ensemble coverage and the Laplace fallback are Phase 1.

M4-R1 now separates input rank diagnostics from nonlinear model capacity.
Phase 0 freezes an absolute parameter budget after the gauge audit and held-out
capacity ablations. Dense learned projections and compact frozen bases are both
supported; the compact option is deliberately conservative, with two tied head
coefficients. A rank fraction remains available only as an explicit legacy
heuristic. Unrestricted demo rules have `parameter_budget_frozen=false` and
cannot clear scientific acceptance.

Fly transfer, external flyvis/Shiu runs, real body adapters, zebrafish records,
full morphological reduction, fast surrogate execution and
multi-host simulation remain later delivery work. These omissions are not concealed
by the test suite or reported as completed phases.

## Validation receipt

On 2026-10-06, all **82 tests passed** (90.76 seconds) on an Apple M4 Pro CPU,
including NWB interchange and two virtual-device distributed solving. Ruff lint
and format checks passed. The CLI demo, canonical Parquet → YAML → simulation
→ Zarr reload workflow, synthetic 8/32/128-neuron benchmark, and wheel/source
build all passed. The demo's normalized training loss decreased from 0.18128 to
0.01100 over ten optimizer steps.

[validation.json](validation.json) records dependency versions and measured
first-call/steady-call timings. The backward benchmark differentiates external
current; rule-to-observation gradients are verified separately by tests and the
demo. Reported storage counts only static neuron/synapse/gap parameter arrays,
not complete state, trained rules or peak allocator memory. These CPU receipts
satisfy the initial reference exit; they do not establish animal-data acceptance.

## Reproduction

```sh
uv sync --locked --extra nwb
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run molc demo --training-steps 10
uv run molc benchmark --sizes 8 32 128 --repeats 10
```

The benchmark warms and synchronizes JAX calls, keeps numerical inputs dynamic,
reports first-call compilation separately, and reports actual solve iterations.
The graph family is synthetic and sparse; it does not establish performance on
the whole fly or mouse connectome. Default float32 runs and float64 references
remain distinct claims.
