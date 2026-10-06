# Phase 1 on C. elegans

Phase 1 runs the full compiler (M4–M9 compiled and simulated, not the Phase 0
linear response) on the Randi et al. 2023 perturbation atlas. The acceptance
test is Section 9.3: on `leave_class_out`, with residuals off and the frozen
pre-registration, the compiler must beat B0, B1 and B2.

Results are not in the repository. The inputs include `restricted`-tier
sources, so every derived artifact is restricted and stays in
`artifacts/phase1/`.

## What is frozen

Everything that defines the comparison comes from Phase 0 and is unchanged:

- targets, labels and noise ceilings (Stage A): trial-averaged ΔF/F0 from the
  atlas, `q < 0.05`, split-half ceilings;
- the five `leave_class_out` and `leave_neuron_out` folds (Stage A);
- the absolute rule budget of 13 parameters and the k = 2 PLM projection
  (Stage B);
- the baselines B0, B1 and B2, which are refit per fold exactly as in
  Phase 0.

## Inputs added in Phase 1

- **Trace ingestion** (`randi_traces.py`). The OSF export (`randi2023_traces`)
  is converted into per-event, per-neuron response windows with label
  confidences. Events are flagged when they match the atlas's inclusion rule,
  and trials with |ΔF/F0| ≥ 5 are flagged as outliers. After those flags, the
  per-pair means reproduce the atlas (r = 0.74–0.82). These traces are used for
  audit and for the exploratory held-out-animal analysis. The pre-registered
  targets remain the atlas values.
- **Curated kinetics** (`worm_kinetics.py`, `phase1_kinetics_curation`).
  - There is one record per modeled gene, assigned by curated functional
    family (spec 10.2).
  - Channel gating, conductances and passive properties come from the
    Nicoletti et al. 2019 and 2024 C. elegans neuron models. Receptor decay
    constants come from C. elegans NMJ and AVA recordings.
  - Families without a C. elegans measurement are labeled `UNMEASURED` and
    get 16× prior covariance. These include K2P, BK and SK gating, the
    sensory channels, most anion receptors, GPCRs and the neuronal [Cl⁻]ᵢ.

## Declared approximations

| Item | Choice | Reason |
|---|---|---|
| Spatial and time resolution | 1 compartment, dt = 10 ms, float64 | Section 12.6 convergence on this graph. Against dt = 5 ms, the worst relative error over responders was 12% at 10 ms and 134% at 20 ms. |
| Gating | one m and one h gate per channel, with time constants independent of voltage | single-gate generic HH form; Ca²⁺ and U-shaped gating dropped |
| Connectome units | 0.01 nS per chemical-synapse EM section; 0.05 nS per gap-junction section | Section counts are not conductances. At 1 nS per section, coupling swamped membrane conductance and the network sat near −25 mV. A scan found the network stable and quiescent (rest near −70 mV) up to about 0.015 nS per section, and non-stationary at 0.02 (35 mV drift over 10 s). 0.01 is on the stable side. |
| Pre-stimulus state | 20 s relaxation from rest, `stop_gradient` | Rest is a fixed point. The gradient through rest is ignored. |
| Stimulus | 0.5 s, 200 pA step | Opsin drive is unknown. The drive gauge below removes its scale. |
| Fluorescence | F = 0.1 + Hill(Ca) (n = 2), with no indicator kernel. Half-saturation is the initial model's median resting calcium | Calcium units are arbitrary. With a fixed 10⁻³ half-saturation, resting calcium (about 3×10⁻⁶) sat far below it, and no decrease could register. The value is computed from the model alone, never from data. |
| Chloride | basal [Cl⁻]ᵢ = 5 mM, the lower bound of the policy prior | Unmeasured. At 10 mM, E_Cl ≈ −62 mV lies above the −70 mV rest, so every anion synapse depolarizes; this is the K3 failure. The choice was made after a smoke run showed no negative predictions, before any fold was scored. |
| Drive gauge | Each simulated column is divided by its simulated self-response (floored at 0.01) and multiplied by the observed autoresponse, with one global gain | Same Section 6.3 gauge as Phase 0. The floor stops near-zero self-responses from producing very large predictions (seen in the smoke run). |
| Training | AdamW, cosine decay, random batches of stimulated columns, few steps | Compute; see below |

## Status at launch

At initialization the simulator predicts almost no negative responses, even
with these settings. Inhibition has to come from the trained rule, which sets
the density of anion versus cation receptors through the k = 2 PLM tokens.
The sign metric is therefore the most likely place for the compiler to lose
to B0.

## Compute

- On this machine (12-core CPU, no usable accelerator for JAX), one simulated
  stimulation (20 s rest plus 30.5 s response at dt = 10 ms) takes about 2 s
  forward.
- A reverse-mode gradient takes roughly 10–20× longer.
- A fold therefore trains for tens of steps, not the full M10 curriculum. A
  4-column batch takes about 2–3 minutes per step.
- The ensembles, multiple shooting and the trajectory and statistic losses of
  M10 are not run. Phase 1 here tests the perturbation pathway only.

## Commands

```bash
molc phase1-fold --fold 0 --steps 30 --batch 4
```

Run all five `leave_class_out` folds, then:

```bash
molc phase1-report
```

`phase1-report` refits B0, B1 and B2 on the same folds and pools the compiler's
held-out predictions. It then reports metrics, ceilings, paired cluster
bootstrap intervals (2,000 draws) and the Section 9.3 verdict.

## Trace ingestion

The traces are not needed for the pre-registered comparison. They are needed
for the audit and the exploratory analyses:

```bash
uv run python -c "from molecular_compiler.randi_traces import ingest; from molecular_compiler.provenance import DataRegister; ingest('data/cache', 'data/worm', DataRegister.load('data-register.json'))"
```

This writes `data/worm/responses/{wt,unc31}.npz` and an ingestion report.
The OSF archives are expected, already extracted, under
`data/cache/randi/exported_data` and `exported_data_unc31`. Their SHA-256
values are in `data-register.json`.

## Memory

- A reverse-mode gradient through a 3,050-step response window would store
  the full simulator state at every step. The per-edge synaptic state alone is
  3,709 edges × 97 receptors in float64.
- With per-step checkpointing only, a 4-column batch needed about 45 GB per
  process.
- The driver therefore checkpoints at two levels: segments of 50 steps, then
  steps inside each segment. Peak memory is about 4 GB per process. A 4-column
  gradient step takes about 140 s alone, or about 250–350 s with three folds
  sharing the CPU. The first step includes about 5 minutes of compilation.

## Run state (2026-10-06)

- The five `leave_class_out` folds were launched three at a time (folds 0–2,
  then 3–4) with `--steps 30 --batch 4`. Logs are in
  `artifacts/phase1/logs/`.
- Work was paused after the first training step of folds 0–2. The processes
  were suspended with SIGSTOP, not killed, so `kill -CONT` on their PIDs
  resumes them where they stopped.
- If they are lost (reboot, app quit, kill), rerun the missing folds from the
  start. The driver writes a fold's JSON only at the end and keeps no
  mid-fold checkpoint.
- Remaining: about 5 hours of compute for all five folds, then
  `molc phase1-report`. `leave_neuron_out` folds are not scheduled.
- Not yet done:
  - the exploratory re-run of Phase 0 on the outlier-cleaned, trial-level data
    with a held-out-animal split;
  - the M10 residual analysis;
  - the 90% family-recovery check on the curated library (spec 10.2).
