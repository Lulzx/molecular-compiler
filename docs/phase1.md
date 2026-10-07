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
| Spatial and time resolution | 1 compartment, dt = 10 ms, float64 | Against dt = 5 ms, the worst relative error over responders was 12% at 10 ms and 134% at 20 ms. The Section 12.6 test as written passes on the untrained model (see the audit findings for the caveat). |
| Gating | one m and one h gate per channel, with time constants independent of voltage | single-gate generic HH form; Ca²⁺ and U-shaped gating dropped |
| Connectome units | 0.01 nS per chemical-synapse EM section; 0.05 nS per gap-junction section | Section counts are not conductances. At 1 nS per section, coupling swamped membrane conductance and the network sat near −25 mV. A scan found the network stable and quiescent (rest near −70 mV) up to about 0.015 nS per section, and non-stationary at 0.02 (35 mV drift over 10 s). 0.01 is on the stable side. |
| Pre-stimulus state | 20 s relaxation from rest, `stop_gradient` | Rest is a fixed point. The gradient through rest is ignored. |
| Stimulus | 0.5 s, 200 pA step | Opsin drive is unknown. The drive gauge below removes its scale. |
| Fluorescence | F = 0.1 + Hill(Ca) (n = 2), with no indicator kernel. Half-saturation is the initial model's median resting calcium | Calcium units are arbitrary. With a fixed 10⁻³ half-saturation, resting calcium (about 3×10⁻⁶) sat far below it, and no decrease could register. The value is computed from the model alone, never from data. |
| Chloride | basal [Cl⁻]ᵢ = 5 mM, the lower bound of the policy prior | Unmeasured. At 10 mM, E_Cl ≈ −62 mV lies above the −70 mV rest, so every anion synapse depolarizes; this is the K3 failure. The choice was made after a smoke run showed no negative predictions, before any fold was scored. |
| Drive gauge | Each simulated column is divided by its simulated self-response (floored at 0.01) and multiplied by the observed autoresponse, with one global gain | Same Section 6.3 gauge as Phase 0. The floor stops near-zero self-responses from producing very large predictions (seen in the smoke run). |
| Training | AdamW, cosine decay, 60 steps of 4 random stimulated columns (Revision 12; was 30) | Compute; see below |
| Training adequacy | 8 seeded training columns per fold, evaluated forward-only every 10 updates. Converged if the loss falls less than 2% over the last 20 updates | Revision 12. A fail with any unconverged fold is reported as `fail_undertraining_not_excluded` |
| Assumption variants | Retrained on the same folds with [Cl⁻]ᵢ = 3 mM (`cl3`), [Cl⁻]ᵢ = 8 mM (`cl8`), and Hill n = 1 (`hill1`) | Revision 12 sensitivity analysis; does not gate acceptance |

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

| Command | What it does | Output (restricted, local) |
|---|---|---|
| `molc traces-ingest` | Converts the OSF trace export into per-event response windows | `data/worm/responses/` |
| `molc phase1-audit` | Pre-training audit. Reads no targets. Reports the library, family recovery, K3, the resting state, and edges net-inhibitory at rest | `artifacts/phase1/audit.json` |
| `molc phase1-convergence` | The Section 12.6 test on the untrained model: halve dt, then separately add two compartments | `artifacts/phase1/convergence.json` |
| `molc phase1-convergence --trained F.json` | The same test with a base fold's trained parameters, run on every base fold (required before acceptance) | `artifacts/phase1/convergence-trained-<fold>.json` |
| `molc phase1-fold --fold F [--variant V]` | Trains one frozen fold, base or variant `cl3`/`cl8`/`hill1`. Checkpoints after every step and resumes from a matching checkpoint. Records the monitor loss and the training-convergence verdict | `artifacts/phase1/<split>[-V]-fold<F>.json`, `.ckpt` |
| `molc phase1-report` | Refits B0, B1 and B2 on the same folds and pools held-out predictions. Reports metrics, ceilings, paired cluster bootstrap intervals (2,000 draws), the Section 9.3 verdict and its interpretation under the training-adequacy rule, gauge-invariant metrics and the M10 residual analysis. Completed variants appear under `sensitivity` | `artifacts/phase1/phase1-report.json` |
| `molc heldout-animals` | Exploratory (`pre_registered: false`). Phase 0 linear-response models fit on training animals and scored on held-out animals | `artifacts/phase1/held-out-animals.json` |

`scripts/phase1-queue.sh` runs everything in order:
1. it waits for any running `molc individual-state`;
2. it trains the five base `leave_class_out` folds three at a time;
3. it runs the trained-model convergence test and writes the report;
4. it trains the 15 variant folds and rewrites the report.

Logs go to `artifacts/phase1/logs/`. Rerunning the script resumes: an
interrupted fold continues from its checkpoint, and a finished fold is
retrained from its final checkpoint without new updates. A checkpoint written
with different steps, batch, seed, learning rate or monitor columns is
refused.

### Gauge-invariant metrics

The drive gauge scales each simulated column by the measured autoresponse.
Three metrics are unchanged by any positive per-column scale:
- detection AUROC within each stimulated column, averaged over columns;
- amplitude correlation within each column, averaged over columns;
- sign accuracy.

The report gives all three for the compiler and B0–B2, with paired cluster
bootstrap. A molecular advantage that appears only in the pooled metrics
depends on the gauge.

### Residual analysis

Residuals are off in Phase 1, so the M10 analysis regresses held-out errors
(observed − compiler) on per-pair features the model does not use:

- `peptide_coupling`: released peptide × matched receptor × potency. Peptides
  are off (K2).
- `no_direct_connection`: neither a chemical synapse nor a gap junction joins
  the pair.
- `log_autoresponse`: a trend here means the drive gauge is inadequate.

A feature is a candidate addition when its Bonferroni-corrected p-value is
below 0.05.

## Trace ingestion

The traces are not needed for the pre-registered comparison. They are needed
for the audit and the exploratory analyses:

```bash
uv run molc traces-ingest
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

## Pre-training audit findings

- **Family recovery (spec 10.2).**
  - The pooled leave-one-out rate passes the 90% threshold. GPCRs dominate
    that rate, and they recover almost perfectly.
  - Ligand-gated receptors are below 90%, at about 85%, and transporters are
    about 71%.
  - The anion-receptor families that decide sign are weakest: GABA_anion
    about 50%, GluCl_anion about 67%, ACh_anion about 83%.
  - So PLM neighbors must not replace curated labels for receptors (M5-R1
    keeps labels first).
- **Chloride bug, fixed.**
  - The "neutral" non-chloride transporters were given a 10 mM chloride
    equilibrium. That was neutral only while basal [Cl⁻]ᵢ was 10 mM. Once
    basal was set to 5 mM, they pulled [Cl⁻]ᵢ back up.
  - `build_worm_library` now sets them to the basal value it is given.
  - Edges that are net-inhibitory at rest rose from under 1% to about 9%.
  - The first fold launch used the bug, so its runs were discarded (logs in
    `artifacts/phase1/obsolete/`).

- **Section 12.6 convergence.**
  - `molc phase1-convergence` passed both refinements (halved dt;
    n_comp + 2) on 24 seeded columns of the untrained model.
  - The untrained model's metrics are near chance, so this is weak evidence.
    Two extra compartments changed individual predictions substantially
    while leaving the metrics nearly unchanged.
  - Repeat the test on a trained fold before any acceptance claim.

## Run state (2026-10-07)

- No Phase 1 fold has completed. The first launch was discarded because of
  the chloride bug above.
- Revision 12 (training adequacy, variants, gauge-invariant metrics) was
  fixed before any fold was scored.
- `scripts/phase1-queue.sh` was launched to start once the Track I0 run
  finishes.
- Estimated wall time with three folds in parallel and about 300 s per step:
  - base folds: about 11 h (two waves of 60 steps);
  - variants: about 28 h more.
- `leave_neuron_out` folds are not scheduled.
