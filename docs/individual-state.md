# Track I: durable individual state

Track I (spec Section 10.4) asks two questions:

1. What compact per-animal state predicts an individual's own later
   responses, and does it persist?
2. What is the smallest virtual body that keeps an emulated individual
   within its own state envelope?

Stage I0 addresses the first question on recorded data. Stages I1, the
restore test in the simulator, and I2, the body ladder, wait for a trained
Phase 1 model with dynamic slow state.

The Stage I0 design, gates and decision rules were committed in spec
Revision 10 before any code read per-animal response deviations.
Revision 11 replaced I0-3 before any I0 output existed. Under Revision 10, an
interval containing zero counted as durable, which only shows that drift was
not detected. The first run was stopped before it wrote results, and its log
is in `artifacts/individual/obsolete/`. The pooled
metrics of `molc heldout-animals` had been seen. The analysis code was
committed before its first run on real data. Results are restricted and stay
in `artifacts/individual/`.

## Command

```bash
uv run molc individual-state   # writes artifacts/individual/individual-state.json
```

It needs the ingested traces (`molc traces-ingest`; see
[phase1.md](phase1.md)). The full nested selection takes about 40 minutes on
a 12-core CPU. Each run also writes per-trial squared errors to
`individual-errors.npz` so the gates can be recomputed without refitting.

## What it does

- **Data.** Wild-type Randi et al. traces: atlas-included, non-outlier trials
  with label confidence ≥ 0.95, using the per-trial mean ΔF/F0 over 30 s.
  Self-responses are excluded. Event ids increase with stimulation time in
  every animal. In the one export file whose volume list is not increasing,
  the extra entry falls outside the recording window and never becomes an
  event.
- **Folds.** The five `leave_animal_out` folds of `molc heldout-animals`
  (same seed). Animals with at least 6 events are scored (96 of 109).
- **Within-animal splits.**
  - `chronological`: fit on the first ⌊n/2⌋ events, score the rest.
  - `interleaved`: fit on even-position events, score odd ones.
- **Models.** Each trial of pair p in animal a is predicted as P_p plus a
  deviation:

  | Model | Deviation | Latent size |
  |---|---|---|
  | `pop` | none | 0 |
  | `gain` | (u − 1)·P_p, ridge toward u = 1 | 1 |
  | `factor-r` | V_p·u_a; V from ridge ALS on training animals, u_a from ridge on the fit half | r ∈ {1, 2, 4, 8, 16} |
  | `swap-r` | as `factor-r`, scored with another held-out animal's u (seeded derangement within the fold) | r |

  P is the training-animal pair mean shrunk by κ. Pairs never seen in
  training get P = 0 and V = 0.
- **Selection.** For every outer fold and model, κ, λ_V and λ_u are chosen
  from {0.1, 1, 10} by the chronological protocol on an inner 4-fold
  leave-animal-out split of the training animals. The criterion is the
  summed squared error on the score halves.
- **Metric.** The gain over `pop` is G = 1 − ΣSE_model / ΣSE_pop on the score
  halves, pooled over animals. Intervals come from 2,000 bootstrap draws that
  resample animals. The ceiling is 1 − σ²_trial / MSE_pop, where σ²_trial is
  the pooled within-animal, within-pair trial variance.

## Gates

r\* is the smallest rank whose chronological G is within one bootstrap SE of
the best rank's G.

| Gate | Pass | Interpretation |
|---|---|---|
| I0-1 | G(factor-r\*) > 0 **and** G(factor-r\*) − G(swap-r\*) > 0 (95% intervals) | A compact state carries to unseen pairs and belongs to that animal |
| I0-2 | G(factor-r\*) − G(gain) > 0 | More than one global scale. A scalar gain alone is not counted as biological evidence (indicator-gain confound, spec 10.2) |
| I0-3 | Persistence ratio R = G_chron / G_interleaved, with animals bootstrapped jointly. `durable`: the 95% interval of R is above 0.75 and at most 2.5% of draws have G_interleaved ≤ 0. `drifts`: the interval of R is below 1, or the interval of G_chron − G_interleaved is below 0. Otherwise `inconclusive` | Persistence over one recording session only (median about 32 min). Not detecting drift is not evidence of persistence |

**Covariate audit.** Each latent (`gain` and `factor-r*`, fit on the early
half) is regressed on:
- confident trace count;
- recording duration;
- recording date;
- strain-folder batch.

An in-sample R² above 0.5 marks the latent as measurement state. The spec
says "mostly predicted"; the 0.5 cut-off is this document's reading of that
phrase, fixed before the run. A permutation p-value (1,000 shuffles) is
reported alongside.

## Synthetic checks

`tests/test_individual.py` plants individual latents in synthetic trials.
- A planted rank-2 latent passes I0-1 and I0-2.
- With no latent, I0-1 fails.
- A latent that flips sign halfway through the recording is classified by
  I0-3 as `drifts`.
- A latent that shrinks to 40% halfway through is not classified as
  `durable`.
- r\* can land one rank above the planted rank. With a fixed ridge, the extra
  dimensions recover some of the shrinkage, so the 1-SE rule's r\* is an
  upper bound on the knee, not the knee itself.

## Results

Pending: the first run is in progress.

## Stage I2 command (`molc track-i2`)

```bash
uv run molc track-i2 --trained artifacts/phase1/<fold>.json \
    --recordings <atanas dir> --labels <neuropal labels.json> \
    [--manifest <receipt.json>] [--decoder decoder.npz] \
    [--variant base] [--horizons 60 300 full] [--output artifacts/track-i/i2-report.json]
```

Implemented in `minimum_body.py`; not yet run on real data. The simulator is
rebuilt from a trained Phase 1 fold result (`--variant` must match the fold's).
Recordings are read from `gcamp/trace_array_original` and aligned to the
simulator on canonical neuron names present in every animal.

- **Part (i)** is `sustain.spontaneous_activity_check` against all recorded
  animals that cover H, on dF/F0 with F0 the 10th percentile over [0, H].
- **Part (ii)** is `not_evaluable` unless `--decoder` supplies a decoder and
  per-animal latents (npz: `weights`, `bias`, `mean`, `neurons`, `animals`,
  `latents`; the readout acts on per-neuron window means). Stage I0 found no
  usable latent (spec Revision 13), so a default run reports it as such. It
  is never counted as a pass or a fail.
- **Verdicts:** `sustains_this_worm`, `sustains_a_worm`,
  `sustains_a_worm_identity_not_evaluable`, `fails`. The report gives, per
  animal and H, the lowest rung that sustains *a* worm (part i) and, when
  part (ii) is evaluable, the lowest that sustains *this* worm, each with its
  M13-R4 body-model defects. Every rung up to and beyond the first pass is
  recorded with its M13-R2 fields (boundary condition, rung, H).
- **Declared approximations** (also listed in the report): L1 is one
  population-fitted per-neuron current found by damped diagonal-Jacobian
  relaxation on mean recorded dF/F0; L2 maps head angle to SMDD/SMDV and its
  absolute value to DVA (illustrative, not anatomical); L3 uses the reduced
  body's illustrative territories; L4 is `unavailable` because the simulator
  has no modulator-target input (a stand-in linear map to tonic current can be
  supplied); L5 is `unavailable` unless a body factory is configured. No
  per-animal parameter enters the emulation, so L0, L1 and L3 runs are shared
  by all animals.
- Closed-loop rungs step the simulator in Python, so full-length L3 and L4
  runs at the Phase 1 timestep are slow.
- Output carries `data_tier: restricted`. Tested only on a tiny synthetic
  network and synthetic HDF5 (`tests/test_minimum_body.py`).
