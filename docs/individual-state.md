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
a 12-core CPU when the machine is otherwise idle. Each run also writes per-trial squared errors to
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

## Results (2026-10-07)

The numerical report is restricted and stays in `artifacts/individual/`. The
outcomes are recorded in spec §10.4 (Revision 13):

- **I0-1 fails.** No factor rank beats the population model or the swap
  control. Trial noise leaves room for about a third of the population
  model's error to be explained, and the latent explains essentially none of
  it.
- **I0-2 fails.** The scalar gain beats every factor model. The gain is small
  but reliably positive, the same under both splits, and not predicted by
  recording covariates. It cannot be separated from indicator expression, so
  it does not count as biological evidence.
- **I0-3 is inconclusive.** There is no factor state to test for persistence.
- **Grid edge.** Inner selection always chose the largest κ and usually the
  largest ridge values. More shrinkage moves factor gains toward zero, so the
  conclusion holds.

What this does and does not show: about 14% of pairs repeat within an animal.
A state that is specific to individual pairs, and not shared through a basis,
would be invisible to this design. The result is "not detected", not
"absent". The next candidate source is spontaneous activity in the Atanas et
al. recordings, which needs its own pre-registered stage.

The run took about 40 minutes.
