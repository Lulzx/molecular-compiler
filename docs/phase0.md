# Phase 0 on public C. elegans data

This file describes how Phase 0 is run. Numerical results are not in the
repository: several inputs are `restricted`-tier (Section 12.8), so every
derived report inherits that tier and stays local, in `artifacts/phase0/`.

## Order of operations

1. **Register sources.** `data-register.json` lists every input with source,
   version, license and tier. Ingestion refuses unregistered data (M1-R5).
2. **Ingest.** `molecular-compiler worm-ingest --checkpoint-hash <sha256> --device mps`
   converts the cached sources into canonical Parquet tables under `data/worm/`
   (gitignored). It writes an ingestion report with source hashes, the c302
   cross-check (M1-R6), gene counts, dropped peptide pairs and every declared
   approximation.
3. **Freeze Stage A.** `molecular-compiler phase0-register` writes
   `configs/phase0-stage-a.json` with exclusive create. It holds the 118-class
   partition, the leave-class-out and leave-neuron-out folds, the K1–K4
   thresholds, metric and ceiling definitions, and the Stage B selection rules.
   It was committed before any analysis read the response atlas.
4. **Run.** `molecular-compiler phase0-run` executes every analysis against the
   frozen Stage A. It writes `artifacts/phase0/phase0-report.{json,md}` and
   `exploratory.json`, then freezes `configs/phase0-stage-b.json`
   (architecture, absolute parameter budget and metric targets).

## Sources

| Dataset ID | Content | Tier |
|---|---|---|
| `cook2019_hermaphrodite` | Cook et al. 2019 SI 5, corrected July 2020 | restricted |
| `cect_c302_reference` | ConnectomeToolbox reader cache, for M1-R6 | restricted |
| `cengen_taylor2021` | CeNGEN threshold-2 class TPM | restricted |
| `beets2023_peptide_gpcr` | Peptide–GPCR EC50 screen | restricted |
| `fenyves2020_polarity` | Transmitter identity, receptor polarity | open |
| `randi2023_signal_propagation` | Response atlas, wild type and unc-31 | open |
| `ripoll_sanchez2023_peptide_connectome` | Long-range peptide network | open |
| `wormbase_ws286` | Gene name and sequence-name map | restricted |
| `uniprot_celegans` | Protein sequences (release recorded) | open |
| `esm2_t33_650M_UR50D` | ESM-2 650M weights, SHA-256 recorded | open |

The cache layout is fixed by `worm_public.SOURCE_FILES`. The raw files come from
the `wormneuroatlas` 0.0.7.3 and `cect` 0.3.5 wheels on PyPI, the fair-esm
download server and the UniProt REST API. The wheels are unpacked for their data
only. The GPL-licensed `wormneuroatlas` code is not imported.

## Declared approximations

These are listed in every ingestion report:

- Synapse size is the count of EM serial sections.
- Per-synapse positions, path distances and diameters are unavailable, so
  there is no dendritic attenuation.
- Gap-junction section counts stand in for contact area.
- AWC left and right both use the mean of the AWC ON and OFF profiles.
- Expression is log1p of CeNGEN TPM.
- Ortholog groups are left unmapped.

Phase 0 response models are a steady-state linear response of the compiler
(`linear_response.py`). There are no kinetics or time course. Each prediction
is scaled by the stimulated neuron's own response, which is the Section 6.3
drive gauge.

## Analyses

- **K1.** Rank and participation ratio of X, with peptide rows accumulated
  implicitly.
- **K2.** Compiler vs B4 on held-out pairs with and without a direct wired
  connection. Also an unc-31 check: the wild-type-fitted model is scored on
  unc-31 pairs with peptide release on and off.
- **K3.** Chloride sweep over the policy prior for Glu and GABA edges.
  Anion-receptor edges whose E_Cl interval overlaps the rest range are
  ambiguous. The receptor mixture is reported as well.
- **K4.** Convex-hull and novel-gene diagnostics per fold, plus the training
  participation ratio.
- **Gauge audit.**
  - Indicator gain: a rab-3 expression proxy. GCaMP6s and the GUR-3/PRDX-2
    QF driver both use the rab-3 promoter.
  - Opsin drive: estimated from each stimulated neuron's own response.
  - Both are tested against expression PCs with permutation p-values.
- **PLM family recovery.** Leave-one-out 1-NN over gene families. No curated
  kinetics library exists yet, so the families come from gene nomenclature.
- **S-R4.** Jaxley interface audit (conditions 1–3), plus a float64
  single-cell comparison against the in-house loop.
- **Baselines.** Held-out comparisons of B0, B1, B2, B4, B5, the compiler and
  six capacity candidates. Each uses a paired cluster bootstrap with 2,000
  draws.
- **B5 check.** Compared qualitatively with Creamer, Leifer & Pillow
  (bioRxiv 10.1101/2024.09.22.614271). The atlas pools animals, so their
  held-out-animal split cannot be reproduced.
- **Peptide network check.** The derived peptide network is compared with the
  Ripoll-Sánchez long-range model.

Exploratory analyses are labeled `pre_registered: false` and never change a
gate.

Rerunning `phase0-run` against an existing `configs/phase0-stage-b.json` fails by design: Stage B is created once and cannot drift. Pass a new `--stage-b` path for a sensitivity re-run.
