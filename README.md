# molecular-compiler

A research framework for compiling connectomes and molecular profiles into
executable nervous-system models.

The compiler takes the wiring of a nervous system (a connectome) and the
molecular makeup of its neurons (gene expression, protein measurements, ion
channel kinetics) and produces a differentiable simulation. The goal is a
simulation whose spontaneous activity, stimulus responses and responses to
single-neuron stimulation match the real animal, without fitting each animal
separately.

What is learned is a set of rules that map molecules to parameters: how a
given channel, receptor or gap junction protein becomes a conductance. The
same rules apply to every neuron, every animal and, if the approach works,
every species.

There is no code yet. The design is in [spec.md](spec.md).

## Pipeline

```
connectome ─┐
expression ─┼─> prepare ──> compile ──> run ──> observe ──> losses
proteins   ─┘   (M1-M3)     (M4-M7)     (M8)    (M9)        (M10)
                             ^   ^                            │
kinetics ────────────────────┘   └──────── gradients ─────────┘
```

- prepare: load and validate the data, then assign each neuron a
  probability distribution over molecular identity.
- compile: apply the learned rules to get channel densities, receptor
  densities, gap junctions and neuropeptide coupling, then reduce each neuron
  type to a fast surrogate model. Types that no surrogate fits run the full
  model.
- run: integrate the network. Voltages coupled by gap junctions are solved
  as one sparse linear system each step.
- observe: convert simulated calcium into predicted fluorescence.
- M11 ranks the next experiments to run. M12 scores the model on held-out
  data. M13 couples the simulation to a body model.

## Targets

| | System | Neurons | Synapses |
|:-:|---|--:|--:|
| 🪱 | C. elegans | 302 | thousands |
| 🪰 | Drosophila optic lobe | ~53,000 | millions |
| 🪰 | Drosophila whole brain | 139,255 | ~50 million |
| 🐟 | Larval zebrafish | ~100,000 | unknown |
| 🐭 | Mouse | ~70 million | ~10^11 |

The worm is used for training and validation. The fly is the first transfer
target.

## Evaluation

Results are reported on held-out neurons, neuron classes, internal states and
species, with all per-animal correction terms switched off. The compiler has
to beat three baselines on held-out neuron classes:

- B0: connectome only, with weights from synapse counts and sign from the
  predicted neurotransmitter.
- B1: the same inputs fed to an unstructured network.
- B2: a model trained on the recordings with no anatomy.

Three more baselines are always reported but don't decide acceptance: the
compiler without neuropeptide signaling, a connectome-constrained model fit
to activity, and, for the fly optic lobe, a task-trained connectome model.

Before any of this, a feasibility phase tests whether the approach can work
at all. It checks how many independent directions the molecular data
contains, whether neuropeptide signaling outside synapses is needed, how many
synapse signs depend on poorly measured chloride levels, and whether the
held-out classes are actually new or just interpolations.

## Related work

The closest existing project is flyvis, which fits a fly visual system model
on top of the connectome but learns parameters per cell type and uses no
molecular data. Jaxley, a differentiable multicompartment simulator in JAX,
will run the full neuron models and serve as the numerical reference. As far
as we found, no existing model learns per-molecule rules that transfer across
species, simulates neuropeptide signaling that doesn't follow the wiring, or
lets synapse sign emerge from reversal potentials. Section 1.5 of the spec
has the full comparison.

## Stack

Python and JAX. Rules, simulator and training form one differentiable
program. Native kernels will be added only where profiling at fly scale
shows they are needed.

## Design decisions

Section 12 of the spec records the main choices, with the reason for each
and the result that would reopen it:

- Genes are represented by ESM-2 650M protein embeddings, computed once
  before training.
- Each neuron type gets a simple fixed ODE surrogate if one fits, a small
  neural ODE if not, and the full model otherwise.
- Uncertainty comes from an ensemble of 5 models. It counts as calibrated
  only after a coverage test on held-out data passes; until then it is
  reported as a sensitivity probe.
- At fly scale, the voltage solve uses conjugate gradient with one
  preconditioner block per neuron.
- In the worm, neuropeptide coupling has no distance limit. In the fly and
  larger brains, it falls off with a learned decay length.
- The default timestep is 0.5 ms for the worm, 0.1 ms for the fly and
  0.05 ms for zebrafish, each confirmed by a convergence test.
- The rule network sees each neuron's molecules only, not its neighbors in
  the connectome.
- Every dataset is entered in a data register with its license before use.
  Published models are trained only on data that can be redistributed.

## Status

Specification only (revision 5). See [spec.md](spec.md) for modules, data
schemas, training, evaluation, phases and references.

## License

Apache 2.0. See [LICENSE](LICENSE).
