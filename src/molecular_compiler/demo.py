"""Run a synthetic end-to-end compiler and train the rules through the simulator."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from molecular_compiler import ObservationModel, Stimulus, compile, observe, simulate
from molecular_compiler.fixtures import synthetic_system
from molecular_compiler.training import train


def run(output="artifacts/demo", training_steps=5):
    graph, rules, kinetics = synthetic_system()
    resolution = replace(
        __import__("molecular_compiler").ResolutionPolicy.default(), n_comp=1
    )
    stimulus = Stimulus(pulses=((0, 0.001, 0.006, 5.0),))
    observation = ObservationModel(half_saturation=1e-6)
    sim = compile(graph, rules, kinetics, resolution=resolution)
    trajectory = simulate(sim, stimulus, 0.01)
    recording = observe(trajectory, observation)
    target_params = jax.tree.map(lambda x: x + 0.1, rules.params)
    target = observe(
        simulate(
            compile(
                graph, rules.with_params(target_params), kinetics, resolution=resolution
            ),
            stimulus,
            0.01,
        ),
        observation,
    ).y

    def loss(params, _):
        predicted = observe(
            simulate(
                compile(
                    graph, rules.with_params(params), kinetics, resolution=resolution
                ),
                stimulus,
                0.01,
            ),
            observation,
        ).y
        return jnp.mean((predicted - target) ** 2) / jnp.maximum(
            jnp.mean(target**2), 1e-12
        )

    initial_loss = float(loss(rules.params, 0))
    updated, history = train(
        rules.params, loss, steps=training_steps, learning_rate=0.01, weight_decay=0
    )
    final_loss = float(loss(updated, 0))
    result = {
        "data_kind": "synthetic",
        "n_neurons": sim.n_neurons,
        "n_synapses": len(sim.syn_pre_idx),
        "parameter_count": rules.parameter_count,
        "duration_s": 0.01,
        "initial_training_loss": initial_loss,
        "final_training_loss": final_loss,
        "history": history,
        "metadata": trajectory.metadata,
        "voltage_range_mV": [
            float(trajectory.voltage.min()),
            float(trajectory.voltage.max()),
        ],
        "scientific_acceptance": False,
    }
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        path / "trajectory.npz",
        t=trajectory.t,
        voltage=trajectory.voltage,
        calcium=trajectory.calcium,
        fluorescence=recording.y,
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
