"""M13 body protocol and explicit actuator/sensor mappings."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

BodyState = dict
SensoryInput = np.ndarray
MotorOutput = np.ndarray


@runtime_checkable
class BodyModel(Protocol):
    def reset(self, state0: BodyState) -> SensoryInput: ...
    def step(self, motor: MotorOutput, dt_s: float) -> SensoryInput: ...


@dataclass
class MappedBodyAdapter:
    """Wrap a biomechanical backend with explicit mappings to neuron currents.

    backend implements reset(state0) and step(actuator_values, dt_s), returning
    a sensor vector. External BAAIWorm/flygym wrappers must implement that
    contract and record their backend version and model provenance.
    """

    backend: object
    n_neurons: int
    motor_indices: tuple
    sensory_indices: tuple
    motor_gain: float = 1.0
    sensory_gain_pA: float = 1.0
    backend_provenance: dict | None = None

    def __post_init__(self):
        if self.n_neurons < 1 or any(
            i < 0 or i >= self.n_neurons
            for i in self.motor_indices + self.sensory_indices
        ):
            raise ValueError("invalid body neuron mapping")
        if not self.backend_provenance:
            raise ValueError("body adapter requires backend version and provenance")

    def _map_sensory(self, sensors):
        sensors = np.asarray(sensors)
        if sensors.shape != (len(self.sensory_indices),) or not np.all(
            np.isfinite(sensors)
        ):
            raise ValueError("body sensor vector differs from sensory mapping")
        result = np.zeros(self.n_neurons)
        result[list(self.sensory_indices)] = self.sensory_gain_pA * sensors
        return result

    def reset(self, state0):
        return self._map_sensory(self.backend.reset(state0))

    def step(self, motor, dt_s):
        motor = np.asarray(motor)
        if motor.shape != (self.n_neurons,) or dt_s <= 0:
            raise ValueError("invalid motor vector or timestep")
        actuators = self.motor_gain * np.tanh(
            (motor[list(self.motor_indices)] + 40) / 10
        )
        return self._map_sensory(self.backend.step(actuators, dt_s))


@dataclass
class SpringBody:
    """Synthetic damped body for testing closed-loop wiring only."""

    position: float = 0.0
    velocity: float = 0.0

    def reset(self, state0):
        self.position = state0.get("position", 0.0)
        self.velocity = 0.0
        return np.array([self.position])

    def step(self, motor, dt_s):
        acceleration = float(np.mean(motor)) - self.position - 0.5 * self.velocity
        self.velocity += dt_s * acceleration
        self.position += dt_s * self.velocity
        return np.array([self.position])
