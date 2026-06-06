"""Small variational quantum policies with NumPy and optional Qiskit backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Backend = Literal["numpy", "qiskit"]

N_QUBITS = 2
N_LAYERS = 3
N_FEATURES = N_QUBITS * N_LAYERS
PARAMETERS_PER_LAYER = N_QUBITS * 2
PARAMETER_COUNT = N_LAYERS * PARAMETERS_PER_LAYER + N_QUBITS


def _ry(angle: float) -> NDArray[np.complex128]:
    half = angle / 2.0
    return np.array(
        [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )


def _rz(angle: float) -> NDArray[np.complex128]:
    half = angle / 2.0
    return np.array(
        [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]],
        dtype=np.complex128,
    )


def _apply_single_qubit(
    state: NDArray[np.complex128], gate: NDArray[np.complex128], qubit: int
) -> None:
    bit = 1 << qubit
    for low_index in range(state.size):
        if low_index & bit:
            continue
        high_index = low_index | bit
        low = state[low_index]
        high = state[high_index]
        state[low_index] = gate[0, 0] * low + gate[0, 1] * high
        state[high_index] = gate[1, 0] * low + gate[1, 1] * high


def _apply_cnot(state: NDArray[np.complex128], control: int, target: int) -> None:
    control_bit = 1 << control
    target_bit = 1 << target
    for index in range(state.size):
        if index & control_bit and not index & target_bit:
            paired_index = index | target_bit
            state[index], state[paired_index] = state[paired_index], state[index]


def _z_expectation(state: NDArray[np.complex128], qubit: int) -> float:
    probabilities = np.abs(state) ** 2
    signs = np.array(
        [1.0 if not index & (1 << qubit) else -1.0 for index in range(state.size)]
    )
    return float(np.dot(probabilities, signs))


@dataclass(slots=True)
class VariationalQuantumPolicy:
    """A shallow data re-uploading circuit used as a two-axis movement policy."""

    parameters: FloatArray
    backend: Backend = "numpy"

    def __post_init__(self) -> None:
        self.parameters = np.asarray(self.parameters, dtype=np.float64)
        if self.parameters.shape != (PARAMETER_COUNT,):
            raise ValueError(
                f"Expected {PARAMETER_COUNT} parameters, got {self.parameters.shape}"
            )
        if self.backend not in ("numpy", "qiskit"):
            raise ValueError(f"Unsupported backend: {self.backend}")

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        *,
        scale: float = 0.35,
        backend: Backend = "numpy",
    ) -> "VariationalQuantumPolicy":
        parameters = rng.normal(0.0, scale, size=PARAMETER_COUNT)
        parameters[-N_QUBITS:] = 0.0
        return cls(parameters, backend)

    def act(self, observations: FloatArray) -> FloatArray:
        features = np.asarray(observations, dtype=np.float64)
        if features.shape != (N_FEATURES,):
            raise ValueError(f"Expected {N_FEATURES} features, got {features.shape}")
        features = np.clip(features, -1.0, 1.0)

        if self.backend == "qiskit":
            expectations = self._qiskit_expectations(features)
        else:
            expectations = self._numpy_expectations(features)

        biases = self.parameters[-N_QUBITS:]
        return np.tanh(1.5 * expectations + biases)

    def _numpy_expectations(self, features: FloatArray) -> FloatArray:
        state = np.zeros(2**N_QUBITS, dtype=np.complex128)
        state[0] = 1.0
        rotations = self.parameters[:-N_QUBITS].reshape(N_LAYERS, N_QUBITS, 2)

        for layer in range(N_LAYERS):
            for qubit in range(N_QUBITS):
                feature = features[layer * N_QUBITS + qubit]
                _apply_single_qubit(state, _ry(np.pi * feature), qubit)
                _apply_single_qubit(state, _ry(rotations[layer, qubit, 0]), qubit)
                _apply_single_qubit(state, _rz(rotations[layer, qubit, 1]), qubit)
            if layer % 2 == 0:
                _apply_cnot(state, 0, 1)
            else:
                _apply_cnot(state, 1, 0)

        return np.array(
            [_z_expectation(state, qubit) for qubit in range(N_QUBITS)],
            dtype=np.float64,
        )

    def _qiskit_expectations(self, features: FloatArray) -> FloatArray:
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp, Statevector
        except ImportError as exc:
            raise RuntimeError(
                "The Qiskit backend requires `pip install 'quantumbot[quantum]'`."
            ) from exc

        circuit = QuantumCircuit(N_QUBITS)
        rotations = self.parameters[:-N_QUBITS].reshape(N_LAYERS, N_QUBITS, 2)
        for layer in range(N_LAYERS):
            for qubit in range(N_QUBITS):
                feature = features[layer * N_QUBITS + qubit]
                circuit.ry(np.pi * feature, qubit)
                circuit.ry(rotations[layer, qubit, 0], qubit)
                circuit.rz(rotations[layer, qubit, 1], qubit)
            if layer % 2 == 0:
                circuit.cx(0, 1)
            else:
                circuit.cx(1, 0)

        state = Statevector.from_instruction(circuit)
        observables = (SparsePauliOp("IZ"), SparsePauliOp("ZI"))
        return np.array(
            [float(np.real(state.expectation_value(op))) for op in observables],
            dtype=np.float64,
        )
