"""Independent qutrit density-matrix oracle for device-neutral circuits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from general_noise_conformance import ExpectedDistribution
from numpy.typing import NDArray

type Matrix = NDArray[np.complex128]
type QutritState = tuple[int, ...]

LEAKED = 2
NUMERICAL_PROBABILITY_TOLERANCE = 1e-12


def _roundoff_clipped_probability(value: float, *, context: str) -> float:
    """Clip floating-point residue while rejecting material nonphysical values."""
    tolerance = NUMERICAL_PROBABILITY_TOLERANCE
    if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
        message = f"{context} is outside the physical probability range: {value}"
        raise ValueError(message)
    return min(1.0, max(0.0, value))


def rx(theta: float) -> Matrix:
    """Return the standard one-qubit X rotation."""
    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=np.complex128)


def ry(theta: float) -> Matrix:
    """Return the standard one-qubit Y rotation."""
    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def rz(theta: float) -> Matrix:
    """Return the standard one-qubit Z rotation."""
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def hadamard() -> Matrix:
    """Return the standard one-qubit Hadamard unitary."""
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


def controlled_x() -> Matrix:
    """Return a controlled-X unitary ordered as control then target."""
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.complex128,
    )


PAULIS: dict[str, Matrix] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


@dataclass(frozen=True)
class QutritNoise:
    """Independent parameters for preparation, one-qubit, and readout channels."""

    preparation_probability: float = 0.0
    preparation_leakage_ratio: float = 0.0
    p1: float = 0.0
    p1_pauli_model: tuple[tuple[str, float], ...] = (("X", 1 / 3), ("Y", 1 / 3), ("Z", 1 / 3))
    p1_emission_ratio: float = 0.0
    p1_emission_model: tuple[tuple[str, float], ...] = (("L", 1.0),)
    p1_seepage_probability: float = 0.0
    p2: float = 0.0
    p2_pauli_model: tuple[tuple[str, float], ...] = tuple(
        (first + second, 1.0 / 15.0) for first, second in product("IXYZ", repeat=2) if first + second != "II"
    )
    p2_emission_ratio: float = 0.0
    p2_emission_model: tuple[tuple[str, float], ...] = (("LL", 1.0),)
    p2_seepage_probability: float = 0.0
    measurement_0_to_1: float = 0.0
    measurement_1_to_0: float = 0.0


class QutritReference:
    """Exact density-matrix evolution over ``|0>``, ``|1>``, and ``|L>``."""

    def __init__(self, n_qubits: int, noise: QutritNoise | None = None) -> None:
        if n_qubits <= 0:
            message = "a reference circuit must contain at least one qubit"
            raise ValueError(message)
        self.n_qubits = n_qubits
        self.noise = noise or QutritNoise()
        self._states = tuple(product(range(3), repeat=n_qubits))
        self._indices = {state: index for index, state in enumerate(self._states)}
        dimension = 3**n_qubits
        self.rho = np.zeros((dimension, dimension), dtype=np.complex128)
        self.rho[0, 0] = 1.0

    def _full_unitary(self, local: Matrix, qubits: tuple[int, ...]) -> Matrix:
        """Embed a computational-space unitary, acting as identity on leaked inputs."""
        dimension = len(self._states)
        full = np.zeros((dimension, dimension), dtype=np.complex128)
        width = len(qubits)
        expected = 2**width
        if local.shape != (expected, expected):
            message = f"expected a {expected}x{expected} local unitary"
            raise ValueError(message)
        for column, before in enumerate(self._states):
            local_before = tuple(before[qubit] for qubit in qubits)
            if LEAKED in local_before:
                full[column, column] = 1.0
                continue
            input_index = sum(bit << (width - offset - 1) for offset, bit in enumerate(local_before))
            for local_after in product(range(2), repeat=width):
                output_index = sum(bit << (width - offset - 1) for offset, bit in enumerate(local_after))
                after = list(before)
                for qubit, value in zip(qubits, local_after, strict=True):
                    after[qubit] = value
                full[self._indices[tuple(after)], column] = local[output_index, input_index]
        return full

    def _conjugated(self, rho: Matrix, local: Matrix, qubits: tuple[int, ...]) -> Matrix:
        unitary = self._full_unitary(local, qubits)
        return unitary @ rho @ unitary.conj().T

    def _projected(self, rho: Matrix, qubit: int, values: set[int]) -> Matrix:
        mask = np.array([state[qubit] in values for state in self._states], dtype=np.float64)
        return rho * np.outer(mask, mask)

    def _projected_targets(self, rho: Matrix, required: dict[int, set[int]]) -> Matrix:
        mask = np.array(
            [all(state[qubit] in values for qubit, values in required.items()) for state in self._states],
            dtype=np.float64,
        )
        return rho * np.outer(mask, mask)

    def _replaced(self, rho: Matrix, qubit: int, replacement: Matrix) -> Matrix:
        """Trace out one qutrit and tensor in a replacement single-qutrit state."""
        if replacement.shape != (3, 3):
            message = "replacement state must be a 3x3 density matrix"
            raise ValueError(message)
        output = np.zeros_like(rho)
        for row_rest in product(range(3), repeat=self.n_qubits - 1):
            for column_rest in product(range(3), repeat=self.n_qubits - 1):
                traced = 0.0j
                for value in range(3):
                    row = list(row_rest)
                    column = list(column_rest)
                    row.insert(qubit, value)
                    column.insert(qubit, value)
                    traced += rho[self._indices[tuple(row)], self._indices[tuple(column)]]
                for row_value, column_value in product(range(3), repeat=2):
                    row = list(row_rest)
                    column = list(column_rest)
                    row.insert(qubit, row_value)
                    column.insert(qubit, column_value)
                    output[self._indices[tuple(row)], self._indices[tuple(column)]] = (
                        traced * replacement[row_value, column_value]
                    )
        return output

    def _faulted(self, rho: Matrix, qubit: int, fault: str) -> Matrix:
        if fault == "L":
            leaked = np.zeros((3, 3), dtype=np.complex128)
            leaked[LEAKED, LEAKED] = 1.0
            return self._replaced(rho, qubit, leaked)
        try:
            pauli = PAULIS[fault]
        except KeyError as error:
            message = f"unsupported qutrit-reference fault {fault!r}"
            raise ValueError(message) from error
        return self._conjugated(rho, pauli, (qubit,))

    def _two_qubit_faulted(self, rho: Matrix, qubits: tuple[int, int], fault: str) -> Matrix:
        if len(fault) != 2:
            message = f"two-qubit fault must contain two symbols, got {fault!r}"
            raise ValueError(message)
        updated = rho
        for qubit, symbol in zip(qubits, fault, strict=True):
            if symbol != "I":
                updated = self._faulted(updated, qubit, symbol)
        return updated

    def reset(self, qubit: int) -> QutritReference:
        """Apply the exact noisy preparation channel."""
        noise = self.noise
        zero = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
        one = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
        leaked = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
        self.rho = (
            (1.0 - noise.preparation_probability) * self._replaced(self.rho, qubit, zero)
            + noise.preparation_probability
            * (1.0 - noise.preparation_leakage_ratio)
            * self._replaced(self.rho, qubit, one)
            + noise.preparation_probability * noise.preparation_leakage_ratio * self._replaced(self.rho, qubit, leaked)
        )
        return self

    def one_qubit_gate(self, qubit: int, unitary: Matrix) -> QutritReference:
        """Apply PECOS general-noise one-qubit semantics to an arbitrary unitary."""
        noise = self.noise
        computational = self._projected(self.rho, qubit, {0, 1})
        leaked = self._projected(self.rho, qubit, {LEAKED})
        ideal = self._conjugated(computational, unitary, (qubit,))
        updated = (1.0 - noise.p1) * ideal

        for fault, probability in noise.p1_pauli_model:
            updated += (
                noise.p1
                * (1.0 - noise.p1_emission_ratio)
                * probability
                * self._faulted(
                    ideal,
                    qubit,
                    fault,
                )
            )
        for fault, probability in noise.p1_emission_model:
            updated += (
                noise.p1
                * noise.p1_emission_ratio
                * probability
                * self._faulted(
                    computational,
                    qubit,
                    fault,
                )
            )

        emission_on_leaked = noise.p1 * noise.p1_emission_ratio
        mixed = np.diag([0.5, 0.5, 0.0]).astype(np.complex128)
        seeped = self._replaced(leaked, qubit, mixed)
        updated += (1.0 - emission_on_leaked) * leaked
        updated += emission_on_leaked * (
            (1.0 - noise.p1_seepage_probability) * leaked + noise.p1_seepage_probability * seeped
        )
        self.rho = updated
        return self

    def two_qubit_gate(self, qubits: tuple[int, int], unitary: Matrix) -> QutritReference:
        """Apply PECOS general-noise two-qubit semantics to an arbitrary unitary."""
        noise = self.noise
        first, second = qubits
        computational = self._projected_targets(self.rho, {first: {0, 1}, second: {0, 1}})
        ideal = self._conjugated(computational, unitary, qubits)
        updated = (1.0 - noise.p2) * ideal

        for fault, probability in noise.p2_pauli_model:
            updated += (
                noise.p2
                * (1.0 - noise.p2_emission_ratio)
                * probability
                * self._two_qubit_faulted(
                    ideal,
                    qubits,
                    fault,
                )
            )
        for fault, probability in noise.p2_emission_model:
            updated += (
                noise.p2
                * noise.p2_emission_ratio
                * probability
                * self._two_qubit_faulted(
                    computational,
                    qubits,
                    fault,
                )
            )

        emission_on_leaked = noise.p2 * noise.p2_emission_ratio
        mixed = np.diag([0.5, 0.5, 0.0]).astype(np.complex128)
        for leaked_qubits in ({first}, {second}, {first, second}):
            required = {qubit: ({LEAKED} if qubit in leaked_qubits else {0, 1}) for qubit in qubits}
            leaked_component = self._projected_targets(self.rho, required)
            seeped = leaked_component
            for qubit in leaked_qubits:
                seeped = (1.0 - noise.p2_seepage_probability) * seeped + (
                    noise.p2_seepage_probability * self._replaced(seeped, qubit, mixed)
                )
            updated += (1.0 - emission_on_leaked) * leaked_component
            updated += emission_on_leaked * seeped

        self.rho = updated
        return self

    def measurement_distribution(self, qubits: tuple[int, ...]) -> ExpectedDistribution:
        """Return exact noisy Boolean measurement probabilities."""
        probabilities: dict[tuple[int, ...], float] = {}
        diagonal = np.real_if_close(np.diag(self.rho)).real
        for state, raw_state_probability in zip(self._states, diagonal, strict=True):
            state_probability = _roundoff_clipped_probability(
                float(raw_state_probability),
                context=f"density-matrix diagonal for state {state}",
            )
            if state_probability == 0.0:
                continue
            per_qubit: list[tuple[tuple[int, float], ...]] = []
            for qubit in qubits:
                value = state[qubit]
                if value == LEAKED:
                    # PECOS regular measurement first maps leakage to Boolean 1,
                    # then applies the ordinary 1 -> 0 readout channel.
                    per_qubit.append(
                        ((0, self.noise.measurement_1_to_0), (1, 1.0 - self.noise.measurement_1_to_0)),
                    )
                elif value == 0:
                    per_qubit.append(
                        ((0, 1.0 - self.noise.measurement_0_to_1), (1, self.noise.measurement_0_to_1)),
                    )
                else:
                    per_qubit.append(
                        ((0, self.noise.measurement_1_to_0), (1, 1.0 - self.noise.measurement_1_to_0)),
                    )
            for choices in product(*per_qubit):
                outcome = tuple(choice[0] for choice in choices)
                probability = state_probability
                for _, readout_probability in choices:
                    probability *= readout_probability
                probabilities[outcome] = probabilities.get(outcome, 0.0) + probability

        normalization = sum(probabilities.values())
        if not math.isclose(normalization, 1.0, abs_tol=NUMERICAL_PROBABILITY_TOLERANCE):
            message = f"qutrit reference lost probability mass: trace-derived outcomes sum to {normalization}"
            raise ValueError(message)
        clipped = {
            outcome: _roundoff_clipped_probability(probability, context=f"measurement outcome {outcome}")
            for outcome, probability in probabilities.items()
        }
        if not math.isclose(sum(clipped.values()), 1.0, abs_tol=NUMERICAL_PROBABILITY_TOLERANCE):
            message = "qutrit reference roundoff clipping changed total probability mass"
            raise ValueError(message)
        return ExpectedDistribution(clipped)
