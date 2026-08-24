"""Exact device-neutral reference model for computational-basis circuits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from general_noise_conformance import ExpectedDistribution

type BasisState = tuple[int, ...]
type StateDistribution = dict[BasisState, float]

LEAKED = 2


def _accumulate(distribution: StateDistribution, state: BasisState, probability: float) -> None:
    if probability > 1e-15:
        distribution[state] = distribution.get(state, 0.0) + probability


def _updated(state: BasisState, qubit: int, value: int) -> BasisState:
    values = list(state)
    values[qubit] = value
    return tuple(values)


def _fault(value: int, pauli: str) -> int:
    if pauli == "L":
        return LEAKED
    if value == LEAKED or pauli in {"I", "Z"}:
        return value
    if pauli in {"X", "Y"}:
        return 1 - value
    message = f"unsupported basis-state fault {pauli!r}"
    raise ValueError(message)


@dataclass(frozen=True)
class BasisNoise:
    """Independent parameters for the exact basis-state reference model."""

    preparation_probability: float = 0.0
    preparation_leakage_ratio: float = 0.0
    p1: float = 0.0
    p1_pauli_model: tuple[tuple[str, float], ...] = (("X", 1 / 3), ("Y", 1 / 3), ("Z", 1 / 3))
    p1_emission_ratio: float = 0.0
    p1_emission_model: tuple[tuple[str, float], ...] = (("L", 1.0),)
    p1_seepage_probability: float = 0.0
    measurement_0_to_1: float = 0.0
    measurement_1_to_0: float = 0.0


class BasisStateReference:
    """Propagate an exact distribution over ``0``, ``1``, and leaked states.

    This model intentionally shares no implementation code with PECOS. It specifies
    the observable semantics of preparation, one-qubit gate, leakage, seepage, and
    readout channels for circuits that remain in the computational basis.
    """

    def __init__(self, n_qubits: int, noise: BasisNoise | None = None) -> None:
        if n_qubits <= 0:
            message = "a reference circuit must contain at least one qubit"
            raise ValueError(message)
        self.noise = noise or BasisNoise()
        self.distribution: StateDistribution = {(0,) * n_qubits: 1.0}

    def reset(self, qubit: int) -> BasisStateReference:
        """Apply noisy preparation, including reset from a leaked state."""
        noise = self.noise
        updated: StateDistribution = {}
        for state, weight in self.distribution.items():
            _accumulate(updated, _updated(state, qubit, 0), weight * (1.0 - noise.preparation_probability))
            _accumulate(
                updated,
                _updated(state, qubit, 1),
                weight * noise.preparation_probability * (1.0 - noise.preparation_leakage_ratio),
            )
            _accumulate(
                updated,
                _updated(state, qubit, LEAKED),
                weight * noise.preparation_probability * noise.preparation_leakage_ratio,
            )
        self.distribution = updated
        return self

    def one_qubit_gate(self, qubit: int, *, flips_basis: bool) -> BasisStateReference:
        """Apply a noisy one-qubit gate that either flips or preserves basis values."""
        noise = self.noise
        updated: StateDistribution = {}
        for state, weight in self.distribution.items():
            before = state[qubit]
            if before == LEAKED:
                seepage = noise.p1 * noise.p1_emission_ratio * noise.p1_seepage_probability
                _accumulate(updated, state, weight * (1.0 - seepage))
                _accumulate(updated, _updated(state, qubit, 0), weight * seepage / 2.0)
                _accumulate(updated, _updated(state, qubit, 1), weight * seepage / 2.0)
                continue

            ideal = 1 - before if flips_basis else before
            _accumulate(updated, _updated(state, qubit, ideal), weight * (1.0 - noise.p1))

            pauli_weight = weight * noise.p1 * (1.0 - noise.p1_emission_ratio)
            for fault, probability in noise.p1_pauli_model:
                _accumulate(updated, _updated(state, qubit, _fault(ideal, fault)), pauli_weight * probability)

            emission_weight = weight * noise.p1 * noise.p1_emission_ratio
            for fault, probability in noise.p1_emission_model:
                # Emission replaces rather than follows the requested gate.
                _accumulate(updated, _updated(state, qubit, _fault(before, fault)), emission_weight * probability)

        self.distribution = updated
        return self

    def measurement_distribution(self, qubits: tuple[int, ...]) -> ExpectedDistribution:
        """Return the exact noisy Boolean-readout distribution for selected qubits."""
        noise = self.noise
        outcomes: dict[tuple[int, ...], float] = {}
        for state, state_probability in self.distribution.items():
            per_qubit: list[tuple[tuple[int, float], ...]] = []
            for qubit in qubits:
                value = state[qubit]
                if value == LEAKED:
                    per_qubit.append(((1, 1.0),))
                elif value == 0:
                    per_qubit.append(
                        ((0, 1.0 - noise.measurement_0_to_1), (1, noise.measurement_0_to_1)),
                    )
                else:
                    per_qubit.append(
                        ((0, noise.measurement_1_to_0), (1, 1.0 - noise.measurement_1_to_0)),
                    )
            for choices in product(*per_qubit):
                outcome = tuple(choice[0] for choice in choices)
                probability = state_probability
                for _, readout_probability in choices:
                    probability *= readout_probability
                outcomes[outcome] = outcomes.get(outcome, 0.0) + probability
        return ExpectedDistribution(outcomes)
