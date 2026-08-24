"""Generated device-neutral circuit matrices for broad noise conformance."""

from __future__ import annotations

import math
import random
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from general_noise_conformance import ConformanceExperiment
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import comptime, result
from guppylang.std.quantum import cx, measure, qubit
from guppylang.std.quantum import rx as quantum_rx
from guppylang.std.quantum import ry as quantum_ry
from guppylang.std.quantum import rz as quantum_rz
from pecos_selene_general_noise import GeneralNoiseParameters
from qutrit_reference import QutritNoise, QutritReference, controlled_x, rx, ry, rz
from selene_sim import Stim
from selene_sim.build import build

if TYPE_CHECKING:
    from qutrit_reference import Matrix
    from selene_sim.instance import SeleneInstance

type OperationSequence = tuple[int, ...]

LOCAL_GATES: tuple[Matrix, ...] = (
    rx(math.pi / 2),
    rx(-math.pi / 2),
    ry(math.pi / 2),
    ry(-math.pi / 2),
    rz(math.pi / 2),
    rz(-math.pi / 2),
    rx(math.pi),
    ry(math.pi),
    rz(math.pi),
)
N_LOCAL_GATES = len(LOCAL_GATES)
CX_0_1 = 2 * N_LOCAL_GATES
CX_1_0 = CX_0_1 + 1


@guppy
def _apply_standard_gate(q: qubit, operation: int) -> None:
    """Compile one encoded local gate using only standard quantum operations."""
    if operation == 0:
        quantum_rx(q, pi / 2)
    elif operation == 1:
        quantum_rx(q, -pi / 2)
    elif operation == 2:
        quantum_ry(q, pi / 2)
    elif operation == 3:
        quantum_ry(q, -pi / 2)
    elif operation == 4:
        quantum_rz(q, pi / 2)
    elif operation == 5:
        quantum_rz(q, -pi / 2)
    elif operation == 6:
        quantum_rx(q, pi)
    elif operation == 7:
        quantum_ry(q, pi)
    else:
        quantum_rz(q, pi)


def _one_qubit_program(operations: OperationSequence) -> SeleneInstance:
    @guppy
    def main() -> None:
        q = qubit()
        encoded = comptime(list(operations))
        for index in range(comptime(len(operations))):
            _apply_standard_gate(q, encoded[index])
        result("outcome", measure(q).read())

    return build(main.compile())


def _two_qubit_program(operations: OperationSequence) -> SeleneInstance:
    @guppy
    def main() -> None:
        q0 = qubit()
        q1 = qubit()
        encoded = comptime(list(operations))
        for index in range(comptime(len(operations))):
            operation = encoded[index]
            if operation == comptime(CX_0_1):
                cx(q0, q1)
            elif operation == comptime(CX_1_0):
                cx(q1, q0)
            elif operation < comptime(N_LOCAL_GATES):
                _apply_standard_gate(q0, operation)
            else:
                _apply_standard_gate(q1, operation - comptime(N_LOCAL_GATES))
        result("q0", measure(q0).read())
        result("q1", measure(q1).read())

    return build(main.compile())


def _generated_one_qubit_sequences() -> tuple[OperationSequence, ...]:
    """Generate reproducible mixed-axis sequences with sensitive ideal readout."""
    rng = random.Random(1729)
    sequences = []
    candidate_id = 0
    while len(sequences) < 12:
        sequence = []
        for layer in range(5 + candidate_id % 6):
            axis = (candidate_id + layer) % 3
            sign = rng.randrange(2)
            sequence.append(2 * axis + sign)
        state = np.array([1.0, 0.0], dtype=np.complex128)
        for operation in sequence:
            state = LOCAL_GATES[operation] @ state
        probability_zero = float(abs(state[0]) ** 2)
        if probability_zero < 1e-12 or probability_zero > 1.0 - 1e-12:
            sequences.append(tuple(sequence))
        candidate_id += 1
    return tuple(sequences)


def _generated_two_qubit_sequences() -> tuple[OperationSequence, ...]:
    """Generate parity-sensitive Clifford circuits around one entangling gate."""

    def encode(site: int, gate: int) -> int:
        return site * N_LOCAL_GATES + gate

    sequences = []
    for case_id in range(12):
        control = case_id % 2
        target = 1 - control
        sequence = [
            encode(control, 2 + (case_id // 2) % 2),
            encode(control, 8) if case_id % 3 == 0 else encode(target, 8),
        ]
        if case_id % 4 >= 2:
            sequence.append(encode(target, 6))
        sequence.append(CX_0_1 if control == 0 else CX_1_0)
        if case_id % 3 == 1:
            sequence.append(encode(control, 6))
        elif case_id % 3 == 2:
            sequence.append(encode(target, 7))
        sequence.extend((encode(0, 8), encode(1, 8)))
        sequences.append(tuple(sequence))
    return tuple(sequences)


ONE_QUBIT_SEQUENCES = _generated_one_qubit_sequences()
TWO_QUBIT_SEQUENCES = _generated_two_qubit_sequences()
ONE_QUBIT_RUNNERS = tuple(_one_qubit_program(sequence) for sequence in ONE_QUBIT_SEQUENCES)
TWO_QUBIT_RUNNERS = tuple(_two_qubit_program(sequence) for sequence in TWO_QUBIT_SEQUENCES)

ONE_QUBIT_NOISE = QutritNoise(
    preparation_probability=0.08,
    preparation_leakage_ratio=0.3,
    p1=0.18,
    p1_pauli_model=(("X", 0.2), ("Y", 0.3), ("Z", 0.5)),
    p1_emission_ratio=0.25,
    p1_emission_model=(("X", 0.4), ("Y", 0.2), ("L", 0.4)),
    p1_seepage_probability=0.35,
    measurement_0_to_1=0.06,
    measurement_1_to_0=0.11,
)
ONE_QUBIT_PARAMETERS = (
    GeneralNoiseParameters()
    .with_p_prep(0.08)
    .with_prep_leak_ratio(0.3)
    .with_p1(0.18)
    .with_p1_pauli_model({"X": 0.2, "Y": 0.3, "Z": 0.5})
    .with_p1_emission_ratio(0.25)
    .with_p1_emission_model({"X": 0.4, "Y": 0.2, "L": 0.4})
    .with_p1_seepage_prob(0.35)
    .with_p_meas_0(0.06)
    .with_p_meas_1(0.11)
)

UNIFORM_P2_MODEL = {first + second: 1.0 / 15.0 for first, second in product("IXYZ", repeat=2) if first + second != "II"}
TWO_QUBIT_NOISE = QutritNoise(p2=0.3)
TWO_QUBIT_PARAMETERS = GeneralNoiseParameters().with_p2(0.3).with_p2_pauli_model(UNIFORM_P2_MODEL)


def _one_qubit_reference(operations: OperationSequence, noise: QutritNoise) -> QutritReference:
    reference = QutritReference(1, noise).reset(0)
    for operation in operations:
        reference.one_qubit_gate(0, LOCAL_GATES[operation])
    return reference


def _two_qubit_reference(operations: OperationSequence, noise: QutritNoise) -> QutritReference:
    reference = QutritReference(2, noise).reset(0).reset(1)
    for operation in operations:
        if operation == CX_0_1:
            reference.two_qubit_gate((0, 1), controlled_x())
        elif operation == CX_1_0:
            reference.two_qubit_gate((1, 0), controlled_x())
        else:
            site, local = divmod(operation, N_LOCAL_GATES)
            reference.one_qubit_gate(site, LOCAL_GATES[local])
    return reference


@pytest.mark.parametrize("case_id", range(len(ONE_QUBIT_SEQUENCES)))
@pytest.mark.parametrize(("error_seed", "simulator_seed"), [(601, 607), (613, 617)])
def test_generated_one_qubit_matrix_matches_qutrit_reference(
    case_id: int,
    error_seed: int,
    simulator_seed: int,
) -> None:
    """Generated mixed-axis circuits agree with the exact qutrit oracle."""
    operations = ONE_QUBIT_SEQUENCES[case_id]
    experiment = ConformanceExperiment(
        runner=ONE_QUBIT_RUNNERS[case_id],
        n_qubits=1,
        result_tags=("outcome",),
        parameters=ONE_QUBIT_PARAMETERS,
        expected=_one_qubit_reference(operations, ONE_QUBIT_NOISE).measurement_distribution((0,)),
        comparison=_one_qubit_reference(operations, QutritNoise()).measurement_distribution((0,)),
        shots=2048,
        seed=error_seed + case_id,
    )
    experiment.assert_conforms(Stim(random_seed=simulator_seed + case_id), n_processes=2)


@pytest.mark.parametrize("case_id", range(len(TWO_QUBIT_SEQUENCES)))
def test_generated_entangling_matrix_matches_qutrit_reference(case_id: int) -> None:
    """Generated parity-sensitive circuits agree for uniform two-qubit noise."""
    operations = TWO_QUBIT_SEQUENCES[case_id]
    experiment = ConformanceExperiment(
        runner=TWO_QUBIT_RUNNERS[case_id],
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=TWO_QUBIT_PARAMETERS,
        expected=_two_qubit_reference(operations, TWO_QUBIT_NOISE).measurement_distribution((0, 1)),
        comparison=_two_qubit_reference(operations, QutritNoise()).measurement_distribution((0, 1)),
        shots=4096,
        seed=631 + case_id,
    )
    experiment.assert_conforms(Stim(random_seed=647 + case_id), n_processes=2)
