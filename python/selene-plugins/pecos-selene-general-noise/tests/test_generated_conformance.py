"""Generated device-neutral circuit matrices for broad noise conformance."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from general_noise_conformance import ConformanceExperiment
from guppylang import guppy
from guppylang.std.angles import angle, pi
from guppylang.std.builtins import comptime, result
from guppylang.std.qsystem.functional import zz_phase
from guppylang.std.quantum import measure, qubit
from guppylang.std.quantum import rx as quantum_rx
from guppylang.std.quantum import ry as quantum_ry
from guppylang.std.quantum import rz as quantum_rz
from pecos_selene_general_noise import GeneralNoiseParameters
from qutrit_reference import QutritNoise, QutritReference, rx, ry, rz, rzz
from selene_sim import Stim
from selene_sim.build import build

if TYPE_CHECKING:
    from general_noise_conformance import ExpectedDistribution
    from qutrit_reference import Matrix
    from selene_sim.instance import SeleneInstance

type OperationSequence = tuple[int, ...]

pytestmark = pytest.mark.slow

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
RZZ_0_1 = 2 * N_LOCAL_GATES
RZZ_1_0 = RZZ_0_1 + 1


@dataclass(frozen=True)
class NoiseProfile:
    """One independently specified channel configuration for the circuit matrix."""

    name: str
    noise: QutritNoise
    parameters: GeneralNoiseParameters
    comparison: QutritNoise
    channels: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedCase:
    """A sensitive profile/circuit pair retained by the generated matrix."""

    profile: NoiseProfile
    case_id: int


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
            if operation == comptime(RZZ_0_1):
                q0, q1 = zz_phase(q0, q1, angle(0.5))
            elif operation == comptime(RZZ_1_0):
                q1, q0 = zz_phase(q1, q0, angle(0.5))
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
    # Short probes keep gate-replacement and axis-asymmetry faults observable;
    # longer seeded sequences below exercise accumulation and mixed axes.
    sequences = [(6,), (7,), (6, 4), (7, 4), (0, 0), (2, 2)]
    candidate_id = 0
    while len(sequences) < 15:
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

    sequences = [
        (encode(0, 2), encode(1, 2), RZZ_0_1, encode(0, 3), encode(1, 3)),
        (encode(0, 0), encode(1, 2), RZZ_0_1, encode(0, 1), encode(1, 3)),
        (encode(0, 2), RZZ_0_1, encode(0, 3)),
    ]
    case_id = 0
    while len(sequences) < 25:
        control = case_id % 2
        target = 1 - control
        sequence = [
            encode(control, 2 + (case_id // 2) % 2),
            encode(control, 8) if case_id % 3 == 0 else encode(target, 8),
        ]
        if case_id % 4 >= 2:
            sequence.append(encode(target, 6))
        sequence.append(RZZ_0_1 if control == 0 else RZZ_1_0)
        if case_id % 3 == 1:
            sequence.append(encode(control, 6))
        elif case_id % 3 == 2:
            sequence.append(encode(target, 7))
        sequence.extend(
            (
                encode(0, 2 + case_id % 2),
                encode(1, (case_id // 2) % 2),
            ),
        )
        sequences.append(tuple(sequence))
        case_id += 1
    return tuple(sequences)


ONE_QUBIT_SEQUENCES = _generated_one_qubit_sequences()
TWO_QUBIT_SEQUENCES = _generated_two_qubit_sequences()


@cache
def _one_qubit_runner(case_id: int) -> SeleneInstance:
    return _one_qubit_program(ONE_QUBIT_SEQUENCES[case_id])


@cache
def _two_qubit_runner(case_id: int) -> SeleneInstance:
    return _two_qubit_program(TWO_QUBIT_SEQUENCES[case_id])


FULL_ONE_QUBIT_NOISE = QutritNoise(
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
FULL_ONE_QUBIT_PARAMETERS = (
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

UNIFORM_P1_MODEL = (("X", 1.0 / 3.0), ("Y", 1.0 / 3.0), ("Z", 1.0 / 3.0))
UNIFORM_P2_MODEL = {first + second: 1.0 / 15.0 for first, second in product("IXYZ", repeat=2) if first + second != "II"}
UNIFORM_P2_REFERENCE = tuple(UNIFORM_P2_MODEL.items())

ONE_QUBIT_PROFILES = (
    NoiseProfile(
        "preparation",
        QutritNoise(preparation_probability=0.25),
        GeneralNoiseParameters().with_p_prep(0.25),
        QutritNoise(),
        ("preparation",),
    ),
    NoiseProfile(
        "preparation-leakage",
        QutritNoise(preparation_probability=0.25, preparation_leakage_ratio=0.8),
        GeneralNoiseParameters().with_p_prep(0.25).with_prep_leak_ratio(0.8),
        QutritNoise(preparation_probability=0.25),
        ("preparation-leakage",),
    ),
    NoiseProfile(
        "readout-symmetric",
        QutritNoise(measurement_0_to_1=0.15, measurement_1_to_0=0.15),
        GeneralNoiseParameters().with_p_meas(0.15),
        QutritNoise(),
        ("readout",),
    ),
    NoiseProfile(
        "readout-asymmetric",
        QutritNoise(measurement_0_to_1=0.0, measurement_1_to_0=0.35),
        GeneralNoiseParameters().with_p_meas_0(0.0).with_p_meas_1(0.35),
        QutritNoise(measurement_0_to_1=0.175, measurement_1_to_0=0.175),
        ("readout",),
    ),
    NoiseProfile(
        "p1-symmetric-pauli",
        QutritNoise(p1=0.3, p1_pauli_model=UNIFORM_P1_MODEL),
        GeneralNoiseParameters().with_p1(0.3).with_p1_pauli_model(dict(UNIFORM_P1_MODEL)),
        QutritNoise(),
        ("single-qubit-pauli",),
    ),
    NoiseProfile(
        "p1-asymmetric-pauli",
        QutritNoise(p1=0.5, p1_pauli_model=(("X", 1.0),)),
        GeneralNoiseParameters().with_p1(0.5).with_p1_pauli_model({"X": 1.0}),
        QutritNoise(p1=0.5, p1_pauli_model=(("Z", 1.0),)),
        ("single-qubit-pauli",),
    ),
    NoiseProfile(
        "p1-emission",
        QutritNoise(
            p1=0.5,
            p1_pauli_model=(("Z", 1.0),),
            p1_emission_ratio=1.0,
            p1_emission_model=(("Z", 1.0),),
        ),
        GeneralNoiseParameters()
        .with_p1(0.5)
        .with_p1_pauli_model({"Z": 1.0})
        .with_p1_emission_ratio(1.0)
        .with_p1_emission_model({"Z": 1.0}),
        QutritNoise(p1=0.5, p1_pauli_model=(("Z", 1.0),)),
        ("single-qubit-emission",),
    ),
    NoiseProfile(
        "p1-emission-leakage",
        QutritNoise(p1=0.3, p1_emission_ratio=0.75, p1_emission_model=(("L", 1.0),)),
        GeneralNoiseParameters().with_p1(0.3).with_p1_emission_ratio(0.75).with_p1_emission_model({"L": 1.0}),
        QutritNoise(p1=0.3, p1_emission_ratio=0.75, p1_emission_model=(("X", 1.0),)),
        ("single-qubit-emission", "gate-leakage"),
    ),
    NoiseProfile(
        "p1-seepage",
        QutritNoise(
            preparation_probability=0.35,
            preparation_leakage_ratio=1.0,
            p1=0.35,
            p1_emission_ratio=1.0,
            p1_emission_model=(("L", 1.0),),
            p1_seepage_probability=0.7,
        ),
        GeneralNoiseParameters()
        .with_p_prep(0.35)
        .with_prep_leak_ratio(1.0)
        .with_p1(0.35)
        .with_p1_emission_ratio(1.0)
        .with_p1_emission_model({"L": 1.0})
        .with_p1_seepage_prob(0.7),
        QutritNoise(
            preparation_probability=0.35,
            preparation_leakage_ratio=1.0,
            p1=0.35,
            p1_emission_ratio=1.0,
            p1_emission_model=(("L", 1.0),),
        ),
        ("single-qubit-seepage",),
    ),
    NoiseProfile(
        "full-one-qubit",
        FULL_ONE_QUBIT_NOISE,
        FULL_ONE_QUBIT_PARAMETERS,
        QutritNoise(),
        ("combined-channels",),
    ),
)

TWO_QUBIT_PROFILES = (
    NoiseProfile(
        "p2-symmetric-pauli",
        QutritNoise(p2=0.3, p2_pauli_model=UNIFORM_P2_REFERENCE),
        GeneralNoiseParameters().with_p2(0.3).with_p2_pauli_model(UNIFORM_P2_MODEL),
        QutritNoise(),
        ("two-qubit-pauli",),
    ),
    NoiseProfile(
        "p2-asymmetric-pauli",
        QutritNoise(p2=0.6, p2_pauli_model=(("XI", 1.0),)),
        GeneralNoiseParameters().with_p2(0.6).with_p2_pauli_model({"XI": 1.0}),
        QutritNoise(p2=0.6, p2_pauli_model=(("IX", 1.0),)),
        ("two-qubit-pauli",),
    ),
    NoiseProfile(
        "p2-emission",
        QutritNoise(
            p2=1.0,
            p2_pauli_model=(("XI", 0.5), ("IZ", 0.5)),
            p2_emission_ratio=1.0,
            p2_emission_model=(("XI", 0.5), ("IZ", 0.5)),
        ),
        GeneralNoiseParameters()
        .with_p2(1.0)
        .with_p2_pauli_model({"XI": 0.5, "IZ": 0.5})
        .with_p2_emission_ratio(1.0)
        .with_p2_emission_model({"XI": 0.5, "IZ": 0.5}),
        QutritNoise(p2=1.0, p2_pauli_model=(("XI", 0.5), ("IZ", 0.5))),
        ("two-qubit-emission",),
    ),
    NoiseProfile(
        "p2-emission-leakage",
        QutritNoise(
            p2=0.3,
            p2_emission_ratio=0.75,
            p2_emission_model=(("IL", 0.4), ("LI", 0.3), ("LL", 0.3)),
        ),
        GeneralNoiseParameters()
        .with_p2(0.3)
        .with_p2_emission_ratio(0.75)
        .with_p2_emission_model({"IL": 0.4, "LI": 0.3, "LL": 0.3}),
        QutritNoise(p2=0.3, p2_emission_ratio=0.75, p2_emission_model=(("XI", 0.5), ("IZ", 0.5))),
        ("two-qubit-emission", "gate-leakage"),
    ),
    NoiseProfile(
        "p2-seepage",
        QutritNoise(
            preparation_probability=1.0,
            preparation_leakage_ratio=1.0,
            p2=1.0,
            p2_emission_ratio=1.0,
            p2_emission_model=(("LL", 1.0),),
            p2_seepage_probability=1.0,
        ),
        GeneralNoiseParameters()
        .with_p_prep(1.0)
        .with_prep_leak_ratio(1.0)
        .with_p2(1.0)
        .with_p2_emission_ratio(1.0)
        .with_p2_emission_model({"LL": 1.0})
        .with_p2_seepage_prob(1.0),
        QutritNoise(
            preparation_probability=1.0,
            preparation_leakage_ratio=1.0,
            p2=1.0,
            p2_emission_ratio=1.0,
            p2_emission_model=(("LL", 1.0),),
        ),
        ("two-qubit-seepage",),
    ),
    NoiseProfile(
        "p1+p2",
        QutritNoise(p1=0.12, p2=0.2, p2_pauli_model=UNIFORM_P2_REFERENCE),
        GeneralNoiseParameters().with_p1(0.12).with_p2(0.2).with_p2_pauli_model(UNIFORM_P2_MODEL),
        QutritNoise(),
        ("combined-channels",),
    ),
    NoiseProfile(
        "p2+spam",
        QutritNoise(
            preparation_probability=0.08,
            p2=0.2,
            p2_pauli_model=UNIFORM_P2_REFERENCE,
            measurement_0_to_1=0.06,
            measurement_1_to_0=0.11,
        ),
        GeneralNoiseParameters()
        .with_p_prep(0.08)
        .with_p2(0.2)
        .with_p2_pauli_model(UNIFORM_P2_MODEL)
        .with_p_meas_0(0.06)
        .with_p_meas_1(0.11),
        QutritNoise(),
        ("combined-channels",),
    ),
)


def _one_qubit_reference(operations: OperationSequence, noise: QutritNoise) -> QutritReference:
    reference = QutritReference(1, noise).reset(0)
    for operation in operations:
        reference.one_qubit_gate(0, LOCAL_GATES[operation])
    return reference


def _two_qubit_reference(operations: OperationSequence, noise: QutritNoise) -> QutritReference:
    reference = QutritReference(2, noise).reset(0).reset(1)
    for operation in operations:
        if operation == RZZ_0_1:
            reference.two_qubit_gate((0, 1), rzz(math.pi / 2.0))
        elif operation == RZZ_1_0:
            reference.two_qubit_gate((1, 0), rzz(math.pi / 2.0))
        else:
            site, local = divmod(operation, N_LOCAL_GATES)
            reference.one_qubit_gate(site, LOCAL_GATES[local])
    return reference


def _expected(profile: NoiseProfile, operations: OperationSequence, *, n_qubits: int) -> ExpectedDistribution:
    if n_qubits == 1:
        return _one_qubit_reference(operations, profile.noise).measurement_distribution((0,))
    return _two_qubit_reference(operations, profile.noise).measurement_distribution((0, 1))


def _comparison(profile: NoiseProfile, operations: OperationSequence, *, n_qubits: int) -> ExpectedDistribution:
    if n_qubits == 1:
        return _one_qubit_reference(operations, profile.comparison).measurement_distribution((0,))
    return _two_qubit_reference(operations, profile.comparison).measurement_distribution((0, 1))


def _sensitive_cases(
    profiles: tuple[NoiseProfile, ...],
    sequences: tuple[OperationSequence, ...],
    *,
    n_qubits: int,
    shots: int,
) -> tuple[object, ...]:
    """Retain observable cases and make channel coverage visible to pytest."""
    cases = []
    counts = {profile.name: 0 for profile in profiles}
    for profile in profiles:
        for case_id, operations in enumerate(sequences):
            expected = _expected(profile, operations, n_qubits=n_qubits)
            comparison = _comparison(profile, operations, n_qubits=n_qubits)
            try:
                expected.assert_sensitive_to(comparison, shots=shots)
            except AssertionError:
                continue
            evidence = f"{profile.name}-{case_id}"
            marks = [
                pytest.mark.noise_channel(channel, oracle="qutrit", evidence=evidence) for channel in profile.channels
            ]
            cases.append(pytest.param(GeneratedCase(profile, case_id), id=f"{profile.name}-{case_id}", marks=marks))
            counts[profile.name] += 1
    insufficient = {name: count for name, count in counts.items() if count < 3}
    if insufficient:
        message = f"generated profiles need at least three sensitive circuits: {insufficient}"
        raise AssertionError(message)
    return tuple(cases)


ONE_QUBIT_CASES = _sensitive_cases(ONE_QUBIT_PROFILES, ONE_QUBIT_SEQUENCES, n_qubits=1, shots=2048)
TWO_QUBIT_CASES = _sensitive_cases(TWO_QUBIT_PROFILES, TWO_QUBIT_SEQUENCES, n_qubits=2, shots=4096)


@pytest.mark.parametrize("case", ONE_QUBIT_CASES)
@pytest.mark.parametrize(
    ("error_seed", "simulator_seed"),
    [(601, 607), (613, 617), (619, 631)],
    ids=("seed-1", "seed-2", "seed-3"),
)
def test_generated_one_qubit_matrix_matches_qutrit_reference(
    case: GeneratedCase,
    error_seed: int,
    simulator_seed: int,
) -> None:
    """Sensitive one-qubit channel/circuit pairs agree with the qutrit oracle."""
    operations = ONE_QUBIT_SEQUENCES[case.case_id]
    experiment = ConformanceExperiment(
        runner=_one_qubit_runner(case.case_id),
        n_qubits=1,
        result_tags=("outcome",),
        parameters=case.profile.parameters,
        expected=_expected(case.profile, operations, n_qubits=1),
        comparison=_comparison(case.profile, operations, n_qubits=1),
        shots=2048,
        seed=error_seed + case.case_id,
    )
    experiment.assert_conforms(Stim(random_seed=simulator_seed + case.case_id), n_processes=2)


@pytest.mark.parametrize("case", TWO_QUBIT_CASES)
@pytest.mark.parametrize(
    ("error_seed", "simulator_seed"),
    [(641, 647), (653, 659), (661, 673)],
    ids=("seed-1", "seed-2", "seed-3"),
)
def test_generated_entangling_matrix_matches_qutrit_reference(
    case: GeneratedCase,
    error_seed: int,
    simulator_seed: int,
) -> None:
    """Sensitive two-qubit channel/circuit pairs agree with the qutrit oracle."""
    operations = TWO_QUBIT_SEQUENCES[case.case_id]
    experiment = ConformanceExperiment(
        runner=_two_qubit_runner(case.case_id),
        n_qubits=2,
        result_tags=("q0", "q1"),
        parameters=case.profile.parameters,
        expected=_expected(case.profile, operations, n_qubits=2),
        comparison=_comparison(case.profile, operations, n_qubits=2),
        shots=4096,
        seed=error_seed + case.case_id,
    )
    experiment.assert_conforms(Stim(random_seed=simulator_seed + case.case_id), n_processes=2)
