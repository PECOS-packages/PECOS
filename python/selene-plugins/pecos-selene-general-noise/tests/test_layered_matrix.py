"""Seeded three- and four-qubit layered semantic conformance."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import TYPE_CHECKING

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
    from qutrit_reference import Matrix
    from selene_sim.instance import SeleneInstance

pytestmark = pytest.mark.slow

type OperationSequence = tuple[int, ...]

LOCAL_GATES: tuple[Matrix, ...] = (
    rx(math.pi / 2.0),
    rx(-math.pi / 2.0),
    ry(math.pi / 2.0),
    ry(-math.pi / 2.0),
    rz(math.pi / 2.0),
    rz(-math.pi / 2.0),
)
N_LOCAL_GATES = len(LOCAL_GATES)
RZZ_BASE = 4 * N_LOCAL_GATES
RZZ_0_1 = RZZ_BASE
RZZ_1_2 = RZZ_BASE + 1
RZZ_2_3 = RZZ_BASE + 2


@guppy
def _apply_local(q: qubit, operation: int) -> None:
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
    else:
        quantum_rz(q, -pi / 2)


def _seeded_operations(n_qubits: int, layers: int) -> OperationSequence:
    """Generate one reproducible nearest-neighbor layered circuit."""
    rng = random.Random(20_260_831 + 100 * n_qubits + layers)
    operations = []
    for layer in range(layers):
        operations.extend(site * N_LOCAL_GATES + rng.randrange(N_LOCAL_GATES) for site in range(n_qubits))
        edges = ((0, 1), (2, 3)) if layer % 2 == 0 else ((1, 2),)
        for first, second in edges:
            if second < n_qubits:
                operations.append({(0, 1): RZZ_0_1, (1, 2): RZZ_1_2, (2, 3): RZZ_2_3}[(first, second)])
        operations.extend(site * N_LOCAL_GATES + rng.randrange(N_LOCAL_GATES) for site in reversed(range(n_qubits)))
    return tuple(operations)


def _three_qubit_program(operations: OperationSequence) -> SeleneInstance:
    @guppy
    def main() -> None:
        q0 = qubit()
        q1 = qubit()
        q2 = qubit()
        encoded = comptime(list(operations))
        for index in range(comptime(len(operations))):
            operation = encoded[index]
            if operation < comptime(N_LOCAL_GATES):
                _apply_local(q0, operation)
            elif operation < comptime(2 * N_LOCAL_GATES):
                _apply_local(q1, operation - comptime(N_LOCAL_GATES))
            elif operation < comptime(3 * N_LOCAL_GATES):
                _apply_local(q2, operation - comptime(2 * N_LOCAL_GATES))
            elif operation == comptime(RZZ_0_1):
                q0, q1 = zz_phase(q0, q1, angle(0.5))
            else:
                q1, q2 = zz_phase(q1, q2, angle(0.5))
        result("q0", measure(q0).read())
        result("q1", measure(q1).read())
        result("q2", measure(q2).read())

    return build(main.compile())


def _four_qubit_program(operations: OperationSequence) -> SeleneInstance:
    @guppy
    def main() -> None:
        q0 = qubit()
        q1 = qubit()
        q2 = qubit()
        q3 = qubit()
        encoded = comptime(list(operations))
        for index in range(comptime(len(operations))):
            operation = encoded[index]
            if operation < comptime(N_LOCAL_GATES):
                _apply_local(q0, operation)
            elif operation < comptime(2 * N_LOCAL_GATES):
                _apply_local(q1, operation - comptime(N_LOCAL_GATES))
            elif operation < comptime(3 * N_LOCAL_GATES):
                _apply_local(q2, operation - comptime(2 * N_LOCAL_GATES))
            elif operation < comptime(4 * N_LOCAL_GATES):
                _apply_local(q3, operation - comptime(3 * N_LOCAL_GATES))
            elif operation == comptime(RZZ_0_1):
                q0, q1 = zz_phase(q0, q1, angle(0.5))
            elif operation == comptime(RZZ_1_2):
                q1, q2 = zz_phase(q1, q2, angle(0.5))
            else:
                q2, q3 = zz_phase(q2, q3, angle(0.5))
        result("q0", measure(q0).read())
        result("q1", measure(q1).read())
        result("q2", measure(q2).read())
        result("q3", measure(q3).read())

    return build(main.compile())


@dataclass(frozen=True)
class LayeredProfile:
    """One combined profile used across three- and four-qubit workloads."""

    name: str
    noise: QutritNoise
    parameters: GeneralNoiseParameters


UNIFORM_P2 = tuple(
    (first + second, 1.0 / 15.0) for first, second in product("IXYZ", repeat=2) if first + second != "II"
)

LAYERED_PROFILES = (
    LayeredProfile(
        "p1+spam",
        QutritNoise(preparation_probability=0.08, p1=0.18, measurement_0_to_1=0.06, measurement_1_to_0=0.1),
        GeneralNoiseParameters().with_p_prep(0.08).with_p1(0.18).with_p_meas_0(0.06).with_p_meas_1(0.1),
    ),
    LayeredProfile(
        "p2+spam",
        QutritNoise(preparation_probability=0.08, p2=0.22, measurement_0_to_1=0.06, measurement_1_to_0=0.1),
        GeneralNoiseParameters().with_p_prep(0.08).with_p2(0.22).with_p_meas_0(0.06).with_p_meas_1(0.1),
    ),
    LayeredProfile(
        "p1+p2",
        QutritNoise(p1=0.14, p2=0.2, p2_pauli_model=UNIFORM_P2),
        GeneralNoiseParameters().with_p1(0.14).with_p2(0.2).with_p2_pauli_model(dict(UNIFORM_P2)),
    ),
    LayeredProfile(
        "full",
        QutritNoise(
            preparation_probability=0.08,
            preparation_leakage_ratio=0.25,
            p1=0.14,
            p1_emission_ratio=0.2,
            p1_emission_model=(("X", 0.5), ("L", 0.5)),
            p1_seepage_probability=0.3,
            p2=0.2,
            p2_pauli_model=UNIFORM_P2,
            p2_emission_ratio=0.2,
            p2_emission_model=(("XI", 0.5), ("IL", 0.5)),
            p2_seepage_probability=0.3,
            measurement_0_to_1=0.06,
            measurement_1_to_0=0.1,
        ),
        GeneralNoiseParameters()
        .with_p_prep(0.08)
        .with_prep_leak_ratio(0.25)
        .with_p1(0.14)
        .with_p1_emission_ratio(0.2)
        .with_p1_emission_model({"X": 0.5, "L": 0.5})
        .with_p1_seepage_prob(0.3)
        .with_p2(0.2)
        .with_p2_pauli_model(dict(UNIFORM_P2))
        .with_p2_emission_ratio(0.2)
        .with_p2_emission_model({"XI": 0.5, "IL": 0.5})
        .with_p2_seepage_prob(0.3)
        .with_p_meas_0(0.06)
        .with_p_meas_1(0.1),
    ),
)


def _reference(n_qubits: int, operations: OperationSequence, noise: QutritNoise) -> QutritReference:
    reference = QutritReference(n_qubits, noise)
    for qubit_id in range(n_qubits):
        reference.reset(qubit_id)
    for operation in operations:
        if operation < RZZ_BASE:
            site, local = divmod(operation, N_LOCAL_GATES)
            reference.one_qubit_gate(site, LOCAL_GATES[local])
        elif operation == RZZ_0_1:
            reference.two_qubit_gate((0, 1), rzz(math.pi / 2.0))
        elif operation == RZZ_1_2:
            reference.two_qubit_gate((1, 2), rzz(math.pi / 2.0))
        else:
            reference.two_qubit_gate((2, 3), rzz(math.pi / 2.0))
    return reference


@cache
def _runner(n_qubits: int, layers: int) -> SeleneInstance:
    operations = _seeded_operations(n_qubits, layers)
    if n_qubits == 3:
        return _three_qubit_program(operations)
    return _four_qubit_program(operations)


@dataclass(frozen=True)
class LayeredCase:
    profile_id: int
    n_qubits: int
    layers: int


def _layered_cases() -> tuple[object, ...]:
    cases = []
    profile_counts = {profile.name: 0 for profile in LAYERED_PROFILES}
    for profile_id, profile in enumerate(LAYERED_PROFILES):
        for n_qubits, layers in product((3, 4), (1, 5)):
            operations = _seeded_operations(n_qubits, layers)
            qubits = tuple(range(n_qubits))
            expected = _reference(n_qubits, operations, profile.noise).measurement_distribution(qubits)
            comparison = _reference(n_qubits, operations, QutritNoise()).measurement_distribution(qubits)
            try:
                expected.assert_sensitive_to(comparison, shots=8192)
            except AssertionError:
                continue
            evidence = f"{profile.name}-{n_qubits}q-{layers}-layers"
            cases.append(
                pytest.param(
                    LayeredCase(profile_id, n_qubits, layers),
                    id=evidence,
                    marks=pytest.mark.noise_channel("layered-multiqubit", oracle="qutrit", evidence=evidence),
                ),
            )
            profile_counts[profile.name] += 1
    insufficient = {name: count for name, count in profile_counts.items() if count < 3}
    if insufficient:
        message = f"layered profiles need at least three sensitive circuits: {insufficient}"
        raise AssertionError(message)
    return tuple(cases)


LAYERED_CASES = _layered_cases()


@pytest.mark.parametrize("case", LAYERED_CASES)
@pytest.mark.parametrize(
    ("error_seed", "simulator_seed"),
    [(1009, 1013), (1019, 1021), (1031, 1033)],
    ids=("seed-1", "seed-2", "seed-3"),
)
def test_seeded_multiqubit_layered_matrix(
    case: LayeredCase,
    error_seed: int,
    simulator_seed: int,
) -> None:
    """Three- and four-qubit layered workloads agree with the qutrit oracle."""
    profile = LAYERED_PROFILES[case.profile_id]
    operations = _seeded_operations(case.n_qubits, case.layers)
    qubits = tuple(range(case.n_qubits))
    experiment = ConformanceExperiment(
        runner=_runner(case.n_qubits, case.layers),
        n_qubits=case.n_qubits,
        result_tags=tuple(f"q{qubit_id}" for qubit_id in qubits),
        parameters=profile.parameters,
        expected=_reference(case.n_qubits, operations, profile.noise).measurement_distribution(qubits),
        comparison=_reference(case.n_qubits, operations, QutritNoise()).measurement_distribution(qubits),
        shots=8192,
        seed=error_seed + 10 * case.n_qubits + case.layers,
    )
    experiment.assert_conforms(
        Stim(random_seed=simulator_seed + 10 * case.n_qubits + case.layers),
        n_processes=2,
    )
