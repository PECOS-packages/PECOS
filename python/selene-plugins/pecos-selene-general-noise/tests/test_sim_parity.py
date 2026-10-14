"""Statistical parity between PECOS ``sim()`` and the Selene plugin route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pecos_selene_statevec as selene_statevec
import pytest
from general_noise_conformance import ExpectedDistribution, OutcomeSample
from guppylang import guppy
from guppylang.std.angles import angle, pi
from guppylang.std.builtins import result
from guppylang.std.qsystem.functional import zz_phase
from guppylang.std.quantum import cx, measure, qubit, rz
from pecos_selene_general_noise import GeneralNoiseParameters, GeneralNoisePlugin
from selene_sim.build import build

pecos = pytest.importorskip(
    "pecos",
    reason="cross-runtime parity requires the optional quantum-pecos package",
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from selene_sim.instance import SeleneInstance


@guppy
def measurement_probe() -> None:
    """Measure a freshly prepared zero."""
    q = qubit()
    result("outcome", measure(q).read())


@guppy
def one_qubit_probe() -> None:
    """Expose an X fault after a diagonal one-qubit gate."""
    q = qubit()
    rz(q, pi / 2)
    result("outcome", measure(q).read())


@guppy
def two_qubit_probe() -> None:
    """Expose an XI fault after one primitive two-qubit gate."""
    q0 = qubit()
    q1 = qubit()
    q0, q1 = zz_phase(q0, q1, angle(0.5))
    result("q0", measure(q0).read())
    result("q1", measure(q1).read())


@guppy
def repetition_code_probe() -> None:
    """Measure one repetition-code check round under a combined profile."""
    data_0 = qubit()
    data_1 = qubit()
    data_2 = qubit()
    check_0 = qubit()
    check_1 = qubit()

    cx(data_0, check_0)
    cx(data_1, check_0)
    cx(data_1, check_1)
    cx(data_2, check_1)

    result("check_0", measure(check_0).read())
    result("check_1", measure(check_1).read())
    result("data_0", measure(data_0).read())
    result("data_1", measure(data_1).read())
    result("data_2", measure(data_2).read())


@dataclass(frozen=True)
class ParityCase:
    """One program and noise profile shared by both runtime routes."""

    program: object
    runner: SeleneInstance
    n_qubits: int
    result_tags: tuple[str, ...]
    parameters: GeneralNoiseParameters
    expected: ExpectedDistribution | None


CASES = (
    ParityCase(
        measurement_probe,
        build(measurement_probe.compile()),
        1,
        ("outcome",),
        GeneralNoiseParameters().with_p_meas_0(0.25),
        ExpectedDistribution({(0,): 0.75, (1,): 0.25}),
    ),
    ParityCase(
        one_qubit_probe,
        build(one_qubit_probe.compile()),
        1,
        ("outcome",),
        GeneralNoiseParameters().with_p1(0.3).with_p1_pauli_model({"X": 1.0}),
        ExpectedDistribution({(0,): 0.7, (1,): 0.3}),
    ),
    ParityCase(
        two_qubit_probe,
        build(two_qubit_probe.compile()),
        2,
        ("q0", "q1"),
        GeneralNoiseParameters().with_p2(0.3).with_p2_pauli_model({"XI": 1.0}),
        ExpectedDistribution({(0, 0): 0.7, (1, 0): 0.3}),
    ),
)

QEC_CASE = ParityCase(
    repetition_code_probe,
    build(repetition_code_probe.compile()),
    5,
    ("check_0", "check_1"),
    GeneralNoiseParameters()
    .with_p_prep(0.08)
    .with_prep_leak_ratio(0.25)
    .with_p1(0.14)
    .with_p1_pauli_model({"X": 0.2, "Y": 0.3, "Z": 0.5})
    .with_p1_emission_ratio(0.2)
    .with_p1_emission_model({"X": 0.5, "L": 0.5})
    .with_p1_seepage_prob(0.3)
    .with_p2(0.2)
    .with_p2_pauli_model({"XI": 0.25, "IX": 0.25, "XX": 0.25, "ZZ": 0.25})
    .with_p2_emission_ratio(0.2)
    .with_p2_emission_model({"XI": 0.5, "IL": 0.5})
    .with_p2_seepage_prob(0.3)
    .with_p_meas_0(0.06)
    .with_p_meas_1(0.1),
    None,
)


def _columns_to_sample(
    columns: Mapping[str, list[object]],
    result_tags: tuple[str, ...],
    shots: int,
) -> OutcomeSample:
    """Convert PECOS column-oriented results to the conformance sample format."""
    assert all(len(columns[tag]) == shots for tag in result_tags)
    return OutcomeSample(
        tuple(tuple(int(columns[tag][shot]) for tag in result_tags) for shot in range(shots)),
    )


def _sample_selene(case: ParityCase, *, shots: int, seed: int) -> OutcomeSample:
    """Run a case through Selene's error-model plugin boundary."""
    simulator = selene_statevec.StateVecPlugin(random_seed=seed + 1)
    error_model = GeneralNoisePlugin(parameters=case.parameters, random_seed=seed)
    outcomes = []
    for shot in case.runner.run_shots(
        simulator=simulator,
        error_model=error_model,
        n_qubits=case.n_qubits,
        n_shots=shots,
        n_processes=1,
    ):
        tagged = dict(shot)
        outcomes.append(tuple(int(tagged[tag]) for tag in case.result_tags))
    return OutcomeSample(tuple(outcomes))


def _sample_pecos(case: ParityCase, *, shots: int, seed: int) -> OutcomeSample:
    """Run the same case through PECOS ``sim()`` and its native noise model."""
    columns = (
        pecos.sim(case.program)
        .qubits(case.n_qubits)
        .quantum(pecos.state_vector())
        .noise(case.parameters)
        .seed(seed)
        .workers(1)
        .run(shots)
        .to_dict()
    )
    return _columns_to_sample(columns, case.result_tags, shots)


@pytest.mark.parametrize("case", CASES, ids=("measurement", "one-qubit", "two-qubit"))
def test_general_noise_parameters_have_runtime_parity(case: ParityCase) -> None:
    """Both public execution paths implement the same configured distribution."""
    shots = 2048
    seed = 1701

    selene_sample = _sample_selene(case, shots=shots, seed=seed)
    pecos_sample = _sample_pecos(case, shots=shots, seed=seed)

    assert case.expected is not None
    case.expected.assert_matches(selene_sample)
    case.expected.assert_matches(pecos_sample)
    selene_sample.assert_statistically_equivalent(pecos_sample)


@pytest.mark.slow
def test_combined_qec_workload_has_runtime_parity() -> None:
    """A combined leakage-bearing profile agrees across a QEC gate stream."""
    shots = 8192
    seed = 1901

    selene_sample = _sample_selene(QEC_CASE, shots=shots, seed=seed)
    pecos_sample = _sample_pecos(QEC_CASE, shots=shots, seed=seed)

    assert len(selene_sample.counts) > 1
    assert len(pecos_sample.counts) > 1
    selene_sample.assert_statistically_equivalent(
        pecos_sample,
        failure_probability=1e-4,
    )
