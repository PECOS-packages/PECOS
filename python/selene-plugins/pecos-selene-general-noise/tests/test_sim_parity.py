"""Statistical parity between PECOS ``sim()`` and the Selene plugin route."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import pecos
import pecos_selene_statevec as selene_statevec
import pytest
from basis_state_reference import BasisNoise, BasisStateReference
from general_noise_conformance import ExpectedDistribution, OutcomeSample
from guppylang import guppy
from guppylang.std.angles import angle, pi
from guppylang.std.builtins import result
from guppylang.std.qsystem.functional import zz_phase
from guppylang.std.quantum import cx, measure, qubit, rz, x
from pecos_selene_general_noise import GeneralNoiseParameters, GeneralNoisePlugin
from selene_sim.build import build

if TYPE_CHECKING:
    from collections.abc import Mapping

SELENE_ERROR_SEED = 1701
SELENE_SIMULATOR_SEED = 1709
PECOS_SEED = 1721


@guppy
def measurement_probe() -> None:
    """Measure a freshly prepared zero and a flipped one."""
    zero = qubit()
    one = qubit()
    x(one)
    result("zero", measure(zero).read())
    result("one", measure(one).read())


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
def leaked_flip_probe() -> None:
    """Separate a leaked preparation, which ignores the flip, from a bit-flipped one."""
    q = qubit()
    x(q)
    result("outcome", measure(q).read())


@guppy
def leaked_idle_probe() -> None:
    """Give a leaked preparation one diagonal gate in which to seep back."""
    q = qubit()
    rz(q, pi / 2)
    result("outcome", measure(q).read())


@guppy
def leaked_control_probe() -> None:
    """Pass a possibly leaked control through one two-qubit interaction."""
    control = qubit()
    target = qubit()
    cx(control, target)
    result("control", measure(control).read())
    result("target", measure(target).read())


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


@dataclass(frozen=True, eq=False)
class ParityCase:
    """One program and noise profile shared by both runtime routes.

    ``mutants`` are profiles that differ from ``parameters`` in a single channel. A
    route that implemented one of them instead of ``parameters`` must fail the parity
    check at ``shots``; the slow lane verifies that it does.
    """

    name: str
    program: object
    n_qubits: int
    result_tags: tuple[str, ...]
    parameters: GeneralNoiseParameters
    expected: ExpectedDistribution | None
    shots: int
    mutants: Mapping[str, GeneralNoiseParameters]


MEASUREMENT_NOISE = GeneralNoiseParameters().with_p_meas_0(0.25).with_p_meas_1(0.1)
ONE_QUBIT_NOISE = GeneralNoiseParameters().with_p1(0.3).with_p1_pauli_model({"X": 1.0})
TWO_QUBIT_NOISE = GeneralNoiseParameters().with_p2(0.3).with_p2_pauli_model({"XI": 1.0})
PREPARATION_LEAKAGE_NOISE = GeneralNoiseParameters().with_p_prep(0.4).with_prep_leak_ratio(0.5)
ONE_QUBIT_SEEPAGE_NOISE = (
    GeneralNoiseParameters()
    .with_p_prep(0.5)
    .with_prep_leak_ratio(1.0)
    .with_p1(0.5)
    .with_p1_emission_ratio(1.0)
    .with_p1_emission_model({"Z": 1.0})
    .with_p1_seepage_prob(1.0)
)
TWO_QUBIT_SEEPAGE_NOISE = (
    GeneralNoiseParameters()
    .with_p_prep(0.3)
    .with_prep_leak_ratio(1.0)
    .with_p2(0.5)
    .with_p2_emission_ratio(1.0)
    .with_p2_emission_model({"IX": 1.0})
    .with_p2_seepage_prob(1.0)
)
QEC_NOISE = (
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
    .with_p_meas_0(0.05)
    .with_p_meas_1(0.2)
)


def _one_qubit_reference(noise: BasisNoise, *, flips: tuple[bool, ...]) -> ExpectedDistribution:
    reference = BasisStateReference(1, noise).reset(0)
    for flips_basis in flips:
        reference.one_qubit_gate(0, flips_basis=flips_basis)
    return reference.measurement_distribution((0,))


CASES = (
    ParityCase(
        "measurement",
        measurement_probe,
        2,
        ("zero", "one"),
        MEASUREMENT_NOISE,
        ExpectedDistribution({(0, 1): 0.675, (1, 1): 0.225, (0, 0): 0.075, (1, 0): 0.025}),
        4096,
        {"symmetric-readout": MEASUREMENT_NOISE.with_p_meas_1(0.25)},
    ),
    ParityCase(
        "one-qubit",
        one_qubit_probe,
        1,
        ("outcome",),
        ONE_QUBIT_NOISE,
        ExpectedDistribution({(0,): 0.7, (1,): 0.3}),
        4096,
        {"phase-faults": ONE_QUBIT_NOISE.with_p1_pauli_model({"Z": 1.0})},
    ),
    ParityCase(
        "two-qubit",
        two_qubit_probe,
        2,
        ("q0", "q1"),
        TWO_QUBIT_NOISE,
        ExpectedDistribution({(0, 0): 0.7, (1, 0): 0.3}),
        4096,
        {"swapped-pauli-order": TWO_QUBIT_NOISE.with_p2_pauli_model({"IX": 1.0})},
    ),
    ParityCase(
        "preparation-leakage",
        leaked_flip_probe,
        1,
        ("outcome",),
        PREPARATION_LEAKAGE_NOISE,
        _one_qubit_reference(
            BasisNoise(preparation_probability=0.4, preparation_leakage_ratio=0.5),
            flips=(True,),
        ),
        4096,
        {
            "bit-flip-preparation": PREPARATION_LEAKAGE_NOISE.with_prep_leak_ratio(0.0),
            "leaked-preparation": PREPARATION_LEAKAGE_NOISE.with_prep_leak_ratio(1.0),
        },
    ),
    ParityCase(
        "one-qubit-seepage",
        leaked_idle_probe,
        1,
        ("outcome",),
        ONE_QUBIT_SEEPAGE_NOISE,
        _one_qubit_reference(
            BasisNoise(
                preparation_probability=0.5,
                preparation_leakage_ratio=1.0,
                p1=0.5,
                p1_emission_ratio=1.0,
                p1_emission_model=(("Z", 1.0),),
                p1_seepage_probability=1.0,
            ),
            flips=(False,),
        ),
        4096,
        {"no-seepage": ONE_QUBIT_SEEPAGE_NOISE.with_p1_seepage_prob(0.0)},
    ),
    # The leaked-control outcome depends on how each route lowers ``cx`` to native
    # gates, so this case has no independent analytic reference.
    ParityCase(
        "two-qubit-seepage",
        leaked_control_probe,
        2,
        ("control", "target"),
        TWO_QUBIT_SEEPAGE_NOISE,
        None,
        4096,
        {"no-seepage": TWO_QUBIT_SEEPAGE_NOISE.with_p2_seepage_prob(0.0)},
    ),
)

QEC_CASE = ParityCase(
    "repetition-code",
    repetition_code_probe,
    5,
    ("check_0", "check_1", "data_0", "data_1", "data_2"),
    QEC_NOISE,
    None,
    32768,
    {
        "no-one-qubit-emission": QEC_NOISE.with_p1_emission_ratio(0.0),
        "symmetric-readout": QEC_NOISE.with_p_meas_1(0.05),
        "single-two-qubit-pauli": QEC_NOISE.with_p2_pauli_model({"XI": 1.0}),
    },
)

ALL_CASES = (
    *(pytest.param(case, id=case.name) for case in CASES),
    pytest.param(QEC_CASE, id=QEC_CASE.name, marks=pytest.mark.slow),
)


@cache
def _sample_selene(case: ParityCase) -> OutcomeSample:
    """Run a case through Selene's error-model plugin boundary."""
    simulator = selene_statevec.StateVecPlugin(random_seed=SELENE_SIMULATOR_SEED)
    error_model = GeneralNoisePlugin(parameters=case.parameters, random_seed=SELENE_ERROR_SEED)
    outcomes = []
    for shot in build(case.program.compile()).run_shots(
        simulator=simulator,
        error_model=error_model,
        n_qubits=case.n_qubits,
        n_shots=case.shots,
        n_processes=1,
    ):
        tagged = dict(shot)
        outcomes.append(tuple(int(tagged[tag]) for tag in case.result_tags))
    return OutcomeSample(tuple(outcomes))


def _sample_pecos(case: ParityCase, parameters: GeneralNoiseParameters) -> OutcomeSample:
    """Run the case's program through PECOS ``sim()`` and its native noise model."""
    columns = (
        pecos.sim(case.program)
        .qubits(case.n_qubits)
        .quantum(pecos.state_vector())
        .noise(parameters)
        .seed(PECOS_SEED)
        .workers(1)
        .run(case.shots)
        .to_dict()
    )
    assert all(len(columns[tag]) == case.shots for tag in case.result_tags)
    return OutcomeSample(
        tuple(tuple(int(columns[tag][shot]) for tag in case.result_tags) for shot in range(case.shots)),
    )


@pytest.mark.parametrize("case", ALL_CASES)
def test_general_noise_parameters_have_runtime_parity(case: ParityCase) -> None:
    """Both public execution paths implement the same configured distribution."""
    selene_sample = _sample_selene(case)
    pecos_sample = _sample_pecos(case, case.parameters)

    if case.expected is not None:
        case.expected.assert_matches(selene_sample)
        case.expected.assert_matches(pecos_sample)
    selene_sample.assert_statistically_equivalent(pecos_sample)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "mutant"),
    [pytest.param(case, mutant, id=f"{case.name}-{mutant}") for case in (*CASES, QEC_CASE) for mutant in case.mutants],
)
def test_parity_check_rejects_single_channel_mutants(case: ParityCase, mutant: str) -> None:
    """A route that implemented a one-channel variant of the profile would be caught."""
    selene_sample = _sample_selene(case)
    mutant_sample = _sample_pecos(case, case.mutants[mutant])

    with pytest.raises(AssertionError, match="empirical distributions differ"):
        selene_sample.assert_statistically_equivalent(mutant_sample)
