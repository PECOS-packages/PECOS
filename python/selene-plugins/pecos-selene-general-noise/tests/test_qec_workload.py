"""Device-neutral QEC integration checks for the general-noise plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from general_noise_conformance import ConformanceExperiment, ExpectedDistribution
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit, x
from pecos_selene_general_noise import GeneralNoiseParameters
from selene_sim import Stim
from selene_sim.build import build

if TYPE_CHECKING:
    from selene_sim.instance import SeleneInstance


@guppy
def repetition_code_syndrome_round() -> None:
    """Measure the two checks of a three-qubit repetition-code block."""
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


@guppy
def repetition_code_middle_fault() -> None:
    """Measure both checks after a known X fault on the middle data qubit."""
    data_0 = qubit()
    data_1 = qubit()
    data_2 = qubit()
    check_0 = qubit()
    check_1 = qubit()

    x(data_1)
    cx(data_0, check_0)
    cx(data_1, check_0)
    cx(data_1, check_1)
    cx(data_2, check_1)

    result("check_0", measure(check_0).read())
    result("check_1", measure(check_1).read())
    result("data_0", measure(data_0).read())
    result("data_1", measure(data_1).read())
    result("data_2", measure(data_2).read())


SYNDROME_RUNNER = build(repetition_code_syndrome_round.compile())
FAULT_RUNNER = build(repetition_code_middle_fault.compile())
NO_SYNDROME = ExpectedDistribution({(0, 0): 1.0})
MIDDLE_DATA_SYNDROME = ExpectedDistribution({(1, 1): 1.0})


@pytest.mark.parametrize(
    ("runner", "parameters", "expected", "comparison"),
    [
        pytest.param(
            SYNDROME_RUNNER,
            GeneralNoiseParameters(),
            NO_SYNDROME,
            MIDDLE_DATA_SYNDROME,
            id="noiseless",
        ),
        pytest.param(
            FAULT_RUNNER,
            GeneralNoiseParameters(),
            MIDDLE_DATA_SYNDROME,
            NO_SYNDROME,
            id="known-middle-data-x",
        ),
        pytest.param(
            SYNDROME_RUNNER,
            GeneralNoiseParameters().with_p_meas_0(1.0),
            MIDDLE_DATA_SYNDROME,
            NO_SYNDROME,
            id="certain-readout-faults",
        ),
    ],
)
def test_repetition_code_syndrome_round(
    runner: SeleneInstance,
    parameters: GeneralNoiseParameters,
    expected: ExpectedDistribution,
    comparison: ExpectedDistribution,
) -> None:
    """A generic QEC check round preserves and detects the expected syndromes."""
    experiment = ConformanceExperiment(
        runner=runner,
        n_qubits=5,
        result_tags=("check_0", "check_1"),
        parameters=parameters,
        expected=expected,
        comparison=comparison,
        shots=64,
    )
    experiment.assert_conforms(Stim(random_seed=709))


@pytest.mark.slow
def test_repetition_code_syndrome_distribution() -> None:
    """Independent readout faults produce the analytic syndrome mixture."""
    experiment = ConformanceExperiment(
        runner=SYNDROME_RUNNER,
        n_qubits=5,
        result_tags=("check_0", "check_1"),
        parameters=GeneralNoiseParameters().with_p_meas_0(0.2),
        expected=ExpectedDistribution({(0, 0): 0.64, (0, 1): 0.16, (1, 0): 0.16, (1, 1): 0.04}),
        comparison=NO_SYNDROME,
        shots=1024,
        seed=719,
    )
    experiment.assert_conforms(Stim(random_seed=727), n_processes=2)


@pytest.mark.slow
def test_distribution_is_worker_count_independent() -> None:
    """Every Selene worker layout satisfies the same error-model distribution."""
    experiment = ConformanceExperiment(
        runner=SYNDROME_RUNNER,
        n_qubits=5,
        result_tags=("check_0", "check_1"),
        parameters=GeneralNoiseParameters().with_p_meas_0(0.3),
        expected=ExpectedDistribution({(0, 0): 0.49, (0, 1): 0.21, (1, 0): 0.21, (1, 1): 0.09}),
        comparison=NO_SYNDROME,
        shots=1024,
        seed=733,
    )
    experiment.assert_conforms(Stim(random_seed=739), n_processes=1)
    experiment.assert_conforms(Stim(random_seed=739), n_processes=2)
