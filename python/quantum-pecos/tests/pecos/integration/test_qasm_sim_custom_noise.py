"""Built-in noise builders apply their configured measurement errors."""

import pytest
from pecos import Qasm, biased_depolarizing_noise, depolarizing_noise, general_noise, qasm_engine


@pytest.fixture(params=[0, 1], ids=["initial-zero", "initial-one"])
def initial_bit(request) -> int:
    return request.param


def _measure_with_noise(initial_bit: int, noise) -> list[int]:
    program = Qasm.from_string(f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        {"x q[0];" if initial_bit else ""}
        measure q -> c;
    """)
    return qasm_engine().program(program).to_sim().noise(noise).seed(42).run(16).to_dict()["c"]


@pytest.mark.parametrize("probability", [0, 1], ids=["no-error", "always-flip"])
def test_depolarizing_measurement_configuration(probability: int, initial_bit: int) -> None:
    noise = depolarizing_noise().with_uniform_probability(0.0).with_p_meas(probability)
    assert _measure_with_noise(initial_bit, noise) == [initial_bit ^ probability] * 16


@pytest.mark.parametrize(
    "noise_factory",
    [
        pytest.param(lambda: biased_depolarizing_noise().with_uniform_probability(0.0), id="biased-depolarizing"),
        pytest.param(general_noise, id="general"),
    ],
)
@pytest.mark.parametrize(
    ("p_meas_0", "p_meas_1"),
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    ids=["no-errors", "flip-zero-only", "flip-one-only", "flip-both"],
)
def test_asymmetric_measurement_configuration(noise_factory, p_meas_0: int, p_meas_1: int, initial_bit: int) -> None:
    noise = noise_factory().with_p_meas_0(p_meas_0).with_p_meas_1(p_meas_1)
    flip = p_meas_1 if initial_bit else p_meas_0
    assert _measure_with_noise(initial_bit, noise) == [initial_bit ^ flip] * 16
