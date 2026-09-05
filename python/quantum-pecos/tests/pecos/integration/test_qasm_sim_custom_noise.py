"""Built-in noise builders apply their configured measurement errors."""

import pytest
from pecos import Qasm, biased_depolarizing_noise, depolarizing_noise, general_noise, qasm_engine


@pytest.mark.parametrize(
    "noise_factory",
    [
        pytest.param(
            lambda p: depolarizing_noise().with_uniform_probability(0.0).with_p_meas(p),
            id="depolarizing",
        ),
        pytest.param(
            lambda p: biased_depolarizing_noise().with_uniform_probability(0.0).with_p_meas_0(p).with_p_meas_1(p),
            id="biased-depolarizing",
        ),
        pytest.param(lambda p: general_noise().with_p_meas_0(p).with_p_meas_1(p), id="general"),
    ],
)
@pytest.mark.parametrize("probability", [0, 1], ids=["no-error", "always-flip"])
@pytest.mark.parametrize("initial_bit", [0, 1])
def test_noise_builder_measurement_configuration(noise_factory, probability: int, initial_bit: int) -> None:
    program = Qasm.from_string(f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        {"x q[0];" if initial_bit else ""}
        measure q -> c;
    """)
    results = qasm_engine().program(program).to_sim().noise(noise_factory(probability)).seed(42).run(16)
    assert results.to_dict()["c"] == [initial_bit ^ probability] * 16
