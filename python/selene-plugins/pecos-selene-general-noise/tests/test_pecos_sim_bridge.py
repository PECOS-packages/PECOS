"""PECOS ``sim()`` accepts the Selene plugin's general-noise parameters."""

from __future__ import annotations

import pecos
import pytest
from pecos_selene_general_noise import GeneralNoiseParameters


def test_pecos_sim_accepts_general_noise_parameters_directly() -> None:
    """The shared parameters can be passed directly to PECOS ``sim().noise()``."""
    program = pecos.Qasm.from_string(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
        """,
    )
    parameters = GeneralNoiseParameters().with_p_meas_0(1.0)

    direct = pecos.sim(program).noise(parameters).seed(17).run(4).to_dict()
    explicit = pecos.sim(program).noise(parameters.to_pecos_builder()).seed(17).run(4).to_dict()

    assert direct == explicit == {"c": [1] * 4}


def test_pecos_sim_rejects_selene_only_local_groups() -> None:
    """Direct PECOS use must not silently discard Selene-owned topology."""
    program = pecos.Qasm.from_string("OPENQASM 2.0; qreg q[1];")
    parameters = GeneralNoiseParameters().with_local_crosstalk_groups((0, 1))

    with pytest.raises(ValueError, match=r"local_groups.*Selene-adapter topology"):
        pecos.sim(program).noise(parameters)


@pytest.mark.parametrize(
    ("gate", "parameters"),
    [
        ("PZ", GeneralNoiseParameters().with_p_prep(1.0).with_prep_leak_ratio(0.0)),
        ("MZ", GeneralNoiseParameters().with_p_meas(1.0)),
        ("CNOT", GeneralNoiseParameters().with_p2(1.0).with_p2_pauli_model({"XI": 1.0})),
    ],
)
def test_noiseless_gate_bridge_suppresses_faults_in_pecos(gate: str, parameters: GeneralNoiseParameters) -> None:
    """Selene gate names reach the real immutable builder and suppress their faults."""
    program = pecos.Qasm.from_string(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        reset q;
        cx q[0], q[1];
        measure q -> c;
        """,
    )
    noisy_builder = parameters.to_pecos_builder(seed=17)
    quiet_builder = parameters.with_noiseless_gate(gate).to_pecos_builder(seed=17)

    noisy = pecos.sim(program).noise(noisy_builder).seed(17).run(4).to_dict()["c"]
    quiet = pecos.sim(program).noise(quiet_builder).seed(17).run(4).to_dict()["c"]

    assert all(value != 0 for value in noisy)
    assert quiet == [0] * 4
