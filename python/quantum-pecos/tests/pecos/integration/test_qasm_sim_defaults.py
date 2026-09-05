"""Behavioral checks for QASM simulation defaults."""

import pytest
from pecos import Qasm, general_noise, qasm_engine


@pytest.mark.parametrize("build_first", [False, True], ids=["direct-run", "build-then-run"])
def test_default_simulation_is_noiseless(build_first: bool) -> None:
    """Default simulation returns every requested shot as integer register values."""
    program = Qasm.from_string("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        measure q -> c;
    """)
    builder = qasm_engine().program(program).to_sim()
    simulation = builder.build() if build_first else builder
    assert simulation.run(16).to_dict()["c"] == [1] * 16


def test_general_noise_defaults_are_noiseless() -> None:
    """General noise requires explicit settings or auto() to introduce errors."""
    program = Qasm.from_string("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        reset q;
        x q[0];
        cx q[0], q[1];
        measure q -> c;
    """)
    results = qasm_engine().program(program).to_sim().noise(general_noise()).seed(42).run(64)
    assert results.to_dict()["c"] == [3] * 64
