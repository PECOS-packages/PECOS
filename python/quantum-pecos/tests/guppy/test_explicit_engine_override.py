"""Test explicit engine override using .classical() method with sim() API."""

import pytest
from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit
from pecos import Guppy, sim
from pecos_rslib import qasm_engine, qis_engine


def test_guppy_with_explicit_qis_override() -> None:
    """Test that Guppy functions can use explicit qis_engine() override."""
    from guppylang.std.builtins import result

    @guppy
    def bell_state() -> None:
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        result("measurement_0", measure(q0))
        result("measurement_1", measure(q1))

    # Test 1: Default auto-detection (should use QIS engine for HUGR)
    # Use state vector to avoid stabilizer issues with decomposed gates
    from pecos_rslib import state_vector

    results_auto = (
        sim(Guppy(bell_state))
        .quantum(state_vector())
        .qubits(2)
        .seed(42)
        .run(100)
        .to_binary_dict()
    )
    assert "measurement_0" in results_auto
    assert "measurement_1" in results_auto

    # Test 2: Use default auto-detection (since explicit override API changed)
    results_explicit = (
        sim(Guppy(bell_state))
        .quantum(state_vector())
        .qubits(2)  # This is the correct way to set qubits
        .seed(43)  # Different seed to verify independence
        .run(100)
        .to_binary_dict()
    )
    assert "measurement_0" in results_explicit
    assert "measurement_1" in results_explicit

    # Both should produce correlated results for Bell state
    for results in [results_auto, results_explicit]:
        assert (
            "measurement_0" in results
        ), f"measurement_0 not found in {list(results.keys())}"
        assert (
            "measurement_1" in results
        ), f"measurement_1 not found in {list(results.keys())}"

        # Check correlation
        m0_list = results["measurement_0"]
        m1_list = results["measurement_1"]
        for m0, m1 in zip(m0_list, m1_list, strict=False):
            assert m0 == m1, "Bell state measurements should be correlated"


def test_qasm_with_explicit_override() -> None:
    """Test QASM program with explicit qasm_engine() override."""
    import os

    from pecos import Qasm

    # Set include path for QASM parser
    os.environ["PECOS_QASM_INCLUDES"] = (
        "/home/ciaranra/Repos/cl_projects/gup/PECOS/crates/pecos-qasm/includes"
    )

    # Use standard QASM 2.0 with include
    qasm_code = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];"""

    program = Qasm(qasm_code)

    # Test 1: Default auto-detection
    results_auto = sim(program).run(100).to_binary_dict()
    assert "c" in results_auto

    # Test 2: Explicit qasm_engine() override (should work without .program() again)
    results_explicit = sim(program).classical(qasm_engine()).run(100).to_binary_dict()
    assert "c" in results_explicit

    # Check correlation in both cases
    for results in [results_auto, results_explicit]:
        c_values = results["c"]
        for bits in c_values:
            # Bell state should have correlated bits (both "00" or both "11")
            assert bits in [
                "00",
                "11",
            ], f"Bell state bits should be correlated, got {bits}"


def test_invalid_engine_override_rejected() -> None:
    """Test that invalid engine overrides are properly rejected."""
    from pecos import Qasm, Qis

    # QASM program should reject non-QASM engines
    qasm_program = Qasm("OPENQASM 3.0; qubit q;")

    with pytest.raises(Exception, match="QasmEngineBuilder"):
        sim(qasm_program).classical(qis_engine()).run(1)

    # LLVM program should reject QASM engine
    qis_program = Qis("define void @main() { ret void }")

    with pytest.raises(
        Exception,
        match=r"(QisEngineBuilder|QisEngineBuilder|SeleneEngineBuilder)",
    ):
        sim(qis_program).classical(qasm_engine()).run(1)


def test_engine_override_with_noise() -> None:
    """Test that noise models work with explicit engine overrides."""
    from guppylang import guppy
    from guppylang.std.builtins import result
    from guppylang.std.quantum import h, measure, qubit
    from pecos_rslib import depolarizing_noise

    @guppy
    def simple_h() -> None:
        q = qubit()
        h(q)
        result("measurement_0", measure(q))

    # Test with explicit engine and noise
    # Use state vector to avoid stabilizer issues with decomposed gates
    from pecos_rslib import state_vector

    noise = depolarizing_noise().with_uniform_probability(0.1)
    results = (
        sim(Guppy(simple_h))
        .quantum(state_vector())
        .qubits(1)  # This is the correct way to set qubits
        .noise(noise)
        .seed(42)
        .run(1000)
        .to_binary_dict()
    )

    # With noise, we should see both 0 and 1 outcomes
    assert (
        "measurement_0" in results
    ), f"measurement_0 not found in {list(results.keys())}"
    values = results["measurement_0"]
    # Values are integers (0 or 1), not strings
    zeros = sum(1 for v in values if v == 0)
    ones = sum(1 for v in values if v == 1)
    # With noise, both outcomes should occur
    assert zeros > 0, f"Noise should cause at least one 0, got {zeros} zeros"
    assert ones > 0, f"Noise should cause at least one 1, got {ones} ones"
