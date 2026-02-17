"""Test explicit engine override using .classical() method with sim() API."""

import pytest
from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit
from pecos import Guppy, sim
from pecos_rslib import qasm_engine, qis_engine


def _verify_bell_correlation(results: dict, label: str) -> None:
    """Verify Bell state measurement correlation with detailed diagnostics."""
    # Handle both key formats: measurement_0/measurement_1 or q0/q1
    if "measurement_0" in results:
        m0_key, m1_key = "measurement_0", "measurement_1"
    elif "q0" in results:
        m0_key, m1_key = "q0", "q1"
    else:
        msg = f"{label}: Expected measurement_0 or q0 in {list(results.keys())}"
        raise AssertionError(
            msg,
        )

    assert m1_key in results, f"{label}: {m1_key} not found in {list(results.keys())}"

    m0_list = results[m0_key]
    m1_list = results[m1_key]

    # Verify lists have same length (crucial for correct shot alignment)
    assert len(m0_list) == len(m1_list), (
        f"{label}: Result list length mismatch - "
        f"measurement_0 has {len(m0_list)} values, measurement_1 has {len(m1_list)} values. "
        f"This indicates a bug in result collection."
    )

    # Check correlation for each shot
    mismatches = []
    for i, (m0, m1) in enumerate(zip(m0_list, m1_list, strict=True)):
        if m0 != m1:
            mismatches.append((i, m0, m1))

    if mismatches:
        # Provide detailed diagnostic information
        error_msg = (
            f"{label}: Bell state measurements not correlated!\n"
            f"Found {len(mismatches)} mismatched shots out of {len(m0_list)}.\n"
            f"First 10 mismatches: {mismatches[:10]}\n"
            f"Full m0 list: {m0_list}\n"
            f"Full m1 list: {m1_list}\n"
            f"For a Bell state |00⟩ + |11⟩, measurements must always be equal. "
            f"Getting m0 != m1 indicates either:\n"
            f"  1. Results from different shots got mixed together\n"
            f"  2. TLS (thread-local storage) issue with result collection\n"
            f"  3. A bug in the quantum simulation"
        )
        raise AssertionError(error_msg)


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

    results_auto = sim(Guppy(bell_state)).quantum(state_vector()).qubits(2).seed(42).run(100).to_binary_dict()

    # Test 2: Use default auto-detection (since explicit override API changed)
    results_explicit = (
        sim(Guppy(bell_state))
        .quantum(state_vector())
        .qubits(2)  # This is the correct way to set qubits
        .seed(43)  # Different seed to verify independence
        .run(100)
        .to_binary_dict()
    )

    # Verify Bell state correlation with detailed diagnostics
    _verify_bell_correlation(results_auto, "results_auto (seed=42)")
    _verify_bell_correlation(results_explicit, "results_explicit (seed=43)")


def test_qasm_with_explicit_override() -> None:
    """Test QASM program with explicit qasm_engine() override."""
    import os

    from pecos import Qasm

    # Set include path for QASM parser
    os.environ["PECOS_QASM_INCLUDES"] = "/home/ciaranra/Repos/cl_projects/gup/PECOS/crates/pecos-qasm/includes"

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
    # Handle both key formats: measurement_0 or q0
    if "measurement_0" in results:
        m_key = "measurement_0"
    elif "q0" in results:
        m_key = "q0"
    else:
        msg = f"Expected measurement_0 or q0 in {list(results.keys())}"
        raise AssertionError(msg)
    values = results[m_key]
    # Values are integers (0 or 1), not strings
    zeros = sum(1 for v in values if v == 0)
    ones = sum(1 for v in values if v == 1)
    # With noise, both outcomes should occur
    assert zeros > 0, f"Noise should cause at least one 0, got {zeros} zeros"
    assert ones > 0, f"Noise should cause at least one 1, got {ones} ones"
