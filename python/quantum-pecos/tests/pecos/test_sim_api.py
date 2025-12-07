"""Test the new sim(program) API."""

from pecos_rslib import (
    Qasm,
    Qis,
    depolarizing_noise,
    qasm_engine,
    sim,
    sparse_stabilizer,
    state_vector,
)


def test_sim_with_qasm_program() -> None:
    """Test sim() with QASM program auto-detection."""
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h q[0];
    measure q[0] -> c[0];
    """

    # Test auto-detection
    results = sim(Qasm.from_string(qasm_code)).run(100)
    assert len(results) == 100

    # Test with configuration
    results = sim(Qasm.from_string(qasm_code)).seed(42).workers(2).run(100)
    assert len(results) == 100

    # Test with noise
    noise_model = depolarizing_noise().with_uniform_probability(0.01)
    results = sim(Qasm.from_string(qasm_code)).noise(noise_model).run(100)
    assert len(results) == 100

    # Test with quantum engine selection
    results = sim(Qasm.from_string(qasm_code)).quantum(state_vector()).run(100)
    assert len(results) == 100


def test_sim_with_llvm_program() -> None:
    """Test sim() with LLVM program auto-detection."""
    llvm_ir = """define void @main() #0 {
  %qubit = call i64 @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__h__body(i64 %qubit)
  %result = call i32 @__quantum__qis__m__body(i64 %qubit, i64 0)
  ret void
}

declare i64 @__quantum__rt__qubit_allocate()
declare void @__quantum__qis__h__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)

attributes #0 = { "EntryPoint" }"""

    # Test auto-detection
    results = sim(Qis.from_string(llvm_ir)).qubits(1).run(100)
    assert len(results) == 100


def test_sim_with_explicit_engine_override() -> None:
    """Test overriding auto-selected engine with classical()."""
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h q[0];
    measure q[0] -> c[0];
    """

    # Override with custom engine configuration
    # (Note: without actual WASM file this would fail, so we just test the API)
    builder = sim(Qasm.from_string(qasm_code)).classical(
        qasm_engine().program(Qasm.from_string(qasm_code)),
    )

    # This verifies the API works, even if execution would fail without WASM
    results = builder.run(100)
    assert len(results) == 100


def test_sim_with_different_quantum_engines() -> None:
    """Test sim() with different quantum engine backends."""
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """

    # State vector backend
    results_sv = sim(Qasm.from_string(qasm_code)).quantum(state_vector()).run(100)
    assert len(results_sv) == 100

    # Sparse stabilizer backend (only works for Clifford circuits)
    results_ss = sim(Qasm.from_string(qasm_code)).quantum(sparse_stabilizer()).run(100)
    assert len(results_ss) == 100


def test_sim_builder_chaining() -> None:
    """Test that all builder methods can be chained."""
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h q[0];
    measure q[0] -> c[0];
    """

    results = (
        sim(Qasm.from_string(qasm_code))
        .seed(12345)
        .workers(4)
        .noise(depolarizing_noise().with_uniform_probability(0.001))
        .quantum(state_vector())
        .qubits(1)
        .run(100)
    )

    assert len(results) == 100
