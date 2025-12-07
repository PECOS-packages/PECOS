"""Tests for quantum engine builders in the unified API."""

import pytest
from pecos_rslib import (
    SparseStabilizerEngineBuilder,
    StateVectorEngineBuilder,
    sparse_stab,
    sparse_stabilizer,
    state_vector,
    Qis,
    Qasm,
    depolarizing_noise,
    qasm_engine,
)


class TestQuantumEngineBuilders:
    """Test quantum engine builders and factory functions."""

    def test_factory_functions_exist(self) -> None:
        """Test that factory functions are available."""
        # These should all be callable
        assert callable(state_vector)
        assert callable(sparse_stabilizer)
        assert callable(sparse_stab)

    def test_builder_classes_exist(self) -> None:
        """Test that builder classes are available."""
        # These should be classes
        assert hasattr(StateVectorEngineBuilder, "__name__")
        assert hasattr(SparseStabilizerEngineBuilder, "__name__")

    def test_state_vector_builder(self) -> None:
        """Test creating state vector engine builder."""
        # Using factory function
        builder1 = state_vector()
        assert builder1 is not None

        # Using class directly
        builder2 = StateVectorEngineBuilder()
        assert builder2 is not None

        # Test with qubits
        builder3 = state_vector().qubits(10)
        assert builder3 is not None

    def test_sparse_stabilizer_builder(self) -> None:
        """Test creating sparse stabilizer engine builder."""
        # Using factory function
        builder1 = sparse_stabilizer()
        assert builder1 is not None

        # Using class directly
        builder2 = SparseStabilizerEngineBuilder()
        assert builder2 is not None

        # Test with qubits
        builder3 = sparse_stabilizer().qubits(5)
        assert builder3 is not None

    def test_sparse_stab_alias(self) -> None:
        """Test that sparse_stab is an alias for sparse_stabilizer."""
        builder1 = sparse_stab()
        builder2 = sparse_stabilizer()
        # Both should create the same type of builder
        assert type(builder1) is type(builder2)

    def test_unified_api_with_quantum_engine(self) -> None:
        """Test using quantum engine builders in the unified API."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        # Test with state vector engine
        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .quantum(state_vector())
            .seed(42)
        )
        results = sim.run(100)
        results_dict = results.to_dict()
        assert "c" in results_dict
        assert len(results_dict["c"]) == 100

        # Test with sparse stabilizer engine
        sim2 = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .quantum(sparse_stabilizer())
            .seed(42)
        )
        results2 = sim2.run(100)
        results2_dict = results2.to_dict()
        assert "c" in results2_dict
        assert len(results2_dict["c"]) == 100

    def test_quantum_engine_with_noise(self) -> None:
        """Test using quantum engines with noise models."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """

        # Create noise model with all required probabilities
        noise = depolarizing_noise().with_uniform_probability(0.01)

        # Test with state vector engine and noise
        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .quantum(state_vector())
            .noise(noise)
            .seed(42)
        )
        results = sim.run(1000)
        results_dict = results.to_dict()
        assert "c" in results_dict
        assert len(results_dict["c"]) == 1000

    def test_llvm_with_quantum_engine(self) -> None:
        """Test LLVM engine with quantum engine builders.

        Note: Currently uses sim() API instead of qis_engine().program().to_sim()
        because the builder API doesn't yet have automatic JIT interface selection.
        """
        # Minimal LLVM IR - single qubit H gate and measurement
        # Uses qmain entry point expected by Helios interface
        llvm_ir = """; ModuleID = 'test_module'
source_filename = "test_module"

@str_r0 = constant [3 x i8] c"r0\\00"

declare void @__quantum__qis__h__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare void @__quantum__rt__result_record_output(i64, i8*)

define i64 @qmain(i64 %arg) #0 {
entry:
    call void @__quantum__qis__h__body(i64 0)
    %result = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    call void @__quantum__rt__result_record_output(i64 0, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @str_r0, i32 0, i32 0))
    ret i64 0
}

attributes #0 = { "EntryPoint" }
"""

        try:
            # Import sim directly from pecos_rslib (Rust implementation)
            from pecos_rslib import sim

            # Create QIS program and run with quantum engine
            # Need to specify number of qubits (1 qubit in this test)
            program = Qis.from_string(llvm_ir)
            results = sim(program).qubits(1).quantum(state_vector()).seed(42).run(100)
            results_dict = results.to_dict()

            # Check results - should have roughly 50/50 distribution due to H gate
            # Note: The result key might be "measurement_0" instead of "r0" depending on backend
            result_key = None
            for key in results_dict.keys():
                if "0" in str(key) or "r0" in str(key):
                    result_key = key
                    break

            assert (
                result_key is not None
            ), f"No measurement result found. Keys: {list(results_dict.keys())}"
            assert len(results_dict[result_key]) == 100

            # Count occurrences
            zeros = sum(1 for r in results_dict[result_key] if r == 0)
            ones = sum(1 for r in results_dict[result_key] if r == 1)
            assert zeros + ones == 100
            # With H gate, should get roughly 50/50 split (allow some variance)
            assert 30 < zeros < 70
            assert 30 < ones < 70

        except (RuntimeError, ImportError, AttributeError, OSError) as e:
            # LLVM runtime not available or not working
            # OSError can occur if LLVM shared libraries are missing
            pytest.skip(f"LLVM runtime not available: {type(e).__name__}: {e}")
