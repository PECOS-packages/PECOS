"""Tests for the modern sim() API."""

import pytest
from pecos_rslib import (
    biased_depolarizing_noise,
    depolarizing_noise,
    general_noise,
    qasm_engine,
    sparse_stabilizer,
    state_vector,
)
from pecos_rslib.programs import Qasm
from pecos_rslib import sim


class TestSimAPI:
    """Test the modern sim() API for QASM simulations."""

    def test_basic_simulation(self) -> None:
        """Test basic QASM simulation with sim() API."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        x q[1];
        measure q -> c;
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)
        results = sim(program).classical(engine).run(10).to_dict()

        # Both qubits should be 1, so c should be 3
        assert "c" in results
        assert all(val == 3 for val in results["c"])

    def test_deterministic_simulation(self) -> None:
        """Test deterministic QASM simulation using seed parameter."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)

        # Run with same seed should give same results
        results1 = sim(program).classical(engine).seed(42).run(100).to_dict()
        results2 = sim(program).classical(engine).seed(42).run(100).to_dict()

        assert results1["c"] == results2["c"]

        # Different seed should give different results (with high probability)
        results3 = sim(program).classical(engine).seed(123).run(100).to_dict()
        # This might fail with very low probability
        assert results1["c"] != results3["c"]

    def test_quantum_engines(self) -> None:
        """Test different quantum engines."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        cnot q[0], q[1];
        measure q -> c;
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)

        # Test with StateVector engine
        results_sv = (
            sim(program).classical(engine).quantum(state_vector()).run(10).to_dict()
        )
        assert all(val == 3 for val in results_sv["c"])  # Both qubits should be 1

        # Test with SparseStabilizer engine
        results_stab = (
            sim(program)
            .classical(engine)
            .quantum(sparse_stabilizer())
            .run(10)
            .to_dict()
        )
        assert all(val == 3 for val in results_stab["c"])  # Both qubits should be 1

    def test_noise_models(self) -> None:
        """Test different noise models."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)

        # Test with no noise - should always measure 1
        results_no_noise = sim(program).classical(engine).run(100).to_dict()
        assert all(val == 1 for val in results_no_noise["c"])

        # Test with depolarizing noise
        noise = depolarizing_noise().with_uniform_probability(0.1)
        results_with_noise = (
            sim(program).classical(engine).noise(noise).seed(42).run(1000).to_dict()
        )

        # With noise, we should sometimes get 0
        ones = sum(results_with_noise["c"])
        zeros = len(results_with_noise["c"]) - ones
        assert zeros > 0  # Should have some errors
        assert ones > zeros  # But most should still be correct

    def test_biased_depolarizing_noise(self) -> None:
        """Test biased depolarizing noise model."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)

        # Test with biased depolarizing noise
        noise = biased_depolarizing_noise().with_uniform_probability(0.05)
        results = (
            sim(program).classical(engine).noise(noise).seed(42).run(1000).to_dict()
        )

        # Should have some errors but mostly correct
        ones = sum(results["c"])
        assert ones > 900  # Most should be correct
        assert ones < 1000  # But some errors

    def test_general_noise_model(self) -> None:
        """Test general noise model."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)

        # Test with general noise model
        noise = general_noise()
        results = sim(program).classical(engine).noise(noise).run(100).to_dict()

        # General noise model may introduce errors even without explicit configuration
        # Just check that we get results
        assert "c" in results
        assert len(results["c"]) == 100

    def test_error_handling(self) -> None:
        """Test error handling for invalid inputs."""
        # Invalid QASM should raise an error
        program = Qasm.from_string("invalid qasm")
        engine = qasm_engine().program(program)
        with pytest.raises((RuntimeError, ValueError)):
            sim(program).classical(engine).run(10).to_dict()

    def test_multiple_registers(self) -> None:
        """Test simulation with multiple classical registers."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c1[1];
        creg c2[1];
        x q[0];
        x q[1];
        measure q[0] -> c1[0];
        measure q[1] -> c2[0];
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)
        results = sim(program).classical(engine).run(10).to_dict()

        # Both registers should measure 1
        assert "c1" in results
        assert "c2" in results
        assert all(val == 1 for val in results["c1"])
        assert all(val == 1 for val in results["c2"])

    def test_large_circuit(self) -> None:
        """Test simulation of a larger circuit."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        creg c[5];

        // Create GHZ state
        h q[0];
        cnot q[0], q[1];
        cnot q[1], q[2];
        cnot q[2], q[3];
        cnot q[3], q[4];

        measure q -> c;
        """

        program = Qasm.from_string(qasm)
        engine = qasm_engine().program(program)
        results = sim(program).classical(engine).seed(42).run(100).to_dict()

        # Should get either all 0s or all 1s (GHZ state)
        for val in results["c"]:
            assert val == 0 or val == 31  # 0b00000 or 0b11111
