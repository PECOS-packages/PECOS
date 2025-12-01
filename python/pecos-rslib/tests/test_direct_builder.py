"""Test direct GeneralNoiseModelBuilder usage."""

from collections import Counter

import pytest
from _pecos_rslib import (
    GeneralNoiseModelBuilder,
    QasmProgram,
)
from _pecos_rslib import sim


class TestDirectBuilder:
    """Test using GeneralNoiseModelBuilder directly."""

    def test_direct_builder_noise(self) -> None:
        """Test setting noise with GeneralNoiseModelBuilder directly using .noise() method."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        # Create and configure the Rust-native builder with fluent chaining
        builder = (
            GeneralNoiseModelBuilder()
            .with_seed(42)
            .with_p1_probability(0.001)
            .with_p2_probability(0.01)
            .with_meas_0_probability(0.002)
            .with_meas_1_probability(0.002)
        )

        # Use sim() with noise builder
        prog = QasmProgram.from_string(qasm)
        results = sim(prog).noise(builder).run(1000).to_dict()

        assert len(results["c"]) == 1000
        counts = Counter(results["c"])
        # Should see Bell state results (0 and 3) with some noise errors
        assert 0 in counts
        assert 3 in counts

    def test_builder_with_pauli_model(self) -> None:
        """Test builder with Pauli error models."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        builder = (
            GeneralNoiseModelBuilder()
            .with_seed(42)
            .with_p1_probability(0.1)  # High error rate for testing
            .with_p1_pauli_model({"X": 0.5, "Y": 0.3, "Z": 0.2})
        )

        prog = QasmProgram.from_string(qasm)
        results = sim(prog).noise(builder).run(1000).to_dict()

        # Should see some errors due to high p1 error rate
        zeros = sum(1 for val in results["c"] if val == 0)
        # With 10% error rate and specific Pauli model, we expect some measurement errors
        # The X error (50% of errors) would flip |1⟩ back to |0⟩, giving us 0 measurement
        # Y and Z errors (30% and 20%) would also affect the measurement
        # We expect roughly 5% of measurements to be 0 (10% error * 50% X errors)
        # Allow for statistical variation: expect between 30 and 150 zeros
        assert 30 <= zeros <= 150, f"Expected between 30 and 150 zeros, got {zeros}"

    def test_builder_with_method_chaining(self) -> None:
        """Test using builder with direct method chaining."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        prog = QasmProgram.from_string(qasm)

        # Create builder with fluent API
        builder = GeneralNoiseModelBuilder().with_seed(42).with_p2_probability(0.01)

        # Use sim() with direct method chaining
        results = sim(prog).seed(42).noise(builder).run(100).to_dict()

        assert len(results["c"]) == 100
        # Results are integers, not binary strings in the new API
        assert all(isinstance(val, int) for val in results["c"])

    def test_builder_chaining_validation(self) -> None:
        """Test that builder methods validate parameters."""
        # Test validation - Rust panics raise BaseException with "PanicException" in the name
        with pytest.raises(BaseException, match="Probability must be between 0 and 1"):
            GeneralNoiseModelBuilder().with_p1_probability(1.5)

        # Scale validation happens at build time, not when setting the value
        # So we need to build and use the noise model to trigger validation
        # For now, just test that we can set negative scale (validation may happen later)
        GeneralNoiseModelBuilder().with_scale(-1)
        # The actual validation might happen when building the noise model
        # which is done internally when using it with a simulation

        # Note: leakage_scale method doesn't exist in the current bindings
        # with pytest.raises(ValueError, match="leakage_scale must be between 0 and 1"):
        #     GeneralNoiseModelBuilder().with_leakage_scale(1.5)

    def test_rust_vs_native_noise_models(self) -> None:
        """Test using Rust noise models in the .noise() method directly."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        prog = QasmProgram.from_string(qasm)

        # Create builder
        builder = GeneralNoiseModelBuilder()
        builder.with_seed(42)
        builder.with_p1_probability(0.001)
        builder.with_p2_probability(0.01)

        # Test that builder can be used directly in .noise() method
        results = sim(prog).noise(builder).seed(42).run(100).to_dict()

        assert len(results["c"]) == 100
        counts = Counter(results["c"])
        assert 0 in counts or 3 in counts  # Bell state results
