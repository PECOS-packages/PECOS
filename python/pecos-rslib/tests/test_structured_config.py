"""Test structured configuration for sim() with direct method chaining."""

from collections import Counter

import pytest
from pecos_rslib import (
    biased_depolarizing_noise,
    depolarizing_noise,
    general_noise,
)
from pecos_rslib.programs import Qasm
from pecos_rslib import sim


class TestDirectMethodChaining:
    """Test the direct method chaining configuration approach."""

    def test_general_noise_model_builder_basic(self) -> None:
        """Test basic general_noise() usage."""
        noise = (
            general_noise()
            .with_seed(42)
            .with_p1_probability(0.001)
            .with_p2_probability(0.01)
            .with_meas_0_probability(0.002)
            .with_meas_1_probability(0.002)
        )

        # The noise object is already a builder, can be used directly
        # Test that it's a valid builder by checking it has builder methods
        assert hasattr(noise, "with_seed")
        assert hasattr(noise, "with_p1_probability")

    def test_general_noise_model_builder_validation(self) -> None:
        """Test general_noise() parameter validation."""
        builder = general_noise()

        # Test invalid probability values
        # Rust panics raise BaseException
        with pytest.raises(BaseException, match=r".*"):  # Rust panic - any error message
            builder.with_p1_probability(-0.1)  # Negative probability

        builder = general_noise()
        with pytest.raises(BaseException, match=r".*"):  # Rust panic - any error message
            builder.with_p2_probability(1.5)  # > 1 probability

    def test_direct_noise_builder_with_sim(self) -> None:
        """Test using builders directly with sim()."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        prog = Qasm.from_string(qasm)

        # Create a configured noise builder
        noise = general_noise().with_seed(42).with_p1_probability(0.001).with_p2_probability(0.01)

        # Use the builder directly with sim()
        results = sim(prog).noise(noise).run(1000).to_dict()

        assert "c" in results
        assert len(results["c"]) == 1000

        # Check for Bell state with some noise
        counts = Counter(results["c"])
        assert 0 in counts  # 00
        assert 3 in counts  # 11

    def test_depolarizing_noise_builder(self) -> None:
        """Test depolarizing_noise() function."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        prog = Qasm.from_string(qasm)

        # Create builder with specific config
        noise = depolarizing_noise().with_seed(42).with_uniform_probability(0.1)

        results = sim(prog).seed(42).noise(noise).run(1000).to_dict()

        # Should see some errors with 10% error rate
        zeros = sum(1 for val in results["c"] if val == 0)
        assert 50 < zeros < 200

    def test_biased_depolarizing_builder(self) -> None:
        """Test biased_depolarizing_noise() function."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """

        prog = Qasm.from_string(qasm)

        # Create builder with uniform probability
        noise = biased_depolarizing_noise().with_seed(42).with_uniform_probability(0.05)

        results = sim(prog).noise(noise).run(1000).to_dict()

        assert "c" in results
        assert len(results["c"]) == 1000

    def test_complex_circuit_with_noise(self) -> None:
        """Test more complex circuit with noise."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];

        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];

        measure q -> c;
        """

        prog = Qasm.from_string(qasm)

        # Configure general noise with specific parameters
        noise = (
            general_noise()
            .with_seed(123)
            .with_p1_probability(0.005)
            .with_p2_probability(0.02)
            .with_meas_0_probability(0.01)
            .with_meas_1_probability(0.01)
        )

        results = sim(prog).noise(noise).run(1000).to_dict()

        counts = Counter(results["c"])

        # Should see mostly GHZ states (000 and 111) with some errors
        assert 0 in counts  # 000
        assert 7 in counts  # 111

        # But also some error states due to noise
        error_states = [k for k in counts.keys() if k not in [0, 7]]
        assert len(error_states) > 0
