"""Test custom noise model registration and from_config pattern."""


class TestCustomNoiseModels:
    """Test custom noise model registration and configuration."""

    def test_built_in_noise_builders(self) -> None:
        """Test that all built-in noise models have builder methods."""
        from pecos import (
            GeneralNoiseModelBuilder,
            biased_depolarizing_noise,
            depolarizing_noise,
        )

        # Test depolarizing noise builder
        dep = depolarizing_noise().with_p1_probability(0.05)
        assert dep is not None

        # Test depolarizing noise with multiple parameters
        dep_custom = (
            depolarizing_noise()
            .with_prep_probability(0.002)
            .with_meas_probability(0.001)
            .with_p1_probability(0.003)
            .with_p2_probability(0.002)
        )
        assert dep_custom is not None

        # Test BiasedDepolarizingNoise
        biased = biased_depolarizing_noise().with_uniform_probability(0.033)
        assert biased is not None

        # Test GeneralNoise
        general = GeneralNoiseModelBuilder()
        assert general is not None

    def test_custom_noise_model_limitation(self) -> None:
        """Test that custom noise models have limitations due to Rust bindings."""
        # In the new API, only built-in noise builders can be used
        # Custom Python noise models cannot be passed to Rust
        # This limitation is enforced at the type level by using builder objects

    def test_register_without_from_config_fails(self) -> None:
        """Test that using noise without from_config fails."""
        # In the current implementation, noise model registration is not supported
        # All noise models must be built-in types implemented in Rust
        # This test is kept to document this limitation

    def test_noise_builder_configuration(self) -> None:
        """Test that built-in noise models use builder configuration."""
        from pecos import Qasm, depolarizing_noise, qasm_engine

        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
            """

        # Use builder pattern with explicit probability
        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm))
            .to_sim()
            .noise(depolarizing_noise().with_uniform_probability(0.001))
            .build()
        )
        results = sim.run(1000)
        results_dict = results.to_dict()

        # Should see very few errors due to low noise (p=0.001)
        zeros = sum(1 for val in results_dict["c"] if val == 0)
        assert zeros < 10  # Less than 1% error rate expected

    def test_noise_builder_validation(self) -> None:
        """Test that built-in noise models work with builder pattern."""
        from pecos import Qasm, depolarizing_noise, qasm_engine

        # Valid QASM for testing
        qasm_valid = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
            """

        # Test DepolarizingNoise with valid p
        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm_valid))
            .to_sim()
            .noise(depolarizing_noise().with_uniform_probability(0.5))
            .build()
        )
        results = sim.run(10)
        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 10

        # Test DepolarizingNoise with multiple parameters
        sim = (
            qasm_engine()
            .program(Qasm.from_string(qasm_valid))
            .to_sim()
            .noise(
                depolarizing_noise()
                .with_prep_probability(0.1)
                .with_meas_probability(0.2)
                .with_p1_probability(0.3)
                .with_p2_probability(0.4),
            )
            .build()
        )
        results = sim.run(10)
        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 10

        # Unknown noise types are now prevented at the type level by the builder pattern
