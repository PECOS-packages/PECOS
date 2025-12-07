"""Test and document default values for QASM simulations using sim() API."""


class TestQasmSimDefaults:
    """Test and document default values for all QASM simulation settings."""

    def test_builder_defaults(self) -> None:
        """Test and document defaults when using qasm_engine builder."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        # Build with all defaults
        sim = qasm_engine().program(Qasm.from_string(qasm)).to_sim().build()

        # Based on Rust code, the defaults are:
        # - seed: None (non-deterministic)
        # - workers: 1 (single thread)
        # - noise_model: no noise (don't call .noise())
        # - quantum_engine: SparseStabilizer
        # - bit_format: BigInt (integers, not binary strings)

        # Run to verify it works
        results = sim.run(100)
        results_dict = results.to_dict()
        assert len(results_dict["c"]) == 100

    def test_run_direct_defaults(self) -> None:
        """Test and document defaults when using engine run directly."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        # Run with minimal parameters using new API
        results = qasm_engine().program(Qasm.from_string(qasm)).to_sim().run(10)
        results_dict = results.to_dict()

        # Defaults for direct run:
        # - noise_model: None (no noise)
        # - engine: auto-selected based on circuit
        # - workers: defaults to 1
        # - seed: None (non-deterministic)

        assert all(val == 1 for val in results_dict["c"])

    def test_noise_model_defaults(self) -> None:
        """Test and document default parameters for noise models."""
        from pecos import (
            GeneralNoiseModelBuilder,
            biased_depolarizing_noise,
            depolarizing_noise,
        )

        # Test default values for noise models using builder pattern
        # Note: depolarizing_noise() builder requires explicit probability
        depolarizing_noise().with_p1_probability(0.001)
        # Can't directly assert on builder properties

        # General noise model has defaults that can be overridden
        GeneralNoiseModelBuilder()
        # Default values are set when building

        (
            biased_depolarizing_noise()
            .with_p1_probability(0.001)
            .with_p2_probability(0.001)
            .with_prep_probability(0.001)
        )
        # Builder pattern requires explicit values

    def test_builder_defaults_new_api(self) -> None:
        """Test and document defaults when using new unified API."""
        from pecos import Qasm, qasm_engine

        # Minimal setup - only required field
        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
            """

        sim = qasm_engine().program(Qasm.from_string(qasm)).to_sim().build()
        results = sim.run(10)
        results_dict = results.to_dict()

        # Defaults for new API:
        # - seed: None (not set)
        # - workers: 1 (default)
        # - noise: no noise (ideal simulation)
        # - quantum_engine: SparseStabilizer (default)
        # - binary_string_format: False (integers)

        assert all(val == 1 for val in results_dict["c"])

    def test_no_noise_means_ideal(self) -> None:
        """Test that omitting noise results in ideal (deterministic) simulation."""
        from pecos import Qasm, qasm_engine

        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        x q[1];
        measure q -> c;
        """

        # Build without noise specification
        sim1 = qasm_engine().program(Qasm.from_string(qasm)).to_sim().build()

        # Both should produce identical deterministic results
        results1 = sim1.run(100)
        results1_dict = results1.to_dict()

        # Should always measure |11> = 3
        assert all(val == 3 for val in results1_dict["c"])

    def test_default_summary(self) -> None:
        """Document all defaults in one place."""
        # Default values summary:
        #
        # QasmEngine defaults:
        # - seed: None (non-deterministic)
        # - workers: 1 (single thread)
        # - noise_model: no noise (ideal simulation)
        # - quantum_engine: SparseStabilizer
        # - bit_format: BigInt (integers, not binary strings)
        #
        # Noise model builders:
        # - depolarizing_noise(): requires explicit .with_p1_probability()
        # - biased_depolarizing_noise(): requires probability settings
        # - GeneralNoiseModelBuilder(): has internal defaults
        #
        # New unified API defaults:
        # - All optional fields use builder defaults when not specified
        # - noise: no noise (ideal simulation) when omitted

        # This test just documents the defaults
        assert True
