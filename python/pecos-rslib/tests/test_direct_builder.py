"""Test direct GeneralNoiseModelBuilder usage."""

from collections import Counter

import pytest
from pecos_rslib import (
    GeneralNoiseModelBuilder,
    Qasm,
)
from pecos_rslib import sim


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
            .with_p1(0.001)
            .with_p2(0.01)
            .with_p_meas_0(0.002)
            .with_p_meas_1(0.002)
        )

        # Use sim() with noise builder
        prog = Qasm.from_string(qasm)
        results = sim(prog).noise(builder).run(1000).to_dict()

        assert len(results["c"]) == 1000
        counts = Counter(results["c"])
        # Should see Bell state results (0 and 3) with some noise errors
        assert 0 in counts
        assert 3 in counts

    def test_noise_accepts_parameter_conversion_protocol(self) -> None:
        """Portable parameter objects are converted when attached to ``sim()``."""

        class Parameters:
            def __init__(self) -> None:
                self.converted = False

            def __pecos_noise_builder__(self) -> GeneralNoiseModelBuilder:
                self.converted = True
                return GeneralNoiseModelBuilder().with_seed(17).with_p_meas_0(1.0)

        program = Qasm.from_string("""
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
            """)
        parameters = Parameters()

        simulation = sim(program).noise(parameters)

        assert parameters.converted
        assert simulation.run(4).to_dict()["c"] == [1] * 4

    def test_noise_propagates_parameter_conversion_errors(self) -> None:
        """A conversion error reaches the caller without being obscured."""

        class InvalidParameters:
            def __pecos_noise_builder__(self) -> GeneralNoiseModelBuilder:
                message = "portable noise configuration is invalid"
                raise ValueError(message)

        program = Qasm.from_string("OPENQASM 2.0; qreg q[1];")

        with pytest.raises(ValueError, match="portable noise configuration is invalid"):
            sim(program).noise(InvalidParameters())

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
            .with_p1(0.1)  # High error rate for testing
            .with_p1_pauli_model({"X": 0.5, "Y": 0.3, "Z": 0.2})
        )

        prog = Qasm.from_string(qasm)
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

        prog = Qasm.from_string(qasm)

        # Create builder with fluent API
        builder = GeneralNoiseModelBuilder().with_seed(42).with_p2(0.01)

        # Use sim() with direct method chaining
        results = sim(prog).seed(42).noise(builder).run(100).to_dict()

        assert len(results["c"]) == 100
        # Results are integers, not binary strings in the new API
        assert all(isinstance(val, int) for val in results["c"])

    def test_builder_chaining_validation(self) -> None:
        """Test that builder methods validate parameters."""
        # Setter validation reports ordinary Python ValueError.
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            GeneralNoiseModelBuilder().with_p1(1.5)

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

        prog = Qasm.from_string(qasm)

        # Create builder
        builder = GeneralNoiseModelBuilder()
        builder.with_seed(42)
        builder.with_p1(0.001)
        builder.with_p2(0.01)

        # Test that builder can be used directly in .noise() method
        results = sim(prog).noise(builder).seed(42).run(100).to_dict()

        assert len(results["c"]) == 100
        counts = Counter(results["c"])
        assert 0 in counts or 3 in counts  # Bell state results


@pytest.mark.parametrize(
    ("setter", "args", "message"),
    [
        ("with_p1", (1.5,), "Probability must be between 0 and 1"),
        ("with_leakage_scale", (1.5,), "Probability must be between 0 and 1"),
        ("with_seepage_prob", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_prep", (1.5,), "Probability must be between 0 and 1"),
        ("with_prep_leak_ratio", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_prep_crosstalk", (1.5,), "Probability must be between 0 and 1"),
        ("with_p1_emission_ratio", (1.5,), "Probability must be between 0 and 1"),
        ("with_p1_seepage_prob", (1.5,), "Probability must be between 0 and 1"),
        ("with_p2", (1.5,), "Probability must be between 0 and 1"),
        ("with_p2_emission_ratio", (1.5,), "Probability must be between 0 and 1"),
        ("with_p2_seepage_prob", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_meas_0", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_meas_1", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_meas", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_meas_crosstalk_global", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_meas_crosstalk_local", (1.5,), "Probability must be between 0 and 1"),
        ("with_p_meas_crosstalk", (1.5,), "Probability must be between 0 and 1"),
        ("with_average_p1", (1.0,), "Probability must be between 0 and 1"),
        ("with_average_p2", (1.0,), "Probability must be between 0 and 1"),
        (
            "with_measurement_probability",
            (0.2, 1.5),
            "Probability must be between 0 and 1",
        ),
        ("with_p_meas_crosstalk_scale", (-1.0,), "must be non-negative"),
        ("with_p_prep_crosstalk_scale", (-1.0,), "must be non-negative"),
        ("with_p1_pauli_model", ({"bad": 1.0},), "invalid key"),
        ("with_p1_emission_model", ({"X": 2.0},), "total weight 2"),
        ("with_p2_pauli_model", ({"XA": 1.0},), "invalid character"),
        (
            "with_p2_emission_model",
            ({"XI": -1.0},),
            "weights must be finite and non-negative",
        ),
        (
            "with_p_meas_crosstalk_model",
            ({"0->1": 1.0},),
            "weighted_map cannot be empty",
        ),
        (
            "with_p_idle_linear",
            (-1.0, {"X": 1.0}),
            "linear idling rate must be non-negative",
        ),
        ("with_p_idle_sin_squared", (0.1, {"A": 1.0}), "invalid key"),
        (
            "with_p_idle_coherent",
            (0.1, {"RX": -1.0}),
            "multiplier.*must be finite and non-negative",
        ),
        ("with_p2_angle_power", (0.0,), "must be positive"),
        ("with_idle_after_2q", (-1.0,), "Duration must be finite and non-negative"),
    ],
)
def test_invalid_setters_raise_value_error_at_call(setter, args, message) -> None:
    builder = GeneralNoiseModelBuilder()
    with pytest.raises(ValueError, match=message):
        getattr(builder, setter)(*args)
    # A rejected immutable update does not poison the original builder.
    builder.with_config_json("{}")


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ("{", "EOF while parsing an object"),
        ('{"unknown": 1}', "unknown field `unknown`"),
        (
            '{"single_qubit": {"probability": 1.5}}',
            "Probability must be between 0 and 1",
        ),
        ('{"measurement": {"local_groups": [[0]]}}', "unknown field `local_groups`"),
        (
            '{"preparation": {"average_crosstalk_probability": 0.3, "scale": 0.01}}',
            "average preparation crosstalk probability must be between 0 and 5/18",
        ),
    ],
)
def test_config_json_rejects_invalid_input(configuration: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GeneralNoiseModelBuilder().with_config_json(configuration)


def test_config_json_composes_immutably_and_defers_cross_field_validation() -> None:
    original = GeneralNoiseModelBuilder().with_p_meas(1.0)
    configured = original.with_config_json('{"measurement": {"p0_to_1": 0.0}}')
    program = Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; measure q -> c;')
    assert sim(program).noise(original.with_config_json("{}")).run(2).to_dict()["c"] == [1, 1]
    assert sim(program).noise(configured).run(2).to_dict()["c"] == [0, 0]
    intermediate = GeneralNoiseModelBuilder().with_p1(0.8).with_config_json('{"scaling": {"overall": 2.0}}')
    repaired = intermediate.with_p1_scale(0.5)
    sim(program).noise(repaired).run(1)
    with pytest.raises(ValueError, match="scaled one-qubit probability"):
        sim(program).noise(intermediate).run(1)
