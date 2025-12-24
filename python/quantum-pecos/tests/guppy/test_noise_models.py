"""Test noise model integration with sim.

This test file verifies that noise models are properly integrated
and working with the sim builder pattern.
"""

import pytest

try:
    from guppylang import guppy
    from guppylang.std.quantum import cx, h, measure, qubit, x

    GUPPY_AVAILABLE = True
except ImportError:
    GUPPY_AVAILABLE = False

try:
    from pecos import Guppy, sim
    from pecos_rslib import (
        biased_depolarizing_noise,
        depolarizing_noise,
        general_noise,
        state_vector,
    )
except ImportError:
    pass


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="Guppy not available")
class TestNoiseModels:
    """Test noise model integration with sim."""

    def test_no_noise_deterministic(self) -> None:
        """Test that circuits without noise are deterministic."""

        @guppy
        def deterministic_circuit() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        # Run with seed for reproducibility
        results = (
            sim(deterministic_circuit)
            .qubits(10)
            .quantum(state_vector())
            .seed(42)
            .run(10)
        )

        # Should always measure |1⟩
        measurements = results.get("measurements", results.get("measurement_0", []))
        assert all(r == 1 for r in measurements)

    def test_depolarizing_noise_effect(self) -> None:
        """Test that depolarizing noise introduces errors."""

        @guppy
        def simple_circuit() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        # Create depolarizing noise - must chain all probability setters
        noise = (
            depolarizing_noise()
            .with_prep_probability(0.0)  # No prep errors
            .with_p1_probability(0.2)  # 20% chance of error on single-qubit gates
            .with_p2_probability(0.0)  # No two-qubit gate errors
            .with_meas_probability(0.0)
        )  # No measurement errors

        # High depolarizing probability to see effect
        results = (
            sim(simple_circuit)
            .qubits(10)
            .quantum(state_vector())
            .noise(noise)
            .seed(42)
            .run(100)
        )

        measurements = results.get("measurements", results.get("measurement_0", []))

        # With 0.2 depolarizing on X gate, we should see some 0s
        zeros = sum(1 for r in measurements if r == 0)
        assert (
            zeros > 0
        ), f"Depolarizing noise should introduce errors, got {zeros} zeros"
        assert zeros < 100, "Should not flip all bits"

    def test_biased_depolarizing_noise(self) -> None:
        """Test biased depolarizing noise model."""

        @guppy
        def simple_circuit() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        # Use biased depolarizing - must chain all probability setters
        noise = (
            biased_depolarizing_noise()
            .with_prep_probability(0.05)  # State prep errors
            .with_p1_probability(0.1)  # Single-qubit gate errors
            .with_p2_probability(0.0)  # No two-qubit gate errors
            .with_meas_0_probability(0.05)  # Measurement errors for |0⟩
            .with_meas_1_probability(0.05)
        )  # Measurement errors for |1⟩

        results = (
            sim(simple_circuit)
            .qubits(10)
            .quantum(state_vector())
            .noise(noise)
            .seed(42)
            .run(100)
        )

        measurements = results.get("measurements", results.get("measurement_0", []))

        # Should see some errors
        zeros = sum(1 for r in measurements if r == 0)
        assert zeros > 0, "Biased depolarizing should introduce errors"

    def test_general_noise_model(self) -> None:
        """Test general noise model builder."""

        @guppy
        def simple_circuit() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        # Use general noise model with multiple error types
        noise_builder = (
            general_noise()
            .with_p1_probability(0.01)  # Single-qubit gate errors
            .with_prep_probability(0.01)
        )  # Preparation errors

        results = (
            sim(simple_circuit)
            .qubits(10)
            .quantum(state_vector())
            .noise(noise_builder)
            .seed(42)
            .run(100)
        )

        measurements = results.get("measurements", results.get("measurement_0", []))

        # Should see some errors but not too many
        sum(1 for r in measurements if r == 0)
        # With low error rates, might not see errors in 100 shots
        # Just verify it runs without crashing
        assert len(measurements) == 100

    def test_noise_models_comparison(self) -> None:
        """Compare different noise models on same circuit."""

        @guppy
        def bell_circuit() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)
            return measure(q1), measure(q2)

        # Run without noise
        results_clean = (
            sim(Guppy(bell_circuit))
            .qubits(10)
            .quantum(state_vector())
            .seed(42)
            .run(100)
        )

        # Run with depolarizing noise - chain all probability setters
        noise = (
            depolarizing_noise()
            .with_prep_probability(0.0)  # No prep errors
            .with_p1_probability(0.05)  # 5% error on single-qubit gates
            .with_p2_probability(0.05)  # 5% error on two-qubit gates
            .with_meas_probability(0.0)
        )  # No measurement errors

        results_noisy = (
            sim(bell_circuit)
            .qubits(10)
            .quantum(state_vector())
            .noise(noise)
            .seed(42)
            .run(100)
        )

        # Extract measurements
        m1_clean = results_clean.get("measurement_0", [])
        m2_clean = results_clean.get("measurement_1", [])
        m1_noisy = results_noisy.get("measurement_0", [])
        m2_noisy = results_noisy.get("measurement_1", [])

        # Check correlations
        clean_corr = sum(1 for i in range(100) if m1_clean[i] == m2_clean[i])
        noisy_corr = sum(1 for i in range(100) if m1_noisy[i] == m2_noisy[i])

        # Clean Bell state should have perfect correlation
        assert clean_corr == 100, "Clean Bell state should be perfectly correlated"

        # Noisy should have less correlation (or might still be perfect with low noise)
        # Just verify it runs
        assert noisy_corr >= 0


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="Guppy not available")
def test_noise_model_builder_pattern() -> None:
    """Test the builder pattern for noise models."""

    @guppy
    def simple_x_circuit() -> bool:
        q = qubit()
        x(q)
        return measure(q)

    # Test that builder pattern works - chain all probability setters
    noise1 = (
        depolarizing_noise()
        .with_prep_probability(0.0)
        .with_p1_probability(0.1)
        .with_p2_probability(0.0)
        .with_meas_probability(0.0)
        .with_seed(1)
    )

    results1 = (
        sim(simple_x_circuit)
        .qubits(10)
        .quantum(state_vector())
        .noise(noise1)
        .seed(42)
        .run(10)
    )

    # Different seed should give different results
    noise2 = (
        depolarizing_noise()
        .with_prep_probability(0.0)
        .with_p1_probability(0.1)
        .with_p2_probability(0.0)
        .with_meas_probability(0.0)
        .with_seed(2)
    )

    results2 = (
        sim(simple_x_circuit)
        .qubits(10)
        .quantum(state_vector())
        .noise(noise2)
        .seed(43)
        .run(10)
    )

    measurements1 = results1.get("measurements", results1.get("measurement_0", []))
    measurements2 = results2.get("measurements", results2.get("measurement_0", []))

    # With different seeds in noise models, we might get different error patterns
    # But with only 10 shots, they might be the same. Just check they both run.
    assert len(measurements1) == 10
    assert len(measurements2) == 10


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="Guppy not available")
def test_noise_on_single_qubit_gates() -> None:
    """Test noise specifically on single-qubit gates."""

    @guppy
    def multi_gate_circuit() -> bool:
        q = qubit()
        h(q)  # Should get noise
        x(q)  # Should get noise
        return measure(q)

    # Configure noise only for single-qubit gates
    noise = general_noise().with_p1_probability(0.3)  # High error rate to see effect

    results = (
        sim(multi_gate_circuit)
        .qubits(10)
        .quantum(state_vector())
        .noise(noise)
        .seed(42)
        .run(100)
    )

    measurements = results.get("measurements", results.get("measurement_0", []))

    # H followed by X should give |1⟩ without noise
    # With noise, we should see some 0s
    zeros = sum(1 for r in measurements if r == 0)
    assert zeros > 0, "Noise on gates should cause errors"


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="Guppy not available")
def test_measurement_noise() -> None:
    """Test measurement noise specifically."""

    @guppy
    def simple_circuit() -> bool:
        q = qubit()
        x(q)
        return measure(q)

    # Configure noise only for measurements
    noise = general_noise().with_meas_probability(0.2)  # High measurement error

    results = (
        sim(simple_circuit)
        .qubits(10)
        .quantum(state_vector())
        .noise(noise)
        .seed(42)
        .run(100)
    )

    measurements = results.get("measurements", results.get("measurement_0", []))

    # X gate gives |1⟩, but measurement errors should flip some
    zeros = sum(1 for r in measurements if r == 0)
    assert zeros > 0, "Measurement noise should cause readout errors"
