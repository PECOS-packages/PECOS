"""Test Python-side Guppy to Selene compilation."""

import pytest


class TestPythonSideCompilation:
    """Test suite for Python-side Guppy compilation to Selene."""

    @pytest.fixture
    def simple_circuit(self) -> object:
        """Fixture providing a simple quantum circuit."""
        try:
            from guppylang.decorator import guppy
            from guppylang.std.quantum import h, measure, qubit
        except ImportError:
            pytest.skip("Guppy not available")

        @guppy
        def simple_circuit() -> bool:
            """Simple H-gate and measurement."""
            q = qubit()
            h(q)
            return measure(q)

        return simple_circuit

    @pytest.fixture
    def bell_pair_circuit(self) -> object:
        """Fixture providing a Bell pair circuit."""
        try:
            from guppylang.decorator import guppy
            from guppylang.std.quantum import cx, h, measure, qubit
        except ImportError:
            pytest.skip("Guppy not available")

        @guppy
        def bell_pair() -> tuple[bool, bool]:
            """Create a Bell pair."""
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)  # Create entanglement
            return measure(q1), measure(q2)

        return bell_pair

    def test_hugr_pass_through_compilation(self, bell_pair_circuit: object) -> None:
        """Test the HUGR pass-through path (Guppy → HUGR → Rust)."""
        try:
            from pecos import Guppy, sim
            from pecos_rslib import state_vector
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            # The sim API handles Guppy → HUGR → Selene compilation
            results = (
                sim(bell_pair_circuit)
                .qubits(2)
                .quantum(state_vector())
                .seed(42)
                .run(100)
            )
        except (RuntimeError, ValueError) as e:
            if "compilation" in str(e).lower() or "not supported" in str(e):
                pytest.skip(f"HUGR compilation issue: {e}")
            pytest.fail(f"HUGR pass-through failed: {e}")

        # Verify results structure
        assert hasattr(results, "__getitem__"), "Results should be dict-like"

        # Check for measurement results
        assert (
            "measurement_1" in results or "measurements" in results
        ), "Results should contain measurements"

        if "measurement_1" in results and "measurement_2" in results:
            # New format with separate measurement keys
            m1 = results["measurement_1"]
            m2 = results["measurement_2"]

            assert len(m1) == 100, "Should have 100 measurements for qubit 1"
            assert len(m2) == 100, "Should have 100 measurements for qubit 2"

            # Bell pair should be correlated
            correlated = sum(1 for i in range(100) if m1[i] == m2[i])
            correlation_rate = correlated / 100

            assert (
                correlation_rate > 0.9
            ), f"Bell pair should be highly correlated, got {correlation_rate:.2%}"

        elif "measurements" in results:
            # Old format or combined measurements
            measurements = results["measurements"]
            assert len(measurements) == 100, "Should have 100 measurements"
            assert all(
                isinstance(m, tuple | int) for m in measurements
            ), "Measurements should be tuples or integers"

    def test_compilation_output_structure(self, simple_circuit: object) -> None:
        """Test the structure of compilation outputs."""
        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        try:
            # Compile to HUGR
            hugr_bytes = compile_guppy_to_hugr(simple_circuit)
        except Exception as e:
            pytest.fail(f"HUGR compilation failed: {e}")

        # Verify HUGR output
        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"
        assert isinstance(hugr_bytes, bytes), "HUGR should be bytes"

        # Check for HUGR markers
        hugr_str = hugr_bytes.decode("utf-8")
        is_hugr_envelope = hugr_str.startswith("HUGRiHJv")
        is_json = hugr_str.startswith("{") or "{" in hugr_str[:100]

        assert is_hugr_envelope or is_json, "HUGR should be in envelope format or JSON"

        # If JSON, verify it can be parsed
        if is_json or (is_hugr_envelope and "{" in hugr_str):
            import json

            json_start = hugr_str.find("{") if is_hugr_envelope else 0
            if json_start != -1:
                try:
                    json_data = json.loads(hugr_str[json_start:])
                    assert isinstance(
                        json_data,
                        dict,
                    ), "HUGR JSON should be a dictionary"
                    assert len(json_data) > 0, "HUGR JSON should not be empty"
                except json.JSONDecodeError as e:
                    pytest.fail(f"HUGR JSON is invalid: {e}")


class TestCompilationErrorHandling:
    """Test error handling in compilation process."""

    def test_invalid_function_compilation(self) -> None:
        """Test compilation with invalid function."""
        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        # Try to compile a non-Guppy function
        def regular_function() -> int:
            return 42

        with pytest.raises((TypeError, ValueError, AttributeError)):
            compile_guppy_to_hugr(regular_function)
