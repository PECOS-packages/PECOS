"""Test the complete Guppy → HUGR → Standard QIR → PECOS pipeline.

This tests the new Standard QIR+ architecture implementation.
"""

import subprocess
from pathlib import Path

import pytest
from guppylang.std.builtins import array
from guppylang.std.builtins import result as record_result


def decode_integer_results(results: list[int], n_bits: int) -> list[tuple[bool, ...]]:
    """Decode integer-encoded results back to tuples of booleans."""
    decoded = []
    for val in results:
        bits = [bool(val & (1 << i)) for i in range(n_bits)]
        decoded.append(tuple(bits))
    return decoded


class TestGuppyLLVMPipeline:
    """Test suite for the Guppy to LLVM compilation pipeline."""

    def test_backend_availability(self) -> None:
        """Test that backends are properly detected."""
        from pecos import get_guppy_backends

        backends = get_guppy_backends()

        # Check that we get a dictionary with expected keys
        assert isinstance(
            backends,
            dict,
        ), "get_guppy_backends should return a dictionary"
        assert "guppy_available" in backends, "Should have 'guppy_available' key"
        assert "rust_backend" in backends, "Should have 'rust_backend' key"

        # These should be boolean values
        assert isinstance(
            backends["guppy_available"],
            bool,
        ), "guppy_available should be boolean"
        assert isinstance(
            backends["rust_backend"],
            bool,
        ), "rust_backend should be boolean"

        # If guppy is available, rust backend should also be available in most cases
        assert not backends["guppy_available"] or backends["rust_backend"], "Guppy requires the Rust backend"

    def test_bell_state_execution(self) -> None:
        """Test Bell state creation and measurement correlation."""
        from guppylang import guppy
        from guppylang.std.quantum import cx, h, measure, qubit
        from pecos import Guppy, sim
        from pecos_rslib import state_vector

        @guppy
        def bell_state() -> tuple[bool, bool]:
            """Create a Bell state and measure both qubits."""
            q0, q1 = qubit(), qubit()
            h(q0)
            cx(q0, q1)
            output_value = measure(q0).read(), measure(q1).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

        # Execute the Bell state circuit
        result = sim(Guppy(bell_state)).qubits(10).quantum(state_vector()).seed(42).run(100).to_dict()

        # Verify we got results
        assert result is not None, "Should get execution results"

        # Measurements format is [[m0, m1], [m0, m1], ...]
        measurements = result["outcome"]
        assert len(measurements) == 100, "Should have 100 measurements"

        # Check correlation (Bell state should be perfectly correlated)
        correlated = sum(1 for m in measurements if m[0] == m[1])
        correlation_rate = correlated / 100
        assert (
            correlation_rate > 0.95
        ), f"Bell state measurements should be highly correlated, got {correlation_rate:.2%}"

    def test_rust_compilation_check(self) -> None:
        """Test that Rust components compile properly."""
        # Check if cargo is available
        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, "Cargo not available"

        # Check if we're in a Rust project
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        cargo_toml = project_root / "Cargo.toml"

        assert cargo_toml.exists(), "Not in a Rust project directory"

        # Check metadata to verify the project structure
        result = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version=1"],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        assert result.returncode == 0, f"Cargo metadata should succeed, got error: {result.stderr[:500]}"

        # Verify output is valid JSON (basic check)
        assert result.stdout.startswith("{"), "Cargo metadata should return JSON"
        assert '"packages"' in result.stdout, "Metadata should contain packages info"


@pytest.mark.parametrize(
    ("n_qubits", "expected_avg"),
    [
        (1, 0.5),  # Single qubit in superposition
        (2, 1.0),  # Two qubits in superposition
        (3, 1.5),  # Three qubits in superposition
    ],
)
def test_superposition_statistics(n_qubits: int, expected_avg: float) -> None:
    """Test that qubits in superposition give expected statistics."""
    from guppylang import guppy
    from guppylang.std.quantum import h, measure, qubit
    from pecos import Guppy, sim
    from pecos_rslib import state_vector

    # Create a function that measures n qubits in superposition
    if n_qubits == 1:

        def superposition_test() -> bool:
            q = qubit()
            h(q)
            output_value = measure(q).read()
            record_result("outcome", output_value)
            return output_value

    elif n_qubits == 2:

        def superposition_test() -> tuple[bool, bool]:
            q1, q2 = qubit(), qubit()
            h(q1)
            h(q2)
            output_value = measure(q1).read(), measure(q2).read()
            record_result("outcome", array(output_value[0], output_value[1]))
            return output_value

    else:  # n_qubits == 3

        def superposition_test() -> tuple[bool, bool, bool]:
            q1, q2, q3 = qubit(), qubit(), qubit()
            h(q1)
            h(q2)
            h(q3)
            output_value = measure(q1).read(), measure(q2).read(), measure(q3).read()
            record_result("outcome", array(output_value[0], output_value[1], output_value[2]))
            return output_value

    from pecos.guppy_gen import variant_scoped

    superposition_test = guppy(variant_scoped(superposition_test, n_qubits))

    # Run the test
    result = sim(superposition_test).qubits(10).quantum(state_vector()).seed(42).run(1000).to_dict()

    # Calculate average number of 1s
    # Measurements format is [[m0], [m0], ...] for single qubit
    # or [[m0, m1], [m0, m1], ...] for multiple qubits
    measurements = result["outcome"]

    if n_qubits == 1:
        ones_count = sum(measurements)
        avg_ones = ones_count / 1000
    else:
        # For multiple qubits, sum up all the 1s from each shot
        total_ones = sum(sum(m) for m in measurements)
        avg_ones = total_ones / 1000

    # Check that average is close to expected (allowing for statistical variation)
    assert (
        abs(avg_ones - expected_avg) < 0.1
    ), f"Average should be close to {expected_avg} for {n_qubits} qubits, got {avg_ones:.3f}"
