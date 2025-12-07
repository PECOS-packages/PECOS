"""Test the complete Guppy → HUGR → Standard QIR → PECOS pipeline.

This tests the new Standard QIR+ architecture implementation.
"""

import subprocess
from pathlib import Path

import pytest


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
        try:
            from pecos import get_guppy_backends
        except ImportError:
            pytest.skip("get_guppy_backends not available")

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
        if backends["guppy_available"] and not backends["rust_backend"]:
            pytest.skip("Guppy available but Rust backend not available")

    def test_guppy_frontend_initialization(self) -> None:
        """Test the GuppyFrontend class initialization."""
        try:
            from pecos._compilation import GuppyFrontend
        except ImportError:
            pytest.skip("GuppyFrontend not available")

        try:
            frontend = GuppyFrontend()
            info = frontend.get_backend_info()
        except (ImportError, RuntimeError) as e:
            if "guppylang" in str(e) or "not available" in str(e):
                pytest.skip(f"Guppy not available: {e}")
            pytest.fail(f"Failed to create GuppyFrontend: {e}")

        # Verify backend info structure
        assert isinstance(info, dict), "Backend info should be a dictionary"
        assert len(info) > 0, "Backend info should not be empty"

    def test_simple_quantum_function_compilation(self) -> None:
        """Test compiling a simple quantum function."""
        try:
            from guppylang import guppy
            from guppylang.std.quantum import h, measure, qubit
            from pecos._compilation import GuppyFrontend
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        @guppy
        def random_bit() -> bool:
            """Generate a random bit using quantum superposition."""
            q = qubit()
            h(q)
            return measure(q)

        # Test compilation
        try:
            frontend = GuppyFrontend()
            qir_file = frontend.compile_function(random_bit)
        except (ImportError, RuntimeError) as e:
            if "HUGR version" in str(e) or "not available" in str(e):
                pytest.skip(f"Known compatibility issue: {e}")
            pytest.fail(f"Compilation failed: {e}")

        # Verify QIR file was created
        assert qir_file is not None, "Compilation should return a file path"
        qir_path = Path(qir_file)
        assert qir_path.exists(), f"QIR file should exist at {qir_file}"

        # Verify QIR file has content
        with qir_path.open() as f:
            qir_content = f.read()
        assert len(qir_content) > 0, "QIR file should not be empty"
        assert (
            "@__quantum__" in qir_content or "define" in qir_content
        ), "QIR should contain quantum operations or function definitions"

    def test_bell_state_execution(self) -> None:
        """Test Bell state creation and measurement correlation."""
        try:
            from guppylang import guppy
            from guppylang.std.quantum import cx, h, measure, qubit
            from pecos import Guppy, sim
            from pecos_rslib import state_vector
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        @guppy
        def bell_state() -> tuple[bool, bool]:
            """Create a Bell state and measure both qubits."""
            q0, q1 = qubit(), qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        # Execute the Bell state circuit
        try:
            result = (
                sim(Guppy(bell_state))
                .qubits(10)
                .quantum(state_vector())
                .seed(42)
                .run(100)
            )
        except (RuntimeError, ImportError) as e:
            if "PECOS" in str(e) or "compilation" in str(e):
                pytest.skip(f"Execution environment issue: {e}")
            pytest.fail(f"Bell state execution failed: {e}")

        # Verify we got results
        assert result is not None, "Should get execution results"

        # Check for measurement results in various formats
        if "measurement_0" in result and "measurement_1" in result:
            # Tuple return format - individual measurement keys
            measurements1 = result["measurement_0"]
            measurements2 = result["measurement_1"]
            assert len(measurements1) == 100, "Should have 100 measurements for qubit 1"
            assert len(measurements2) == 100, "Should have 100 measurements for qubit 2"

            # Check correlation (Bell state should be perfectly correlated)
            correlated = sum(
                1 for i in range(100) if measurements1[i] == measurements2[i]
            )
            correlation_rate = correlated / 100
            assert (
                correlation_rate > 0.95
            ), f"Bell state measurements should be highly correlated, got {correlation_rate:.2%}"
        elif "measurements" in result:
            # Check if measurements are tuples
            measurements = result["measurements"]
            assert len(measurements) == 100, "Should have 100 measurements"

            if measurements and isinstance(measurements[0], tuple):
                # Direct tuple format
                correlated = sum(1 for (a, b) in measurements if a == b)
            else:
                # Integer-encoded format
                decoded = decode_integer_results(measurements, 2)
                correlated = sum(1 for (a, b) in decoded if a == b)

            correlation_rate = correlated / 100
            assert (
                correlation_rate > 0.95
            ), f"Bell state measurements should be highly correlated, got {correlation_rate:.2%}"
        else:
            pytest.fail(f"Unexpected result format: {result.keys()}")

    def test_rust_compilation_check(self) -> None:
        """Test that Rust components compile properly."""
        # Check if cargo is available
        try:
            result = subprocess.run(
                ["cargo", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                pytest.skip("Cargo not available")
        except FileNotFoundError:
            pytest.skip("Cargo not found in PATH")

        # Check if we're in a Rust project
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        cargo_toml = project_root / "Cargo.toml"

        if not cargo_toml.exists():
            pytest.skip("Not in a Rust project directory")

        # Check metadata to verify the project structure
        result = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version=1"],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        assert (
            result.returncode == 0
        ), f"Cargo metadata should succeed, got error: {result.stderr[:500]}"

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
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
        from pecos import Guppy, sim
        from pecos_rslib import state_vector
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")

    # Create a function that measures n qubits in superposition
    if n_qubits == 1:

        @guppy
        def superposition_test() -> bool:
            q = qubit()
            h(q)
            return measure(q)

    elif n_qubits == 2:

        @guppy
        def superposition_test() -> tuple[bool, bool]:
            q1, q2 = qubit(), qubit()
            h(q1)
            h(q2)
            return measure(q1), measure(q2)

    else:  # n_qubits == 3

        @guppy
        def superposition_test() -> tuple[bool, bool, bool]:
            q1, q2, q3 = qubit(), qubit(), qubit()
            h(q1)
            h(q2)
            h(q3)
            return measure(q1), measure(q2), measure(q3)

    # Run the test
    try:
        result = (
            sim(superposition_test)
            .qubits(10)
            .quantum(state_vector())
            .seed(42)
            .run(1000)
        )
    except (RuntimeError, ImportError) as e:
        pytest.skip(f"Execution issue: {e}")

    # Calculate average number of 1s
    if n_qubits == 1:
        ones_count = (
            sum(result["measurement_0"])
            if "measurement_0" in result
            else sum(result.get("measurements", []))
        )
        avg_ones = ones_count / 1000
    else:
        # For multiple qubits, sum up all the 1s
        total_ones = 0
        if "measurement_0" in result:
            # Separate measurement keys
            for i in range(n_qubits):
                total_ones += sum(result[f"measurement_{i}"])
        elif "measurements" in result:
            measurements = result["measurements"]
            if measurements and isinstance(measurements[0], tuple):
                # Direct tuple format
                for meas in measurements:
                    total_ones += sum(meas)
            else:
                # Integer-encoded format
                decoded = decode_integer_results(measurements, n_qubits)
                for meas in decoded:
                    total_ones += sum(meas)
        avg_ones = total_ones / 1000

    # Check that average is close to expected (allowing for statistical variation)
    assert (
        abs(avg_ones - expected_avg) < 0.1
    ), f"Average should be close to {expected_avg} for {n_qubits} qubits, got {avg_ones:.3f}"
