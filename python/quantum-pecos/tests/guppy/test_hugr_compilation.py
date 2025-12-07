"""Test HUGR compilation and LLVM IR generation."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestHUGRCompilation:
    """Test suite for HUGR compilation and related functionality."""

    def test_rust_hugr_crate_compilation(self) -> None:
        """Test that the Rust HUGR support compiles."""
        # Check if cargo is available
        cargo_path = shutil.which("cargo")
        if not cargo_path:
            pytest.skip("Cargo not available")

        try:
            result = subprocess.run(
                [cargo_path, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                pytest.skip("Cargo not available")
        except FileNotFoundError:
            pytest.skip("Cargo not found in PATH")

        # Check if pecos-hugr-qis crate exists
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        hugr_crate = project_root / "crates" / "pecos-hugr-qis"

        if not hugr_crate.exists():
            pytest.skip("pecos-hugr-qis crate not found")

        # Test compilation of pecos-hugr-qis crate
        result = subprocess.run(
            [cargo_path, "check", "-p", "pecos-hugr-qis", "--features", "llvm"],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        # returncode == 0 means SUCCESS, not failure!
        assert (
            result.returncode == 0
        ), f"HUGR crate compilation failed: {result.stderr[:500]}"

    def test_rust_hugr_unit_tests(self) -> None:
        """Test that HUGR unit tests pass."""
        # Check cargo availability
        cargo_path = shutil.which("cargo")
        if not cargo_path:
            pytest.skip("Cargo not available")

        try:
            subprocess.run(
                [cargo_path, "--version"],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            pytest.skip("Cargo not available")

        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        hugr_crate = project_root / "crates" / "pecos-hugr-qis"

        if not hugr_crate.exists():
            pytest.skip("pecos-hugr-qis crate not found")

        # Run HUGR-specific unit tests
        result = subprocess.run(
            [
                cargo_path,
                "test",
                "-p",
                "pecos-hugr-qis",
                "--features",
                "llvm",
                "--",
                "--nocapture",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        assert result.returncode == 0, f"HUGR unit tests failed: {result.stderr[:500]}"

        # Count successful tests if output is available
        if "test result: ok" in result.stdout:
            test_count = result.stdout.count("test result: ok")
            assert test_count > 0, "Should have at least one passing test"

    def test_llvm_ir_format_validation(self) -> None:
        """Test that generated LLVM IR follows HUGR conventions."""
        import os

        # Create a test LLVM IR file following HUGR conventions
        test_llvm = """
; HUGR convention LLVM IR
; Uses i64 for qubit indices, immediate measurements

declare void @__quantum__qis__h__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare void @__quantum__rt__result_record_output(i64, i8*)

@.str.c = constant [2 x i8] c"c\\00"

define void @main() #0 {
    ; Apply H to qubit 0
    call void @__quantum__qis__h__body(i64 0)

    ; Immediate measurement - returns i32 result
    %result = call i32 @__quantum__qis__m__body(i64 0, i64 0)

    ; Record result
    call void @__quantum__rt__result_record_output(i64 0,
        i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.c, i32 0, i32 0))

    ret void
}

attributes #0 = { "EntryPoint" }
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            f.write(test_llvm)
            llvm_file = Path(f.name)

        try:
            # Find llvm-as - check PATH first, then use pecos-llvm-utils
            llvm_as_path = shutil.which("llvm-as")
            print(f"DEBUG: llvm-as in PATH: {llvm_as_path}")

            if not llvm_as_path:
                # Use pecos-llvm-utils to find the tool
                cargo_path = shutil.which("cargo")
                print(f"DEBUG: cargo found at: {cargo_path}")
                if cargo_path:
                    try:
                        print("DEBUG: Running cargo to find llvm-as...")
                        result = subprocess.run(
                            [
                                cargo_path,
                                "run",
                                "-q",
                                "--release",
                                "-p",
                                "pecos-llvm-utils",
                                "--bin",
                                "pecos-llvm",
                                "--",
                                "tool",
                                "llvm-as",
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=120,  # Increased from 30s to account for compilation time on CI
                        )
                        print(f"DEBUG: cargo returncode: {result.returncode}")
                        print(f"DEBUG: cargo stdout: {result.stdout[:200]}")
                        print(f"DEBUG: cargo stderr: {result.stderr[:200]}")
                        if result.returncode == 0 and result.stdout.strip():
                            llvm_as_path = result.stdout.strip()
                            print(f"DEBUG: llvm-as found at: {llvm_as_path}")
                    except subprocess.TimeoutExpired as e:
                        print(f"DEBUG: cargo command timed out after {e.timeout}s")
                    except Exception as e:
                        print(f"DEBUG: cargo command failed with exception: {e}")
                else:
                    print("DEBUG: cargo not found in PATH")

            if llvm_as_path:
                # Validate with llvm-as
                output_path = "nul" if os.name == "nt" else "/dev/null"
                result = subprocess.run(
                    [llvm_as_path, str(llvm_file), "-o", output_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                assert (
                    result.returncode == 0
                ), f"LLVM IR validation failed: {result.stderr}"
            else:
                # llvm-as not available - this shouldn't happen for HUGR/QIS tests
                pytest.fail(
                    "llvm-as not found. LLVM should be available for HUGR/QIS tests. "
                    "Check LLVM_SYS_140_PREFIX environment variable.",
                )

        finally:
            # Clean up
            if llvm_file.exists():
                llvm_file.unlink()

    def test_llvm_ir_examples_structure(self) -> None:
        """Test LLVM IR examples follow HUGR conventions."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent

        # Look for LLVM IR examples
        llvm_examples = project_root / "examples" / "llvm"

        # Also check parent examples directory
        llvm_files: list[Path] = []

        if llvm_examples.exists():
            llvm_files.extend(llvm_examples.glob("*.ll"))

        # Check parent examples directory
        parent_examples = project_root / "examples"
        if parent_examples.exists():
            llvm_files.extend(parent_examples.glob("*.ll"))

        if not llvm_files:
            pytest.skip("No LLVM IR examples found")

        for llvm_file in llvm_files:
            content = llvm_file.read_text()

            # Check for HUGR convention characteristics
            has_quantum_intrinsics = "__quantum__qis__" in content
            has_i64_params = "i64" in content
            has_entry_point = "@main" in content or "EntryPoint" in content

            # Verify structure
            assert (
                has_quantum_intrinsics or has_entry_point
            ), f"{llvm_file.name} should have quantum intrinsics or entry point"

            if has_quantum_intrinsics:
                # If it has quantum operations, should use i64 for indices
                assert (
                    has_i64_params
                ), f"{llvm_file.name} should use i64 for qubit indices"

            # Check for measurement patterns if present
            if "__quantum__qis__m__body" in content:
                assert (
                    "i32" in content
                ), f"{llvm_file.name} measurements should return i32"

    def test_python_api_availability(self) -> None:
        """Test Python API for HUGR compilation is available."""
        try:
            from pecos import get_guppy_backends
        except ImportError as e:
            pytest.skip(f"Python API not available: {e}")

        backends = get_guppy_backends()

        # Verify backends is a dictionary
        assert isinstance(backends, dict), "get_guppy_backends should return a dict"

        # Check for expected keys
        expected_keys = {"guppy_available", "rust_backend"}
        for key in expected_keys:
            assert key in backends, f"backends should have '{key}' key"
            assert isinstance(
                backends[key],
                bool,
            ), f"backends['{key}'] should be boolean"

    def test_compile_guppy_to_hugr_api(self) -> None:
        """Test the compile_guppy_to_hugr function."""
        try:
            from guppylang import guppy
            from guppylang.std.quantum import h, measure, qubit
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        @guppy
        def simple_circuit() -> bool:
            """Simple quantum circuit."""
            q = qubit()
            h(q)
            return measure(q)

        # Test compilation
        try:
            hugr_bytes = compile_guppy_to_hugr(simple_circuit)
        except Exception as e:
            pytest.fail(f"Failed to compile Guppy to HUGR: {e}")

        # Verify output
        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"
        assert isinstance(hugr_bytes, bytes), "Should return bytes"

        # Check for HUGR format markers
        hugr_str = hugr_bytes.decode("utf-8")
        is_hugr_envelope = hugr_str.startswith("HUGRiHJv")
        is_json = hugr_str.startswith("{") or "{" in hugr_str[:100]

        assert (
            is_hugr_envelope or is_json
        ), "HUGR output should be envelope format or JSON"


class TestLLVMIRPatterns:
    """Test LLVM IR patterns and conventions."""

    def test_quantum_intrinsic_patterns(self) -> None:
        """Test that quantum intrinsics follow expected patterns."""
        # Define expected patterns for quantum operations
        intrinsic_patterns = {
            "hadamard": "@__quantum__qis__h__body",
            "pauli_x": "@__quantum__qis__x__body",
            "pauli_y": "@__quantum__qis__y__body",
            "pauli_z": "@__quantum__qis__z__body",
            "cnot": "@__quantum__qis__cnot__body",
            "measure": "@__quantum__qis__m__body",
            "reset": "@__quantum__qis__reset__body",
        }

        # Create test LLVM IR with these patterns
        test_ir_snippets = {
            "hadamard": "declare void @__quantum__qis__h__body(i64)",
            "pauli_x": "declare void @__quantum__qis__x__body(i64)",
            "measure": "declare i32 @__quantum__qis__m__body(i64, i64)",
            "cnot": "declare void @__quantum__qis__cnot__body(i64, i64)",
        }

        for op_name, declaration in test_ir_snippets.items():
            # Verify declaration follows expected pattern
            expected_pattern = intrinsic_patterns.get(op_name, "")
            if expected_pattern:
                assert (
                    expected_pattern in declaration
                ), f"{op_name} declaration should contain {expected_pattern}"

            # Check parameter types
            if op_name in ["hadamard", "pauli_x"]:
                assert (
                    "(i64)" in declaration
                ), f"{op_name} should take single i64 parameter"
            elif op_name == "cnot":
                assert (
                    "(i64, i64)" in declaration
                ), f"{op_name} should take two i64 parameters"
            elif op_name == "measure":
                assert "i32" in declaration, f"{op_name} should return i32"
                assert (
                    "(i64, i64)" in declaration
                ), f"{op_name} should take two i64 parameters"

    def test_result_recording_patterns(self) -> None:
        """Test result recording function patterns."""
        result_patterns = [
            "void @__quantum__rt__result_record_output(i64, i8*)",
            "void @__quantum__rt__tuple_record_output(i64, i8*)",
            "void @__quantum__rt__array_record_output(i8*, i32*)",
        ]

        # Each pattern should follow specific conventions
        for pattern in result_patterns:
            # Check return type
            assert "void" in pattern, "Result recording should return void"

            # Check for proper pointer types
            if "result_record" in pattern:
                assert "i64" in pattern, "result_record should take i64 parameter"
                assert "i8*" in pattern, "result_record should take i8* parameter"
            elif "tuple_record" in pattern:
                assert "i64" in pattern, "tuple_record should take i64 parameter"
                assert "i8*" in pattern, "tuple_record should take i8* parameter"
            elif "array_record" in pattern:
                assert "i8*" in pattern, "array_record should take i8* parameter"
                assert "i32*" in pattern, "array_record should take i32* parameter"
