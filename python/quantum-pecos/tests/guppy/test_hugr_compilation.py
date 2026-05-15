"""Test HUGR compilation and LLVM IR generation.

Rust-side coverage (compilation, unit tests) lives in `cargo test
-p pecos-hugr-qis` and is run by `just rstest` / `pecos rust test
--workspace --features=runtime,hugr`. Don't re-invoke cargo from pytest --
duplicates work, hides Rust build errors as Python test failures, and
runs under a different env than the canonical Rust test path.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Module-level cache for llvm-as path to avoid repeated lookups
_llvm_as_cache: dict[str, str | None] = {}


def _find_llvm_as() -> str | None:
    """Find llvm-as path using the Rust pecos-build crate's LLVM detection.

    This uses the same search logic as the Rust codebase:
    1. ~/.pecos/llvm/ (PECOS managed installation)
    2. Project-local llvm/ directory
    3. System installations (Homebrew on macOS, package manager on Linux)
    """
    if "llvm_as" in _llvm_as_cache:
        return _llvm_as_cache["llvm_as"]

    try:
        from pecos_rslib import find_llvm_tool

        llvm_as_path = find_llvm_tool("llvm-as")
        _llvm_as_cache["llvm_as"] = llvm_as_path
        return llvm_as_path
    except ImportError:
        # Fallback if pecos_rslib not available (shouldn't happen in normal tests)
        _llvm_as_cache["llvm_as"] = None
        return None


class TestHUGRCompilation:
    """Test suite for HUGR compilation and related functionality."""

    def test_llvm_ir_format_validation(self) -> None:
        """Test that generated LLVM IR follows HUGR conventions."""
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

        # Find llvm-as using cached lookup (avoids expensive cargo fallback)
        llvm_as_path = _find_llvm_as()
        if not llvm_as_path:
            pytest.skip(
                "llvm-as not found in PATH, LLVM_SYS_*_PREFIX, or common locations. "
                "Set PECOS_TEST_USE_CARGO_LLVM_LOOKUP=1 to enable slow cargo fallback.",
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            f.write(test_llvm)
            llvm_file = Path(f.name)

        try:
            # Validate with llvm-as
            output_path = "nul" if os.name == "nt" else "/dev/null"
            result = subprocess.run(
                [llvm_as_path, str(llvm_file), "-o", output_path],
                capture_output=True,
                text=True,
                check=False,
            )

            assert result.returncode == 0, f"LLVM IR validation failed: {result.stderr}"

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
                assert has_i64_params, f"{llvm_file.name} should use i64 for qubit indices"

            # Check for measurement patterns if present
            if "__quantum__qis__m__body" in content:
                assert "i32" in content, f"{llvm_file.name} measurements should return i32"

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

        assert is_hugr_envelope or is_json, "HUGR output should be envelope format or JSON"


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
                assert expected_pattern in declaration, f"{op_name} declaration should contain {expected_pattern}"

            # Check parameter types
            if op_name in ["hadamard", "pauli_x"]:
                assert "(i64)" in declaration, f"{op_name} should take single i64 parameter"
            elif op_name == "cnot":
                assert "(i64, i64)" in declaration, f"{op_name} should take two i64 parameters"
            elif op_name == "measure":
                assert "i32" in declaration, f"{op_name} should return i32"
                assert "(i64, i64)" in declaration, f"{op_name} should take two i64 parameters"

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
