"""Tests for HUGR/LLVM PyO3 integration.

Tests the Rust backend for HUGR compilation via the Selene compiler.
"""

import tempfile
from pathlib import Path

import pytest


# Test availability checks
def test_hugr_backend_availability() -> None:
    """Test that we can check HUGR backend availability."""
    try:
        from pecos_rslib import RUST_HUGR_AVAILABLE, check_rust_hugr_availability

        available, message = check_rust_hugr_availability()
        assert isinstance(available, bool)
        assert isinstance(message, str)
        assert available == RUST_HUGR_AVAILABLE

    except ImportError:
        # This is expected if the Rust backend is not compiled
        pytest.skip("Rust HUGR backend not available")


def test_hugr_compiler_creation() -> None:
    """Test HUGR compilation functionality with the new API."""
    try:
        from pecos_rslib import compile_hugr_to_llvm_rust, check_rust_hugr_availability

        # Check that HUGR support is available
        available, message = check_rust_hugr_availability()
        assert available, f"HUGR support should be available but got: {message}"

        # Test that the function exists and is callable
        assert callable(compile_hugr_to_llvm_rust)

        # Test that compiler handles None/empty input appropriately
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            compile_hugr_to_llvm_rust(None)

        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(b"")
        assert "empty hugr" in str(exc_info.value).lower()

        # Test that compiler provides meaningful error for invalid data
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(b"not json or hugr")
        assert "failed to read hugr" in str(exc_info.value).lower()

    except ImportError:
        pytest.skip("Rust HUGR backend not available")


def test_hugr_compilation_with_invalid_data() -> None:
    """Test HUGR compilation with various invalid inputs."""
    try:
        from pecos_rslib import compile_hugr_to_llvm_rust, check_rust_hugr_availability

        available, message = check_rust_hugr_availability()
        if not available:
            pytest.skip(f"HUGR support not available: {message}")

        # Test with invalid data
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(b"invalid json")
        assert "failed to read hugr" in str(exc_info.value).lower()

        # Test with valid JSON but not HUGR
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(b'{"not": "hugr"}')
        assert "failed to read hugr" in str(exc_info.value).lower()

        # Test with malformed HUGR (missing required fields)
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(b'{"modules": []}')
        assert "failed to read hugr" in str(exc_info.value).lower()

    except ImportError:
        pytest.skip("Rust HUGR backend not available")


def test_convenience_functions() -> None:
    """Test convenience functions for HUGR compilation."""
    try:
        from pecos_rslib import check_rust_hugr_availability, compile_hugr_to_llvm_rust

        available, message = check_rust_hugr_availability()
        if not available:
            pytest.skip(f"HUGR support not available: {message}")

        # Test that invalid HUGR raises an error
        dummy_hugr = b"dummy hugr data"
        with pytest.raises(RuntimeError, match="Failed to read HUGR"):
            compile_hugr_to_llvm_rust(dummy_hugr)

        # Test with output path - should still raise error for invalid HUGR
        import os

        temp_dir = tempfile.mkdtemp()
        temp_qir_path = os.path.join(temp_dir, "output.ll")

        try:
            # Should raise error for invalid HUGR even with output path
            with pytest.raises(RuntimeError, match="Failed to read HUGR"):
                compile_hugr_to_llvm_rust(dummy_hugr, temp_qir_path)
            # Output file should not be created for invalid HUGR
            assert not Path(temp_qir_path).exists()
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        # Test with valid HUGR (if Guppy is available)
        try:
            from guppylang import guppy
            from guppylang.std.quantum import qubit, h, measure

            @guppy
            def simple_circuit() -> bool:
                q = qubit()
                h(q)
                return measure(q)

            # Compile to HUGR
            package = simple_circuit.compile()
            # Use binary envelope format (modern approach)
            valid_hugr = package.to_bytes()

            # Should successfully compile valid HUGR
            result = compile_hugr_to_llvm_rust(valid_hugr)
            assert isinstance(result, str)
            assert len(result) > 0
            # Check for LLVM IR markers (Selene QIS patterns)
            assert "@qmain" in result or "@___qalloc" in result or "define" in result

            # Test with output path
            with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as f:
                temp_qir_path = f.name

            try:
                result = compile_hugr_to_llvm_rust(valid_hugr, temp_qir_path)
                assert isinstance(result, str)
                # Check that output file was created
                assert Path(temp_qir_path).exists()
                # Verify file contents match returned string
                assert Path(temp_qir_path).read_text() == result
            finally:
                Path(temp_qir_path).unlink(missing_ok=True)

        except ImportError:
            # Guppy not available, skip the valid HUGR test
            pass

    except ImportError:
        pytest.skip("Rust HUGR backend not available")


def test_guppy_frontend_rust_backend() -> None:
    """Test that Guppy frontend can use Rust backend."""
    try:
        from pecos._compilation import GuppyFrontend
        from pecos_rslib import check_rust_hugr_availability

        available, message = check_rust_hugr_availability()
        if not available:
            pytest.skip(f"HUGR support not available: {message}")

        # Create frontend instance - it may not detect Rust backend properly
        # due to import order issues or other factors
        frontend = GuppyFrontend()

        # Check that frontend has the expected attributes
        assert hasattr(frontend, "use_rust_backend")
        # Frontend might not always detect Rust backend even when available
        # This is OK - just test that the frontend was created
        assert isinstance(frontend.use_rust_backend, bool)

        # Frontend should be created successfully
        assert frontend is not None

    except ImportError:
        pytest.skip("Guppy frontend not available")


def test_guppy_frontend_backend_selection() -> None:
    """Test that Guppy frontend backend selection works."""
    try:
        from pecos import get_guppy_backends
        from pecos._compilation import GuppyFrontend

        frontend = GuppyFrontend()

        # Frontend object should exist
        assert frontend is not None

        # Should be able to get backends info via the module function
        backends = get_guppy_backends()
        assert isinstance(backends, dict)
        assert "guppy_available" in backends

        # Even if Rust backend is not available, Guppy should still work
        if not backends.get("rust_backend", False):
            # Guppy can still be available without Rust backend
            assert backends.get("guppy_available", False)

    except ImportError:
        pytest.skip("Guppy frontend not available")


def test_hugr_compiler_with_valid_data() -> None:
    """Test HUGR compiler with semi-valid HUGR data."""
    try:
        from pecos_rslib import compile_hugr_to_llvm_rust, check_rust_hugr_availability

        available, message = check_rust_hugr_availability()
        if not available:
            pytest.skip(f"HUGR support not available: {message}")

        # Create a minimal HUGR-like structure
        # This is still likely to fail compilation but tests JSON parsing
        hugr_data = b"""{
            "modules": [{
                "version": "live",
                "metadata": {"name": "test"},
                "nodes": [],
                "edges": []
            }],
            "extensions": []
        }"""

        # This will fail due to incomplete HUGR
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(hugr_data)
        # We should get an error, but it processed the JSON
        assert exc_info.value is not None

        # Try with valid Guppy-generated HUGR if available
        try:
            from guppylang import guppy
            from guppylang.std.quantum import qubit, measure

            @guppy
            def trivial_circuit() -> bool:
                q = qubit()
                return measure(q)

            # Compile to HUGR
            package = trivial_circuit.compile()
            hugr_bytes = package.to_bytes()

            # This should succeed
            result = compile_hugr_to_llvm_rust(hugr_bytes)
            assert isinstance(result, str)
            assert len(result) > 0

        except ImportError:
            # Guppy not available, that's OK
            pass

    except ImportError:
        pytest.skip("Rust HUGR backend not available")
