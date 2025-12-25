"""Tests for HUGR/LLVM PyO3 integration.

Tests the Rust backend for HUGR compilation via the Selene compiler.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit

from pecos_rslib import compile_hugr_to_qis


def test_hugr_compiler_creation() -> None:
    """Test HUGR compilation functionality with the new API."""
    # Test that the function exists and is callable
    assert callable(compile_hugr_to_qis)

    # Test that compiler handles None/empty input appropriately
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        compile_hugr_to_qis(None)

    with pytest.raises(RuntimeError) as exc_info:
        compile_hugr_to_qis(b"")
    assert "empty hugr" in str(exc_info.value).lower()

    # Test that compiler provides meaningful error for invalid data
    with pytest.raises(RuntimeError) as exc_info:
        compile_hugr_to_qis(b"not json or hugr")
    assert "failed to read hugr" in str(exc_info.value).lower()


def test_hugr_compilation_with_invalid_data() -> None:
    """Test HUGR compilation with various invalid inputs."""
    # Test with invalid data
    with pytest.raises(RuntimeError) as exc_info:
        compile_hugr_to_qis(b"invalid json")
    assert "failed to read hugr" in str(exc_info.value).lower()

    # Test with valid JSON but not HUGR
    with pytest.raises(RuntimeError) as exc_info:
        compile_hugr_to_qis(b'{"not": "hugr"}')
    assert "failed to read hugr" in str(exc_info.value).lower()

    # Test with malformed HUGR (missing required fields)
    with pytest.raises(RuntimeError) as exc_info:
        compile_hugr_to_qis(b'{"modules": []}')
    assert "failed to read hugr" in str(exc_info.value).lower()


def test_convenience_functions() -> None:
    """Test convenience functions for HUGR compilation."""
    # Test that invalid HUGR raises an error
    dummy_hugr = b"dummy hugr data"
    with pytest.raises(RuntimeError, match="Failed to read HUGR"):
        compile_hugr_to_qis(dummy_hugr)

    # Test with output path - should still raise error for invalid HUGR
    temp_dir = tempfile.mkdtemp()
    temp_qir_path = Path(temp_dir) / "output.ll"

    try:
        # Should raise error for invalid HUGR even with output path
        with pytest.raises(RuntimeError, match="Failed to read HUGR"):
            compile_hugr_to_qis(dummy_hugr, str(temp_qir_path))
        # Output file should not be created for invalid HUGR
        assert not temp_qir_path.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Test with valid HUGR from Guppy

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
    result = compile_hugr_to_qis(valid_hugr)
    assert isinstance(result, str)
    assert len(result) > 0
    # Check for LLVM IR markers (Selene QIS patterns)
    assert "@qmain" in result or "@___qalloc" in result or "define" in result

    # Test with output path
    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as f:
        temp_qir_path = Path(f.name)

    try:
        result = compile_hugr_to_qis(valid_hugr, str(temp_qir_path))
        assert isinstance(result, str)
        # Check that output file was created
        assert temp_qir_path.exists()
        # Verify file contents match returned string
        assert temp_qir_path.read_text() == result
    finally:
        temp_qir_path.unlink(missing_ok=True)


def test_guppy_frontend_rust_backend() -> None:
    """Test that Guppy frontend can use Rust backend."""
    from pecos._compilation import GuppyFrontend

    # Create frontend instance - Rust backend is always available
    frontend = GuppyFrontend()

    # Check that frontend has the expected attributes
    assert hasattr(frontend, "use_rust_backend")
    assert frontend.use_rust_backend is True

    # Frontend should be created successfully
    assert frontend is not None


def test_guppy_frontend_backend_selection() -> None:
    """Test that Guppy frontend backend selection works."""
    from pecos import get_guppy_backends
    from pecos._compilation import GuppyFrontend

    frontend = GuppyFrontend()

    # Frontend object should exist
    assert frontend is not None

    # Should be able to get backends info via the module function
    backends = get_guppy_backends()
    assert isinstance(backends, dict)
    assert backends["guppy_available"] is True
    assert backends["rust_backend"] is True


def test_hugr_compiler_with_valid_data() -> None:
    """Test HUGR compiler with semi-valid HUGR data."""
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
        compile_hugr_to_qis(hugr_data)
    # We should get an error, but it processed the JSON
    assert exc_info.value is not None

    # Try with valid Guppy-generated HUGR

    @guppy
    def trivial_circuit() -> bool:
        q = qubit()
        return measure(q)

    # Compile to HUGR
    package = trivial_circuit.compile()
    hugr_bytes = package.to_bytes()

    # This should succeed
    result = compile_hugr_to_qis(hugr_bytes)
    assert isinstance(result, str)
    assert len(result) > 0
