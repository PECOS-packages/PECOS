"""Basic infrastructure tests for Guppy integration.

These are pytest-compatible tests.
"""

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.optional_dependency

# Add PECOS to path
PECOS_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PECOS_ROOT / "python" / "quantum-pecos" / "src"))


def test_python_imports() -> None:
    """Test that basic Python imports work."""
    # If we get here, imports worked
    assert True


def test_backend_detection() -> None:
    """Test backend detection functionality."""
    from pecos import get_guppy_backends

    backends = get_guppy_backends()

    # Should return a dict with the expected keys
    assert isinstance(backends, dict)
    assert "guppy_available" in backends
    assert "rust_backend" in backends
    # External tools are no longer tracked - only Rust backend is used

    # These should be boolean values
    assert isinstance(backends["guppy_available"], bool)
    assert isinstance(backends["rust_backend"], bool)


def test_guppy_frontend_creation() -> None:
    """Test that GuppyFrontend can be created."""
    pytest.importorskip("guppylang")
    from pecos._compilation import GuppyFrontend

    # Since guppy_frontend.py is already imported with GUPPY_AVAILABLE=False,
    # we need to check if it would fail
    try:
        frontend = GuppyFrontend()
        # Should be able to get backend info
        info = frontend.get_backend_info()
        assert isinstance(info, dict)
        assert "backend" in info

        # Clean up
        frontend.cleanup()
    except ImportError as e:
        if "guppylang is not available" in str(e):
            pytest.skip("GuppyFrontend import check happens at module import time")


def test_guppy_import_if_available() -> None:
    """Test Guppy import if available (may be skipped)."""
    try:
        from guppylang import guppy

        # If we get here, guppylang is available
        @guppy
        def simple_func(x: int) -> int:
            return x + 1

        # Function should be decorated (check for guppy-specific attributes)
        assert hasattr(simple_func, "wrapped") or str(type(simple_func)).startswith(
            "<class 'guppylang",
        )

    except ImportError:
        # Guppy not available, skip this test
        import pytest

        pytest.skip("guppylang not available")
