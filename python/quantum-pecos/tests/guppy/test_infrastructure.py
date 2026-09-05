"""Basic infrastructure tests for Guppy integration.

These are pytest-compatible tests.
"""

import pytest

pytestmark = pytest.mark.optional_dependency


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

    frontend = GuppyFrontend()
    try:
        # Should be able to get backend info
        info = frontend.get_backend_info()
        assert isinstance(info, dict)
        assert "backend" in info

    finally:
        frontend.cleanup()


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
