"""Basic infrastructure tests for Guppy integration.

These are pytest-compatible tests.
"""

from guppylang import guppy


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


def test_guppy_function_decoration() -> None:
    """The required Guppy dependency exposes compilable decorated functions."""

    @guppy
    def simple_func(x: int) -> int:
        return x + 1

    assert callable(simple_func.compile)
