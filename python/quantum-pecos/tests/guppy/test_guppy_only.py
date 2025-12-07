"""Guppy-only tests that don't require full PECOS installation."""

import pytest
from pecos import get_guppy_backends


def test_guppy_available() -> None:
    """Test if Guppy is available in the environment."""
    try:
        from guppylang import guppy

        @guppy
        def test_func(x: int) -> int:
            return x + 1

        # Function should be a GuppyDefinition
        assert hasattr(test_func, "id") or hasattr(test_func, "compile")

    except ImportError:
        pytest.skip("guppylang not available - install with: uv pip install guppylang")


def test_backend_detection_minimal() -> None:
    """Test backend detection without full PECOS."""
    backends = get_guppy_backends()

    # Should return a dict
    assert isinstance(backends, dict)
    assert "guppy_available" in backends
    assert "rust_backend" in backends
