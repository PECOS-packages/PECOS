"""Basic infrastructure tests for Guppy integration.

These are pytest-compatible tests.
"""

from guppylang import guppy


def test_guppy_function_decoration() -> None:
    """The required Guppy dependency exposes compilable decorated functions."""

    @guppy
    def simple_func(x: int) -> int:
        return x + 1

    assert callable(simple_func.compile)
