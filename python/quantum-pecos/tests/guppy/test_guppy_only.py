"""Guppy-only tests that don't require full PECOS installation."""


def test_guppy_available() -> None:
    """Test if Guppy is available in the environment."""
    from guppylang import guppy

    @guppy
    def test_func(x: int) -> int:
        return x + 1

    # Function should be a GuppyDefinition
    assert hasattr(test_func, "id") or hasattr(test_func, "compile")
