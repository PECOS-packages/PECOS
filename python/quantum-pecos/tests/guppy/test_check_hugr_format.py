"""Check HUGR format from guppylang."""

import pytest


def test_check_hugr_format() -> None:
    """Check what HUGR format guppylang produces."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
    except ImportError:
        pytest.skip("guppylang not available")

    @guppy
    def simple() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR (a hugr.package.Package)
    hugr = simple.compile()

    # Binary Model envelope should be non-empty bytes
    hugr_bytes = hugr.to_bytes()
    assert isinstance(hugr_bytes, bytes), "to_bytes() should return bytes"
    assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

    # Inspect the Package structure directly
    assert len(hugr.modules) >= 1, "HUGR should contain at least one module"
    assert sum(1 for _ in hugr.modules[0].nodes()) > 0, "First module should contain at least one node"
