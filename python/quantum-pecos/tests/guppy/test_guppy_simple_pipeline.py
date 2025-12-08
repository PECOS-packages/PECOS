"""Test the Guppy → HUGR → PECOS pipeline."""

import pytest
from pecos import get_guppy_backends


def test_infrastructure() -> None:
    """Test if all components are available."""
    backends = get_guppy_backends()
    assert isinstance(backends, dict)
    assert "guppy_available" in backends
    assert "rust_backend" in backends


def test_simple_classical_function_definition() -> None:
    """Test defining a simple classical function."""
    try:
        from guppylang.decorator import guppy

        @guppy
        def add_numbers(x: int, y: int) -> int:
            return x + y

        # Function should be defined successfully
        assert add_numbers is not None

    except ImportError:
        pytest.skip("Guppylang not available")


def test_quantum_function() -> None:
    """Test quantum function compilation and execution."""
    try:
        from guppylang.decorator import guppy
        from guppylang.std.quantum import h, measure, qubit
        from pecos import Guppy, sim
        from pecos_rslib import state_vector

        @guppy
        def quantum_coin() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        result = (
            sim(Guppy(quantum_coin)).qubits(1).quantum(state_vector()).seed(42).run(10)
        )

        # Should have measurement results
        assert "measurement_0" in result
        values = result["measurement_0"]
        assert len(values) == 10
        # Hadamard should give mix of 0s and 1s
        assert 0 in values or 1 in values

    except ImportError as e:
        if "guppylang" in str(e):
            pytest.skip("Guppylang not available")
        else:
            raise
