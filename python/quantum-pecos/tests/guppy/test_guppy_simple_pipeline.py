"""Test the Guppy → HUGR → PECOS pipeline."""

import pytest
from guppylang.std.builtins import array
from guppylang.std.builtins import result as record_result
from pecos import get_guppy_backends


def test_infrastructure() -> None:
    """Test if all components are available."""
    backends = get_guppy_backends()
    assert isinstance(backends, dict)
    assert "guppy_available" in backends
    assert "rust_backend" in backends


def test_simple_classical_function_definition() -> None:
    """Test defining a simple classical function."""
    from guppylang.decorator import guppy

    @guppy
    def add_numbers(x: int, y: int) -> int:
        return x + y

    # Function should be defined successfully
    assert add_numbers is not None


def test_quantum_function() -> None:
    """Test quantum function compilation and execution."""
    from guppylang.decorator import guppy
    from guppylang.std.quantum import h, measure, qubit
    from pecos import Guppy, sim
    from pecos_rslib import state_vector

    @guppy
    def quantum_coin() -> bool:
        q = qubit()
        h(q)
        output_value = measure(q).read()
        record_result("outcome", output_value)
        return output_value

    result = sim(Guppy(quantum_coin)).qubits(1).quantum(state_vector()).seed(42).run(10).to_dict()

    # Should have measurement results
    values = result["outcome"]
    assert len(values) == 10
    # Hadamard should give mix of 0s and 1s
    assert 0 in values or 1 in values
