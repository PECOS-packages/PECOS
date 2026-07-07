"""Tests for constrained-ancilla surface-code Guppy generation."""

from __future__ import annotations

import pytest


def test_surface_qubit_count_respects_ancilla_budget() -> None:
    """The optional budget caps peak live ancillas without changing data qubits."""
    from pecos.guppy import get_num_qubits

    assert get_num_qubits(7) == 97
    assert get_num_qubits(9) == 161
    assert get_num_qubits(9, ancilla_budget=17) == 98
    assert get_num_qubits(9, ancilla_budget=999) == 161


def test_surface_ancilla_budget_must_be_positive() -> None:
    """Reject nonsensical budgets early."""
    from pecos.guppy import get_num_qubits

    with pytest.raises(ValueError, match="ancilla_budget must be >= 1"):
        get_num_qubits(3, ancilla_budget=0)


def test_constrained_ancilla_surface_code_compiles_to_hugr() -> None:
    """A budgeted surface memory experiment should still be valid Guppy/HUGR."""
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from pecos.guppy import make_surface_code

    program = make_surface_code(distance=3, num_rounds=1, basis="Z", ancilla_budget=2)
    hugr = compile_guppy_to_hugr(program)

    assert len(hugr) > 0


def test_constrained_ancilla_surface_code_traces_to_native_tick_circuit() -> None:
    """Budgeted Guppy surface programs should work through traced-QIS DEM plumbing."""
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

    patch = SurfacePatch.create(distance=3)
    circuit = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=1,
        basis="Z",
        ancilla_budget=2,
        circuit_source="traced_qis",
    )

    assert circuit.get_meta("ancilla_budget") == "2"
    assert int(circuit.get_meta("num_measurements")) > 0


def test_constrained_szz_surface_code_traces_to_native_tick_circuit() -> None:
    """Budgeted SZZ/SZZdg surface programs should share the constrained Guppy path."""
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model

    patch = SurfacePatch.create(distance=3)
    circuit = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=1,
        basis="Z",
        ancilla_budget=2,
        circuit_source="traced_qis",
        interaction_basis="szz",
    )

    gate_counts = circuit.gate_counts_by_type()
    assert circuit.get_meta("ancilla_budget") == "2"
    assert int(circuit.get_meta("num_measurements")) > 0
    assert gate_counts.get("RZZ", 0) > 0
    assert gate_counts.get("CX", 0) == 0
