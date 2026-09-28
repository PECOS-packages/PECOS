"""Quantum declarations retain upstream schema validation and Python semantics."""

import pytest
from pecos.classical_interpreters.phir_classical_interpreter import PhirClassicalInterpreter
from pecos.reps.pyphir import PyPHIR


@pytest.mark.parametrize("explicit_type", [False, True])
def test_optional_quantum_type(explicit_type: bool) -> None:
    """Omitted quantum types preserve declaration-order IDs in the reference reader."""
    declaration = {"data": "qvar_define", "variable": "a", "size": 2}
    if explicit_type:
        declaration["data_type"] = "qubits"
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"data": "qvar_define", "variable": "z", "size": 2}, declaration],
    }
    parsed = PyPHIR.from_phir(program)
    assert parsed.num_qubits == 4
    assert parsed.qvar_meta["a"].data_type == "qubits"
    assert parsed.qvar_meta["a"].qubit_ids == [2, 3]
    assert PhirClassicalInterpreter().init(program) == 4


@pytest.mark.parametrize("data_type", ["u32", None])
def test_wrong_quantum_type(data_type: str | None) -> None:
    """A present wrong type retains the reference reader's existing rejection."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"data": "qvar_define", "variable": "q", "data_type": data_type, "size": 2}],
    }
    for reader in [PyPHIR.from_phir, PhirClassicalInterpreter().init]:
        with pytest.raises(Exception, match="Do not know handle qvar type"):
            reader(program)


@pytest.mark.parametrize("fields", [{}, {"size": 0}, {"size": -1}, {"size": "2"}])
def test_quantum_size_schema_remains_enforced(fields: dict) -> None:
    """The full upstream schema still rejects missing, zero and malformed sizes."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"data": "qvar_define", "variable": "q", **fields}],
    }
    with pytest.raises(ValueError, match="size"):
        PhirClassicalInterpreter().init(program)


@pytest.mark.parametrize("size", [2, 3])
def test_duplicate_quantum_declarations(size: int) -> None:
    """Python preserves its reference behavior of overwriting the name mapping."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "variable": "q", "size": 2},
            {"data": "qvar_define", "variable": "q", "size": size},
        ],
    }
    parsed = PyPHIR.from_phir(program)
    assert parsed.num_qubits == 2 + size
    assert parsed.qvar_meta["q"].qubit_ids == list(range(2, 2 + size))
    assert PhirClassicalInterpreter().init(program) == 2 + size


def test_classical_type_remains_required() -> None:
    """The optional quantum type does not loosen the classical schema."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"data": "cvar_define", "variable": "c", "size": 2}],
    }
    with pytest.raises(ValueError, match="data_type"):
        PhirClassicalInterpreter().init(program)
