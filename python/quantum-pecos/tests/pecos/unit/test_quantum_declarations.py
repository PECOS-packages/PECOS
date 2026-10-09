"""PHIR-JSON declaration shape and program-wide name uniqueness."""

import json
from collections.abc import Callable
from pathlib import Path

import pecos
import pytest
from pecos.circuits.qc2phir import to_phir_dict
from pecos.classical_interpreters.phir_classical_interpreter import PhirClassicalInterpreter
from pecos.reps.pyphir import PyPHIR
from pecos.typing import PhirModel
from pecos_rslib import PhirJsonEngine, RustPhirClassicalInterpreter


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


@pytest.mark.parametrize(
    "fields",
    [{}, {"size": 0}, {"size": -1}, {"size": True}, {"size": False}, {"size": 2.0}, {"size": "2"}, {"size": None}],
)
def test_quantum_size_schema_remains_enforced(fields: dict) -> None:
    """The full upstream schema still rejects missing, zero and malformed sizes."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"data": "qvar_define", "variable": "q", **fields}],
    }
    with pytest.raises(ValueError, match="Quantum register 'q' requires a positive integer size"):
        PyPHIR.from_phir(program)
    with pytest.raises(ValueError, match="size"):
        PhirClassicalInterpreter().init(program)


@pytest.mark.parametrize("size", [2, 3])
def test_duplicate_quantum_declarations(size: int) -> None:
    """Identical and different-sized quantum redeclarations are invalid."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "variable": "q", "size": 2},
            {"data": "qvar_define", "variable": "q", "size": size},
        ],
    }
    for reader in [PyPHIR.from_phir, PhirClassicalInterpreter().init]:
        with pytest.raises(
            ValueError,
            match="Variable 'q' is already declared as quantum; cannot redeclare as quantum",
        ):
            reader(program)


def test_classical_type_remains_required() -> None:
    """The optional quantum type does not loosen the classical schema."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [{"data": "cvar_define", "variable": "c", "size": 2}],
    }
    with pytest.raises(ValueError, match="data_type"):
        PhirClassicalInterpreter().init(program)


@pytest.mark.parametrize(
    "metadata",
    [{}, {"qvar_spec": {"q": 0}}, {"cvar_spec": {"c": 0}}, {"qvar_spec": {"q": 0}, "cvar_spec": {"c": 0}}],
)
def test_empty_circuit_round_trip(metadata: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact emitted document is also consumed by all four Rust entry points."""
    monkeypatch.setattr(pecos, "__version__", "fixture")
    generated = to_phir_dict(pecos.QuantumCircuit(**metadata))
    fixture = (
        Path(__file__).resolve().parents[5] / "crates/pecos-phir-json/tests/fixtures/empty_quantum_circuit.phir.json"
    )
    assert generated == json.loads(fixture.read_text())
    PhirModel.model_validate(generated)
    assert PyPHIR.from_phir(generated).num_qubits == 0
    assert PhirClassicalInterpreter().init(generated) == 0


def test_empty_registers_do_not_remove_nonempty_registers() -> None:
    """Omission is limited to empty registers, including explicit classical sizes."""
    generated = to_phir_dict(pecos.QuantumCircuit(qvar_spec={"empty_q": 0, "q": 2}, cvar_spec={"empty_c": 0, "c": 2}))
    assert [op["variable"] for op in generated["ops"]] == ["q", "c"]
    PhirModel.model_validate(generated)
    assert PhirClassicalInterpreter().init(generated) == 2


@pytest.mark.parametrize(
    ("first", "second"),
    [("quantum", "quantum"), ("classical", "classical"), ("quantum", "classical"), ("classical", "quantum")],
)
@pytest.mark.parametrize(
    "reader",
    [
        PyPHIR.from_phir,
        PhirClassicalInterpreter().init,
        RustPhirClassicalInterpreter().init,
        lambda program: PhirJsonEngine(json.dumps(program)),
        lambda program: PhirJsonEngine.create_with_validation_disabled(json.dumps(program)),
        PhirModel.model_validate,
        lambda program: PhirModel.model_validate_json(json.dumps(program)),
    ],
)
def test_redeclaration_matrix(first: str, second: str, reader: Callable[[dict], object]) -> None:
    """All entry points reject every shape with the name and both declaration kinds."""

    def declaration(kind: str) -> dict:
        return {
            "data": "qvar_define" if kind == "quantum" else "cvar_define",
            "data_type": "qubits" if kind == "quantum" else "u32",
            "variable": "a",
            "size": 1,
        }

    program = {"format": "PHIR/JSON", "version": "0.1.0", "ops": [declaration(first), declaration(second)]}
    with pytest.raises(ValueError, match=f"Variable 'a' is already declared as {first}; cannot redeclare as {second}"):
        reader(program)


@pytest.mark.parametrize("quantum_name", ["q", "__q0__"])
@pytest.mark.parametrize("interpreter", [PhirClassicalInterpreter(), RustPhirClassicalInterpreter()])
def test_distinct_declarations_and_repeated_references(
    interpreter: PhirClassicalInterpreter | RustPhirClassicalInterpreter,
    quantum_name: str,
) -> None:
    """Several quantum and classical names execute identically through both interpreters."""
    from pecos.engines.hybrid_engine import HybridEngine

    fixture = (
        Path(__file__).resolve().parents[5] / "crates/pecos-phir-json/tests/fixtures/unique_declarations.phir.json"
    )
    program = json.loads(fixture.read_text().replace('"q"', json.dumps(quantum_name)))
    result = HybridEngine(cinterp=interpreter).run(program, shots=2, seed=42, return_int=True)
    assert result == {"a": [7, 7], "b": [2, 2], "m": [1, 1]}


def test_unreachable_redeclaration() -> None:
    """Control flow does not create another declaration namespace."""
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "variable": "a", "size": 1},
            {
                "block": "if",
                "condition": {"cop": "==", "args": [0, 1]},
                "true_branch": [{"data": "cvar_define", "data_type": "u32", "variable": "a", "size": 1}],
            },
        ],
    }
    with pytest.raises(ValueError, match="Variable 'a' is already declared as quantum; cannot redeclare as classical"):
        PyPHIR.from_phir(program)
