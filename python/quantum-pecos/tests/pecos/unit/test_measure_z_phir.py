"""PHIR accepts MeasureZ only at its input boundaries, preserving Measure output."""

import copy
import json
from collections import OrderedDict

import pecos as pc
import pytest
from pecos._phir_normalization import canonicalize_measurement_qops
from pecos.classical_interpreters.phir_classical_interpreter import PhirClassicalInterpreter
from pecos.engines.hybrid_engine import HybridEngine
from pecos.noise.depolarizing_error_model import DepolarizingErrorModel
from pecos.reps.pyphir.pyphir import PyPHIR
from pecos.typing import PhirModel
from pecos_rslib import GateType, RustPhirClassicalInterpreter, validate_phir_classical_inputs
from pecos_rslib.quantum import TickCircuit


def _program(name: str, block: str | None = None) -> dict:
    measure = {"qop": name, "args": [["q", 0]], "returns": [["m", 0]]}
    if block in {"sequence", "qparallel"}:
        measure = {"block": block, "ops": [measure]}
    elif block == "if":
        measure = {"block": "if", "condition": {"cop": "==", "args": [0, 0]}, "true_branch": [measure]}
    return {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 1},
            {"qop": "X", "args": [["q", 0]]},
            measure,
        ],
    }


def test_gate_type_measure_z_attribute() -> None:
    assert GateType.MeasureZ == GateType.Measure


@pytest.mark.parametrize("name", ["Measure", "MeasureZ"])
def test_gate_type_measure_z_registry(name: str) -> None:
    circuit = TickCircuit()
    circuit.tick().add_gate(name, [0])
    assert [gate.gate_type for _, gate in circuit.gate_batches()] == [GateType.MeasureZ]


@pytest.mark.parametrize("block", [None, "sequence", "qparallel", "if"])
def test_phir_model_reexports_canonical_measure(block: str | None) -> None:
    source = _program("MeasureZ", block)
    untouched = copy.deepcopy(source)
    canonical = PhirModel.model_validate(source).model_dump_json()
    assert canonical == PhirModel.model_validate(_program("Measure", block)).model_dump_json()
    assert '"qop":"Measure"' in canonical
    assert "MeasureZ" not in canonical
    assert source == untouched


@pytest.mark.parametrize("interpreter", ["python", "rust"])
@pytest.mark.parametrize("phir_validate", [False, True])
@pytest.mark.parametrize("measurement_noise", [False, True])
@pytest.mark.parametrize("block", [None, "sequence", "qparallel", "if"])
def test_phir_measure_z_reaches_measurement_noise(interpreter, phir_validate, measurement_noise, block) -> None:
    def run(name):
        noise = (
            DepolarizingErrorModel({"p1": 0.0, "p2": 0.0, "p_prep": 0.0, "p_meas": 1.0}) if measurement_noise else None
        )
        classical = PhirClassicalInterpreter() if interpreter == "python" else RustPhirClassicalInterpreter()
        classical.phir_validate = phir_validate
        engine = HybridEngine(cinterp=classical, qsim="stabilizer", error_model=noise)
        return engine.run(_program(name, block), shots=3, seed=1314, return_int=True)

    expected = {"m": [0 if measurement_noise else 1] * 3}
    assert run("MeasureZ") == run("Measure") == expected


def test_native_phir_measure_z() -> None:
    def run(name):
        return pc.sim(pc.PhirJson(json.dumps(_program(name)))).seed(1314).run(3).to_dict()

    assert run("MeasureZ") == run("Measure") == {"m": [1, 1, 1]}


def test_pyphir_direct_boundary_preserves_input() -> None:
    source = _program("MeasureZ")
    untouched = copy.deepcopy(source)
    program = PyPHIR.from_phir(source)
    assert program.ops[-1].name == "Measure"
    assert source == untouched


def test_normalization_preserves_canonical_program_identity() -> None:
    source = _program("Measure", "if")
    assert canonicalize_measurement_qops(source) is source


def test_normalization_visits_nested_false_branch() -> None:
    source = _program("Measure", "if")
    source["ops"][-1]["false_branch"] = [
        {
            "block": "sequence",
            "ops": [{"qop": "MeasureZ", "args": [["q", 0]], "returns": [["m", 0]]}],
        },
    ]
    untouched = copy.deepcopy(source)
    normalized = canonicalize_measurement_qops(source)
    assert normalized["ops"][-1]["false_branch"][0]["ops"][0]["qop"] == "Measure"
    assert source == untouched


def test_normalization_preserves_ordered_dict_types() -> None:
    source = json.loads(json.dumps(_program("MeasureZ", "if")), object_pairs_hook=OrderedDict)
    untouched = copy.deepcopy(source)
    normalized = canonicalize_measurement_qops(source)
    assert type(normalized) is OrderedDict
    assert type(normalized["ops"][-1]) is OrderedDict
    assert type(normalized["ops"][-1]["true_branch"][0]) is OrderedDict
    assert normalized["ops"][-1]["true_branch"][0]["qop"] == "Measure"
    assert source == untouched
    assert list(normalized) == list(source)
    assert normalized["ops"][0] is source["ops"][0]
    assert canonicalize_measurement_qops(normalized) is normalized


def test_normalization_copies_only_changed_paths() -> None:
    source = _program("MeasureZ", "if")
    source["metadata"] = {"qop": "MeasureZ"}
    source["ops"][-1]["false_branch"] = [{"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]}]
    untouched = copy.deepcopy(source)
    normalized = canonicalize_measurement_qops(source)

    assert source == untouched
    assert normalized is not source
    assert normalized["ops"] is not source["ops"]
    assert normalized["metadata"] is source["metadata"]
    for original, canonical in zip(source["ops"][:-1], normalized["ops"][:-1], strict=True):
        assert canonical is original
    original_block = source["ops"][-1]
    block = normalized["ops"][-1]
    assert block is not original_block
    assert block["condition"] is original_block["condition"]
    assert block["false_branch"] is original_block["false_branch"]
    assert block["true_branch"] is not original_block["true_branch"]
    assert block["true_branch"][0] is not original_block["true_branch"][0]
    assert block["true_branch"][0]["qop"] == "Measure"
    assert canonicalize_measurement_qops(normalized) is normalized


@pytest.mark.parametrize("name", ["MeasureZed", "UnknownGate"])
@pytest.mark.parametrize("boundary", ["model", "pyphir", "native"])
def test_unknown_measurement_spelling_rejected(name, boundary) -> None:
    source = _program(name)

    def load():
        if boundary == "model":
            PhirModel.model_validate(source)
        elif boundary == "pyphir":
            PyPHIR.from_phir(source)
        else:
            pc.sim(pc.PhirJson(json.dumps(source))).run(1)

    with pytest.raises((ValueError, RuntimeError)):
        load()


def _malformed_program(name: str, block: str | None = None) -> dict:
    source = _program(name)
    source["ops"][0]["size"] = 2
    source["ops"][1]["size"] = 2
    measure = source["ops"].pop()
    measure["args"] = [["q", 0], ["q", 1]]
    if block is not None:
        measure = {"block": "sequence", "ops": [measure]}
    source["ops"].append(measure)
    return source


@pytest.mark.parametrize("interpreter", ["python", "rust"])
@pytest.mark.parametrize("phir_validate", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("block", [None, "sequence"])
def test_malformed_measure_z_interpreter_diagnostics(interpreter, phir_validate, as_json, block) -> None:
    # Also guard shared measurement validation preceding format/schema checks.
    for format_name in ["PHIR/JSON", "unsupported"]:
        errors = []
        for name in ["Measure", "MeasureZ"]:
            source = _malformed_program(name, block)
            source["format"] = format_name
            classical = PhirClassicalInterpreter() if interpreter == "python" else RustPhirClassicalInterpreter()
            classical.phir_validate = phir_validate
            with pytest.raises(ValueError, match="Measure args") as error:
                classical.init(json.dumps(source) if as_json else source)
            errors.append((type(error.value), str(error.value)))
        assert errors[0] == errors[1]


@pytest.mark.parametrize("block", [None, "sequence"])
def test_malformed_measure_z_rust_validator_diagnostics(block) -> None:
    errors = []
    for name in ["Measure", "MeasureZ"]:
        with pytest.raises(ValueError, match="Measure args") as error:
            validate_phir_classical_inputs(json.dumps(_malformed_program(name, block)))
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]
