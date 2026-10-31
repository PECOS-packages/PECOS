"""cvar_export selects final values consistently across all Python PHIR runners."""

import json

import pecos as pc
import pytest
from pecos.classical_interpreters.phir_classical_interpreter import PhirClassicalInterpreter
from pecos.engines.hybrid_engine import HybridEngine
from pecos.reps.pyphir.data_types import ExportVar
from pecos.reps.pyphir.pyphir import PyPHIR
from pecos_rslib import qasm_to_phir_json_py


def program(ops):
    return {"format": "PHIR/JSON", "version": "0.1.0", "ops": ops}


def declarations():
    return [
        {"data": "qvar_define", "variable": "q", "size": 1},
        *[{"data": "cvar_define", "variable": name, "data_type": "u32", "size": 2} for name in ("m", "a", "scratch")],
    ]


def export(*variables):
    return {"data": "cvar_export", "variables": list(variables)}


def rename(source, output):
    return {"data": "cvar_export", "variables": [source], "to": [output]}


def normalize(result, widths):
    """Compare raw logical bits while retaining separate assertions on public formats."""
    return {
        name: [
            (int(value, 2) if isinstance(value, str) else int(value)) & ((1 << widths[name]) - 1) for value in values
        ]
        for name, values in result.items()
    }


def run(source, runner, *, return_int=True, shots=2):
    if runner == "native":
        return pc.sim(pc.PhirJson(json.dumps(source))).quantum(pc.state_vector()).seed(1314).run(shots).to_dict()
    engine = (
        HybridEngine(qsim="stabilizer") if runner == "default" else HybridEngine(cinterp="python", qsim="stabilizer")
    )
    return engine.run(source, shots=shots, return_int=return_int, seed=1314)


def load(source, runner):
    if runner == "native":
        pc.sim(pc.PhirJson(json.dumps(source))).build()
    else:
        engine = HybridEngine() if runner == "default" else HybridEngine(cinterp="python")
        engine.initialize_sim_components(source)


def assert_output(result, expected, widths, *, bitstrings=False, shots=2):
    assert set(result) == set(expected)
    assert normalize(result, widths) == {name: [value] * shots for name, value in expected.items()}
    for name, values in result.items():
        for value in values:
            if bitstrings:
                assert isinstance(value, str)
                assert len(value) == widths[name]
                assert set(value) <= {"0", "1"}
            else:
                assert not isinstance(value, str)


COPY = {"cop": "Result", "args": ["a"], "returns": ["copy"]}
CASES = [
    pytest.param([], {"m": 1, "a": 2, "scratch": 1}, id="no_export_compatibility"),
    pytest.param([export("m", "a")], {"m": 1, "a": 2}, id="subset"),
    pytest.param([rename("m", "out")], {"out": 1}, id="rename"),
    pytest.param([export("m"), export("a")], {"m": 1, "a": 2}, id="separate_exports"),
    pytest.param([export("m"), COPY], {"m": 1, "copy": 0}, id="export_and_result"),
    pytest.param([export()], {}, id="empty"),
    pytest.param([export(), COPY], {"copy": 0}, id="empty_and_result"),
    pytest.param([rename("m", "out"), rename("m", "other")], {"out": 1, "other": 1}, id="multiple_outputs"),
    pytest.param([export("m"), export("m")], {"m": 1}, id="duplicate"),
]


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("return_int", [True, False])
@pytest.mark.parametrize(("exports", "expected"), CASES)
def test_measurement_exports(runner, return_int, exports, expected):
    """Final measurements, Result snapshots, empty sets, aliases and repeated shots."""
    ops = declarations() + exports + [{"qop": "X", "args": [["q", 0]]}]
    ops += [
        {"qop": "Measure", "args": [["q", 0]], "returns": [[name, bit]]}
        for name, bit in (("m", 0), ("a", 1), ("scratch", 0))
    ]
    result = run(program(ops), runner, return_int=return_int)
    assert_output(result, expected, dict.fromkeys(expected, 2), bitstrings=runner != "native" and not return_int)


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("return_int", [True, False])
def test_nested_exports_and_unexecuted_result(runner, return_int):
    """Both branches select outputs, and an unexecuted Result reports its zero value."""
    source = program(
        [
            *declarations(),
            {
                "block": "if",
                "condition": {"cop": "==", "args": [0, 1]},
                "true_branch": [{"block": "sequence", "ops": [export("m"), COPY]}],
                "false_branch": [rename("a", "other")],
            },
            {"cop": "=", "args": [3], "returns": ["m"]},
        ],
    )
    result = run(source, runner, return_int=return_int)
    assert_output(
        result,
        {"m": 3, "other": 0, "copy": 0},
        {"m": 2, "other": 2, "copy": 2},
        bitstrings=runner != "native" and not return_int,
    )


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("return_int", [True, False])
def test_signed_wide_and_measurement_prefix(runner, return_int):
    """Renames use source widths and signedness, and explicit measurement_* exports survive."""
    ops = [
        {"data": "cvar_define", "variable": name, "data_type": dtype, "size": size}
        for name, dtype, size in (("signed", "i32", 4), ("wide", "u64", 64), ("measurement_user", "u32", 32))
    ]
    ops += [
        rename("signed", "negative"),
        rename("wide", "out"),
        export("measurement_user"),
        rename("measurement_user", "renamed_measurement"),
        {"cop": "=", "args": [-3], "returns": ["signed"]},
        {"cop": "=", "args": [(1 << 63) + 1], "returns": ["wide"]},
        {"cop": "=", "args": [(1 << 31) + 1], "returns": ["measurement_user"]},
    ]
    result = run(program(ops), runner, return_int=return_int)
    expected = {
        "negative": 29,
        "out": (1 << 63) + 1,
        "measurement_user": (1 << 31) + 1,
        "renamed_measurement": (1 << 31) + 1,
    }
    widths = {"negative": 5, "out": 64, "measurement_user": 32, "renamed_measurement": 32}
    assert_output(result, expected, widths, bitstrings=runner != "native" and not return_int)
    if runner != "native" and return_int:
        assert int(result["negative"][0]) == -3


ERRORS = [
    pytest.param([export("missing")], "missing", id="unknown"),
    pytest.param([export("q")], "q", id="quantum_source"),
    pytest.param([{**export("m"), "to": ["out", "extra"]}], "extra", id="length"),
    pytest.param([{**export("m"), "to": [42]}], "42", id="non_string"),
    pytest.param([{**export("m"), "to": None}], "<missing or null>", id="null"),
    pytest.param([rename("m", "out"), rename("a", "out")], "out", id="conflict"),
    pytest.param([rename("m", "scratch")], "scratch", id="declared"),
    pytest.param([rename("m", "q")], "q", id="quantum_output"),
    pytest.param([rename("m", "copy"), COPY], "copy", id="result"),
    pytest.param([rename("m", "_width_out")], "_width_out", id="reserved"),
]


@pytest.mark.parametrize(("directives", "bad"), ERRORS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("loader", ["representation", "interpreter"])
def test_load_time_errors(directives, bad, nested, loader):
    """Load errors name the directive and bad value even in unexecuted branches."""
    ops = declarations()
    ops += (
        [{"block": "if", "condition": {"cop": "==", "args": [0, 1]}, "true_branch": directives}]
        if nested
        else directives
    )
    load = PyPHIR.from_phir if loader == "representation" else PhirClassicalInterpreter().init
    with pytest.raises(ValueError, match="cvar_export") as error:
        load(program(ops))
    assert str(error.value).startswith("cvar_export:")
    assert bad in str(error.value)


def test_representation_and_direct_nested_path():
    """Both conversion paths retain ExportVar and its positional to field."""
    source = program([*declarations(), rename("m", "out")])
    rep = PyPHIR.from_phir(source)
    assert isinstance(rep.ops[0], ExportVar)
    assert rep.ops[0].to == ["out"]
    nested = PyPHIR.handle_op({"block": "sequence", "ops": [rename("a", "other")]}, rep)
    assert isinstance(nested.ops[0], ExportVar)
    rep.ops.append(nested)
    for bad_to in [None, 42, [], [42]]:
        with pytest.raises(ValueError, match="cvar_export") as error:
            PyPHIR.handle_op({**export("m"), "to": bad_to}, rep)
        assert ("<missing or null>" if bad_to is None else str(bad_to)) in str(error.value)
    interpreter = PhirClassicalInterpreter()
    interpreter.init(rep)
    assert_output(
        {n: [v] for n, v in interpreter.results().items()},
        {"out": 0, "other": 0},
        {"out": 2, "other": 2},
        shots=1,
    )


def test_interpreter_reuse_and_result_reset():
    """Program reload clears exports; shot reset keeps Result destinations and zeros their values."""
    interpreter = PhirClassicalInterpreter()
    interpreter.init(program([*declarations(), export("m"), COPY]))
    list(interpreter.execute(interpreter.program.ops))
    assert set(interpreter.results()) == {"m", "copy"}
    interpreter.shot_reinit()
    assert set(interpreter.results()) == {"m", "copy"}
    interpreter.init(program(declarations()))
    assert set(interpreter.results()) == {"m", "a", "scratch"}


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_quantum_circuit_export_subset(runner):
    """qc2phir's ExportCVar selects a strict subset through an actual runner."""
    circuit = pc.QuantumCircuit(cvar_spec={"m": 1, "scratch": 1})
    circuit.append("X", {0})
    circuit.append("measure Z", {0}, var=("m", 0))
    circuit.append("cop", set(), cop_type="ExportCVar", export="m")
    result = run(circuit.to_phir_dict(), runner)
    assert_output(result, {"m": 1}, {"m": 1})


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_qasm_export_all_compatibility(runner):
    """Rust QASM-to-PHIR exports all ordinary registers as before."""
    source = qasm_to_phir_json_py(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg m[1]; creg scratch[1]; x q[0]; measure q[0] -> m[0];',
    )
    result = run(source, runner)
    assert_output(result, {"m": 1, "scratch": 0}, {"m": 1, "scratch": 1})


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_qparallel_export(runner):
    """Static export directives are accepted in qparallel blocks too."""
    source = program([*declarations(), {"block": "qparallel", "ops": [export("m")]}])
    assert_output(run(source, runner), {"m": 0}, {"m": 2})


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_declaration_in_unexecuted_branch(runner):
    """An exported branch-local declaration has its zero-initialized value."""
    source = program(
        [
            {
                "block": "if",
                "condition": {"cop": "==", "args": [0, 1]},
                "true_branch": [
                    {"data": "cvar_define", "variable": "inner", "data_type": "u32", "size": 4},
                    rename("inner", "out"),
                ],
            },
        ],
    )
    assert_output(run(source, runner), {"out": 0}, {"out": 4})


def test_python_result_order_follows_declarations():
    """Selection retains the existing Python declaration order."""
    source = program([*declarations(), export("a", "m")])
    result = run(source, "python")
    assert list(result) == ["m", "a"]
    assert_output(result, {"m": 0, "a": 0}, {"m": 2, "a": 2})


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("return_int", [True, False])
@pytest.mark.parametrize("indexed", [False, True])
def test_result_destination_exported_under_its_own_name(runner, return_int, indexed):
    """An exported variable can also receive an executed Result copy."""
    result_op = (
        {"cop": "Result", "args": [["m", 0]], "returns": [["a", 1]]}
        if indexed
        else {"cop": "Result", "args": ["m"], "returns": ["a"]}
    )
    source = program(
        [
            *declarations(),
            export("a"),
            rename("a", "alias"),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            result_op,
        ],
    )
    value = 2 if indexed else 1
    assert_output(
        run(source, runner, return_int=return_int),
        {"a": value, "alias": value},
        {"a": 2, "alias": 2},
        bitstrings=runner != "native" and not return_int,
    )


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_own_export_survives_unexecuted_result(runner):
    """An unexecuted Result does not prevent the variable's explicit export."""
    source = program(
        [
            *declarations(),
            rename("a", "a"),
            {
                "block": "if",
                "condition": {"cop": "==", "args": [0, 1]},
                "true_branch": [{"cop": "Result", "args": ["m"], "returns": ["a"]}],
            },
        ],
    )
    assert_output(run(source, runner), {"a": 0}, {"a": 2})


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("exports", [("m",), ("m", "m"), ("m", "k"), ("k", "m")])
@pytest.mark.parametrize("conditional", [False, True])
def test_zero_size_circuit_exports(runner, exports, conditional):
    """Omitted registers cannot appear in exports; empty-only requests still select nothing."""
    from pecos._phir_exports import all_operations

    circuit = pc.QuantumCircuit(cvar_spec={"m": 0, "k": 1, "scratch": 1})
    circuit.append("X", {0})
    circuit.append("measure Z", {0}, var=("k", 0))
    for name in exports:
        metadata = {"cop_type": "ExportCVar", "export": name}
        if conditional:
            metadata["cond"] = {"op": "==", "a": 0, "b": 1}
        circuit.append("cop", set(), **metadata)
        # Keep exports separate to test dropping an empty directive next to live exports.
        circuit.append("H", {0})
    source = circuit.to_phir_dict()
    nodes = all_operations(source["ops"])
    assert not any(node.get("variable") == "m" for node in nodes)
    directives = [node["variables"] for node in nodes if node.get("data") == "cvar_export"]
    if "k" in exports:
        assert directives == [["k"]]
        expected = {"k": 1}
    else:
        assert directives
        assert all(names == [] for names in directives)
        expected = {}
    assert_output(run(source, runner), expected, dict.fromkeys(expected, 1))


@pytest.mark.parametrize(
    "loader",
    ["representation", "schema", "python", "python_unvalidated", "rust", "rust_unvalidated", "native"],
)
@pytest.mark.parametrize("kind", ["cvar_define", "qvar_define"])
@pytest.mark.parametrize("branch", ["ops", "true_branch", "false_branch"])
def test_nested_declarations_without_exports_fail_at_load(loader, kind, branch):
    """Reject nested declarations at load time even when their branch never executes."""
    from pecos.typing import PhirModel
    from pecos_rslib import PhirJsonEngine
    from pecos_rslib import RustPhirClassicalInterpreter as RustInterpreter

    declaration = {"data": kind, "variable": "inner", "size": 1}
    if kind == "cvar_define":
        declaration["data_type"] = "u32"
    block = {"block": "sequence", "ops": [declaration]}
    if branch != "ops":
        block = {
            "block": "if",
            "condition": {"cop": "==", "args": [int(branch == "false_branch"), 1]},
            "true_branch": [],
            branch: [block],
        }
    source = program([block])
    python_interpreter = PhirClassicalInterpreter()
    rust_interpreter = RustInterpreter()
    python_interpreter.phir_validate = not loader.endswith("unvalidated")
    rust_interpreter.phir_validate = not loader.endswith("unvalidated")
    load = {
        "representation": lambda: PyPHIR.from_phir(source),
        "schema": lambda: PhirModel.model_validate(source),
        "python": lambda: python_interpreter.init(source),
        "python_unvalidated": lambda: python_interpreter.init(source),
        "rust": lambda: rust_interpreter.init(source),
        "rust_unvalidated": lambda: rust_interpreter.init(source),
        "native": lambda: PhirJsonEngine(json.dumps(source)),
    }[loader]
    with pytest.raises(ValueError, match="Nested declaration") as error:
        load()
    assert kind in str(error.value)
    assert "inner" in str(error.value)
    assert "declarations inside blocks are only supported in programs that use cvar_export" in str(error.value)
    assert "move the declaration to the top level" in str(error.value)


@pytest.mark.parametrize(
    ("directive", "field"),
    [
        ({"data": "cvar_export"}, "variables"),
        ({"data": "cvar_export", "variables": None}, "variables"),
        ({"data": "cvar_export", "variables": [None]}, "variables entry"),
        ({**export("m"), "to": [None]}, "to entry"),
        ({**export("m"), "to": None}, "to"),
    ],
)
def test_missing_and_null_export_fields_have_actionable_errors(directive, field):
    """Missing and null values use the same diagnostic words as Rust."""
    source = program([*declarations(), directive])
    with pytest.raises(ValueError, match="cvar_export") as error:
        PyPHIR.from_phir(source)
    assert str(error.value) == f"cvar_export: invalid {field} <missing or null>; supply an array of strings"


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("with_export", [False, True])
def test_branch_result_has_one_value_per_shot(runner, with_export):
    ops = declarations()[:2]
    if with_export:
        ops.append(export("m"))
    ops += [
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {
            "block": "if",
            "condition": {"cop": "==", "args": [["m", 0], 1]},
            "true_branch": [{"cop": "Result", "args": ["m"], "returns": ["copy"]}],
        },
    ]
    result = normalize(run(program(ops), runner, shots=128), {"m": 2, "copy": 2})
    assert all(len(column) == 128 for column in result.values())
    assert set(result["copy"]) == {0, 1}
    if "m" in result:
        assert result["copy"] == result["m"]


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_private_measurement_export(runner):
    source = program(
        [
            {"data": "qvar_define", "variable": "q", "size": 1},
            {"data": "cvar_define", "variable": "__x", "data_type": "u32", "size": 1},
            export("__x"),
            {"qop": "X", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["__x", 0]]},
        ],
    )
    assert_output(run(source, runner), {"__x": 1}, {"__x": 1})


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("returns", [None, [], [["m", 0], ["m", 1]]])
@pytest.mark.parametrize("nested", [False, True])
def test_measure_requires_one_destination_per_qubit_at_load(runner, returns, nested):
    measure = {"qop": "Measure", "args": [["q", 0]]}
    if returns is not None:
        measure["returns"] = returns
    if nested:
        measure = {"block": "if", "condition": {"cop": "==", "args": [0, 1]}, "true_branch": [measure]}
    source = program([*declarations(), measure])
    with pytest.raises((ValueError, RuntimeError), match=r"Measure.*give each measured qubit a destination"):
        load(source, runner)


@pytest.mark.parametrize(("first", "second"), [({}, {"added": 1}), ({"lost": 1}, {})])
def test_accumulator_rejects_changed_keys(first, second):
    engine = HybridEngine(cinterp="python")
    engine.results_accumulator(first)
    with pytest.raises(ValueError, match=r"Shot 2.*(added|lost)"):
        engine.results_accumulator(second)


@pytest.mark.parametrize("runner", ["default", "python", "native"])
@pytest.mark.parametrize("declared", [False, True])
def test_result_source_width_conflict_at_load(runner, declared):
    ops = [
        {"data": "cvar_define", "variable": "a", "data_type": "u32", "size": 2},
        {"data": "cvar_define", "variable": "b", "data_type": "u32", "size": 3},
    ]
    if declared:
        ops.append({"data": "cvar_define", "variable": "copy", "data_type": "u32", "size": 4})
    ops.append(
        {
            "block": "if",
            "condition": {"cop": "==", "args": [0, 1]},
            "true_branch": [
                {"cop": "Result", "args": ["a"], "returns": ["copy"]},
                {"cop": "Result", "args": ["b"], "returns": ["copy"]},
            ],
        },
    )
    with pytest.raises((ValueError, RuntimeError), match=r"copy.*source widths"):
        load(program(ops), runner)


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_unexecuted_result_uses_destination_final_value_and_type(runner):
    source = program(
        [
            {"data": "cvar_define", "variable": "source", "data_type": "i32", "size": 4},
            {"data": "cvar_define", "variable": "declared", "data_type": "i64", "size": 6},
            export(),
            {
                "block": "if",
                "condition": {"cop": "==", "args": [0, 1]},
                "true_branch": [
                    {"cop": "Result", "args": ["source", "source"], "returns": ["created", "declared"]},
                ],
            },
            {"cop": "=", "args": [-2], "returns": ["declared"]},
        ],
    )
    assert_output(
        run(source, runner, return_int=False),
        {"created": 0, "declared": 126},
        {"created": 5, "declared": 7},
        bitstrings=runner != "native",
    )
