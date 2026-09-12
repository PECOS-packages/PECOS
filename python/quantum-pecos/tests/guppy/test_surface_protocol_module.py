# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Compilation, runtime outcomes and physical protocol oracles."""

import ast
import doctest
import re
import sys
from pathlib import Path

import pecos
import pytest
from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.guppy_gen.gadget_render import render_gadget_function, render_surface_gadget_module
from pecos.guppy_gen.protocol_render import load_surface_protocol_module, render_surface_protocol_module
from pecos.guppy_gen.transversal import (
    CSSCodeType,
    get_transversal_num_qubits,
    make_color_transversal_cnot,
    make_color_transversal_cnot_with_x,
    make_css_transversal_cnot,
    make_css_transversal_cnot_with_x,
    make_surface_transversal_cnot,
    make_surface_transversal_cnot_with_x,
)
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch, gadgets
from pecos.testing import (
    assert_same_measurement_partition,
    measurement_partition_from_builder,
    measurement_partition_from_trace,
)
from pecos.tracing import capture_qis_operation_trace, trace_program_to_tick_circuit
from pecos_rslib.quantum import TickCircuit


@pytest.fixture(scope="module")
def patch():
    return SurfacePatch.create(distance=3)


@pytest.fixture(scope="module")
def module(patch):
    return load_surface_protocol_module(patch)


RECIPES = {
    "h": ("make_h_experiment", (2,)),
    "cx": ("make_transversal_cx", (2,)),
    "sz": ("make_sz_teleportation", (2, 2, 2)),
    "t": ("make_t_injection", (2, 2)),
}


def _builder(patch, recipe):
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "D")
    if recipe == "h":
        builder.add_memory("D", 2, "Z")
        builder.add_transversal_h("D")
        builder.add_memory("D", 2, "X")
    else:
        builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
        if recipe == "cx":
            builder.add_memory(["D", "A"], 2, "Z")
            builder.add_transversal_cx("D", "A")
            builder.add_memory(["D", "A"], 2, "Z")
        elif recipe == "sz":
            builder.add_sz_via_teleportation("D", "A", 2, 2)
            builder.add_memory("D", 2, "Z")
        else:
            builder.add_t_via_injection("D", "A", 2, 2)
    return builder


@pytest.mark.parametrize("recipe", RECIPES)
def test_factory_compile(module, recipe):
    name, args = RECIPES[recipe]
    assert module[name](*args).compile() is not None


def test_all_gadget_functions_compile(module, patch):
    source = ast.parse(render_surface_protocol_module(patch))
    expected = {
        "prep_z_basis",
        "prep_x_basis",
        "prep_y_basis",
        "measure_z_basis",
        "measure_x_basis",
        "transversal_h",
        "transversal_cx",
        "apply_logical_x",
        "syndrome_extraction_swapped_a",
        *(f"syndrome_extraction_{scope}" for scope in ("a", "ctrl", "tgt", "data", "anc")),
    }
    rendered = {
        node.name for node in source.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("make_")
    }
    assert rendered == expected
    for name in expected:
        assert module[name].compile_function() is not None


@pytest.mark.parametrize(
    ("factory", "preparations", "readouts"),
    [
        ("make_h_experiment", [("a", "prep_z_basis")], [("measure_x_basis", "a")]),
        (
            "make_transversal_cx",
            [("ctrl", "prep_z_basis"), ("tgt", "prep_z_basis")],
            [("measure_z_basis", "ctrl"), ("measure_z_basis", "tgt")],
        ),
        (
            "make_sz_teleportation",
            [("data", "prep_z_basis"), ("anc", "prep_y_basis")],
            [("measure_z_basis", "anc"), ("measure_z_basis", "data")],
        ),
        (
            "make_t_injection",
            [("data", "prep_z_basis"), ("anc", "prep_x_basis")],
            [("measure_z_basis", "data"), ("measure_z_basis", "anc")],
        ),
    ],
)
def test_factory_preparations_and_readouts(patch, factory, preparations, readouts):
    """Check source structure because current physical readouts cannot distinguish Y from X ancilla prep.

    Measurement partitions and logical Z outcomes are unchanged by that substitution;
    a physical check of the phase would require a Y readout that is not available yet.
    """
    source = ast.parse(render_surface_protocol_module(patch))
    factory_node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == factory)
    assignments = [
        node
        for node in ast.walk(factory_node)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
    ]
    assert [
        (node.targets[0].id, node.value.func.id) for node in assignments if node.value.func.id.startswith("prep_")
    ] == preparations
    assert [
        (node.value.func.id, node.value.args[0].id) for node in assignments if node.value.func.id.startswith("measure_")
    ] == readouts


def test_testing_doctest_discovery():
    """Keep the testing helpers' examples readable by standard doctest discovery."""
    examples = doctest.DocTestFinder().find(pecos.testing)
    assert {
        "pecos.testing.assert_allclose",
        "pecos.testing.assert_array_equal",
        "pecos.testing.assert_array_less",
    } <= {example.name for example in examples}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("factory", "flag"),
    [("make_h_experiment", "logical_x"), ("make_transversal_cx", "control_x")],
)
def test_variants(module, factory, flag, reverse):
    variants = [(1, False), (2, True), (1, True), (2, False)]
    if reverse:
        variants.reverse()
    for rounds, enabled in variants:
        program = module[factory](rounds, **{flag: enabled})
        assert program.compile() is not None
        tc = trace_program_to_tick_circuit(program, 17 if flag == "logical_x" else 26)
        expected = 2 * rounds * 8 + 9
        if flag == "control_x":
            expected *= 2
        assert tc.num_measurements() == expected


@pytest.mark.parametrize("recipe", RECIPES)
def test_measurement_partition(module, patch, recipe):
    name, args = RECIPES[recipe]
    n = patch.geometry.num_qubits if recipe == "h" else get_transversal_num_qubits("surface", 3)
    tag_patches = (
        {"a": "D"} if recipe == "h" else {"ctrl": "D", "tgt": "A"} if recipe == "cx" else {"data": "D", "anc": "A"}
    )
    builder = _builder(patch, recipe)
    expected = measurement_partition_from_builder(builder)
    actual = measurement_partition_from_trace(module[name](*args), n, tag_patches)
    assert_same_measurement_partition(actual, expected)
    assert (
        sum(map(len, actual.values()))
        == sum(map(len, expected.values()))
        == builder.to_tick_circuit().num_measurements()
    )


@pytest.mark.parametrize("distance", [3, 5])
def test_peak_allocation(distance):
    module = load_surface_protocol_module(SurfacePatch.create(distance=distance))
    chunks = capture_qis_operation_trace(
        module["make_transversal_cx"](2),
        get_transversal_num_qubits("surface", distance),
    )
    live = set()
    peak = 0
    # _qis_trace_replay.py replays lowered PZ allocations and MZ readouts.
    # These factories destructively measure every allocation; no reset occurs while live.
    for chunk in chunks:
        for gate in chunk["lowered_quantum_ops"]:
            qubits = set(gate["qubits"])
            if gate["gate_type"] in {"PZ", "QAlloc"}:
                assert not live.intersection(qubits)
                live.update(qubits)
                peak = max(peak, len(live))
            elif gate["gate_type"] == "MZ":
                assert qubits <= live
                live.difference_update(qubits)
    assert not live
    assert peak == get_transversal_num_qubits("surface", distance)
    assert peak == get_transversal_num_qubits(CSSCodeType.SURFACE, distance)
    assert get_transversal_num_qubits("color", 3) == 18
    assert get_transversal_num_qubits(CSSCodeType.COLOR, 5) == 38


def _results(program, n):
    return (
        pecos.sim(program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(n)
        .seed(123)
        .run(32)
        .to_shot_map()
        .to_dict()
    )


def _assert_parity(rows, support, expected):
    assert len(rows) == 32
    assert [sum(row[q] for q in support) % 2 for row in rows] == [expected] * 32


def _sideband_tags(scopes, *, swapped=False):
    first = "z" if swapped else "x"
    orientation = "swapped:" if swapped else ""
    return {
        f"{scope}:{orientation}s{family}{i}:meas:{i + (0 if family == first else 4)}"
        for scope in scopes
        for family in ("x", "z")
        for i in range(4)
    }


@pytest.mark.parametrize("logical_x", [False, True])
def test_h_outcome(module, patch, logical_x):
    """Check logical readout and H's exchange of stabilizer signs.

    Wrong-register ancilla Hadamards can destroy stabilizer-sign correlations
    across H while preserving logical parity. The syndrome history therefore
    catches the mutation even when every final logical bit remains correct.
    """
    results = _results(module["make_h_experiment"](2, logical_x=logical_x), 17)
    # H maps the initial logical Z to current logical X on the same support.
    support = patch.geometry.logical_z.data_qubits
    _assert_parity(results["final_a"], support, int(logical_x))
    assert set(results) == (
        {"synx_a", "synz_a", "final_a"} | _sideband_tags(("a",)) | _sideband_tags(("a",), swapped=True)
    )
    # Full rounds preserve stabilizer signs, exchanging the families across H.
    width = len(patch.geometry.x_stabilizers)
    for sx, sz in zip(results["synx_a"], results["synz_a"], strict=True):
        assert sx[:width] == sx[width : 2 * width] == sz[2 * width : 3 * width] == sz[3 * width :]
        assert sz[:width] == sz[width : 2 * width] == sx[2 * width : 3 * width] == sx[3 * width :]


@pytest.mark.parametrize("control_x", [False, True])
def test_cx_outcome(module, patch, control_x):
    results = _results(module["make_transversal_cx"](2, control_x=control_x), 26)
    for label in ("ctrl", "tgt"):
        _assert_parity(results[f"final_{label}"], patch.geometry.logical_z.data_qubits, int(control_x))
    assert set(results) == {
        f"{tag}_{label}" for tag in ("synx", "synz", "final") for label in ("ctrl", "tgt")
    } | _sideband_tags(("ctrl", "tgt"))


@pytest.mark.parametrize("recipe", ["sz", "t"])
def test_teleportation_outcome(module, patch, recipe):
    """Check measurement grouping and Z preservation only."""
    name, args = RECIPES[recipe]
    results = _results(module[name](*args), 26)
    _assert_parity(results["final_data"], patch.geometry.logical_z.data_qubits, 0)
    assert set(results) == {
        f"{tag}_{label}" for tag in ("synx", "synz", "final") for label in ("data", "anc")
    } | _sideband_tags(("data", "anc"))


def test_sidebands_and_memory_parity(patch):
    source = render_surface_protocol_module(patch)
    scopes = {"a", "ctrl", "tgt", "data", "anc"}
    sidebands = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("make_"):
            continue
        calls = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "output"
        ]
        if node.name.startswith(("syndrome_extraction", "init_")):
            scope = node.name.rsplit("_", 1)[1]
            assert scope in scopes
            assert calls
            for call in calls:
                tag = ast.literal_eval(call.args[0])
                orientation = "swapped:" if "_swapped_" in node.name else ""
                assert re.fullmatch(rf"{scope}:{orientation}s[xz][0-9]+:(init:)?meas:[0-9]+", tag)
                sidebands.append(tag)
        else:
            assert not calls
    assert len(sidebands) == 48
    assert not re.search(r'output\("s[xz][0-9]+:', source)
    memory = render_surface_gadget_module(patch)
    golden = Path(__file__).parents[1] / "qec/surface/goldens/gadget_parity/guppy_d3.py.txt"
    assert memory == golden.read_text()
    assert 'output("sx0:meas:0"' in memory


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_surface_transversal_cnot(3, 1),
        lambda: make_surface_transversal_cnot_with_x(3, 1),
        lambda: make_css_transversal_cnot("surface", 3, 1),
        lambda: make_css_transversal_cnot(CSSCodeType.SURFACE, 3, 1),
        lambda: make_css_transversal_cnot_with_x("surface", 3, 1),
        lambda: make_css_transversal_cnot_with_x(CSSCodeType.SURFACE, 3, 1),
        lambda: make_color_transversal_cnot(3, 1),
        lambda: make_color_transversal_cnot_with_x(3, 1),
    ],
)
def test_public_api(factory):
    assert factory().compile() is not None


@pytest.mark.parametrize(
    ("dx", "dz", "rotated", "clause"),
    [(1, 1, True, "distance >= 3"), (3, 5, True, "square"), (4, 4, True, "odd"), (3, 3, False, "rotated=True")],
)
def test_invalid_patch(dx, dz, rotated, clause):
    with pytest.raises(ValueError, match=clause):
        render_surface_protocol_module(SurfacePatch.create(dx=dx, dz=dz, rotated=rotated))


@pytest.mark.parametrize("tag", ["a:sx0:init:meas:0", "a:swapped:sx0:init:meas:0"])
def test_partition_rejects_init_sidebands(monkeypatch, tag):
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [40])
    records = [{"name": tag, "values": [False], "result_ids": [40]}]
    monkeypatch.setattr(
        "pecos.testing._trace_program_to_tick_circuit_with_result_traces",
        lambda _program, _num_qubits: (circuit, records),
    )
    with pytest.raises(
        ValueError,
        match=r"Init sidebands are not partitioned: .*may share no repeated label, "
        "so the repeated-label rule cannot separate them",
    ):
        measurement_partition_from_trace("program", 1, {"a": "D"})


def test_partition_uses_result_ids(monkeypatch):
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [40, 7])
    circuit.tick().mz_with_ids([0, 1], [90, 12])
    records = [
        {"name": "a:sx0:meas:0", "values": [False], "result_ids": [7]},
        {"name": "a:sz0:meas:1", "values": [False], "result_ids": [40]},
        {"name": "final_a", "values": [False, False], "result_ids": [12, 90]},
    ]
    calls = []

    def trace(program, num_qubits):
        calls.append((program, num_qubits))
        return circuit, records

    monkeypatch.setattr("pecos.testing._trace_program_to_tick_circuit_with_result_traces", trace)
    actual = measurement_partition_from_trace("program", 2, {"a": "D"})
    assert calls == [("program", 2)]
    assert actual == {("D", "X", 0): (1,), ("D", "Z", 0): (0,), ("D", "final", 0): (2, 3)}
    # Identical values, group sizes and total count cannot hide swapped memberships.
    changed = {**actual, ("D", "X", 0): (0,), ("D", "Z", 0): (1,)}
    with pytest.raises(AssertionError, match=r"\('D', 'X', 0\): \(1,\) != \(0,\)"):
        assert_same_measurement_partition(actual, changed)


@pytest.mark.parametrize("ids", [[], [99], [True]])
@pytest.mark.parametrize("tag", ["final_a", "a:sx0:meas:0"])
def test_unsupported_measurement_provenance(monkeypatch, ids, tag):
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [40])
    records = [{"name": tag, "values": [False], "result_ids": ids}]
    monkeypatch.setattr(
        "pecos.testing._trace_program_to_tick_circuit_with_result_traces",
        lambda _program, _num_qubits: (circuit, records),
    )
    with pytest.raises(ValueError, match="Measurement partition"):
        measurement_partition_from_trace("program", 1, {"a": "D"})


@pytest.mark.parametrize(("recipe", "keys", "total"), [("h", 9, 41), ("cx", 18, 82), ("sz", 22, 98), ("t", 18, 82)])
def test_builder_partition(patch, recipe, keys, total):
    builder = _builder(patch, recipe)
    partition = measurement_partition_from_builder(builder)
    assert len(partition) == keys
    assert sum(map(len, partition.values())) == total
    assert sorted(ordinal for group in partition.values() for ordinal in group) == list(range(total))
    assert partition == measurement_partition_from_builder(builder)
    if recipe == "h":
        assert partition["D", "X", 2] == (16, 17, 18, 19)
        assert partition["D", "Z", 2] == (20, 21, 22, 23)
    if recipe == "sz":
        assert partition["A", "final", 0] == tuple(range(64, 73))
        assert partition["D", "X", 4] == (73, 74, 75, 76)
        assert partition["D", "final", 0] == tuple(range(89, 98))


def test_partition_round_occurrences_and_coverage(monkeypatch):
    circuit = TickCircuit()
    circuit.tick().mz_with_ids(list(range(6)), [5, 4, 3, 2, 1, 0])
    records = [
        {"name": "a:sx0:meas:0", "values": [False], "result_ids": [5]},
        {"name": "a:sz0:meas:1", "values": [False], "result_ids": [4]},
        {"name": "a:sx0:meas:0", "values": [False], "result_ids": [3]},
        {"name": "a:sz0:meas:1", "values": [False], "result_ids": [2]},
        {"name": "final_a", "values": [False, False], "result_ids": [1, 0]},
    ]
    monkeypatch.setattr(
        "pecos.testing._trace_program_to_tick_circuit_with_result_traces",
        lambda _program, _num_qubits: (circuit, records),
    )
    assert measurement_partition_from_trace("program", 6, {"a": "D"}) == {
        ("D", "X", 0): (0,),
        ("D", "Z", 0): (1,),
        ("D", "X", 1): (2,),
        ("D", "Z", 1): (3,),
        ("D", "final", 0): (4, 5),
    }
    records[-1]["result_ids"] = [1, 1]
    with pytest.raises(ValueError, match="cover each of 6 ordinals exactly once"):
        measurement_partition_from_trace("program", 6, {"a": "D"})


def test_partition_missing_key():
    with pytest.raises(AssertionError, match=r"\('D', 'X', 0\): None != \(0,\)"):
        assert_same_measurement_partition({}, {("D", "X", 0): (0,)})


@pytest.mark.parametrize("basis", [None, "Z", "X"])
@pytest.mark.parametrize("swapped", [False, True])
def test_scoped_gadget_preserves_labels(patch, basis, swapped):
    allocation = gadgets.default_allocation(patch)
    gadget = (
        gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=swapped)
        if basis is None
        else gadgets.init_syndrome_gadget(patch, allocation, basis=basis, x_z_swapped=swapped)
    )
    plain = render_gadget_function(gadget)
    scoped = render_gadget_function(gadget, tag_scope="ctrl")
    prefix = "ctrl:"
    orientation = "swapped:" if swapped else ""
    label = next(step.label for step in gadget.steps if step.op_type.name == "MEASURE")
    assert any(f'output("{orientation}{label}:' in line for line in plain)
    assert any(f'output("ctrl:{orientation}{label}:' in line for line in scoped)
    expected = [
        line.replace(f"def {gadget.name}(", f"def {gadget.name}_ctrl(").replace('output("', f'output("{prefix}')
        for line in plain
    ]
    assert scoped == expected


def test_partition_rounds_across_orientation_change(monkeypatch):
    circuit = TickCircuit()
    circuit.tick().mz_with_ids(list(range(18)), list(range(18)))
    records = []
    for round_index in range(4):
        labels = ("sx0", "sx1", "sz0", "sz1") if round_index < 2 else ("sz0", "sz1", "sx0", "sx1")
        orientation = "" if round_index < 2 else "swapped:"
        for ordinal, label in enumerate(labels):
            records.append(
                {
                    "name": f"a:{orientation}{label}:meas:{ordinal}",
                    "values": [False],
                    "result_ids": [4 * round_index + ordinal],
                },
            )
        # Arrays neither supply provenance nor delimit rounds.
        records.extend(
            [
                {"name": "synx_a", "values": [False, False], "result_ids": []},
                {"name": "synz_a", "values": [False, False], "result_ids": []},
            ],
        )
    records.append({"name": "final_a", "values": [False, False], "result_ids": [16, 17]})
    monkeypatch.setattr(
        "pecos.testing._trace_program_to_tick_circuit_with_result_traces",
        lambda _program, _num_qubits: (circuit, records),
    )
    assert measurement_partition_from_trace("program", 18, {"a": "D"}) == {
        ("D", "X", 0): (0, 1),
        ("D", "Z", 0): (2, 3),
        ("D", "X", 1): (4, 5),
        ("D", "Z", 1): (6, 7),
        ("D", "X", 2): (8, 9),
        ("D", "Z", 2): (10, 11),
        ("D", "X", 3): (12, 13),
        ("D", "Z", 3): (14, 15),
        ("D", "final", 0): (16, 17),
    }


def _factory_structure(source, factory):
    """Retain every body statement, including loops, conditionals and output calls."""
    outer = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == factory)
    inner = next(node for node in outer.body if isinstance(node, ast.FunctionDef))
    return [ast.unparse(node) for node in inner.body[1:]]


def _expected_rounds(count, *scopes, swapped=False):
    orientation = "swapped_" if swapped else ""
    body = "\n".join(
        f"    syn = syndrome_extraction_{orientation}{scope}({scope})\n"
        f"    output('synx_{scope}', syn.synx)\n"
        f"    output('synz_{scope}', syn.synz)"
        for scope in scopes
    )
    return f"for _ in range(comptime({count})):\n{body}"


def _expected_readout(scope, basis="z"):
    return [f"final = measure_{basis}_basis({scope})", f"output('final_{scope}', final)"]


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            "make_h_experiment",
            [
                "a = prep_z_basis()",
                "if comptime(logical_x):\n    apply_logical_x(a)",
                _expected_rounds("num_rounds", "a"),
                "transversal_h(a)",
                _expected_rounds("num_rounds", "a", swapped=True),
                *_expected_readout("a", "x"),
            ],
        ),
        (
            "make_transversal_cx",
            [
                "ctrl = prep_z_basis()",
                "if comptime(control_x):\n    apply_logical_x(ctrl)",
                "tgt = prep_z_basis()",
                _expected_rounds("num_rounds", "ctrl", "tgt"),
                "transversal_cx(ctrl, tgt)",
                _expected_rounds("num_rounds", "ctrl", "tgt"),
                *_expected_readout("ctrl"),
                *_expected_readout("tgt"),
            ],
        ),
        (
            "make_sz_teleportation",
            [
                "data = prep_z_basis()",
                "anc = prep_y_basis()",
                _expected_rounds("rounds_before", "data", "anc"),
                "transversal_cx(data, anc)",
                _expected_rounds("rounds_after", "data", "anc"),
                *_expected_readout("anc"),
                _expected_rounds("trailing_data_rounds", "data"),
                *_expected_readout("data"),
            ],
        ),
        (
            "make_t_injection",
            [
                "data = prep_z_basis()",
                "anc = prep_x_basis()",
                _expected_rounds("rounds_before", "data", "anc"),
                "transversal_cx(data, anc)",
                _expected_rounds("rounds_after", "data", "anc"),
                *_expected_readout("data"),
                *_expected_readout("anc"),
            ],
        ),
    ],
)
def test_factory_ordered_body(patch, factory, expected):
    """Pin the exact preparation, rounds, gate, rounds and readout sequence."""
    assert _factory_structure(render_surface_protocol_module(patch), factory) == expected


def test_loader_removes_failed_module(tmp_path):
    name = "pecos._generated.failed_protocol_test"
    with pytest.raises(RuntimeError, match="broken module"):
        load_guppy_source("partially_built = True\nraise RuntimeError('broken module')\n", tmp_path / "broken.py", name)
    assert name not in sys.modules
    loaded = load_guppy_source("complete = True\n", tmp_path / "fixed.py", name)
    try:
        assert loaded["complete"] is True
    finally:
        sys.modules.pop(name)


def test_shared_module_directory():
    from pecos.guppy_gen.protocol_render import _get_temp_dir as protocol_dir
    from pecos.guppy_gen.surface import _get_temp_dir as surface_dir
    from pecos.guppy_gen.transversal import _get_temp_dir as transversal_dir

    assert surface_dir() == transversal_dir() == protocol_dir() == _get_temp_dir()
