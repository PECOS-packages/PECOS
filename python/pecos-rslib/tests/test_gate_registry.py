"""Registry input errors must be ordinary exceptions, before decomposition dispatch."""

import pytest
from pecos_rslib import AngleSource, GateRegistry


@pytest.mark.parametrize(("qubits", "angles"), [([0], []), ([], [0.5]), ([0, 1], [0.5]), ([0], [0.5, 0.5])])
def test_decompose_rejects_wrong_input_counts(qubits, angles):
    registry = GateRegistry()
    registry.define("ROT", 1).angle_arity(1).step_with_angles("RZ", [0], [AngleSource.input(0)]).register_into(registry)
    with pytest.raises(ValueError, match="Gate ROT expects 1 qubits and 1 angles") as error:
        registry.decompose("ROT", qubits, angles)
    assert f"got {len(qubits)} qubits and {len(angles)} angles" in str(error.value)


@pytest.mark.parametrize("source", [AngleSource.input, AngleSource.neg_input])
def test_decompose_rejects_invalid_angle_reference(source):
    registry = GateRegistry()
    registry.define("ROT", 1).angle_arity(1).step_with_angles("RZ", [0], [source(1)]).register_into(registry)
    with pytest.raises(ValueError, match="Gate ROT step references angle index 1.*below 1"):
        registry.decompose("ROT", [0], [0.5])


@pytest.mark.parametrize(
    ("method", "extra"), [("step", ()), ("step_with_angles", ([],)), ("step_with_metadata", ([], {}))]
)
def test_builder_rejects_invalid_qubit_positions(method, extra):
    registry = GateRegistry()
    builder = registry.define("BAD", 1)
    with pytest.raises(ValueError, match="Gate BAD step references qubit index 1.*arity 1"):
        getattr(builder, method)("X", [1], *extra)
    builder.step("X", [0]).register_into(registry)
    assert registry.decompose("BAD", [7], []) == [("X", [7], [], {})]


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("step", ("X", [0])),
        ("step_with_angles", ("X", [0], [])),
        ("step_with_metadata", ("X", [0], [], {})),
        ("angle_arity", (1,)),
    ],
)
def test_consumed_builder_is_runtime_error(method, args):
    registry = GateRegistry()
    builder = registry.define("DONE", 1).step("X", [0])
    builder.register_into(registry)
    with pytest.raises(RuntimeError, match="Builder already consumed"):
        getattr(builder, method)(*args)


@pytest.mark.parametrize("method", ["step", "step_with_angles", "step_with_metadata"])
@pytest.mark.parametrize(
    ("gate", "indices", "message"),
    [("RZ", [0], "RZ expects 1 angle sources; got 0"), ("CX", [0], "CX expects 2 qubits; got 1")],
)
def test_builder_rejects_step_arity_mismatch(method, gate, indices, message):
    registry = GateRegistry()
    builder = registry.define("HR", 1).step("H", [0])
    extra = {"step": (), "step_with_angles": ([],), "step_with_metadata": ([], {})}[method]
    with pytest.raises(ValueError, match=f"Gate HR step {message}"):
        getattr(builder, method)(gate, indices, *extra)
    # Rejection preserves the prefix and does not append the malformed step.
    builder.register_into(registry)
    assert registry.decompose("HR", [0], []) == [("H", [0], [], {})]


def test_registry_quantum_arity_comes_from_definition():
    registry = GateRegistry()
    assert registry.quantum_arity("PARTIAL") is None
    registry.define("PARTIAL", 3).step("H", [0]).register_into(registry)
    assert registry.quantum_arity("PARTIAL") == 3
    registry.define("EMPTY", 0).register_into(registry)
    assert registry.quantum_arity("EMPTY") == 0
    registry.define("PARTIAL", 2).step("H", [0]).register_into(registry)
    assert registry.quantum_arity("PARTIAL") == 2


def test_decompose_angles_use_turns():
    registry = GateRegistry()
    (
        registry.define("TURNS", 1)
        .angle_arity(1)
        .step_with_angles("RX", [0], [AngleSource.input(0)])
        .step_with_angles("RY", [0], [AngleSource.fixed(0.25)])
        .step_with_angles("RZ", [0], [AngleSource.neg_input(0)])
        .register_into(registry)
    )
    assert registry.decompose("TURNS", [0], [0.25]) == [
        ("RX", [0], [0.25], {}),
        ("RY", [0], [0.25], {}),
        ("RZ", [0], [0.75], {}),
    ]
