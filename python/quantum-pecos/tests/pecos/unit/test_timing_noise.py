"""Timing markers are consumed without changing ordinary noise dispatch."""

import re

import pytest
from pecos.machines.generic_machine import GenericMachine
from pecos.noise.depolarizing_error_model import DepolarizingErrorModel
from pecos.noise.generic_error_model import GenericErrorModel
from pecos.reps.pyphir.op_types import QOp


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
@pytest.mark.parametrize("name", ["Idle", "Transport"])
@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(None, id="absent"),
        pytest.param({}, id="empty"),
        pytest.param({"duration": [20, "ns"]}, id="duration"),
        pytest.param({"duration": 0}, id="zero-duration"),
        pytest.param({"duration": None}, id="null-duration"),
        pytest.param({"noiseless": True}, id="noiseless"),
        pytest.param({"noiseless": True, "duration": [20, "ns"]}, id="noiseless-duration"),
    ],
)
def test_timing_consumed(model_type: type, name: str, metadata: dict | None) -> None:
    """Timing markers disappear, including noiseless/default metadata, without truncating a batch."""
    model = model_type({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    model.init(2, GenericMachine(num_qubits=2))
    op = QOp(name=name, args=[0, 1]) if metadata is None else QOp(name=name, args=[0, 1], metadata=metadata)
    before = QOp(name="H", args=[0], metadata={})
    after = QOp(name="X", args=[1], metadata={})
    assert [(item.name, item.args) for item in model.process([before, op, after])] == [("H", [0]), ("X", [1])]


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
@pytest.mark.parametrize("name", ["UnknownGate", "Unsupported", "NotAGate"])
def test_unknown_gate_remains_unknown(model_type: type, name: str) -> None:
    """Unknown names retain the existing exception and diagnostic."""
    model = model_type({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    model.init(2, GenericMachine(num_qubits=2))
    expected = f"This error model doesn't handle gate: {name}!"
    with pytest.raises(Exception, match=f"^{re.escape(expected)}$") as exc:
        model.process([QOp(name=name, args=[0], metadata={})])
    assert type(exc.value) is Exception


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("init |0>", [0]),
        ("Init", [0]),
        ("Init +Z", [0]),
        ("H", [0]),
        ("CNOT", [(0, 1)]),
        ("measure Z", [0]),
        ("Measure", [0]),
        ("Measure +Z", [0]),
    ],
)
def test_ordinary_operations_preserved(model_type: type, name: str, args: list) -> None:
    """The shared dispatch groups preserve ordinary gates and all init/measurement aliases."""
    model = model_type({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    model.init(2, GenericMachine(num_qubits=2))
    op = QOp(
        name=name,
        args=args,
        returns=["result"],
        metadata={"tag": "preserved"},
        angles=(0.5,),
        sim_name="backend_alias",
    )
    assert [
        (item.name, item.args, item.returns, item.metadata, item.angles, item.sim_name) for item in model.process([op])
    ] == [(name, args, ["result"], {"tag": "preserved"}, (0.5,), "backend_alias")]


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
@pytest.mark.parametrize("name", ["init |0>", "Init", "Init +Z"])
def test_preparation_noise_family(model_type: type, name: str) -> None:
    """Preparation aliases apply preparation noise even when gate noise is zero."""
    model = model_type({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 1})
    model.init(2, GenericMachine(num_qubits=2))
    op = QOp(name=name, args=[0], metadata={})
    assert [(item.name, item.args) for item in model.process([op])] == [(name, [0]), ("X", [0])]


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
@pytest.mark.parametrize("name", ["measure Z", "Measure", "Measure +Z"])
def test_measurement_noise_family(model_type: type, name: str) -> None:
    """Measurement aliases flip result bits while preserving their destination and metadata."""
    model = model_type({"p1": 0, "p2": 0, "p_meas": 1, "p_prep": 0})
    model.init(2, GenericMachine(num_qubits=2))
    op = QOp(name=name, args=[0], returns=["result"], metadata={"tag": "preserved"})
    # The emitted name is deliberately not asserted: the two models disagree about whether
    # a measurement alias survives, because only the leakage helper preserves it. Pinning
    # either spelling here would assert that divergence as intended behaviour. What both
    # models must do is flip the result bit while preserving destination and metadata.
    (emitted,) = model.process([op])
    assert (emitted.args, emitted.returns) == ([0], ["result"])
    assert emitted.metadata == {"tag": "preserved", "bitflips": [0]}
