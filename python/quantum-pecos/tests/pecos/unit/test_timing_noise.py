"""Timing operations must fail before noise bypasses or simulator emission."""

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
def test_timing_rejected(model_type: type, name: str, metadata: dict | None) -> None:
    """Known timing operations report missing physics, including bypassed/default metadata."""
    model = model_type({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    model.init(2, GenericMachine(num_qubits=2))
    op = QOp(name=name, args=[0, 1]) if metadata is None else QOp(name=name, args=[0, 1], metadata=metadata)
    expected = f"Noise for timing operation {name} is not implemented"
    if metadata is not None and "duration" in metadata:
        expected += f" (duration={metadata['duration']})"
    with pytest.raises(NotImplementedError, match=f"^{re.escape(expected)}$"):
        model.process([op])


@pytest.mark.parametrize("model_type", [GenericErrorModel, DepolarizingErrorModel])
def test_unknown_gate_remains_unknown(model_type: type) -> None:
    """Unknown names retain the existing exception and diagnostic."""
    model = model_type({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    model.init(2, GenericMachine(num_qubits=2))
    with pytest.raises(Exception, match=r"^This error model doesn't handle gate: UnknownGate!$") as exc:
        model.process([QOp(name="UnknownGate", args=[0], metadata={})])
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
    op = QOp(name=name, args=args, returns=["result"], metadata={})
    assert [(item.name, item.args) for item in model.process([op])] == [(name, args)]
