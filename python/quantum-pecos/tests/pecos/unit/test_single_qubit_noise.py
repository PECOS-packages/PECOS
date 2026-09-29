"""Regression tests for single-qubit replacement operation sequences."""

import pecos as pc
import pytest
from pecos.machines.generic_machine import GenericMachine
from pecos.noise.depolarizing_error_model import DepolarizingErrorModel
from pecos.noise.generic_error_model import GenericErrorModel
from pecos.noise.noise_impl.noise_sq_depolarizing import noise_sq_depolarizing
from pecos.noise.noise_impl.noise_sq_depolarizing_leakage import noise_sq_depolarizing_leakage
from pecos.reps.pyphir.op_types import QOp


class FakeMachine:
    """Expose leak calls and return multiple operations for each call."""

    def __init__(self, pre: tuple[int, ...] = ()) -> None:
        """Start with the supplied pre-existing leakage."""
        self.leaked_qubits = set(pre)
        self.leak_calls: list[set[int]] = []

    def leak(self, qubits: set[int]) -> list[QOp]:
        """Track leakage and return an observable sequence."""
        self.leak_calls.append(set(qubits))
        self.leaked_qubits |= qubits
        return [QOp(name=name, args=sorted(qubits), metadata={}) for name in ("Init", "Z")]


@pytest.mark.parametrize("with_leakage", [False, True])
@pytest.mark.parametrize(
    ("noise_dict", "faults"),
    [
        ({"X": 0.5, "Z": 0.5}, [("X", [0, 3]), ("Z", [1])]),
        ({"X": 1 / 3, "Y": 1 / 3, "Z": 1 / 3}, [("Y", [0]), ("Z", [1]), ("X", [3])]),
    ],
)
def test_distinct_symbols_emit_gate_once(with_leakage: bool, noise_dict: dict, faults: list) -> None:
    """Emit one gate, then grouped faults in draw order, not model or sorted order."""
    pc.random.seed(7)
    op = QOp(name="H", args=[0, 1, 2, 3], metadata={})
    if with_leakage:
        result = noise_sq_depolarizing_leakage(op, 0.8, noise_dict, FakeMachine())
    else:
        result = noise_sq_depolarizing(op, 0.8, noise_dict)
    assert [(item.name, item.args) for item in result] == [("H", [0, 1, 2, 3]), *faults]


@pytest.mark.parametrize(
    ("seed", "p", "noise_dict", "expected", "leak_calls"),
    [
        (
            11,
            0.5,
            {"L": 1.0},
            [("H", [0, 1, 2, 3]), ("Init", [0, 1, 3]), ("Z", [0, 1, 3])],
            [{0, 1, 3}],
        ),
        (
            7,
            0.8,
            {"X": 1 / 3, "Y": 1 / 3, "L": 1 / 3},
            [("H", [0, 1, 2, 3]), ("Y", [0]), ("Init", [1]), ("Z", [1]), ("X", [3])],
            [{1}],
        ),
    ],
)
def test_new_leakage_follows_gate(seed: int, p: float, noise_dict: dict, expected: list, leak_calls: list) -> None:
    """New leakage leaves the gate intact and expands in its symbol's emission slot."""
    pc.random.seed(seed)
    machine = FakeMachine()
    result = noise_sq_depolarizing_leakage(QOp("H", [0, 1, 2, 3], metadata={}), p, noise_dict, machine)
    assert [(item.name, item.args) for item in result] == expected
    assert machine.leak_calls == leak_calls


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_preexisting_leakage_narrows_gate(p: float) -> None:
    """Only inputs leaked on entry are excluded from both the gate and faults."""
    pc.random.seed(7)
    result = noise_sq_depolarizing_leakage(
        QOp("H", [0, 1, 2, 3], metadata={}),
        p,
        {"X": 1.0},
        FakeMachine(pre=(0, 2)),
    )
    expected = [("H", [1, 3])]
    if p:
        expected.append(("X", [1, 3]))
    # The narrowed argument list comes from a set difference, so compare sorted
    # arguments: their order carries no meaning for a batched single-qubit gate.
    assert [(item.name, sorted(item.args)) for item in result] == expected


def test_all_preexisting_leakage_replaces_with_nothing() -> None:
    """An empty replacement contains no zero-argument gate."""
    result = noise_sq_depolarizing_leakage(
        QOp("H", [0, 1, 2, 3], metadata={}),
        0.0,
        {"X": 1.0},
        FakeMachine(pre=(0, 1, 2, 3)),
    )
    assert result == []


@pytest.mark.parametrize(
    "model_cls",
    [GenericErrorModel, DepolarizingErrorModel],
)
def test_model_emits_batched_gate_once(model_cls: type) -> None:
    """A batch drawing several Pauli symbols still carries the gate exactly once."""
    model = model_cls(
        error_params={"p1": 0.6, "p2": 0.0, "p_meas": 0.0, "p_init": 0.0, "p_prep": 0.0},
    )
    model.init(4, machine=GenericMachine(num_qubits=4))
    pc.random.seed(7)
    op = QOp(name="H", args=[0, 1, 2, 3], metadata={})
    result = model.process([op])

    gates = [item for item in result if item.name == "H"]
    faults = {item.name for item in result if item.name != "H"}
    # Without several distinct symbols the gate count would be one regardless,
    # so the assertion below would hold vacuously.
    assert len(faults) > 1, f"seed drew only {faults}; the case under test needs several"
    assert len(gates) == 1
    assert gates[0].args == [0, 1, 2, 3]
    assert result[0] is gates[0], "the gate precedes its faults"


def test_model_replaces_fully_leaked_batch_with_nothing() -> None:
    """A batch whose every qubit is already leaked reaches the simulator as nothing."""
    model = GenericErrorModel(
        error_params={"p1": 0.6, "p2": 0.0, "p_meas": 0.0, "p_init": 0.0, "p_prep": 0.0},
    )
    model.init(4, machine=FakeMachine(pre=(0, 1, 2, 3)))
    pc.random.seed(7)
    result = model.process([QOp(name="H", args=[0, 1, 2, 3], metadata={})])
    assert result == []


def test_narrowing_accepts_absent_metadata() -> None:
    """QOp defaults metadata to None; narrowing must not choke on it."""
    op = QOp(name="H", args=[0, 1, 2, 3])
    assert op.metadata is None
    result = noise_sq_depolarizing_leakage(op, 0.0, {"X": 1.0}, FakeMachine(pre=(0,)))

    (narrowed,) = result
    assert sorted(narrowed.args) == [1, 2, 3]
    assert narrowed.metadata == {}


def test_narrowed_gate_keeps_executable_fields() -> None:
    """Narrowing around a leaked qubit preserves what the simulator needs to run it."""
    op = QOp(
        name="RXY1Q",
        args=[0, 1, 2, 3],
        metadata={"key": 1},
        angles=(3.14159, 0.0),
        sim_name="X",
    )
    result = noise_sq_depolarizing_leakage(op, 0.0, {"X": 1.0}, FakeMachine(pre=(0,)))

    (narrowed,) = result
    assert sorted(narrowed.args) == [1, 2, 3]
    assert narrowed.angles == (3.14159, 0.0)
    assert narrowed.sim_name == "X"
    assert narrowed.metadata == {"key": 1}
    # The narrowed operation must not share mutable state with, or disturb, the original.
    assert narrowed.metadata is not op.metadata
    assert op.args == [0, 1, 2, 3]


@pytest.mark.parametrize("with_leakage", [False, True])
def test_no_fault_returns_none(with_leakage: bool) -> None:
    """Without faults or existing leakage, the caller retains the original gate."""
    op = QOp(name="H", args=[0, 1, 2, 3], metadata={})
    if with_leakage:
        result = noise_sq_depolarizing_leakage(op, 0.0, {"X": 1.0}, FakeMachine())
    else:
        result = noise_sq_depolarizing(op, 0.0, {"X": 1.0})
    assert result is None
