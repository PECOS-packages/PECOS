"""Regression tests for measurement replacement operation sequences."""

import pytest
from pecos.noise.generic_error_model import GenericErrorModel
from pecos.noise.noise_impl.noise_meas_bitflip_leakage import noise_meas_bitflip_leakage
from pecos.reps.pyphir.op_types import QOp


class FakeMachine:
    """Expose measurement leakage calls and their ordered operations."""

    def __init__(self, pre: tuple[int, ...] = ()) -> None:
        """Start with the supplied pre-existing leakage."""
        self.leaked_qubits = set(pre)
        self.meas_leaked_calls: list[set[int]] = []

    def meas_leaked(self, qubits: set[int]) -> list[QOp]:
        """Track resets and return multiple observable operations."""
        self.meas_leaked_calls.append(set(qubits))
        self.leaked_qubits -= qubits
        return [QOp(name=name, args=sorted(qubits), metadata={}) for name in ("Init -Z", "Z")]


@pytest.mark.parametrize(
    ("pre", "p", "expected"),
    [
        pytest.param(
            (0, 9),
            0.0,
            [
                ("Init -Z", [0], None, {}),
                ("Z", [0], None, {}),
                ("Measure", [0, 1, 2, 3], [["m", i] for i in range(4)], {}),
            ],
            id="leak-only",
        ),
        pytest.param(
            (0, 9),
            1.0,
            [
                ("Init -Z", [0], None, {}),
                ("Z", [0], None, {}),
                ("Measure", [0, 1, 2, 3], [["m", i] for i in range(4)], {"bitflips": [0, 1, 2, 3]}),
            ],
            id="leak-and-flips",
        ),
        pytest.param(
            (9,),
            1.0,
            [("Measure", [0, 1, 2, 3], [["m", i] for i in range(4)], {"bitflips": [0, 1, 2, 3]})],
            id="flips-only",
        ),
    ],
)
def test_measurement_sequence(pre: tuple[int, ...], p: float, expected: list) -> None:
    """Reset only leaked inputs, then measure the entire batch exactly once."""
    machine = FakeMachine(pre)
    op = QOp("Measure", [0, 1, 2, 3], returns=[["m", i] for i in range(4)], metadata={})
    result = noise_meas_bitflip_leakage(op, p, machine)
    assert [(item.name, item.args, item.returns, item.metadata) for item in result] == expected
    assert machine.meas_leaked_calls == ([{0}] if 0 in pre else [])


def test_neither_returns_none() -> None:
    """Leakage outside the batch does not replace the original measurement."""
    machine = FakeMachine((9,))
    result = noise_meas_bitflip_leakage(QOp("Measure", [0, 1, 2, 3]), 0.0, machine)
    assert result is None
    assert machine.meas_leaked_calls == []


@pytest.mark.parametrize("p", [0.0, 1.0])
@pytest.mark.parametrize("name", ["measure Z", "Measure +Z"])
def test_preserves_fields_and_metadata(p: float, name: str) -> None:
    """Copy executable fields and isolate metadata with and without bit-flips."""
    op = QOp(name, [0, 1], returns=[["m", 0], ["m", 1]], metadata={"tag": 7}, sim_name="Measure")
    result = noise_meas_bitflip_leakage(op, p, FakeMachine((0,)))
    measurement = result[-1]
    assert (measurement.name, measurement.sim_name, measurement.args, measurement.returns, measurement.metadata) == (
        name,
        "Measure",
        [0, 1],
        [["m", 0], ["m", 1]],
        {"tag": 7, "bitflips": [0, 1]} if p else {"tag": 7},
    )
    measurement.metadata["extra"] = 8
    assert op.metadata == {"tag": 7}


def test_process_emits_leaked_measurement() -> None:
    """The caller retains the full reset and measurement replacement at zero noise."""
    machine = FakeMachine((0, 9))
    model = GenericErrorModel({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    model.init(10, machine)
    op = QOp("Measure", [0, 1, 2, 3], returns=[["m", i] for i in range(4)], metadata={})
    result = model.process([op])
    assert [(item.name, item.args, item.returns, item.metadata) for item in result] == [
        ("Init -Z", [0], None, {}),
        ("Z", [0], None, {}),
        ("Measure", [0, 1, 2, 3], [["m", i] for i in range(4)], {}),
    ]
    assert machine.meas_leaked_calls == [{0}]
