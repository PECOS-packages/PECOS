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


# Qubit ids are neither consecutive nor equal to their position in the batch, and the
# return destinations are unrelated to the qubit ids. An implementation that regenerates
# destinations from the arguments, or that records bit-flips by index rather than by
# qubit id, therefore cannot agree with these expectations by coincidence.
BATCH = [4, 7, 2, 9]
RETURNS = [["c", 3], ["c", 1], ["c", 0], ["c", 2]]


@pytest.mark.parametrize(
    ("pre", "p", "expected"),
    [
        pytest.param(
            (7, 11),
            0.0,
            [
                ("Init -Z", "Init -Z", [7], None, {}),
                ("Z", "Z", [7], None, {}),
                ("Measure", "Measure", BATCH, RETURNS, {}),
            ],
            id="leak-only",
        ),
        pytest.param(
            (7, 11),
            1.0,
            [
                ("Init -Z", "Init -Z", [7], None, {}),
                ("Z", "Z", [7], None, {}),
                ("Measure", "Measure", BATCH, RETURNS, {"bitflips": BATCH}),
            ],
            id="leak-and-flips",
        ),
        pytest.param(
            (11,),
            1.0,
            [("Measure", "Measure", BATCH, RETURNS, {"bitflips": BATCH})],
            id="flips-only",
        ),
    ],
)
def test_measurement_sequence(pre: tuple[int, ...], p: float, expected: list) -> None:
    """Reset only leaked inputs, then measure the entire batch exactly once."""
    machine = FakeMachine(pre)
    op = QOp("Measure", list(BATCH), returns=[list(r) for r in RETURNS], metadata={})
    result = noise_meas_bitflip_leakage(op, p, machine)
    assert [(item.name, item.sim_name, item.args, item.returns, item.metadata) for item in result] == expected
    assert machine.meas_leaked_calls == ([{7}] if 7 in pre else [])


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
    op = QOp(name, [4, 7], returns=[["c", 1], ["c", 0]], metadata={"tag": 7}, sim_name="Measure")
    result = noise_meas_bitflip_leakage(op, p, FakeMachine((4,)))
    # Assert the whole sequence, not just the measurement: checking only result[-1] accepts
    # an implementation that drops the reset prefix or emits it in the wrong order.
    assert [(item.name, item.sim_name, item.args, item.returns, item.metadata) for item in result] == [
        ("Init -Z", "Init -Z", [4], None, {}),
        ("Z", "Z", [4], None, {}),
        (
            name,
            "Measure",
            [4, 7],
            [["c", 1], ["c", 0]],
            {"tag": 7, "bitflips": [4, 7]} if p else {"tag": 7},
        ),
    ]
    result[-1].metadata["extra"] = 8
    assert op.metadata == {"tag": 7}


def test_absent_metadata_is_normalised() -> None:
    """QOp defaults metadata to None; the leakage-only path must not choke on it."""
    op = QOp("Measure", [4, 7], returns=[["c", 1], ["c", 0]])
    assert op.metadata is None
    result = noise_meas_bitflip_leakage(op, 0.0, FakeMachine((4,)))
    assert [(item.name, item.args) for item in result] == [
        ("Init -Z", [4]),
        ("Z", [4]),
        ("Measure", [4, 7]),
    ]
    assert result[-1].metadata == {}


def test_process_emits_leaked_measurement() -> None:
    """The caller retains the full reset and measurement replacement at zero noise."""
    machine = FakeMachine((7, 11))
    model = GenericErrorModel({"p1": 0, "p2": 0, "p_meas": 0, "p_prep": 0})
    model.init(12, machine)
    op = QOp("Measure", list(BATCH), returns=[list(r) for r in RETURNS], metadata={})
    result = model.process([op])
    assert [(item.name, item.sim_name, item.args, item.returns, item.metadata) for item in result] == [
        ("Init -Z", "Init -Z", [7], None, {}),
        ("Z", "Z", [7], None, {}),
        ("Measure", "Measure", BATCH, RETURNS, {}),
    ]
    assert machine.meas_leaked_calls == [{7}]
