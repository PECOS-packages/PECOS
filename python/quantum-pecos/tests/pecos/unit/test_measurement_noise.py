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


class FalseyDict(dict):
    """A valid metadata dict that is falsey, to catch truthiness-based normalisation."""

    def __bool__(self) -> bool:
        """Report falsey while still holding entries."""
        return False


@pytest.mark.parametrize(
    ("metadata", "p", "expected_meta"),
    [
        pytest.param(None, 0.0, {}, id="absent"),
        pytest.param(None, 1.0, {"bitflips": [4, 7]}, id="absent-with-flips"),
        # A falsey dict keeps its entries: `op.metadata or {}` would erase them.
        pytest.param(FalseyDict({"tag": 2}), 0.0, {"tag": 2}, id="falsey-dict"),
        pytest.param(FalseyDict({"tag": 2}), 1.0, {"tag": 2, "bitflips": [4, 7]}, id="falsey-dict-flips"),
    ],
)
def test_metadata_normalised_without_discarding_contents(
    metadata: dict | None,
    p: float,
    expected_meta: dict,
) -> None:
    """Only absent metadata is normalised, and a dict never loses its entries."""
    op = QOp("measure Z", [4, 7], returns=[["c", 1], ["c", 0]], metadata=metadata, sim_name="Measure")
    result = noise_meas_bitflip_leakage(op, p, FakeMachine((4,)))
    # Pin the executable fields too: asserting only names, args and metadata accepts a
    # helper that drops returns or rewrites the simulator gate on this path.
    assert [(item.name, item.sim_name, item.args, item.returns, item.metadata) for item in result] == [
        ("Init -Z", "Init -Z", [4], None, {}),
        ("Z", "Z", [4], None, {}),
        ("measure Z", "Measure", [4, 7], [["c", 1], ["c", 0]], expected_meta),
    ]
    # The input operation keeps its own metadata object and contents.
    assert result[-1].metadata is not op.metadata
    assert op.metadata == metadata


@pytest.mark.parametrize("metadata", [False, 0])
def test_metadata_normalisation_does_not_widen_to_falsey_non_dict(metadata: object) -> None:
    """A falsey non-dict still fails rather than being treated as absent metadata.

    `False` and `0` raised before absent metadata was normalised, and must keep doing so.
    Empty iterables such as `[]` and `""` are accepted by `dict()` and so were already
    accepted before this change; validating the metadata type belongs to #925.
    """
    op = QOp("Measure", [4, 7], returns=[["c", 1], ["c", 0]], metadata=metadata)
    with pytest.raises(TypeError):
        noise_meas_bitflip_leakage(op, 0.0, FakeMachine((4,)))


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
