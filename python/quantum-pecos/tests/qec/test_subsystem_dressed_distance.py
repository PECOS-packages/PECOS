"""Tests for the subsystem_dressed_distance Python binding."""

from pecos.qec import DistanceResult, subsystem_dressed_distance
from pecos.quantum import PauliString

P = PauliString.from_dense_str


def _bacon_shor_3x3():
    """Bacon-Shor on a 3x3 grid; dressed distance 3, logical qubit 1."""

    def q(r, c):
        return r * 3 + c

    gauges = []
    for r in range(3):
        for c in range(2):
            s = ["I"] * 9
            s[q(r, c)] = "Z"
            s[q(r, c + 1)] = "Z"
            gauges.append(P("".join(s)))
    for r in range(2):
        for c in range(3):
            s = ["I"] * 9
            s[q(r, c)] = "X"
            s[q(r + 1, c)] = "X"
            gauges.append(P("".join(s)))
    stabilizers = []
    for r in range(2):
        s = ["I"] * 9
        for c in range(3):
            s[q(r, c)] = "X"
            s[q(r + 1, c)] = "X"
        stabilizers.append(P("".join(s)))
    for c in range(2):
        s = ["I"] * 9
        for r in range(3):
            s[q(r, c)] = "Z"
            s[q(r, c + 1)] = "Z"
        stabilizers.append(P("".join(s)))
    lx = ["I"] * 9
    for c in range(3):
        lx[q(0, c)] = "X"
    lz = ["I"] * 9
    for r in range(3):
        lz[q(r, 0)] = "Z"
    return stabilizers, gauges, [P("".join(lz))], [P("".join(lx))]


def test_bacon_shor_dressed_distance_is_three():
    stabilizers, gauges, lzs, lxs = _bacon_shor_3x3()
    result = subsystem_dressed_distance(9, stabilizers, gauges, lzs, lxs, 9)
    assert isinstance(result, DistanceResult)
    assert result.distance == 3
    assert result.min_weight_operator.weight() == 3


def test_returns_none_past_the_weight_budget():
    stabilizers, gauges, lzs, lxs = _bacon_shor_3x3()
    assert subsystem_dressed_distance(9, stabilizers, gauges, lzs, lxs, 2) is None


def test_rejects_a_logical_that_anticommutes_with_a_stabilizer():
    import pytest

    stabilizers, gauges, lzs, _lxs = _bacon_shor_3x3()
    bad_lx = [P("XIIXIIXII")]  # a single column: anticommutes with the horizontal ZZ gauges/stabs
    with pytest.raises(ValueError, match="anticommute"):
        subsystem_dressed_distance(9, stabilizers, gauges, lzs, bad_lx, 9)
