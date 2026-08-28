# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Regression tests for late measurement values carried across CFG branches."""

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.qsystem import measure_and_reset
from guppylang.std.quantum import discard, h, measure, qubit, x
from pecos import Guppy, sim, state_vector


def test_measurement_value_crosses_branch() -> None:
    """A late measurement value survives an unrelated CFG branch."""

    @guppy
    def main() -> None:
        q = qubit()
        bit = measure_and_reset(q).read()
        flag = True
        if flag:
            x(q)
            marker = 10
        else:
            x(q)
            x(q)
            marker = 20
        result("b", bit)
        result("marker", marker)
        discard(q)

    results = sim(Guppy(main)).quantum(state_vector()).seed(1).run(3).to_dict()
    assert results["b"] == [0, 0, 0]
    assert results["marker"] == [10, 10, 10]


def test_called_measurement_value_crosses_branch() -> None:
    """The same branch carry composes with a surrounding Call frame."""

    @guppy
    def measured_before_branch(q: qubit, flag: bool) -> tuple[bool, int]:
        bit = measure_and_reset(q).read()
        if flag:
            x(q)
            marker = 10
        else:
            x(q)
            x(q)
            marker = 20
        return bit, marker

    @guppy
    def main() -> None:
        q = qubit()
        bit, marker = measured_before_branch(q, True)
        result("b", bit)
        result("marker", marker)
        discard(q)

    results = sim(Guppy(main)).quantum(state_vector()).seed(1).run(3).to_dict()
    assert results["b"] == [0, 0, 0]
    assert results["marker"] == [10, 10, 10]


def test_measurement_value_crosses_dynamic_branch() -> None:
    """Both measurement-derived branch directions carry the later value."""

    @guppy
    def main() -> None:
        flag_q = qubit()
        h(flag_q)
        flag = measure(flag_q).read()

        q = qubit()
        bit = measure_and_reset(q).read()
        if flag:
            x(q)
            marker = 10
        else:
            x(q)
            x(q)
            marker = 20
        result("b", bit)
        result("marker", marker)
        discard(q)

    results = sim(Guppy(main)).quantum(state_vector()).seed(1).run(20).to_dict()
    assert results["b"] == [0] * 20
    assert set(results["marker"]) == {10, 20}


def test_measurement_value_crosses_two_branches() -> None:
    """Ordered replay carries a late value through two branch transitions."""

    @guppy
    def main() -> None:
        q = qubit()
        bit = measure_and_reset(q).read()
        first = True
        if first:
            x(q)
            marker = 10
        else:
            marker = 20
        second = True
        if second:
            x(q)
            marker = marker + 20
        else:
            marker = marker + 40
        result("b", bit)
        result("marker", marker)
        discard(q)

    results = sim(Guppy(main)).quantum(state_vector()).seed(1).run(3).to_dict()
    assert results["b"] == [0, 0, 0]
    assert results["marker"] == [30, 30, 30]


def test_tailloop_remeasurement_uses_last_iteration() -> None:
    """Replay cannot resurrect a superseded TailLoop generation."""

    @guppy
    def main() -> None:
        q = qubit()
        last = False
        for i in range(3):
            if i == 2:
                x(q)
            last = measure_and_reset(q).read()

        flag = True
        if flag:
            x(q)
            marker = 10
        else:
            x(q)
            x(q)
            marker = 20
        result("last", last)
        result("marker", marker)
        discard(q)

    results = sim(Guppy(main)).quantum(state_vector()).seed(1).run(3).to_dict()
    assert results["last"] == [1, 1, 1]
    assert results["marker"] == [10, 10, 10]
