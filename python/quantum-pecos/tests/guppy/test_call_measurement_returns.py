# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Regression tests for measurement-derived values returned through Calls."""

from guppylang import guppy
from guppylang.std.builtins import array, owned, result
from guppylang.std.qsystem import measure_and_reset
from guppylang.std.quantum import cx, discard, h, measure, qubit
from pecos import Guppy, sim, state_vector


def test_measurement_return_crosses_call() -> None:
    """A Call must remain active until its measurement return materializes."""

    @guppy
    def measure_one(q: qubit @ owned) -> bool:
        h(q)
        return measure(q)

    @guppy
    def main() -> None:
        value = measure_one(qubit())
        result("value", value)

    values = sim(Guppy(main)).quantum(state_vector()).seed(1).run(3).to_dict()["value"]
    assert len(values) == 3
    assert set(values) <= {0, 1}


def test_measurement_result_in_same_block() -> None:
    """The direct producer-to-consumer path remains a passing control."""

    @guppy
    def main() -> None:
        q = qubit()
        h(q)
        result("value", measure(q))

    values = sim(Guppy(main)).quantum(state_vector()).seed(1).run(3).to_dict()["value"]
    assert len(values) == 3
    assert set(values) <= {0, 1}


def test_measurement_return_crosses_two_calls() -> None:
    """Pending returns cascade correctly through two nested Call frames."""

    @guppy
    def inner(q: qubit @ owned) -> bool:
        h(q)
        return measure(q)

    @guppy
    def outer(q: qubit @ owned) -> bool:
        return inner(q)

    @guppy
    def main() -> None:
        value = outer(qubit())
        result("value", value)

    values = sim(Guppy(main)).quantum(state_vector()).seed(1).run(3).to_dict()["value"]
    assert len(values) == 3
    assert set(values) <= {0, 1}


def test_struct_helper_and_array_measurement_returns() -> None:
    """Struct, helper Calls, and array returns compose across measurement rounds."""

    @guppy.struct
    class Patch:
        data: array[qubit, 4]

    @guppy
    def stabilizer(ax: qubit, data: array[qubit, 4]) -> bool:
        h(ax)
        cx(ax, data[0])
        h(ax)
        return measure_and_reset(ax)

    @guppy
    def initialize(patch: Patch, ax: qubit) -> array[bool, 2]:
        first = stabilizer(ax, patch.data)
        second = stabilizer(ax, patch.data)
        return array(first, second)

    @guppy
    def final_measurement(patch: Patch @ owned) -> array[bool, 4]:
        return array(measure(q) for q in patch.data)

    @guppy
    def main() -> None:
        patch = Patch(array(qubit() for _ in range(4)))
        ax = qubit()
        result("initial", initialize(patch, ax))
        result("final", final_measurement(patch))
        discard(ax)

    results = sim(Guppy(main)).quantum(state_vector()).seed(1).run(1).to_dict()
    assert len(results["initial"][0]) == 2
    assert len(results["final"][0]) == 4
