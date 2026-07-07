#!/usr/bin/env python3
"""Test for-loop behavior."""

import os

from guppylang.decorator import guppy
from guppylang.std.quantum import h, measure, qubit
from pecos import Guppy, sim
from pecos_rslib import state_vector


@guppy
def loop_with_measure() -> int:
    """For-loop with quantum operations inside."""
    count = 0
    for _i in range(3):
        q = qubit()
        h(q)
        if measure(q):
            count = count + 1
    return count


def test_for_loop_with_measurements() -> None:
    """The loop must run exactly 3 iterations (3 measurements per shot), and
    with H per iteration both outcomes must occur across 20 seeded shots --
    a zero-iteration or frozen loop cannot satisfy either."""
    results = sim(Guppy(loop_with_measure)).qubits(10).quantum(state_vector()).seed(42).run(20).to_dict()
    measurements = results["measurements"]
    assert len(measurements) == 20
    for shot in measurements:
        assert len(shot) == 3, f"expected 3 loop measurements, got {shot}"
    outcomes = {m for shot in measurements for m in shot}
    assert outcomes == {0, 1}, f"H per iteration must yield both outcomes, got {outcomes}"


if __name__ == "__main__":
    os.environ["RUST_LOG"] = "pecos_hugr::engine=debug"
    print("Testing for-loop with measurements...")
    try:
        results = sim(Guppy(loop_with_measure)).qubits(10).quantum(state_vector()).seed(42).run(1).to_dict()
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
