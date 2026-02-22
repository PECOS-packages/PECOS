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


if __name__ == "__main__":
    os.environ["RUST_LOG"] = "pecos_guppy_hugr::engine=debug"
    print("Testing for-loop with measurements...")
    try:
        results = sim(Guppy(loop_with_measure)).qubits(10).quantum(state_vector()).seed(42).run(1).to_dict()
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
