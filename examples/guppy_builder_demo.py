#!/usr/bin/env python3
"""Demonstrate the Guppy simulation builder pattern and performance benefits.

This example shows how the builder pattern improves performance by
compiling once and running multiple times.
"""

import time

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit
from pecos import Hugr
from pecos import sim as build_sim


@guppy
def bell_state() -> tuple[bool, bool]:
    """Create a Bell state."""
    q0, q1 = qubit(), qubit()
    h(q0)
    cx(q0, q1)
    left, right = measure(q0).read(), measure(q1).read()
    result("left", left)
    result("right", right)
    return left, right


@guppy
def ghz_3qubit() -> tuple[bool, bool, bool]:
    """Create a 3-qubit GHZ state."""
    q0, q1, q2 = qubit(), qubit(), qubit()
    h(q0)
    cx(q0, q1)
    cx(q1, q2)
    first, second, third = measure(q0).read(), measure(q1).read(), measure(q2).read()
    result("first", first)
    result("second", second)
    result("third", third)
    return first, second, third


def demo_builder_pattern() -> None:
    """Demonstrate the builder pattern API."""
    print("=== Selene Engine Builder Pattern Demo (New Unified API) ===\n")

    # 1. Build once, run multiple times
    print("1. Building simulation once...")
    start = time.time()
    # Convert Guppy function to HUGR
    hugr_bytes = bell_state.compile().to_bytes()
    hugr_program = Hugr(hugr_bytes)

    # Build simulation using new API
    sim = build_sim(hugr_program).qubits(2).seed(42).build()
    build_time = time.time() - start
    print(f"   Build time: {build_time:.4f}s\n")

    # Run multiple times without recompiling
    print("2. Running multiple shot counts without recompiling:")
    for shots in [100, 1000, 10000]:
        start = time.time()
        results = sim.run(shots)
        run_time = time.time() - start

        # Count correlations
        results_dict = results.to_dict()
        result_values = [
            2 * left + right for left, right in zip(results_dict["left"], results_dict["right"], strict=True)
        ]
        zeros = result_values.count(0)  # |00⟩
        threes = result_values.count(3)  # |11⟩

        print(f"   {shots:5d} shots: {run_time:.4f}s - |00⟩: {zeros}, |11⟩: {threes}")

    print("\n3. Configuration options:")
    # The new API returns ShotVec objects
    results = build_sim(hugr_program).qubits(2).run(10)
    results_dict = results.to_dict()
    result_values = [2 * left + right for left, right in zip(results_dict["left"], results_dict["right"], strict=True)]
    print(f"   Integer format: {result_values}")

    # Note: Binary string format is not directly available in the new API
    # You can convert integers to binary strings if needed
    binary_strings = [format(val, "02b") for val in result_values]
    print(f"   Binary format (converted): {binary_strings}")


def compare_performance() -> None:
    """Compare performance of builder pattern vs direct execution."""
    print("\n=== Performance Comparison ===\n")

    shot_counts = [100, 100, 100]  # Run 3 times with same shots

    # Method 1: Using direct execution (recompiles each time)
    print("1. Using direct execution (recompiles each time):")
    total_time = 0
    for i, shots in enumerate(shot_counts):
        start = time.time()
        # Convert Guppy to HUGR each time
        hugr_bytes = bell_state.compile().to_bytes()
        hugr_program = Hugr(hugr_bytes)
        build_sim(hugr_program).qubits(2).seed(42).run(shots)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"   Run {i+1}: {elapsed:.4f}s")
    print(f"   Total: {total_time:.4f}s\n")

    # Method 2: Using builder pattern (compile once)
    print("2. Using the simulation builder (compile once):")
    start = time.time()
    # Convert once
    hugr_bytes = bell_state.compile().to_bytes()
    hugr_program = Hugr(hugr_bytes)
    sim = build_sim(hugr_program).qubits(2).seed(42).build()
    build_time = time.time() - start
    print(f"   Build: {build_time:.4f}s")

    run_time = 0
    for i, shots in enumerate(shot_counts):
        start = time.time()
        sim.run(shots)
        elapsed = time.time() - start
        run_time += elapsed
        print(f"   Run {i+1}: {elapsed:.4f}s")

    total_builder_time = build_time + run_time
    print(f"   Total: {total_builder_time:.4f}s")

    speedup = total_time / total_builder_time
    print(f"\n   Speedup: {speedup:.2f}x faster with builder pattern!")


def demo_advanced_features() -> None:
    """Demonstrate advanced features."""
    print("\n=== Advanced Features ===\n")

    # 1. Complex circuit with configuration
    print("1. GHZ state with full configuration:")
    # Convert Guppy function to HUGR
    hugr_bytes = ghz_3qubit.compile().to_bytes()
    hugr_program = Hugr(hugr_bytes)

    sim = build_sim(hugr_program).qubits(3).seed(123).workers(2).build()

    results = sim.run(1000)
    results_dict = results.to_dict()
    result_values = [
        4 * first + 2 * second + third
        for first, second, third in zip(
            results_dict["first"],
            results_dict["second"],
            results_dict["third"],
            strict=True,
        )
    ]

    # Count GHZ correlations
    all_zeros = result_values.count(0)  # |000⟩ = 0
    all_ones = result_values.count(7)  # |111⟩ = 7

    print(f"   |000⟩: {all_zeros/len(result_values):.1%}, |111⟩: {all_ones/len(result_values):.1%}")

    # 2. Multiple configurations
    print("\n2. Using multiple configurations:")
    # Convert bell_state once
    hugr_bytes2 = bell_state.compile().to_bytes()
    hugr_program2 = Hugr(hugr_bytes2)

    results = build_sim(hugr_program2).qubits(2).seed(42).workers(4).run(20)
    results_dict = results.to_dict()
    result_values = [2 * left + right for left, right in zip(results_dict["left"], results_dict["right"], strict=True)]
    print(f"   Results: {result_values[:10]}...")

    # 3. Direct run without explicit build
    print("\n3. Direct run (implicit build):")
    results = build_sim(hugr_program2).qubits(2).seed(99).run(50)
    results_dict = results.to_dict()
    result_values = [2 * left + right for left, right in zip(results_dict["left"], results_dict["right"], strict=True)]
    print(f"   Got {len(result_values)} results")


if __name__ == "__main__":
    demo_builder_pattern()
    compare_performance()
    demo_advanced_features()

    print("\n=== Demo Complete ===")
    print("This demo uses sim(Hugr(...)).qubits(N) on the Selene QIS route!")
