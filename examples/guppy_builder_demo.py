#!/usr/bin/env python3
"""Demonstrate the guppy_sim builder pattern and performance benefits.

This example shows how the builder pattern improves performance by
compiling once and running multiple times.
"""

# Add quantum-pecos to path
import sys
import time

from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit

sys.path.append("python/quantum-pecos/src")

from pecos._compilation import GuppyFrontend
from pecos_rslib import selene_engine
from pecos_rslib.programs import HugrProgram


@guppy
def bell_state() -> tuple[bool, bool]:
    """Create a Bell state."""
    q0, q1 = qubit(), qubit()
    h(q0)
    cx(q0, q1)
    return measure(q0), measure(q1)


@guppy
def ghz_3qubit() -> tuple[bool, bool, bool]:
    """Create a 3-qubit GHZ state."""
    q0, q1, q2 = qubit(), qubit(), qubit()
    h(q0)
    cx(q0, q1)
    cx(q1, q2)
    return measure(q0), measure(q1), measure(q2)


def demo_builder_pattern() -> None:
    """Demonstrate the builder pattern API."""
    print("=== Selene Engine Builder Pattern Demo (New Unified API) ===\n")

    # 1. Build once, run multiple times
    print("1. Building simulation once...")
    start = time.time()
    # Convert Guppy function to HUGR
    frontend = GuppyFrontend()
    hugr_bytes = frontend.guppy_to_hugr(bell_state)
    hugr_program = HugrProgram.from_bytes(hugr_bytes)

    # Build simulation using new API
    sim = selene_engine().program(hugr_program).to_sim().seed(42).build()
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
        # The new API returns a dict with register names as keys
        # For a single return value, it's typically under "_result" or similar key
        result_values = next(iter(results_dict.values())) if results_dict else []
        zeros = result_values.count(0)  # |00⟩
        threes = result_values.count(3)  # |11⟩

        print(f"   {shots:5d} shots: {run_time:.4f}s - |00⟩: {zeros}, |11⟩: {threes}")

    print("\n3. Configuration options:")
    # The new API returns ShotVec objects
    results = selene_engine().program(hugr_program).to_sim().run(10)
    results_dict = results.to_dict()
    result_values = next(iter(results_dict.values())) if results_dict else []
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
        frontend = GuppyFrontend()
        hugr_bytes = frontend.guppy_to_hugr(bell_state)
        hugr_program = HugrProgram.from_bytes(hugr_bytes)
        selene_engine().program(hugr_program).to_sim().seed(42).run(shots)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"   Run {i+1}: {elapsed:.4f}s")
    print(f"   Total: {total_time:.4f}s\n")

    # Method 2: Using builder pattern (compile once)
    print("2. Using selene_engine builder (compile once):")
    start = time.time()
    # Convert once
    frontend = GuppyFrontend()
    hugr_bytes = frontend.guppy_to_hugr(bell_state)
    hugr_program = HugrProgram.from_bytes(hugr_bytes)
    sim = selene_engine().program(hugr_program).to_sim().seed(42).build()
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
    frontend = GuppyFrontend()
    hugr_bytes = frontend.guppy_to_hugr(ghz_3qubit)
    hugr_program = HugrProgram.from_bytes(hugr_bytes)

    sim = selene_engine().program(hugr_program).to_sim().seed(123).workers(2).build()

    results = sim.run(1000)
    results_dict = results.to_dict()
    result_values = next(iter(results_dict.values())) if results_dict else []

    # Count GHZ correlations
    all_zeros = result_values.count(0)  # |000⟩ = 0
    all_ones = result_values.count(7)  # |111⟩ = 7

    print(f"   |000⟩: {all_zeros/10:.1%}, |111⟩: {all_ones/10:.1%}")

    # 2. Multiple configurations
    print("\n2. Using multiple configurations:")
    # Convert bell_state once
    frontend2 = GuppyFrontend()
    hugr_bytes2 = frontend2.guppy_to_hugr(bell_state)
    hugr_program2 = HugrProgram.from_bytes(hugr_bytes2)

    results = selene_engine().program(hugr_program2).to_sim().seed(42).workers(4).run(20)
    results_dict = results.to_dict()
    result_values = next(iter(results_dict.values())) if results_dict else []
    print(f"   Results: {result_values[:10]}...")

    # 3. Direct run without explicit build
    print("\n3. Direct run (implicit build):")
    results = selene_engine().program(hugr_program2).to_sim().seed(99).run(50)
    results_dict = results.to_dict()
    result_values = next(iter(results_dict.values())) if results_dict else []
    print(f"   Got {len(result_values)} results")


if __name__ == "__main__":
    demo_builder_pattern()
    compare_performance()
    demo_advanced_features()

    print("\n=== Demo Complete ===")
    print("This demo now uses the new unified selene_engine() API!")
