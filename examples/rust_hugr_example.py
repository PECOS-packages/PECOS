#!/usr/bin/env python3
"""PECOS Rust HUGR Backend Example.

This example demonstrates HUGR compilation and QIS execution in PECOS.

Features demonstrated:
1. Compiling Guppy functions to HUGR
2. Compiling HUGR to QIS (LLVM IR with quantum instructions)
3. Running quantum simulations with the sim() API
"""

from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit
from pecos import Guppy, sim
from pecos_rslib import compile_hugr_to_qis


def example_hugr_compilation() -> None:
    """Demonstrate HUGR to QIS compilation."""
    print("\n=== HUGR Compilation Example ===")

    # Define a simple quantum function
    @guppy
    def quantum_random() -> bool:
        """Generate a random bit using quantum superposition."""
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR
    package = quantum_random.compile()
    hugr_bytes = package.to_bytes()
    print(f"Compiled to HUGR: {len(hugr_bytes)} bytes")

    # Compile HUGR to QIS
    qis_code = compile_hugr_to_qis(hugr_bytes)
    print(f"Compiled to QIS: {len(qis_code)} characters")
    print("QIS preview:")
    print(qis_code[:300] + "..." if len(qis_code) > 300 else qis_code)


def example_simulation() -> None:
    """Demonstrate running simulations with the sim() API."""
    print("\n=== Simulation Example ===")

    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create Bell state and measure both qubits."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)

    # Run simulation
    results = sim(Guppy(bell_state)).seed(42).run(100)
    print(f"Bell state results: {results}")


def example_direct_guppy() -> None:
    """Demonstrate direct Guppy function simulation."""
    print("\n=== Direct Guppy Simulation ===")

    @guppy
    def ghz_state() -> tuple[bool, bool, bool]:
        """Create 3-qubit GHZ state."""
        q0 = qubit()
        q1 = qubit()
        q2 = qubit()
        h(q0)
        cx(q0, q1)
        cx(q1, q2)
        return measure(q0), measure(q1), measure(q2)

    # sim() accepts Guppy functions directly
    results = sim(ghz_state).seed(42).run(50)
    print(f"GHZ state results: {results}")


def main() -> None:
    """Run all examples."""
    print("PECOS HUGR Compilation Examples")
    print("=" * 50)

    example_hugr_compilation()
    example_simulation()
    example_direct_guppy()

    print("\n" + "=" * 50)
    print("Examples complete!")


if __name__ == "__main__":
    main()
