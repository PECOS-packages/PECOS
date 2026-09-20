#!/usr/bin/env python3
"""PECOS Guppy Integration Example.

This example demonstrates the complete pipeline from Guppy quantum programming
to execution on PECOS.

Workflow:
1. Write quantum algorithms in Guppy
2. Compile to HUGR intermediate representation
3. Convert HUGR to LLVM IR/QIR
4. Execute on PECOS quantum simulator

Prerequisites:
- Install quantum-pecos: pip install quantum-pecos
- Selene's compiler package is installed with quantum-pecos.
"""

import sys

from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import cx, h, measure, qubit
from pecos import Guppy, sim


def example_bell_state() -> None:
    """Example: Bell state creation and measurement."""

    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        q0 = qubit()
        q1 = qubit()

        # Create entanglement
        h(q0)
        cx(q0, q1)

        # Measure both qubits
        m0 = measure(q0).read()
        m1 = measure(q1).read()

        result("outcome", array(m0, m1))
        return (m0, m1)

    print("\n=== Bell State Example ===")
    print("Guppy function: bell_state")
    print("Expected: Correlated 00 or 11 outcomes")

    data = sim(Guppy(bell_state)).qubits(2).seed(42).run(100).to_dict()
    outcomes = data["outcome"]
    print("[OK] Executed 100 shots")
    print(f"Results: {outcomes[:10]}...")
    correlated = sum(left == right for left, right in outcomes)
    print(f"Correlation rate: {correlated / len(outcomes):.2%}")


def example_quantum_adder() -> None:
    """Example: Simple quantum arithmetic."""

    @guppy
    def quantum_adder() -> bool:
        """Simple quantum computation with classical result."""
        q = qubit()
        h(q)  # Put in superposition
        return measure(q).read()  # Random bit

    print("\n=== Quantum Random Bit Example ===")
    print("Expected: Random 0/1 distribution")

    # This would use the same compilation pipeline
    print("Implementation similar to Bell state example above")


def main() -> int:
    """Run all examples."""
    print("PECOS Guppy Integration Examples")
    print("=" * 40)

    # Run examples
    example_bell_state()
    example_quantum_adder()

    print("\n" + "=" * 40)
    print("Examples complete!")
    print("\nFor full integration:")
    print("Run with: python guppy_integration_example.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
