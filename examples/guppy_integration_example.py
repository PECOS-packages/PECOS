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
- Build hugr-quantum-llvm compiler (or provide path to existing binary).
"""

import sys
from pathlib import Path

from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit
from pecos._compilation import GuppyFrontend


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
        m0 = measure(q0)
        m1 = measure(q1)

        return (m0, m1)

    print("\n=== Bell State Example ===")
    print("Guppy function:", bell_state.__name__)
    print("Expected: Correlated 00 or 11 outcomes")

    # Set up paths to compilation tools
    # These would need to be updated based on your installation
    hugr_compiler = Path(
        "../quantum-compilation-examples/hugr_quantum_llvm/target/release/hugr-to-llvm",
    )
    format_converter = Path("../quantum-compilation-examples/convert_hugr_format.py")

    if not hugr_compiler.exists():
        print(f"[WARNING] HUGR compiler not found at {hugr_compiler}")
        print("Please build hugr-quantum-llvm or update the path")
        return

    if not format_converter.exists():
        print(f"[WARNING] Format converter not found at {format_converter}")
        print("Using compilation without format conversion")
        format_converter = None

    try:
        # Create Guppy frontend
        frontend = GuppyFrontend(
            hugr_to_llvm_binary=hugr_compiler,
            format_converter=format_converter,
        )

        # Compile and run
        results = frontend.compile_and_run(bell_state, shots=100)

        print(f"[OK] Executed {results['shots']} shots")
        print(f"Results: {results['results'][:10]}...")  # Show first 10 results

        # Analyze correlations
        if results["results"]:
            correlated = sum(1 for r in results["results"] if r[0] == r[1])
            correlation_rate = correlated / len(results["results"])
            print(f"Correlation rate: {correlation_rate:.2%}")
            print("Expected: ~100% for ideal Bell state")

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        print("This is expected if compilation tools are not set up")
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}")
        print("This is expected if compilation tools are not set up")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        print("This is expected if compilation tools are not set up")


def example_quantum_adder() -> None:
    """Example: Simple quantum arithmetic."""

    @guppy
    def quantum_adder() -> bool:
        """Simple quantum computation with classical result."""
        q = qubit()
        h(q)  # Put in superposition
        return measure(q)  # Random bit

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
    print("1. Build hugr-quantum-llvm compiler")
    print("2. Update paths in this script")
    print("3. Run with: python guppy_integration_example.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
