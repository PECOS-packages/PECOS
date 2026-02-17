"""Generate a HUGR file with conditionals for testing HugrEngine.

This script creates a simple quantum circuit with a measurement followed by
a conditional X gate on another qubit.

Usage:
    python scripts/generate_conditional_hugr.py

Output:
    crates/pecos/tests/test_data/hugr/conditional_x.hugr
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr import Main as MainType

try:
    from pecos.qeclib import qubit
    from pecos.slr import CReg, If, Main, QReg, SlrConverter
except ImportError as e:
    print(f"Error importing PECOS: {e}")
    print("Make sure you have installed quantum-pecos:")
    print("  cd python/quantum-pecos && pip install -e .")
    sys.exit(1)


def _create_conditional_x_circuit() -> MainType:
    """Create a circuit: H q0 -> Measure q0 -> If result: X q1."""
    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Apply H to first qubit
        qubit.H(q[0]),
        # Measure first qubit
        qubit.Measure(q[0]) > c[0],
        # Conditional X on second qubit based on measurement
        If(c[0])
        .Then(
            qubit.X(q[1]),
            qubit.Measure(q[1]) > c[1],
        )
        .Else(
            # Else block needs to consume q[1] for linearity
            qubit.Measure(q[1])
            > c[1],
        ),
    )


def _create_simple_conditional() -> MainType:
    """Create a simpler conditional circuit."""
    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Measure first qubit
        qubit.H(q[0]),
        qubit.Measure(q[0]) > c[0],
        # Both branches measure q[1]
        If(c[0])
        .Then(
            qubit.X(q[1]),  # X gate if measured 1
            qubit.Measure(q[1]) > c[1],
        )
        .Else(
            qubit.Measure(q[1]) > c[1],  # Just measure if 0
        ),
    )


def main() -> None:
    """Generate the conditional HUGR test file."""
    output_dir = Path(__file__).parent.parent / "crates" / "pecos" / "tests" / "test_data" / "hugr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to create and compile the conditional circuit
    print("Creating conditional circuit...")
    try:
        prog = _create_simple_conditional()
        print("Program created successfully")

        # Generate Guppy code first
        guppy_code = SlrConverter(prog).guppy()
        print(f"Guppy code:\n{guppy_code}\n")

        # Compile to HUGR
        print("Compiling to HUGR...")
        hugr_package = SlrConverter(prog).hugr()

        if hugr_package is None:
            print("Error: HUGR compilation returned None")
            sys.exit(1)

        # Serialize the package to bytes
        print(f"Package type: {type(hugr_package)}")

        # Try to serialize using to_bytes or package.to_bytes
        if hasattr(hugr_package, "to_bytes"):
            hugr_bytes = hugr_package.to_bytes()
        elif hasattr(hugr_package, "package") and hasattr(
            hugr_package.package,
            "to_bytes",
        ):
            hugr_bytes = hugr_package.package.to_bytes()
        else:
            # Try serialization via model dump
            if hasattr(hugr_package, "to_json"):
                hugr_bytes = hugr_package.to_json().encode("utf-8")
            else:
                print(f"Cannot serialize package. Methods: {dir(hugr_package)}")
                sys.exit(1)

        # Save to file
        output_path = output_dir / "conditional_x.hugr"
        with output_path.open("wb") as f:
            f.write(hugr_bytes)

        print(f"HUGR saved to: {output_path}")
        print(f"File size: {len(hugr_bytes)} bytes")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
