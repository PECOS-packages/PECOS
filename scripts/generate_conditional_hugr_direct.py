"""Generate a HUGR file with conditionals using guppylang directly.

This script creates a simple quantum circuit with a measurement followed by
a conditional X gate on another qubit.

Usage:
    python scripts/generate_conditional_hugr_direct.py

Output:
    crates/pecos/tests/test_data/hugr/conditional_x.hugr
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

try:
    from guppylang.decorator import guppy
    from guppylang.std import quantum
    from guppylang.std.builtins import array, result
except ImportError as e:
    print(f"Error importing guppylang: {e}")
    print("Make sure guppylang is installed")
    sys.exit(1)


@guppy
def main() -> None:
    """Conditional X circuit: H q0 -> Measure q0 -> If result: X q1."""
    q = array(quantum.qubit() for _ in range(2))
    c = array(False for _ in range(2))

    # Unpack for individual access
    q_0, q_1 = q
    c_0, c_1 = c

    # Apply H to first qubit
    quantum.h(q_0)

    # Measure first qubit
    c_0 = quantum.measure(q_0)

    # Conditional X on second qubit based on measurement
    if c_0:
        quantum.x(q_1)
        c_1 = quantum.measure(q_1)
    else:
        c_1 = quantum.measure(q_1)

    # Store results
    c = array(c_0, c_1)
    result("c", c)


def main_script() -> None:
    """Run the main script to generate the HUGR file."""
    output_dir = (
        Path(__file__).parent.parent
        / "crates"
        / "pecos"
        / "tests"
        / "test_data"
        / "hugr"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling to HUGR...")
    try:
        hugr_package = main.compile()
        print(f"Package type: {type(hugr_package)}")
        print(
            f"Package methods: {[m for m in dir(hugr_package) if not m.startswith('_')]}",
        )

        # Serialize to bytes
        if hasattr(hugr_package, "to_bytes"):
            hugr_bytes = hugr_package.to_bytes()
        else:
            print("Looking for serialization method...")
            # Try JSON serialization
            if hasattr(hugr_package, "to_json"):
                hugr_bytes = hugr_package.to_json().encode("utf-8")
            else:
                print("Cannot find serialization method")
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
    main_script()
