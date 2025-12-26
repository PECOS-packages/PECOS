"""Generate a HUGR file with conditionals using guppylang directly.

This is a standalone script that only requires guppylang (not quantum-pecos).

Usage:
    pip install guppylang
    python scripts/generate_conditional_standalone.py
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
    print("Install with: pip install guppylang")
    sys.exit(1)


def main() -> None:
    """Generate the conditional HUGR test file."""

    @guppy
    def circuit() -> None:
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
        hugr_package = circuit.compile()
        print(f"Package type: {type(hugr_package)}")

        # List available methods
        public_methods = [m for m in dir(hugr_package) if not m.startswith("_")]
        print(f"Package methods: {public_methods}")

        # Try different serialization methods
        hugr_bytes = None

        if hasattr(hugr_package, "to_bytes"):
            print("Using to_bytes()")
            hugr_bytes = hugr_package.to_bytes()
        elif hasattr(hugr_package, "serialize"):
            print("Using serialize()")
            hugr_bytes = hugr_package.serialize()
        elif hasattr(hugr_package, "to_json"):
            print("Using to_json()")
            hugr_json = hugr_package.to_json()
            hugr_bytes = hugr_json.encode("utf-8")
        else:
            # Look for nested package attribute
            if hasattr(hugr_package, "package"):
                pkg = hugr_package.package
                print(f"Found package attr: {type(pkg)}")
                if hasattr(pkg, "to_bytes"):
                    hugr_bytes = pkg.to_bytes()
                elif hasattr(pkg, "serialize"):
                    hugr_bytes = pkg.serialize()

        if hugr_bytes is None:
            print("Could not serialize HUGR package")
            print(f"All attributes: {dir(hugr_package)}")
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
