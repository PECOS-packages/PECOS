#!/usr/bin/env uv run python
"""Generate HUGR test data files for documentation examples.

This script creates HUGR files needed by documentation code examples.
Output goes to docs/assets/test-data/
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from guppylang import guppy
    from guppylang.std.builtins import array, result
    from guppylang.std.quantum import cx, measure, qubit
except ImportError as e:
    print(f"Error: Could not import guppylang: {e}")
    print("Please install guppylang: uv pip install guppylang")
    sys.exit(1)


def generate_repetition_code_hugr() -> str:
    """Generate HUGR for the distance-3 repetition code from getting-started.md."""

    @guppy
    def repetition_code() -> None:
        """Distance-3 repetition code with syndrome extraction.

        3 data qubits encode logical |0⟩ = |000⟩
        2 ancilla qubits measure parity between adjacent data qubits
        """
        # 3 data qubits encode logical |0⟩ = |000⟩
        d0, d1, d2 = qubit(), qubit(), qubit()

        # 2 ancillas for syndrome extraction
        s0, s1 = qubit(), qubit()

        # Measure parity between adjacent data qubits
        cx(d0, s0)
        cx(d1, s0)
        cx(d1, s1)
        cx(d2, s1)

        # Extract syndromes as an array
        result("syndrome", array(measure(s0), measure(s1)))

        # Measure data qubits (required by Guppy)
        _ = measure(d0), measure(d1), measure(d2)

    # Compile to HUGR Package
    compiled = repetition_code.compile()

    # Use to_str() for text envelope format
    return compiled.to_str()


def main() -> int:
    """Generate all test data files for documentation."""
    # Determine output directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "docs" / "assets" / "test-data"

    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating HUGR test data in: {output_dir}")

    # Generate repetition code
    print("\nGenerating repetition_code.hugr...")
    try:
        hugr_str = generate_repetition_code_hugr()
        output_file = output_dir / "repetition_code.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        # Verify format
        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except (OSError, ValueError, RuntimeError) as e:
        print(f"  Error generating repetition code: {e}")
        traceback.print_exc()
        return 1

    print("\nSuccessfully generated HUGR test data files!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
