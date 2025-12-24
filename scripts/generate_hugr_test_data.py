#!/usr/bin/env uv run python
"""Generate HUGR test data files using guppylang.

This script creates the HUGR test data files needed for PECOS tests:
- bell_state.hugr: Bell state circuit (H on q0, CNOT(q0, q1))
- single_hadamard.hugr: Single Hadamard gate
- ghz_state.hugr: 3-qubit GHZ state

The files are generated using the HUGR envelope format which is the modern
standard that can be loaded by PECOS compilers.
"""

import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from guppylang import guppy
    from guppylang.std.quantum import cx, h, measure, qubit
except ImportError as e:
    print(f"Error: Could not import guppylang: {e}")
    print("Please install guppylang: uv pip install guppylang")
    sys.exit(1)


def generate_bell_state_hugr() -> str:
    """Generate HUGR for Bell state circuit."""

    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create a Bell state: |00⟩ + |11⟩."""
        q0 = qubit()
        q1 = qubit()

        # Create Bell state
        h(q0)
        cx(q0, q1)

        # Measure both qubits
        m0 = measure(q0)
        m1 = measure(q1)

        return m0, m1

    # Compile to HUGR Package
    compiled = bell_state.compile()

    # Use to_str() for text envelope format (human-readable and git-friendly)
    # This is the modern replacement for to_json()
    return compiled.to_str()


def generate_single_hadamard_hugr() -> str:
    """Generate HUGR for single Hadamard gate."""

    @guppy
    def single_hadamard() -> bool:
        """Apply Hadamard gate to a single qubit."""
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR Package
    compiled = single_hadamard.compile()

    # Use to_str() for text envelope format
    return compiled.to_str()


def generate_ghz_state_hugr() -> str:
    """Generate HUGR for 3-qubit GHZ state."""

    @guppy
    def ghz_state() -> tuple[bool, bool, bool]:
        """Create a 3-qubit GHZ state: |000⟩ + |111⟩."""
        q0 = qubit()
        q1 = qubit()
        q2 = qubit()

        # Create GHZ state
        h(q0)
        cx(q0, q1)
        cx(q1, q2)

        # Measure all qubits
        m0 = measure(q0)
        m1 = measure(q1)
        m2 = measure(q2)

        return m0, m1, m2

    # Compile to HUGR Package
    compiled = ghz_state.compile()

    # Use to_str() for text envelope format
    return compiled.to_str()


def main() -> int:
    """Generate all test data files."""
    # Determine output directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "crates" / "pecos" / "tests" / "test_data" / "hugr"

    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating HUGR test data in: {output_dir}")

    # Back up old files if they exist
    for filename in ["bell_state.hugr", "single_hadamard.hugr", "ghz_state.hugr"]:
        old_file = output_dir / filename
        if old_file.exists():
            backup_file = output_dir / f"{filename}.backup"
            print(f"Backing up {filename} to {filename}.backup")
            old_file.rename(backup_file)

    # Generate Bell state
    print("\nGenerating bell_state.hugr...")
    try:
        hugr_str = generate_bell_state_hugr()
        output_file = output_dir / "bell_state.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        # Verify format
        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:  # noqa: BLE001
        # Broad exception catch is intentional - we want to handle any compilation/serialization error
        print(f"  Error generating Bell state: {e}")
        return 1

    # Generate single Hadamard
    print("\nGenerating single_hadamard.hugr...")
    try:
        hugr_str = generate_single_hadamard_hugr()
        output_file = output_dir / "single_hadamard.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        # Verify format
        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:  # noqa: BLE001
        # Broad exception catch is intentional - we want to handle any compilation/serialization error
        print(f"  Error generating single Hadamard: {e}")
        return 1

    # Generate GHZ state
    print("\nGenerating ghz_state.hugr...")
    try:
        hugr_str = generate_ghz_state_hugr()
        output_file = output_dir / "ghz_state.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        # Verify format
        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:  # noqa: BLE001
        # Broad exception catch is intentional - we want to handle any compilation/serialization error
        print(f"  Error generating GHZ state: {e}")
        return 1

    print("\nSuccessfully generated all HUGR test data files!")
    print("\nNext steps:")
    print("1. Run the Rust tests:")
    print("   cargo test -p pecos --test hugr_integration_test")
    print("2. Run the Python tests:")
    print("   uv run pytest python/quantum-pecos/tests/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
