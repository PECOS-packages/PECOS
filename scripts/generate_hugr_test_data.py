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
    from guppylang.std.angles import angle
    from guppylang.std.builtins import array, owned, result
    from guppylang.std.quantum import ch, cx, h, measure, pi, qubit, rx, ry, rz, x
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
        m0 = measure(q0).read()
        m1 = measure(q1).read()

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
        return measure(q).read()

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
        m0 = measure(q0).read()
        m1 = measure(q1).read()
        m2 = measure(q2).read()

        return m0, m1, m2

    # Compile to HUGR Package
    compiled = ghz_state.compile()

    # Use to_str() for text envelope format
    return compiled.to_str()


def generate_rz_x_hugr() -> str:
    """Generate HUGR for Rz(pi) + measure circuit.

    Rz(pi) on |0> applies a phase but keeps the state in |0>,
    so measuring always returns 0. This lets us test that rotation
    gate angle extraction works end-to-end without introducing
    non-determinism.

    To make the result non-trivial, we also apply X to a second
    qubit so we get a deterministic result of 0b10 = 2 (bit 0 = 0,
    bit 1 = 1).
    """

    @guppy
    def rz_x_circuit() -> tuple[bool, bool]:
        """Rz(pi) on q0 (stays |0>), X on q1 (flips to |1>)."""
        q0 = qubit()
        q1 = qubit()
        rz(q0, pi)
        x(q1)
        return measure(q0).read(), measure(q1).read()

    compiled = rz_x_circuit.compile()
    return compiled.to_str()


def generate_simple_conditional_hugr() -> str:
    """Generate HUGR for a simple conditional: if measure=1, apply X.

    This is the simplest possible conditional circuit:
    1. Create a qubit
    2. Apply H (to get 50/50 superposition)
    3. Measure the qubit
    4. If the result is 1, apply X gate to a second qubit
    5. Measure the second qubit

    Expected behavior:
    - If first measurement is 0: second measurement is 0
    - If first measurement is 1: second measurement is 1 (due to X gate)
    """

    @guppy
    def simple_conditional() -> tuple[bool, bool]:
        """Apply X conditionally based on measurement."""
        q0 = qubit()
        q1 = qubit()

        # Put q0 in superposition
        h(q0)

        # Measure q0
        m0 = measure(q0).read()

        # Conditionally apply X to q1
        if m0:
            x(q1)

        # Measure q1
        m1 = measure(q1).read()

        return m0, m1

    compiled = simple_conditional.compile()
    return compiled.to_str()


def generate_conditional_h_hugr() -> str:
    """Generate HUGR for conditional H gate.

    This tests conditional application of a gate that creates superposition:
    1. Create a qubit in |0⟩
    2. Create a control qubit, apply H, measure it
    3. If measurement is 1, apply H to first qubit
    4. Measure first qubit

    Expected behavior:
    - If control is 0: result qubit always measures 0 (stayed in |0⟩)
    - If control is 1: result qubit is 50/50 (H applied, now in superposition)
    """

    @guppy
    def conditional_h() -> tuple[bool, bool]:
        """Apply H conditionally based on measurement."""
        q_result = qubit()
        q_control = qubit()

        # Put control in superposition and measure
        h(q_control)
        m_control = measure(q_control).read()

        # Conditionally apply H to result qubit
        if m_control:
            h(q_result)

        # Measure result qubit
        m_result = measure(q_result).read()

        return m_control, m_result

    compiled = conditional_h.compile()
    return compiled.to_str()


def generate_conditional_branch_hugr() -> str:
    """Generate HUGR for if-else branches with different gates.

    This tests both branches of a conditional:
    1. Create two qubits
    2. Measure first qubit (deterministic 0)
    3. If measurement is 0, apply H to second qubit
    4. If measurement is 1, apply X to second qubit
    5. Measure second qubit

    Since first qubit starts in |0⟩ and we don't apply any gates,
    it always measures 0, so H is always applied to the second qubit.

    Expected behavior:
    - First measurement is always 0
    - Second measurement is 50/50 (H applied)
    """

    @guppy
    def conditional_branch() -> tuple[bool, bool]:
        """Apply different gates based on measurement branch."""
        q0 = qubit()
        q1 = qubit()

        # Measure q0 (will be 0 since it starts in |0⟩)
        m0 = measure(q0).read()

        # Apply different gates based on measurement
        if m0:
            x(q1)  # This branch won't be taken
        else:
            h(q1)  # This branch will be taken

        # Measure q1
        m1 = measure(q1).read()

        return m0, m1

    compiled = conditional_branch.compile()
    return compiled.to_str()


def generate_simple_while_loop_hugr() -> str:
    """Generate HUGR for a simple while loop.

    This tests TailLoop support in HUGR:
    1. Create a qubit, apply H, measure
    2. If result is 0, repeat (continue loop)
    3. If result is 1, exit loop (break)
    4. Return the number of attempts (always 1+ since we exit on success)

    Expected behavior:
    - Each iteration has 50/50 chance to exit (when measure=1)
    - Average iterations: 2 (geometric distribution)
    - Always returns True when loop exits (since we exit on measure=1)

    Note: This generates a TailLoop node in HUGR, not CFG.
    """

    @guppy
    def simple_while_loop() -> bool:
        """Repeat until measurement returns 1."""
        result: bool = False
        while not result:
            q = qubit()
            h(q)
            result = measure(q).read()
        return result

    compiled = simple_while_loop.compile()
    return compiled.to_str()


def generate_function_call_hugr() -> str:
    """Generate HUGR with a user-defined function call.

    This tests Call/FuncDefn support:
    1. Define a function that applies H and returns the qubit
    2. Call that function from main
    3. Measure the result

    Expected behavior:
    - 50/50 measurement outcome (H creates superposition)
    """

    @guppy
    def apply_h(q: qubit @ owned) -> qubit:
        """Apply Hadamard to a qubit."""
        h(q)
        return q

    # Define main using apply_h
    @guppy
    def function_call_main() -> bool:
        """Main function that calls apply_h."""
        q = qubit()
        q = apply_h(q)
        return measure(q).read()

    compiled = function_call_main.compile()
    return compiled.to_str()


def generate_multiple_function_calls_hugr() -> str:
    """Generate HUGR with multiple function calls.

    This tests calling the same function multiple times:
    1. Define apply_h function
    2. Call it on two different qubits
    3. Measure both

    Expected behavior:
    - Both measurements are 50/50 independent
    """

    @guppy
    def apply_h_multi(q: qubit @ owned) -> qubit:
        """Apply Hadamard to a qubit."""
        h(q)
        return q

    @guppy
    def multiple_calls_main() -> tuple[bool, bool]:
        """Call apply_h on two different qubits."""
        q0 = qubit()
        q1 = qubit()
        q0 = apply_h_multi(q0)
        q1 = apply_h_multi(q1)
        return measure(q0).read(), measure(q1).read()

    compiled = multiple_calls_main.compile()
    return compiled.to_str()


def generate_nested_function_calls_hugr() -> str:
    """Generate HUGR with nested function calls.

    This tests function A calling function B:
    1. Define inner_h that applies H
    2. Define outer_func that calls inner_h
    3. Call outer_func from main

    Expected behavior:
    - 50/50 measurement outcome
    """

    @guppy
    def inner_h(q: qubit @ owned) -> qubit:
        """Inner function: apply H."""
        h(q)
        return q

    @guppy
    def outer_func(q: qubit @ owned) -> qubit:
        """Outer function: call inner_h."""
        return inner_h(q)

    @guppy
    def nested_calls_main() -> bool:
        """Main: call outer_func."""
        q = qubit()
        q = outer_func(q)
        return measure(q).read()

    compiled = nested_calls_main.compile()
    return compiled.to_str()


def generate_multi_qubit_function_hugr() -> str:
    """Generate HUGR with a function taking multiple qubits.

    This tests passing multiple qubits to a function:
    1. Define apply_cx that takes 2 qubits and applies CX
    2. Call it from main
    3. Measure both qubits

    Expected behavior:
    - Bell state: both measurements are correlated (00 or 11)
    """

    @guppy
    def apply_cx_func(q0: qubit @ owned, q1: qubit @ owned) -> tuple[qubit, qubit]:
        """Apply CX gate to two qubits."""
        cx(q0, q1)
        return q0, q1

    @guppy
    def multi_qubit_main() -> tuple[bool, bool]:
        """Create Bell state via function call."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        q0, q1 = apply_cx_func(q0, q1)
        return measure(q0).read(), measure(q1).read()

    compiled = multi_qubit_main.compile()
    return compiled.to_str()


def generate_ch_gate_hugr() -> str:
    """Generate a controlled-H circuit used by execution regression tests."""

    @guppy
    def ch_gate() -> tuple[bool, bool]:
        control = qubit()
        target = qubit()
        x(control)
        ch(control, target)
        return measure(control).read(), measure(target).read()

    return ch_gate.compile().to_str()


def generate_forloop_h_hugr() -> str:
    """Generate a fixed-trip loop with one Hadamard per iteration."""

    @guppy
    def forloop_h() -> bool:
        q = qubit()
        for _ in range(3):
            h(q)
        return measure(q).read()

    return forloop_h.compile().to_str()


def generate_rx_pi_tuple_const_hugr() -> str:
    """Generate an Rx(pi) fixture with Guppy 1's measurement representation."""

    @guppy
    def rx_pi_tuple_const() -> bool:
        q = qubit()
        rx(q, pi)
        return measure(q).read()

    return rx_pi_tuple_const.compile().to_str()


def generate_ry_angle_tuple_hugr() -> str:
    """Generate an Ry(pi/2) fixture with a tuple-wrapped angle."""

    @guppy
    def ry_angle_tuple() -> bool:
        q = qubit()
        ry(q, angle(0.5))
        return measure(q).read()

    return ry_angle_tuple.compile().to_str()


def generate_conditional_x_hugr() -> str:
    """Generate a measured conditional-X circuit for engine tests."""

    @guppy
    def conditional_x() -> tuple[bool, bool]:
        control = qubit()
        target = qubit()
        h(control)
        control_measurement = measure(control).read()
        if control_measurement:
            x(target)
        target_measurement = measure(target).read()
        result("c", array(control_measurement, target_measurement))
        return control_measurement, target_measurement

    return conditional_x.compile().to_str()


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
    all_files = [
        "bell_state.hugr",
        "single_hadamard.hugr",
        "ghz_state.hugr",
        "rz_x.hugr",
        "simple_conditional.hugr",
        "conditional_h.hugr",
        "conditional_branch.hugr",
        "simple_while_loop.hugr",
        "function_call.hugr",
        "multiple_function_calls.hugr",
        "nested_function_calls.hugr",
        "multi_qubit_function.hugr",
        "ch_gate.hugr",
        "forloop_h_test.hugr",
        "rx_pi_tuple_const.hugr",
        "ry_angle_tuple.hugr",
        "conditional_x.hugr",
    ]
    for filename in all_files:
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
        # Broad exception catch is intentional - we want to handle any compilation/serialization error
        print(f"  Error generating GHZ state: {e}")
        return 1

    # Generate Rz + X rotation test
    print("\nGenerating rz_x.hugr...")
    try:
        hugr_str = generate_rz_x_hugr()
        output_file = output_dir / "rz_x.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating rz_x: {e}")
        return 1

    # Generate simple conditional (if measure=1, apply X)
    print("\nGenerating simple_conditional.hugr...")
    try:
        hugr_str = generate_simple_conditional_hugr()
        output_file = output_dir / "simple_conditional.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating simple conditional: {e}")
        return 1

    # Generate conditional H gate
    print("\nGenerating conditional_h.hugr...")
    try:
        hugr_str = generate_conditional_h_hugr()
        output_file = output_dir / "conditional_h.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating conditional H: {e}")
        return 1

    # Generate conditional branch (if-else)
    print("\nGenerating conditional_branch.hugr...")
    try:
        hugr_str = generate_conditional_branch_hugr()
        output_file = output_dir / "conditional_branch.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating conditional branch: {e}")
        return 1

    # Generate simple while loop (TailLoop)
    print("\nGenerating simple_while_loop.hugr...")
    try:
        hugr_str = generate_simple_while_loop_hugr()
        output_file = output_dir / "simple_while_loop.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating simple while loop: {e}")
        return 1

    # Generate function call (Call/FuncDefn)
    print("\nGenerating function_call.hugr...")
    try:
        hugr_str = generate_function_call_hugr()
        output_file = output_dir / "function_call.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating function call: {e}")
        return 1

    # Generate multiple function calls
    print("\nGenerating multiple_function_calls.hugr...")
    try:
        hugr_str = generate_multiple_function_calls_hugr()
        output_file = output_dir / "multiple_function_calls.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating multiple function calls: {e}")
        return 1

    # Generate nested function calls
    print("\nGenerating nested_function_calls.hugr...")
    try:
        hugr_str = generate_nested_function_calls_hugr()
        output_file = output_dir / "nested_function_calls.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating nested function calls: {e}")
        return 1

    # Generate multi-qubit function
    print("\nGenerating multi_qubit_function.hugr...")
    try:
        hugr_str = generate_multi_qubit_function_hugr()
        output_file = output_dir / "multi_qubit_function.hugr"
        output_file.write_text(hugr_str)
        print(f"  Created: {output_file} ({len(hugr_str)} chars)")

        if hugr_str.startswith(("HUGR", "{")):
            print("  Valid HUGR format")
        else:
            print(f"  Warning: Unexpected format (starts with: {hugr_str[:20]}...)")
    except Exception as e:
        print(f"  Error generating multi-qubit function: {e}")
        return 1

    additional_fixtures = {
        "ch_gate.hugr": generate_ch_gate_hugr,
        "forloop_h_test.hugr": generate_forloop_h_hugr,
        "rx_pi_tuple_const.hugr": generate_rx_pi_tuple_const_hugr,
        "ry_angle_tuple.hugr": generate_ry_angle_tuple_hugr,
        "conditional_x.hugr": generate_conditional_x_hugr,
    }
    for filename, generate in additional_fixtures.items():
        print(f"\nGenerating {filename}...")
        try:
            hugr_str = generate()
            output_file = output_dir / filename
            output_file.write_text(hugr_str)
            print(f"  Created: {output_file} ({len(hugr_str)} chars)")
        except Exception as e:
            print(f"  Error generating {filename}: {e}")
            return 1

    for filename in all_files:
        output_file = output_dir / filename
        contents = output_file.read_text()
        if not contents.endswith("\n"):
            output_file.write_text(f"{contents}\n")

    print("\nSuccessfully generated all HUGR test data files!")
    print("\nNext steps:")
    print("1. Run the Rust tests:")
    print("   cargo test -p pecos --test hugr_integration_test")
    print("2. Run the Python tests:")
    print("   uv run pytest python/quantum-pecos/tests/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
