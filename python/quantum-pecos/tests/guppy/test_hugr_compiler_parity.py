"""Test parity between Selene's hugr-qis compiler and PECOS Rust HUGR compiler.

This test ensures that both compilers produce functionally equivalent LLVM IR
for the same HUGR input.
"""

from pathlib import Path

import pytest

# Check if we have the required dependencies
try:
    from guppylang import GuppyModule, guppy

    GUPPY_AVAILABLE = True
except ImportError:
    GUPPY_AVAILABLE = False
    GuppyModule = None
    guppy = None

# Import quantum operations separately to avoid import error when guppylang isn't available
if GUPPY_AVAILABLE:
    try:
        from guppylang.stdlib.quantum import cx, h, measure, qubit
    except ImportError:
        # Fallback for different guppylang versions
        from guppylang.std.quantum import cx, h, measure, qubit

try:
    from selene_hugr_qis_compiler import compile_to_llvm_ir as selene_compile

    SELENE_AVAILABLE = True
except ImportError:
    SELENE_AVAILABLE = False

from pecos_rslib import compile_hugr_to_llvm_rust as rust_compile


def normalize_llvm_ir(llvm_ir: str) -> list[str]:
    """Normalize LLVM IR for comparison.

    Removes comments, blank lines, and normalizes whitespace.
    Returns a list of non-comment, non-blank lines.
    """
    lines = []
    for raw_line in llvm_ir.split("\n"):
        # Skip comments and blank lines
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        # Normalize whitespace
        line = " ".join(line.split())
        lines.append(line)
    return lines


def extract_qis_calls(llvm_ir: str) -> list[str]:
    """Extract quantum instruction set calls from LLVM IR.

    This extracts the actual quantum operations which should be equivalent
    between the two compilers.
    """
    import re

    qis_calls = []
    for raw_line in llvm_ir.split("\n"):
        line = raw_line.strip()
        # Look for QIS function calls
        if "call" in line and (
            "___q" in line
            or "___h" in line
            or "___cx" in line
            or "___rzz" in line
            or "___rxy" in line
            or "___m" in line
            or "___rz" in line
            or "___reset" in line
            or "___lazy_measure" in line
        ):
            # Normalize variable names to allow comparison
            # Replace all %variable.names with %VAR
            normalized = re.sub(r"%[a-zA-Z0-9._-]+", "%VAR", line)
            qis_calls.append(normalized)
    return sorted(qis_calls)


def compare_compilers(
    hugr_binary_selene: bytes,
    hugr_binary_rust: bytes,
) -> tuple[bool, str]:
    """Compare outputs from both compilers.

    Args:
        hugr_binary_selene: HUGR in binary envelope format for Selene
        hugr_binary_rust: HUGR in binary envelope format for Rust compiler

    Returns:
        (are_equivalent, diagnostic_message)
    """
    if not SELENE_AVAILABLE:
        return False, "One or both compilers not available"

    try:
        # Compile with Selene's hugr-qis (expects binary format)
        selene_ir = selene_compile(hugr_binary_selene)
    except Exception as e:
        return False, f"Selene compilation failed: {e}"

    try:
        # Compile with our Rust compiler (now also expects envelope format)
        rust_ir = rust_compile(hugr_binary_rust, None)
    except Exception as e:
        return False, f"Rust compilation failed: {e}"

    # Extract QIS calls from both
    selene_qis = extract_qis_calls(selene_ir)
    rust_qis = extract_qis_calls(rust_ir)

    # Check if QIS calls are the same
    if selene_qis == rust_qis:
        return True, "QIS calls match exactly"

    # If not exact match, provide diagnostic info
    selene_set = set(selene_qis)
    rust_set = set(rust_qis)

    only_selene = selene_set - rust_set
    only_rust = rust_set - selene_set

    msg = "QIS calls differ:\n"
    if only_selene:
        msg += f"  Only in Selene: {only_selene}\n"
    if only_rust:
        msg += f"  Only in Rust: {only_rust}\n"

    return False, msg


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="guppylang not available")
@pytest.mark.skipif(
    not SELENE_AVAILABLE,
    reason="selene_hugr_qis_compiler not available",
)
def test_bell_state_compilation_parity() -> None:
    """Test that both compilers produce equivalent LLVM IR for Bell state."""

    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create a Bell state."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        m0 = measure(q0)
        m1 = measure(q1)
        return m0, m1

    # Compile to HUGR
    hugr = bell_state.compile()

    # Get envelope format for both compilers
    hugr_binary = hugr.to_bytes()  # Binary envelope format

    # Compare compilers (both use same format now)
    equivalent, msg = compare_compilers(hugr_binary, hugr_binary)
    assert equivalent, f"Bell state compilation differs: {msg}"


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="guppylang not available")
@pytest.mark.skipif(
    not SELENE_AVAILABLE,
    reason="selene_hugr_qis_compiler not available",
)
def test_single_hadamard_compilation_parity() -> None:
    """Test that both compilers produce equivalent LLVM IR for single Hadamard."""

    @guppy
    def hadamard_test() -> bool:
        """Apply Hadamard and measure."""
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR
    hugr = hadamard_test.compile()

    # Get envelope format for both compilers
    hugr_binary = hugr.to_bytes()  # Binary envelope format

    # Compare compilers (both use same format now)
    equivalent, msg = compare_compilers(hugr_binary, hugr_binary)
    assert equivalent, f"Hadamard compilation differs: {msg}"


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="guppylang not available")
@pytest.mark.skipif(
    not SELENE_AVAILABLE,
    reason="selene_hugr_qis_compiler not available",
)
def test_ghz_state_compilation_parity() -> None:
    """Test that both compilers produce equivalent LLVM IR for GHZ state."""

    @guppy
    def ghz_state() -> tuple[bool, bool, bool]:
        """Create a 3-qubit GHZ state."""
        q0 = qubit()
        q1 = qubit()
        q2 = qubit()
        h(q0)
        cx(q0, q1)
        cx(q1, q2)
        m0 = measure(q0)
        m1 = measure(q1)
        m2 = measure(q2)
        return m0, m1, m2

    # Compile to HUGR
    hugr = ghz_state.compile()

    # Get envelope format for both compilers
    hugr_binary = hugr.to_bytes()  # Binary envelope format

    # Compare compilers (both use same format now)
    equivalent, msg = compare_compilers(hugr_binary, hugr_binary)
    assert equivalent, f"GHZ state compilation differs: {msg}"


@pytest.mark.skipif(
    not SELENE_AVAILABLE,
    reason="selene_hugr_qis_compiler not available",
)
def test_existing_hugr_files_parity() -> None:
    """Test parity using existing HUGR test data files."""
    # Path to test data
    test_data_dir = (
        Path(__file__).parent.parent.parent.parent.parent
        / "crates/pecos/tests/test_data/hugr"
    )

    if not test_data_dir.exists():
        pytest.skip("Test data directory not found")

    # Test each HUGR file
    for hugr_file in test_data_dir.glob("*.hugr"):
        # Skip old format files
        if hugr_file.name.endswith(".old"):
            continue

        hugr_bytes = hugr_file.read_bytes()

        # Check if this is a binary envelope format (starts with "HUGRiHJv")
        if hugr_bytes.startswith(b"HUGRiHJv"):
            # Binary envelope format - both compilers can use this
            equivalent, msg = compare_compilers(hugr_bytes, hugr_bytes)
            assert equivalent, f"HUGR file {hugr_file.name} compilation differs: {msg}"
        else:
            # Try to decode as JSON/text
            try:
                hugr_bytes.decode("utf-8")
                # For text format, skip since we need binary for Selene
                pytest.skip(
                    f"Skipping {hugr_file.name} - text format, need binary for Selene",
                )
            except UnicodeDecodeError:
                pytest.skip(f"Skipping {hugr_file.name} - unknown binary format")


if __name__ == "__main__":
    # Quick manual test
    if GUPPY_AVAILABLE and SELENE_AVAILABLE:

        @guppy
        def test_circuit() -> bool:
            """Simple test circuit with H gate and measurement."""
            q = qubit()
            h(q)
            return measure(q)

        hugr = test_circuit.compile()

        # Get both formats
        hugr_binary = hugr.to_bytes()  # Binary format for Selene
        try:
            hugr_json = hugr.to_json()  # JSON format for Rust compiler
        except AttributeError:
            # If to_json not available, use to_str
            hugr_json = hugr.to_str() if hasattr(hugr, "to_str") else str(hugr)

        print("Comparing compilers for test circuit...")
        equivalent, msg = compare_compilers(hugr_binary, hugr_json)
        print(f"Result: {'MATCH' if equivalent else 'DIFFER'}")
        print(f"Details: {msg}")

        # Show actual outputs for debugging
        if not equivalent:
            print("\n=== Selene LLVM IR ===")
            print(selene_compile(hugr_binary))
            print("\n=== Rust LLVM IR ===")
            print(rust_compile(hugr_json.encode("utf-8"), None))
    else:
        print("Missing dependencies:")
        print(f"  Guppy: {GUPPY_AVAILABLE}")
        print(f"  Selene: {SELENE_AVAILABLE}")
        print("  Rust compiler: Available")
