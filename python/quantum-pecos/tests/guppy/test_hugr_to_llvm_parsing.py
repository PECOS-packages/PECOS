"""Test HUGR 0.13 to LLVM parsing in pecos-selene-engine."""

import pytest


def test_hugr_to_llvm_compilation() -> None:
    """Test actual HUGR to LLVM compilation in Rust."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import cx, h, measure, qubit
        from pecos_rslib import compile_hugr_to_llvm
    except ImportError as e:
        pytest.skip(f"Required imports not available: {e}")

    @guppy
    def bell_state() -> tuple[bool, bool]:
        q1, q2 = qubit(), qubit()
        h(q1)
        cx(q1, q2)
        return measure(q1), measure(q2)

    # Compile to HUGR
    hugr = bell_state.compile()
    # Get string format (uses to_str if available, falls back to to_json)
    if hasattr(hugr, "to_str"):
        hugr_str = hugr.to_str()
        # Check if it's the envelope format with header
        if hugr_str.startswith("HUGRiHJv"):
            # Skip header and find JSON start
            json_start = hugr_str.find("{", 9)
            if json_start != -1:
                hugr_str = hugr_str[json_start:]
            else:
                msg = "Could not find JSON start in HUGR envelope"
                raise ValueError(msg)
    else:
        hugr_str = hugr.to_json()
    hugr_bytes = hugr_str.encode("utf-8")

    # Compile HUGR to LLVM using pecos-selene-engine
    llvm_ir = compile_hugr_to_llvm(hugr_bytes)

    # Verify basic structure - check for Selene QIS patterns
    assert "@___qalloc()" in llvm_ir, "Should have Selene qubit allocation"
    assert (
        "@___rxy" in llvm_ir or "@___rz" in llvm_ir
    ), "Should have Selene rotation gates"
    assert "@___lazy_measure" in llvm_ir, "Should have Selene measurement"

    # Check if we found the main function (entry point) - Selene uses @qmain
    assert "@qmain" in llvm_ir, "Should have Selene qmain entry point"


def test_simple_hadamard_circuit() -> None:
    """Test simple Hadamard circuit compilation."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
        from pecos_rslib import compile_hugr_to_llvm
    except ImportError as e:
        pytest.skip(f"Required imports not available: {e}")

    @guppy
    def hadamard_test() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR
    hugr = hadamard_test.compile()
    # Get string format (uses to_str if available, falls back to to_json)
    if hasattr(hugr, "to_str"):
        hugr_str = hugr.to_str()
        # Check if it's the envelope format with header
        if hugr_str.startswith("HUGRiHJv"):
            # Skip header and find JSON start
            json_start = hugr_str.find("{", 9)
            if json_start != -1:
                hugr_str = hugr_str[json_start:]
            else:
                msg = "Could not find JSON start in HUGR envelope"
                raise ValueError(msg)
    else:
        hugr_str = hugr.to_json()
    hugr_bytes = hugr_str.encode("utf-8")

    # Compile HUGR to LLVM
    llvm_ir = compile_hugr_to_llvm(hugr_bytes)

    # Verify operations - check for Selene QIS patterns
    assert "@___qalloc()" in llvm_ir, "Should have Selene qubit allocation"
    assert (
        "@___rxy" in llvm_ir or "@___rz" in llvm_ir
    ), "Should have Selene rotation gates for H"
    assert "@___lazy_measure" in llvm_ir, "Should have Selene measurement"
