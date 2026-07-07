"""Test HUGR 0.13 to LLVM parsing in pecos-selene-engine."""

import pytest


def test_hugr_to_llvm_compilation() -> None:
    """Test actual HUGR to LLVM compilation in Rust."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import cx, h, measure, qubit
        from pecos_rslib_llvm import compile_hugr_to_qis
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
    hugr_bytes = hugr.to_bytes()

    # Compile HUGR to LLVM using pecos-selene-engine
    llvm_ir = compile_hugr_to_qis(hugr_bytes)

    # Verify basic structure - check for Selene QIS patterns
    assert "@___qalloc()" in llvm_ir, "Should have Selene qubit allocation"
    assert "@___rxy" in llvm_ir or "@___rz" in llvm_ir, "Should have Selene rotation gates"
    assert "@___lazy_measure" in llvm_ir, "Should have Selene measurement"

    # Check if we found the main function (entry point) - Selene uses @qmain
    assert "@qmain" in llvm_ir, "Should have Selene qmain entry point"


def test_simple_hadamard_circuit() -> None:
    """Test simple Hadamard circuit compilation."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
        from pecos_rslib_llvm import compile_hugr_to_qis
    except ImportError as e:
        pytest.skip(f"Required imports not available: {e}")

    @guppy
    def hadamard_test() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR
    hugr = hadamard_test.compile()
    hugr_bytes = hugr.to_bytes()

    # Compile HUGR to LLVM
    llvm_ir = compile_hugr_to_qis(hugr_bytes)

    # Verify operations - check for Selene QIS patterns
    assert "@___qalloc()" in llvm_ir, "Should have Selene qubit allocation"
    assert "@___rxy" in llvm_ir or "@___rz" in llvm_ir, "Should have Selene rotation gates for H"
    assert "@___lazy_measure" in llvm_ir, "Should have Selene measurement"


def test_trace_metadata_helper_uses_public_symbol() -> None:
    """Test that declared trace metadata helpers compile to the public FFI symbol."""
    try:
        from guppylang import guppy
        from guppylang.std.builtins import owned
        from guppylang.std.quantum import h, measure, qubit
        from pecos_rslib_llvm import compile_hugr_to_qis
    except ImportError as e:
        pytest.skip(f"Required imports not available: {e}")

    @guppy.declare
    def pecos_qis_trace_metadata_qubit_hugr(q: qubit @ owned, key: str, value: str) -> qubit: ...

    @guppy
    def metadata_probe() -> None:
        q = qubit()
        q = pecos_qis_trace_metadata_qubit_hugr(q, "source_kind", "szz_host")
        h(q)
        _ = measure(q)

    llvm_ir = compile_hugr_to_qis(metadata_probe.compile().to_bytes())

    assert "@pecos_qis_trace_metadata_qubit_hugr" in llvm_ir
    assert "@__hugr__.pecos_qis_trace_metadata_qubit_hugr" not in llvm_ir


def test_runtime_barrier_pair_helper_uses_public_symbol() -> None:
    """Test that two-qubit runtime-barrier helpers compile to the public FFI symbol."""
    try:
        from guppylang import guppy
        from guppylang.std.builtins import owned
        from guppylang.std.quantum import cx, h, measure, qubit
        from pecos_rslib_llvm import compile_hugr_to_qis
    except ImportError as e:
        pytest.skip(f"Required imports not available: {e}")

    @guppy.declare
    def pecos_qis_runtime_barrier_qubits2_hugr(
        q0: qubit @ owned,
        q1: qubit @ owned,
    ) -> tuple[qubit, qubit]: ...

    @guppy
    def barrier_pair_probe() -> tuple[bool, bool]:
        q0 = qubit()
        q1 = qubit()
        h(q0)
        q0, q1 = pecos_qis_runtime_barrier_qubits2_hugr(q0, q1)
        cx(q0, q1)
        return measure(q0), measure(q1)

    llvm_ir = compile_hugr_to_qis(barrier_pair_probe.compile().to_bytes())

    assert "@pecos_qis_runtime_barrier_qubits2_hugr" in llvm_ir
    assert "@__hugr__.pecos_qis_runtime_barrier_qubits2_hugr" not in llvm_ir
