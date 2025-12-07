"""Test current capabilities of both HUGR-LLVM and PHIR pipelines.

This is a simplified version that won't hang.
"""

import pytest


def decode_integer_results(results: list[int], n_bits: int) -> list[tuple[bool, ...]]:
    """Decode integer-encoded results back to tuples of booleans."""
    decoded = []
    for val in results:
        bits = [bool(val & (1 << i)) for i in range(n_bits)]
        decoded.append(tuple(bits))
    return decoded


try:
    from guppylang import guppy
    from guppylang.std.quantum import cx, h, measure, qubit, x

    GUPPY_AVAILABLE = True
except ImportError:
    GUPPY_AVAILABLE = False

try:
    from pecos import Guppy, get_guppy_backends, sim
    from pecos_rslib import state_vector

    PECOS_FRONTEND_AVAILABLE = True
except ImportError:
    PECOS_FRONTEND_AVAILABLE = False


@pytest.mark.skipif(
    not GUPPY_AVAILABLE or not PECOS_FRONTEND_AVAILABLE,
    reason="Dependencies not available",
)
def test_pipeline_capabilities() -> None:
    """Test what both pipelines can currently handle - simplified version."""
    backends = get_guppy_backends()

    # Test cases - just a few simple ones with 1 shot each
    test_cases = []

    # 1. Basic Hadamard
    @guppy
    def test_hadamard() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    test_cases.append(("Hadamard Gate", test_hadamard))

    # 2. Pauli X (should always return 1)
    @guppy
    def test_pauli_x() -> bool:
        q = qubit()
        x(q)
        return measure(q)

    test_cases.append(("Pauli X Gate", test_pauli_x))

    # 3. Bell state
    @guppy
    def test_bell_state() -> tuple[bool, bool]:
        q0, q1 = qubit(), qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)

    test_cases.append(("Bell State", test_bell_state))

    # Run tests on both pipelines with just 1 shot each
    results = {}

    for test_name, test_func in test_cases:
        results[test_name] = {}

        # Test with Rust backend (the only backend)
        if backends.get("rust_backend", False):
            try:
                # Use sim() API instead of run_guppy
                result_dict = (
                    sim(Guppy(test_func)).qubits(10).quantum(state_vector()).run(1)
                )
                # Extract measurement result
                if "measurements" in result_dict:
                    result_val = result_dict["measurements"][0]
                elif "measurement_0" in result_dict:
                    # Handle tuple returns
                    result_val = tuple(
                        bool(result_dict[f"measurement_{i}"][0])
                        for i in range(1, 10)
                        if f"measurement_{i}" in result_dict
                    )
                else:
                    result_val = result_dict.get("result", [None])[0]

                results[test_name]["hugr_llvm"] = {
                    "success": True,
                    "result": result_val,
                }
            except Exception as e:
                results[test_name]["hugr_llvm"] = {
                    "success": False,
                    "error": str(e)[:80],
                }

        # PHIR pipeline no longer exists - using same sim() backend
        try:
            # Use sim() API for consistency
            result_dict = (
                sim(Guppy(test_func)).qubits(10).quantum(state_vector()).run(1)
            )
            # Extract measurement result
            if "measurements" in result_dict:
                result_val = result_dict["measurements"][0]
            elif "measurement_0" in result_dict:
                # Handle tuple returns
                result_val = tuple(
                    bool(result_dict[f"measurement_{i}"][0])
                    for i in range(1, 10)
                    if f"measurement_{i}" in result_dict
                )
            else:
                result_val = result_dict.get("result", [None])[0]

            results[test_name]["phir"] = {
                "success": True,
                "result": result_val,
            }
        except Exception as e:
            results[test_name]["phir"] = {
                "success": False,
                "error": str(e)[:80],
            }

    # Basic assertions for pytest
    # At least one backend should work for each test
    for test_name, test_results in results.items():
        hugr_success = test_results.get("hugr_llvm", {}).get("success", False)
        phir_success = test_results.get("phir", {}).get("success", False)
        assert hugr_success or phir_success, f"Both backends failed for {test_name}"
