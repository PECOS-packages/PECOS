"""Additional HUGR tests using the current API."""

import pytest


def test_hugr_compilation_with_support() -> None:
    """Test that compilation works when HUGR support IS available."""
    try:
        from pecos_rslib import compile_hugr_to_llvm_rust, check_rust_hugr_availability

        available, message = check_rust_hugr_availability()
        assert available, f"HUGR support should be available but got: {message}"

        # Test that invalid HUGR data raises an error
        dummy_hugr = b"invalid hugr data"
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(dummy_hugr)

        # The error should mention HUGR parsing
        error_msg = str(exc_info.value).lower()
        assert (
            "failed to read hugr" in error_msg or "empty hugr" in error_msg
        ), f"Expected error about HUGR parsing, got: {exc_info.value}"

    except ImportError:
        pytest.skip("Rust HUGR backend not available")


def test_hugr_version_compatibility() -> None:
    """Test HUGR version compatibility handling."""
    try:
        import json

        from pecos_rslib import compile_hugr_to_llvm_rust, check_rust_hugr_availability

        available, message = check_rust_hugr_availability()
        if not available:
            pytest.skip(f"HUGR support not available: {message}")

        # Create HUGR with old version format (simulating old Guppy output)
        old_hugr = {
            "format": "hugr",
            "version": "0.1.0",  # Old version
            "modules": [
                {
                    "name": "test",
                    "nodes": [
                        {
                            "op": "FuncDefn",
                            "name": "test_func",
                            "signature": {
                                "t": "FuncType",
                                "body": {
                                    "input": [],
                                    "output": [{"t": "I", "width": 64}],
                                },
                            },
                        },
                    ],
                },
            ],
        }

        # Try to compile with old version
        hugr_bytes = json.dumps(old_hugr).encode("utf-8")

        # We expect this to fail with parsing error
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(hugr_bytes)

        error_msg = str(exc_info.value).lower()
        # Check that we got a reasonable error
        assert (
            "failed to read hugr" in error_msg or "empty hugr" in error_msg
        ), f"Expected HUGR parsing error, got: {exc_info.value}"

    except ImportError:
        pytest.skip("Rust HUGR backend not available")


def test_hugr_arithmetic_extension_handling() -> None:
    """Test handling of arithmetic extensions."""
    try:
        import json

        from pecos_rslib import compile_hugr_to_llvm_rust, check_rust_hugr_availability

        available, message = check_rust_hugr_availability()
        if not available:
            pytest.skip(f"HUGR support not available: {message}")

        # Create HUGR with arithmetic.int extension
        hugr_with_arithmetic = {
            "format": "hugr",
            "version": "0.20.1",
            "extensions": ["arithmetic.int"],
            "modules": [
                {
                    "name": "test",
                    "nodes": [
                        {
                            "op": "Extension",
                            "extension": "arithmetic.int",
                            "op_name": "iadd",
                            "signature": {
                                "t": "PolyFuncType",
                                "params": [{"t": "TypeBound", "b": "Copyable"}],
                                "body": {
                                    "input": [
                                        {
                                            "t": "Opaque",
                                            "extension": "arithmetic.int",
                                            "name": "int",
                                            "args": [{"t": "BoundedUSize", "size": 6}],
                                        },
                                        {
                                            "t": "Opaque",
                                            "extension": "arithmetic.int",
                                            "name": "int",
                                            "args": [{"t": "BoundedUSize", "size": 6}],
                                        },
                                    ],
                                    "output": [
                                        {
                                            "t": "Opaque",
                                            "extension": "arithmetic.int",
                                            "name": "int",
                                            "args": [{"t": "BoundedUSize", "size": 7}],
                                        },
                                    ],
                                },
                            },
                        },
                    ],
                },
            ],
        }

        hugr_bytes = json.dumps(hugr_with_arithmetic).encode("utf-8")

        # We expect this to fail
        with pytest.raises(RuntimeError) as exc_info:
            compile_hugr_to_llvm_rust(hugr_bytes)

        error_msg = str(exc_info.value).lower()
        # Just check we get a HUGR-related error
        assert (
            "failed to read hugr" in error_msg or "empty hugr" in error_msg
        ), f"Expected HUGR-related error, got: {exc_info.value}"

    except ImportError:
        pytest.skip("Rust HUGR backend not available")
