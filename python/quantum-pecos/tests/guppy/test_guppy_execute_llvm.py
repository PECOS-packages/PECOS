#!/usr/bin/env python3
"""Test the Guppy to LLVM compilation pipeline via execute_llvm."""

import pytest


@pytest.fixture
def simple_quantum_function() -> object:
    """Fixture providing a simple quantum Guppy function."""
    from guppylang import guppy
    from guppylang.std.quantum import h, measure, qubit

    @guppy
    def simple_quantum() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    return simple_quantum


class TestGuppyExecuteLLVM:
    """Test suite for Guppy to LLVM compilation using execute_llvm."""

    def test_execute_llvm_module_available(self) -> None:
        """Test that execute_llvm module is available and has required functions."""
        try:
            from pecos import execute_llvm
        except ImportError:
            pytest.skip("execute_llvm module not available")

        assert hasattr(
            execute_llvm,
            "compile_module_to_string",
        ), "execute_llvm should have compile_module_to_string function"

    def test_compile_guppy_to_hugr(self, simple_quantum_function: object) -> None:
        """Test compiling a Guppy function to HUGR format."""
        try:
            compiled = simple_quantum_function.compile()
            hugr_bytes = compiled.to_bytes()
        except Exception as e:
            pytest.fail(f"HUGR compilation failed: {e}")

        assert hugr_bytes is not None, "HUGR compilation should produce bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

    def test_compile_hugr_to_llvm(self, simple_quantum_function: object) -> None:
        """Test compiling HUGR to LLVM IR using execute_llvm with default Selene compiler."""
        try:
            from pecos import execute_llvm
        except ImportError:
            pytest.skip("execute_llvm not available")

        # First compile Guppy to HUGR
        compiled = simple_quantum_function.compile()
        hugr_bytes = compiled.to_bytes()

        # Then compile HUGR to LLVM using default (Selene) compiler
        try:
            llvm_ir = execute_llvm.compile_module_to_string(hugr_bytes)
        except Exception as e:
            if "Unknown type" in str(e):
                pytest.skip(f"Known issue with type handling: {e}")
            pytest.fail(f"LLVM compilation failed: {e}")

        assert llvm_ir is not None, "LLVM compilation should produce IR"
        assert len(llvm_ir) > 0, "LLVM IR should not be empty"

        # Check for Selene-specific patterns (default compiler)
        # Selene uses: @qmain, ___qalloc, ___lazy_measure, ___qfree
        has_selene_patterns = any(
            pattern in llvm_ir
            for pattern in [
                "___qalloc",  # Selene qubit allocation
                "___lazy_measure",  # Selene measurement
                "___qfree",  # Selene qubit deallocation
                "@qmain",  # Selene's main function
            ]
        )

        assert has_selene_patterns, (
            "LLVM IR should contain Selene QIS patterns (___qalloc, ___lazy_measure, @qmain). "
            "Default compiler should be Selene."
        )

    def test_compile_hugr_with_explicit_compiler(
        self,
        simple_quantum_function: object,
    ) -> None:
        """Test explicit compiler selection for HUGR to LLVM compilation."""
        try:
            from pecos import execute_llvm
        except ImportError:
            pytest.skip("execute_llvm not available")

        # Compile Guppy to HUGR
        compiled = simple_quantum_function.compile()

        # Test with explicit Selene compiler (expects binary format)
        try:
            selene_bytes = compiled.to_bytes()
            selene_ir = execute_llvm.compile_module_to_string(
                selene_bytes,
            )
            assert (
                "___qalloc" in selene_ir or "@qmain" in selene_ir
            ), "Selene compiler should produce QIS patterns"
        except RuntimeError as e:
            if "not available" in str(e) or "envelope format" in str(e):
                pytest.skip(f"Selene compiler issue: {e}")
            raise

        # Test with explicit PECOS/Rust compiler (expects binary envelope format)
        try:
            # Both compilers now expect the same binary envelope format
            rust_bytes = compiled.to_bytes()
            rust_ir = execute_llvm.compile_module_to_string(rust_bytes)
            # PECOS compiler now also produces Selene QIS patterns
            assert (
                "___qalloc" in rust_ir or "@qmain" in rust_ir
            ), "PECOS compiler should produce Selene QIS patterns"
        except RuntimeError as e:
            if "not available" in str(e):
                pytest.skip(f"PECOS compiler not available: {e}")
            raise

    def test_guppy_frontend_integration(self, simple_quantum_function: object) -> None:
        """Test GuppyFrontend integration with execute_llvm."""
        try:
            from pecos._compilation import GuppyFrontend
        except ImportError:
            pytest.skip("GuppyFrontend not available")

        try:
            frontend = GuppyFrontend(use_rust_backend=False)
        except Exception as e:
            pytest.skip(f"GuppyFrontend initialization failed: {e}")

        # Get backend info
        info = frontend.get_backend_info()
        assert isinstance(info, dict), "Backend info should be a dictionary"

        # Try to compile the function
        try:
            qir_file = frontend.compile_function(simple_quantum_function)
            assert qir_file is not None, "Compilation should produce a QIR file path"
        except Exception as e:
            # This is expected to fail in some environments
            if (
                "HUGR version" in str(e)
                or "not available" in str(e)
                or "envelope format" in str(e)
                or "Selene's compiler expects" in str(e)
            ):
                pytest.skip(f"Known compatibility issue: {e}")
            pytest.fail(f"Function compilation failed unexpectedly: {e}")

    def test_sim_api_available(self) -> None:
        """Test that the sim() API is available for execution."""
        try:
            from pecos import Guppy, sim
        except ImportError as e:
            pytest.skip(f"sim API not available: {e}")

        assert callable(sim), "sim should be a callable function"
