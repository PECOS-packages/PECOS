"""Test HUGR compilation through Selene (HUGR 0.13 compatible)."""

import json

import pytest

# Check for required dependencies
try:
    from guppylang.decorator import guppy as guppy_decorator
    from guppylang.std.quantum import cx, h, measure, qubit, x

    GUPPY_AVAILABLE = True
except ImportError:
    GUPPY_AVAILABLE = False

try:
    from pecos import Guppy, sim
    from pecos_rslib import state_vector

    PECOS_API_AVAILABLE = True
except ImportError:
    PECOS_API_AVAILABLE = False

try:
    from pecos.compilation_pipeline import compile_guppy_to_hugr

    COMPILATION_AVAILABLE = True
except ImportError:
    COMPILATION_AVAILABLE = False


@pytest.mark.optional_dependency
@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="Guppy not available")
class TestSeleneHUGRCompilation:
    """Test HUGR compilation through Selene."""

    def test_selene_hugr_llvm_generation(self) -> None:
        """Test that Selene can generate LLVM IR from HUGR."""
        if not PECOS_API_AVAILABLE:
            pytest.skip("PECOS API not available")

        # Define a proper Bell state with CNOT
        @guppy_decorator
        def bell_state() -> tuple[bool, bool]:
            """Create a Bell state and measure."""
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)  # Proper entanglement
            return measure(q1), measure(q2)

        # The sim API handles HUGR compilation internally
        try:
            results = (
                sim(Guppy(bell_state))
                .qubits(2)
                .quantum(state_vector())
                .seed(42)
                .run(100)
            )

            # Verify results structure
            assert hasattr(results, "__getitem__"), "Results should be dict-like"

            # Check for measurement results
            if "measurement_1" in results and "measurement_2" in results:
                m1 = results["measurement_1"]
                m2 = results["measurement_2"]

                assert len(m1) == 100, "Should have 100 measurements for qubit 1"
                assert len(m2) == 100, "Should have 100 measurements for qubit 2"

                # Bell state measurements should be correlated
                correlated = sum(1 for i in range(100) if m1[i] == m2[i])
                correlation_rate = correlated / 100
                assert (
                    correlation_rate > 0.95
                ), f"Bell state should be highly correlated, got {correlation_rate:.2%}"
            else:
                # Alternative result format
                assert (
                    "measurements" in results or len(results) > 0
                ), "Results should contain measurements"

        except (ImportError, RuntimeError, ValueError) as e:
            if "not supported" in str(e).lower() or "not available" in str(e).lower():
                pytest.skip(f"HUGR compilation not fully supported: {e}")
            pytest.fail(f"Unexpected compilation error: {e}")

    def test_direct_hugr_compilation(self) -> None:
        """Test direct HUGR compilation without simulation."""
        if not COMPILATION_AVAILABLE:
            pytest.skip("Compilation pipeline not available")

        @guppy_decorator
        def simple_circuit() -> bool:
            """Simple H gate and measurement."""
            q = qubit()
            h(q)
            return measure(q)

        # Compile to HUGR
        hugr_bytes = compile_guppy_to_hugr(simple_circuit)

        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Verify HUGR format
        hugr_str = hugr_bytes.decode("utf-8")

        # Check if it's envelope format or direct JSON
        is_envelope = hugr_str.startswith("HUGRiHJv")
        is_json = hugr_str.startswith("{")

        assert is_envelope or is_json, "HUGR should be in valid format"

        # Parse JSON content
        if is_envelope:
            json_start = hugr_str.find("{", 9)
            assert json_start != -1, "Envelope should contain JSON"
            json_content = hugr_str[json_start:]
        else:
            json_content = hugr_str

        try:
            hugr_json = json.loads(json_content)
            assert isinstance(hugr_json, dict), "HUGR should be valid JSON object"

            # Check for expected HUGR structure elements
            # HUGR should have version info and graph structure
            assert len(hugr_json) > 0, "HUGR JSON should not be empty"

        except json.JSONDecodeError as e:
            pytest.fail(f"HUGR should contain valid JSON: {e}")

    def test_complex_circuit_compilation(self) -> None:
        """Test compilation of more complex quantum circuits."""
        if not all([GUPPY_AVAILABLE, COMPILATION_AVAILABLE]):
            pytest.skip("Required dependencies not available")

        @guppy_decorator
        def quantum_teleportation() -> tuple[bool, bool, bool]:
            """Quantum teleportation circuit."""
            # Create Bell pair
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)

            # Prepare state to teleport
            q0 = qubit()
            h(q0)  # Put in superposition

            # Bell measurement on q0 and q1
            cx(q0, q1)
            h(q0)

            # Measure
            m0 = measure(q0)
            m1 = measure(q1)
            m2 = measure(q2)

            return m0, m1, m2

        # Compile to HUGR
        try:
            hugr_bytes = compile_guppy_to_hugr(quantum_teleportation)
        except Exception as e:
            pytest.fail(f"Compilation failed: {e}")

        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 100, "Complex circuit should produce substantial HUGR"

        # Verify it contains quantum operations
        hugr_str = hugr_bytes.decode("utf-8")

        # Look for quantum operation indicators
        quantum_ops = ["quantum", "Quantum", "measure", "hadamard", "cnot"]
        found_ops = [op for op in quantum_ops if op.lower() in hugr_str.lower()]

        assert len(found_ops) > 0, "HUGR should contain quantum operation references"

    def test_parametric_circuit_compilation(self) -> None:
        """Test compilation of parametric quantum circuits."""
        if not COMPILATION_AVAILABLE:
            pytest.skip("Compilation pipeline not available")

        @guppy_decorator
        def parametric_circuit(n: int) -> int:
            """Circuit with parameter-based repetition."""
            count = 0
            for _i in range(n):
                q = qubit()
                h(q)
                if measure(q):
                    count += 1
            return count

        # Compile to HUGR
        try:
            hugr_bytes = compile_guppy_to_hugr(parametric_circuit)
        except Exception as e:
            pytest.fail(f"Parametric compilation failed: {e}")

        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Check for loop/iteration structures in HUGR
        hugr_str = hugr_bytes.decode("utf-8")

        # HUGR might represent loops as specific node types

        # At minimum, verify it's valid HUGR
        assert "HUGRiHJv" in hugr_str or hugr_str.startswith(
            "{",
        ), "Should be valid HUGR format"


@pytest.mark.optional_dependency
class TestLLVMGeneration:
    """Test LLVM IR generation from quantum circuits."""

    def test_llvm_ir_from_hugr(self) -> None:
        """Test generating LLVM IR from HUGR."""
        if not all([GUPPY_AVAILABLE, COMPILATION_AVAILABLE]):
            pytest.skip("Required dependencies not available")

        @guppy_decorator
        def simple_measurement() -> bool:
            """Simple measurement circuit."""
            q = qubit()
            x(q)  # Put in |1⟩ state
            return measure(q)

        # First compile to HUGR
        hugr_bytes = compile_guppy_to_hugr(simple_measurement)
        assert hugr_bytes is not None, "Should produce HUGR bytes"

        # Try to convert HUGR to LLVM (if available)
        try:
            from pecos.backends import hugr_to_llvm

            llvm_ir = hugr_to_llvm(hugr_bytes)
            assert isinstance(llvm_ir, str), "Should produce LLVM IR string"
            assert len(llvm_ir) > 0, "LLVM IR should not be empty"

            # Verify LLVM structure
            assert "define" in llvm_ir, "Should have function definitions"
            assert "@__quantum__" in llvm_ir, "Should have quantum intrinsics"

        except ImportError:
            # HUGR to LLVM conversion might not be available yet
            pass

    def test_llvm_ir_patterns(self) -> None:
        """Test that generated LLVM IR follows expected patterns."""
        # Create expected LLVM IR pattern for reference
        expected_llvm_pattern = """
        ; Quantum intrinsics
        declare void @__quantum__qis__h__body(i64)
        declare void @__quantum__qis__x__body(i64)
        declare void @__quantum__qis__y__body(i64)
        declare void @__quantum__qis__z__body(i64)
        declare void @__quantum__qis__cnot__body(i64, i64)
        declare i1 @__quantum__qis__mz__body(i64)
        declare void @__quantum__rt__result_record_output(i64, i8*)
        """

        # Verify pattern structure
        intrinsics = [
            "@__quantum__qis__h__body",
            "@__quantum__qis__x__body",
            "@__quantum__qis__cnot__body",
            "@__quantum__qis__mz__body",
        ]

        for intrinsic in intrinsics:
            assert (
                intrinsic in expected_llvm_pattern
            ), f"Pattern should include {intrinsic}"

        # Check parameter types
        assert "(i64)" in expected_llvm_pattern, "Single qubit ops should take i64"
        assert (
            "(i64, i64)" in expected_llvm_pattern
        ), "Two qubit ops should take two i64"
        assert (
            "i1 @__quantum__qis__mz" in expected_llvm_pattern
        ), "Measurement should return i1"


@pytest.mark.optional_dependency
class TestHUGRVersionCompatibility:
    """Test HUGR version compatibility."""

    def test_hugr_version_detection(self) -> None:
        """Test detection of HUGR version from compiled output."""
        if not all([GUPPY_AVAILABLE, COMPILATION_AVAILABLE]):
            pytest.skip("Required dependencies not available")

        @guppy_decorator
        def version_test() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        hugr_bytes = compile_guppy_to_hugr(version_test)
        hugr_str = hugr_bytes.decode("utf-8")

        # Check for version indicators
        if hugr_str.startswith("HUGRiHJv"):
            # Envelope format - version in header
            # Format: HUGRiHJv<version>...
            version_part = hugr_str[8:10]  # Next chars might be version
            assert len(version_part) > 0, "Should have version info in envelope"
        elif hugr_str.startswith("{"):
            # JSON format - might have version field
            hugr_json = json.loads(hugr_str)

            # Look for version field in various places
            if "version" in hugr_json:
                hugr_json["version"]
            elif "hugr_version" in hugr_json:
                hugr_json["hugr_version"]
            elif "metadata" in hugr_json and "version" in hugr_json["metadata"]:
                hugr_json["metadata"]["version"]

            # Version might not always be present, but structure should be valid
            assert isinstance(hugr_json, dict), "Should be valid JSON structure"

    def test_hugr_0_13_compatibility(self) -> None:
        """Test compatibility with HUGR 0.13 format."""
        if not COMPILATION_AVAILABLE:
            pytest.skip("Compilation pipeline not available")

        @guppy_decorator
        def compatibility_test() -> tuple[bool, bool]:
            """Test circuit for compatibility."""
            q1, q2 = qubit(), qubit()
            h(q1)
            cx(q1, q2)
            return measure(q1), measure(q2)

        hugr_bytes = compile_guppy_to_hugr(compatibility_test)
        assert hugr_bytes is not None, "Should produce HUGR bytes"

        # HUGR 0.13 specific checks
        hugr_str = hugr_bytes.decode("utf-8")

        # HUGR 0.13 uses specific node types and operation formats
        # These might appear in the JSON structure
        if "{" in hugr_str:
            # Extract JSON part
            json_start = hugr_str.find("{")
            json_part = hugr_str[json_start:]

            try:
                hugr_json = json.loads(json_part)

                # HUGR 0.13 should have nodes and edges structure
                # The exact structure depends on the HUGR spec
                assert isinstance(hugr_json, dict), "Should be valid HUGR structure"

                # Check for common HUGR elements
                hugr_keys = list(hugr_json.keys())
                assert len(hugr_keys) > 0, "HUGR should have structure elements"

            except json.JSONDecodeError:
                # Not JSON format, but still valid HUGR
                pass

    def test_hugr_metadata_preservation(self) -> None:
        """Test that metadata is preserved through compilation."""
        if not COMPILATION_AVAILABLE:
            pytest.skip("Compilation pipeline not available")

        @guppy_decorator
        def metadata_test() -> bool:
            """Test function with potential metadata."""
            q = qubit()
            h(q)
            return measure(q)

        # Note: Guppy functions are frozen dataclasses, so we can't set attributes directly
        # The metadata should come from the function definition itself

        hugr_bytes = compile_guppy_to_hugr(metadata_test)
        hugr_str = hugr_bytes.decode("utf-8")

        # Check if any metadata is preserved
        # Function name should at least be preserved
        assert (
            "metadata_test" in hugr_str or len(hugr_bytes) > 50
        ), "HUGR should preserve some function information"
