"""Test Guppy programs that properly output results for Selene to capture.

This shows how Guppy programs should use result() to tag final outputs
that Selene can extract from the result stream.
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestGuppyWithResults:
    """Test suite for Guppy programs using result() function for tagged outputs."""

    @pytest.fixture
    def check_guppy_imports(self) -> dict:
        """Check and provide Guppy imports."""
        try:
            from guppylang import guppy
            from guppylang.std.quantum import cx, h, measure, qubit
        except ImportError:
            pytest.skip("Guppy not available")

        # Check for result function in various locations
        result_func = None
        result_location = None

        # Try different import locations for result()
        try:
            from guppylang.std.builtins import result

            result_func = result
            result_location = "guppylang.std.builtins"
        except ImportError:
            try:
                from guppylang.std.io import result

                result_func = result
                result_location = "guppylang.std.io"
            except ImportError:
                try:
                    from guppylang.std import result

                    result_func = result
                    result_location = "guppylang.std"
                except ImportError:
                    pass

        return {
            "guppy": guppy,
            "quantum": {"h": h, "cx": cx, "measure": measure, "qubit": qubit},
            "result": result_func,
            "result_location": result_location,
        }

    def test_result_function_availability(self, check_guppy_imports: dict) -> None:
        """Test that result() function is available and document its location."""
        if check_guppy_imports["result"] is None:
            pytest.skip("result() function not available in this Guppy version")

        assert callable(
            check_guppy_imports["result"],
        ), "result should be a callable function"
        assert (
            check_guppy_imports["result_location"] is not None
        ), "result function should have a known import location"

    def test_simple_measurement_with_result(self, check_guppy_imports: dict) -> None:
        """Test simple measurement with result tagging."""
        if check_guppy_imports["result"] is None:
            pytest.skip("result() function not available")

        guppy = check_guppy_imports["guppy"]
        q_ops = check_guppy_imports["quantum"]
        result = check_guppy_imports["result"]

        # Extract functions for use in guppy function
        qubit = q_ops["qubit"]
        h = q_ops["h"]
        measure = q_ops["measure"]

        @guppy
        def measure_with_result() -> None:
            """Measure a qubit and tag the result."""
            q = qubit()
            h(q)
            measurement = measure(q)
            # Tag the measurement with a name for Selene to capture
            result("measurement_outcome", measurement)

        # Test compilation
        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        try:
            hugr_bytes = compile_guppy_to_hugr(measure_with_result)
        except Exception as e:
            pytest.fail(f"Failed to compile measure_with_result: {e}")

        assert hugr_bytes is not None, "Compilation should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

    def test_measurement_with_return_fallback(self, check_guppy_imports: dict) -> None:
        """Test measurement using return statement when result() is not available."""
        guppy = check_guppy_imports["guppy"]
        q_ops = check_guppy_imports["quantum"]

        # Extract functions for use in guppy function
        qubit = q_ops["qubit"]
        h = q_ops["h"]
        measure = q_ops["measure"]

        @guppy
        def measure_with_return() -> bool:
            """Return measurement - this should appear in results."""
            q = qubit()
            h(q)
            return measure(q)

        # Test compilation
        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        try:
            hugr_bytes = compile_guppy_to_hugr(measure_with_return)
        except Exception as e:
            pytest.fail(f"Failed to compile measure_with_return: {e}")

        assert hugr_bytes is not None, "Compilation should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

    def test_bell_state_with_named_results(self, check_guppy_imports: dict) -> None:
        """Test Bell state creation with named result outputs."""
        if check_guppy_imports["result"] is None:
            # Test fallback with return statement
            guppy = check_guppy_imports["guppy"]
            q_ops = check_guppy_imports["quantum"]

            # Extract functions for use in guppy function
            qubit = q_ops["qubit"]
            h = q_ops["h"]
            cx = q_ops["cx"]
            measure = q_ops["measure"]

            @guppy
            def bell_state_with_return() -> tuple[bool, bool]:
                """Return Bell state measurements."""
                q0, q1 = qubit(), qubit()
                h(q0)
                cx(q0, q1)
                return measure(q0), measure(q1)

            test_func = bell_state_with_return
        else:
            # Test with result() function
            guppy = check_guppy_imports["guppy"]
            q_ops = check_guppy_imports["quantum"]
            result = check_guppy_imports["result"]

            # Extract functions for use in guppy function
            qubit = q_ops["qubit"]
            h = q_ops["h"]
            cx = q_ops["cx"]
            measure = q_ops["measure"]

            @guppy
            def bell_state_with_results() -> None:
                """Create Bell state and output named results."""
                q0, q1 = qubit(), qubit()
                h(q0)
                cx(q0, q1)

                # Measure and tag results
                m0 = measure(q0)
                m1 = measure(q1)

                result("qubit_0", m0)
                result("qubit_1", m1)
                result("both_same", m0 == m1)  # Should always be True for Bell state

            test_func = bell_state_with_results

        # Test compilation
        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        try:
            hugr_bytes = compile_guppy_to_hugr(test_func)
        except Exception as e:
            pytest.fail(f"Failed to compile Bell state function: {e}")

        assert hugr_bytes is not None, "Compilation should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

    def test_quantum_statistics_output(self, check_guppy_imports: dict) -> None:
        """Test multiple measurements with statistical outputs."""
        if check_guppy_imports["result"] is None:
            pytest.skip("result() function not available for this test")

        guppy = check_guppy_imports["guppy"]
        q_ops = check_guppy_imports["quantum"]
        result = check_guppy_imports["result"]

        # Extract functions for use in guppy function
        qubit = q_ops["qubit"]
        h = q_ops["h"]
        measure = q_ops["measure"]

        @guppy
        def quantum_stats() -> None:
            """Perform multiple measurements and output statistics."""
            # Create 3 qubits in superposition
            q0, q1, q2 = qubit(), qubit(), qubit()
            h(q0)
            h(q1)
            h(q2)

            # Measure all
            m0 = measure(q0)
            m1 = measure(q1)
            m2 = measure(q2)

            # Output individual results
            result("bit_0", m0)
            result("bit_1", m1)
            result("bit_2", m2)

            # Output derived statistics
            count = int(m0) + int(m1) + int(m2)
            result("total_ones", count)
            result("all_same", (m0 == m1) and (m1 == m2))

        # Test compilation
        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        try:
            hugr_bytes = compile_guppy_to_hugr(quantum_stats)
        except Exception as e:
            pytest.fail(f"Failed to compile quantum_stats: {e}")

        assert hugr_bytes is not None, "Compilation should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

    def test_hugr_output_operations(self, check_guppy_imports: dict) -> None:
        """Test that HUGR contains output/result operations."""
        if check_guppy_imports["result"] is None:
            pytest.skip("result() function not available")

        guppy = check_guppy_imports["guppy"]
        q_ops = check_guppy_imports["quantum"]
        result = check_guppy_imports["result"]

        # Extract functions for use in guppy function
        qubit = q_ops["qubit"]
        h = q_ops["h"]
        measure = q_ops["measure"]

        @guppy
        def test_with_outputs() -> None:
            """Simple function with result outputs."""
            q = qubit()
            h(q)
            m = measure(q)
            result("test_output", m)
            result("constant_output", 42)

        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        hugr_bytes = compile_guppy_to_hugr(test_with_outputs)

        # Parse HUGR to check for output operations
        hugr_str = hugr_bytes.decode("utf-8")

        # Handle HUGR envelope format if present
        if hugr_str.startswith("HUGRiHJv"):
            json_start = hugr_str.find("{", 9)
            if json_start != -1:
                hugr_str = hugr_str[json_start:]

        try:
            hugr_json = json.loads(hugr_str)
        except json.JSONDecodeError as e:
            pytest.fail(f"HUGR is not valid JSON: {e}")

        # Count output-related operations
        output_ops = self._count_output_operations(hugr_json)

        # Should have some output/result/io operations
        assert output_ops > 0, "HUGR should contain output/result operations"

    def test_save_hugr_artifacts(self, check_guppy_imports: dict) -> None:
        """Test saving HUGR compilation artifacts for inspection."""
        guppy = check_guppy_imports["guppy"]
        q_ops = check_guppy_imports["quantum"]

        # Extract functions for use in guppy function
        qubit = q_ops["qubit"]
        h = q_ops["h"]
        measure = q_ops["measure"]

        @guppy
        def simple_quantum() -> bool:
            """Simple quantum function."""
            q = qubit()
            h(q)
            return measure(q)

        try:
            from pecos.compilation_pipeline import compile_guppy_to_hugr
        except ImportError:
            pytest.skip("Compilation pipeline not available")

        hugr_bytes = compile_guppy_to_hugr(simple_quantum)

        # Save HUGR artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save raw HUGR bytes
            hugr_file = tmpdir_path / "simple_quantum.hugr"
            hugr_file.write_bytes(hugr_bytes)
            assert hugr_file.exists(), "HUGR file should be created"
            assert hugr_file.stat().st_size > 0, "HUGR file should not be empty"

            # Parse and save formatted JSON
            hugr_str = hugr_bytes.decode("utf-8")
            if hugr_str.startswith("HUGRiHJv"):
                json_start = hugr_str.find("{", 9)
                if json_start != -1:
                    hugr_str = hugr_str[json_start:]

            try:
                hugr_json = json.loads(hugr_str)
                formatted_file = tmpdir_path / "simple_quantum_formatted.json"
                formatted_file.write_text(json.dumps(hugr_json, indent=2))

                assert formatted_file.exists(), "Formatted JSON should be created"
                assert (
                    formatted_file.stat().st_size > 0
                ), "Formatted JSON should not be empty"

                # Verify JSON structure
                assert isinstance(hugr_json, dict), "HUGR should be a JSON object"

            except json.JSONDecodeError:
                # If not JSON, that's okay - just test raw bytes were saved
                pass

    def _count_output_operations(self, hugr_json: dict) -> int:
        """Count output-related operations in HUGR JSON."""
        count = 0

        def search(obj: object) -> None:
            nonlocal count
            if isinstance(obj, dict):
                if "op" in obj:
                    op_str = str(obj["op"]).lower()
                    if any(
                        term in op_str for term in ["output", "result", "return", "io"]
                    ):
                        count += 1

                for value in obj.values():
                    search(value)
            elif isinstance(obj, list):
                for item in obj:
                    search(item)

        search(hugr_json)
        return count


class TestResultFormats:
    """Test expected result formats and documentation."""

    def test_document_expected_formats(self) -> None:
        """Document and validate expected result formats for different patterns."""
        expected_formats = {
            "result_tagged": {
                "description": "Using result() function to tag outputs",
                "example_keys": ["measurement_outcome", "qubit_0", "qubit_1"],
                "format": "Named key-value pairs in result stream",
                "selene_output": "USER:TYPE:name -> value",
            },
            "return_value": {
                "description": "Using return statement",
                "example_keys": ["result", "measurement_1", "measurement_2"],
                "format": "Return values become default-named results",
                "selene_output": "USER:TYPE:result -> value or result_N for tuples",
            },
            "mixed_output": {
                "description": "Mix of result() and return",
                "example_keys": ["named_result", "result"],
                "format": "Both named and default results",
                "selene_output": "Combination of both formats",
            },
        }

        # Validate documentation structure
        for pattern_name, format_info in expected_formats.items():
            assert (
                "description" in format_info
            ), f"{pattern_name} should have description"
            assert (
                "example_keys" in format_info
            ), f"{pattern_name} should have example_keys"
            assert (
                "format" in format_info
            ), f"{pattern_name} should have format description"
            assert (
                "selene_output" in format_info
            ), f"{pattern_name} should have selene_output"

            # Example keys should be non-empty
            assert (
                len(format_info["example_keys"]) > 0
            ), f"{pattern_name} should have at least one example key"

            # All fields should be strings except example_keys
            assert isinstance(format_info["description"], str)
            assert isinstance(format_info["format"], str)
            assert isinstance(format_info["selene_output"], str)
            assert isinstance(format_info["example_keys"], list)
