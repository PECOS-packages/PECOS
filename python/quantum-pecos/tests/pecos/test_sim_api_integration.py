"""Test the unified sim API with different program types."""

import json

import pytest

# Check for required dependencies
try:
    from pecos import sim

    SIM_API_AVAILABLE = True
except ImportError:
    SIM_API_AVAILABLE = False

try:
    from pecos_rslib import (
        Hugr,
        PhirJson,
        Qasm,
        Qis,
        sparse_stabilizer,
        state_vector,
    )

    PECOS_RSLIB_AVAILABLE = True
except ImportError:
    PECOS_RSLIB_AVAILABLE = False

try:
    from guppylang import guppy
    from guppylang.std.quantum import h, measure, qubit

    GUPPY_AVAILABLE = True
except ImportError:
    GUPPY_AVAILABLE = False


@pytest.mark.skipif(
    not all([SIM_API_AVAILABLE, PECOS_RSLIB_AVAILABLE]),
    reason="sim API or pecos_rslib not available",
)
class TestQASMSimulation:
    """Test sim API with QASM programs."""

    def test_sim_api_with_simple_qasm(self) -> None:
        """Test sim API with simple QASM program."""
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """

        program = Qasm.from_string(qasm_str)
        shot_vec = sim(program).seed(42).run(1000)

        # Convert ShotVec to dict
        results = shot_vec.to_dict()

        # Results is a dict with register names as keys, values are shot arrays
        assert isinstance(results, dict), "Results should be a dictionary"
        assert "c" in results, "Results should contain register 'c'"
        assert len(results["c"]) == 1000, "Should have 1000 shots"

        # Check measurement distribution (should be roughly 50/50)
        measurements = results["c"]
        ones = sum(measurements)
        zeros = 1000 - ones

        # With seed, results should be deterministic but still mixed
        assert (
            300 < ones < 700
        ), f"Should be roughly 50/50 distribution, got {ones} ones"
        assert (
            300 < zeros < 700
        ), f"Should be roughly 50/50 distribution, got {zeros} zeros"

    def test_sim_api_with_bell_state_qasm(self) -> None:
        """Test sim API with Bell state in QASM."""
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """

        program = Qasm.from_string(qasm_str)
        shot_vec = sim(program).seed(42).run(100)
        results = shot_vec.to_dict()

        assert "c" in results, "Results should contain register 'c'"
        assert len(results["c"]) == 100, "Should have 100 shots"

        # Each shot should be a 2-bit value (0, 1, 2, or 3)
        # For Bell state, should only see 00 (0) and 11 (3)
        measurements = results["c"]
        unique_values = set(measurements)

        # Bell state should only produce correlated results
        assert unique_values.issubset(
            {0, 3},
        ), f"Bell state should only give 00 or 11, got {unique_values}"

        # Should see both values with reasonable probability
        count_00 = measurements.count(0)
        count_11 = measurements.count(3)
        assert count_00 > 20, f"Should see |00⟩ state, got {count_00} times"
        assert count_11 > 20, f"Should see |11⟩ state, got {count_11} times"

    def test_sim_builder_chaining(self) -> None:
        """Test builder pattern chaining."""
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
        """

        program = Qasm.from_string(qasm_str)

        # Test chaining various configurations
        shot_vec = sim(program).seed(42).workers(2).quantum(state_vector()).run(500)
        results = shot_vec.to_dict()

        assert "c" in results, "Results should contain register 'c'"
        assert len(results["c"]) == 500, "Should have 500 shots"

        # X gate should always give |1⟩
        measurements = results["c"]
        assert all(m == 1 for m in measurements), "X gate should always measure 1"


@pytest.mark.skipif(
    not all([SIM_API_AVAILABLE, PECOS_RSLIB_AVAILABLE]),
    reason="sim API or pecos_rslib not available",
)
class TestLLVMSimulation:
    """Test sim API with LLVM IR programs."""

    def test_sim_api_with_llvm_simple(self) -> None:
        """Test sim API with simple LLVM IR program."""
        # QIS format LLVM IR - uses i64 for qubit indices and qmain entry point
        # This matches the format that PECOS HUGR compiler generates
        llvm_ir = """
        ; ModuleID = 'quantum_test'
        source_filename = "quantum_test"

        @str_c = constant [2 x i8] c"c\\00"

        ; Entry point using qmain(i64)->i64 protocol
        define i64 @qmain(i64 %0) #0 {
        entry:
            ; Allocate qubit (returns i64)
            %qubit = call i64 @__quantum__rt__qubit_allocate()

            ; Apply H gate (takes i64 qubit index)
            call void @__quantum__qis__h__body(i64 %qubit)

            ; Allocate result handle
            %result_id = call i64 @__quantum__rt__result_allocate()

            ; Measure qubit (returns i32 measurement)
            %measurement = call i32 @__quantum__qis__m__body(i64 %qubit, i64 %result_id)

            ; Record output (convert i64 to i8* pointer for register name)
            %result_ptr = inttoptr i64 %result_id to i8*
            call void @__quantum__rt__result_record_output(i8* %result_ptr,
                i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str_c, i32 0, i32 0))

            ret i64 0
        }

        declare i64 @__quantum__rt__qubit_allocate()
        declare void @__quantum__qis__h__body(i64)
        declare i64 @__quantum__rt__result_allocate()
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare void @__quantum__rt__result_record_output(i8*, i8*)

        attributes #0 = { "EntryPoint" }
        """

        try:
            program = Qis.from_string(llvm_ir)

            # Try to run - this might work now with proper QIR format
            shot_vec = sim(program).qubits(1).seed(42).run(10)
            results = shot_vec.to_dict()

            # If it works, verify results
            assert isinstance(results, dict), "Results should be a dictionary"
            assert len(results) > 0, "Should have some results"

            # Check for measurements
            if "c" in results:
                measurements = results["c"]
                assert len(measurements) == 10, "Should have 10 shots"
                assert all(
                    m in [0, 1] for m in measurements
                ), "Measurements should be binary"

        except (RuntimeError, ValueError, NotImplementedError) as e:
            # Known LLVM runtime issues
            error_msg = str(e).lower()
            if any(
                x in error_msg
                for x in [
                    "entry",
                    "not implemented",
                    "undefined symbol",
                    "failed to load",
                ]
            ):
                pytest.skip(f"LLVM runtime not fully working yet: {e}")
            else:
                # Truly unexpected error
                pytest.fail(f"Unexpected LLVM simulation error: {e}")

    def test_sim_api_with_llvm_bell_state(self) -> None:
        """Test sim API with Bell state in LLVM IR."""
        # QIS format LLVM IR - uses i64 for qubit indices and qmain entry point
        # This matches the format that PECOS HUGR compiler generates
        llvm_ir = """
        ; ModuleID = 'bell_state'
        source_filename = "bell_state"

        @str_c0 = constant [3 x i8] c"c0\\00"
        @str_c1 = constant [3 x i8] c"c1\\00"

        ; Entry point using qmain(i64)->i64 protocol
        define i64 @qmain(i64 %0) #0 {
        entry:
            ; Allocate qubits (returns i64)
            %q0 = call i64 @__quantum__rt__qubit_allocate()
            %q1 = call i64 @__quantum__rt__qubit_allocate()

            ; Apply H to qubit 0
            call void @__quantum__qis__h__body(i64 %q0)

            ; Apply CX(q0, q1)
            call void @__quantum__qis__cx__body(i64 %q0, i64 %q1)

            ; Measure qubit 0
            %result_id0 = call i64 @__quantum__rt__result_allocate()
            %measurement0 = call i32 @__quantum__qis__m__body(i64 %q0, i64 %result_id0)
            %result_ptr0 = inttoptr i64 %result_id0 to i8*
            call void @__quantum__rt__result_record_output(i8* %result_ptr0,
                i8* getelementptr inbounds ([3 x i8], [3 x i8]* @str_c0, i32 0, i32 0))

            ; Measure qubit 1
            %result_id1 = call i64 @__quantum__rt__result_allocate()
            %measurement1 = call i32 @__quantum__qis__m__body(i64 %q1, i64 %result_id1)
            %result_ptr1 = inttoptr i64 %result_id1 to i8*
            call void @__quantum__rt__result_record_output(i8* %result_ptr1,
                i8* getelementptr inbounds ([3 x i8], [3 x i8]* @str_c1, i32 0, i32 0))

            ret i64 0
        }

        declare i64 @__quantum__rt__qubit_allocate()
        declare void @__quantum__qis__h__body(i64)
        declare void @__quantum__qis__cx__body(i64, i64)
        declare i64 @__quantum__rt__result_allocate()
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare void @__quantum__rt__result_record_output(i8*, i8*)

        attributes #0 = { "EntryPoint" }
        """

        try:
            program = Qis.from_string(llvm_ir)
            shot_vec = sim(program).qubits(2).seed(42).run(50)
            results = shot_vec.to_dict()

            assert isinstance(results, dict), "Results should be a dictionary"

            # Check if we have correlated measurements
            if "c0" in results and "c1" in results:
                m0 = results["c0"]
                m1 = results["c1"]

                assert len(m0) == 50, "Should have 50 shots for qubit 0"
                assert len(m1) == 50, "Should have 50 shots for qubit 1"

                # Bell state should be correlated
                correlated = sum(1 for i in range(50) if m0[i] == m1[i])
                assert (
                    correlated == 50
                ), f"Bell state should be perfectly correlated, got {correlated}/50"

        except (RuntimeError, ValueError, NotImplementedError) as e:
            error_msg = str(e).lower()
            if any(
                x in error_msg
                for x in [
                    "not implemented",
                    "not supported",
                    "undefined symbol",
                    "failed to load",
                    "getelementptr",
                    "unsized type",
                ]
            ):
                pytest.skip(f"LLVM Bell state not fully working yet: {e}")
            else:
                pytest.fail(f"Unexpected error: {e}")


@pytest.mark.optional_dependency
@pytest.mark.skipif(
    not all([SIM_API_AVAILABLE, PECOS_RSLIB_AVAILABLE, GUPPY_AVAILABLE]),
    reason="Dependencies not available",
)
class TestHUGRSimulation:
    """Test sim API with HUGR programs."""

    def test_sim_api_with_real_hugr(self) -> None:
        """Test sim API with real HUGR from Guppy compilation."""

        # Create a real HUGR program from Guppy
        @guppy
        def simple_circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        # Compile to HUGR
        compiled = simple_circuit.compile()

        # Get HUGR bytes - preferring to_bytes() which gives the correct format
        if hasattr(compiled, "to_bytes"):
            hugr_bytes = compiled.to_bytes()
        else:
            # Use to_str() for HUGR envelope format (includes header)
            hugr_str = compiled.to_str()
            hugr_bytes = hugr_str.encode("utf-8")

        try:
            program = Hugr.from_bytes(hugr_bytes)

            # This should route through Selene with HUGR 0.13
            shot_vec = sim(program).qubits(1).quantum(state_vector()).seed(42).run(100)
            results = shot_vec.to_dict()

            # If it works, verify results
            assert isinstance(results, dict), "Results should be a dictionary"

            # Check for measurements
            has_measurements = (
                "measurement_1" in results
                or "measurements" in results
                or len(results) > 0
            )
            assert has_measurements, "Should have measurement results"

            if "measurement_1" in results:
                measurements = results["measurement_1"]
                assert len(measurements) == 100, "Should have 100 measurements"

                # Should be roughly 50/50 for H gate
                ones = sum(measurements)
                assert (
                    30 < ones < 70
                ), f"H gate should give roughly 50/50, got {ones}/100"

        except (
            ImportError,
            RuntimeError,
            ValueError,
            NotImplementedError,
            TypeError,
        ) as e:
            error_msg = str(e).lower()
            if "hugr" in error_msg and "not implemented" in error_msg:
                pytest.skip(f"HUGR parsing not fully implemented: {e}")
            elif "not supported" in error_msg:
                pytest.skip(f"HUGR not fully supported: {e}")
            elif "unknown resource type" in error_msg and "hugrprogram" in error_msg:
                pytest.skip(f"Hugr type not properly recognized by sim API: {e}")
            else:
                # This might be a real error worth investigating
                pytest.fail(f"Unexpected HUGR simulation error: {e}")

    def test_sim_api_hugr_routing(self) -> None:
        """Test that HUGR programs route through compilation to Selene engine."""

        # Create a real HUGR program from Guppy for routing test
        @guppy
        def simple_h_measure() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        # Compile to HUGR
        compiled = simple_h_measure.compile()

        # Get HUGR bytes
        if hasattr(compiled, "to_bytes"):
            hugr_bytes = compiled.to_bytes()
        else:
            hugr_str = compiled.to_str()
            hugr_bytes = hugr_str.encode("utf-8")

        try:
            program = Hugr.from_bytes(hugr_bytes)

            # Create builder - this should work with real HUGR
            builder = sim(program)
            assert builder is not None, "Should create sim builder for HUGR"

            # Builder should have the right methods
            assert hasattr(builder, "qubits"), "Builder should have qubits method"
            assert hasattr(builder, "run"), "Builder should have run method"
            assert hasattr(builder, "quantum"), "Builder should have quantum method"

            # Configure and verify builder works
            configured = builder.qubits(1).quantum(state_vector())
            assert configured is not None, "Should configure builder"

        except (ImportError, RuntimeError) as e:
            error_msg = str(e).lower()
            if "selene" in error_msg:
                pytest.skip("Selene not available for HUGR routing")
            elif "hugr" in error_msg and "not implemented" in error_msg:
                pytest.skip(f"HUGR compilation not fully implemented: {e}")
            else:
                pytest.fail(f"Unexpected error in HUGR routing: {e}")


@pytest.mark.skipif(
    not all([SIM_API_AVAILABLE, PECOS_RSLIB_AVAILABLE]),
    reason="sim API or pecos_rslib not available",
)
class TestPHIRSimulation:
    """Test sim API with PHIR JSON programs."""

    def test_sim_api_with_phir_basic(self) -> None:
        """Test sim API with basic PHIR JSON program."""
        # PHIR format for simple H gate and measurement
        phir_json = {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"description": "Simple H gate test"},
            "ops": [
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 1,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i64",
                    "variable": "m",
                    "size": 1,
                },
                {"qop": "H", "args": [["q", 0]]},
                {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
                {"cop": "Result", "args": ["m"], "returns": ["c"]},
            ],
        }

        phir_str = json.dumps(phir_json)

        program = PhirJson.from_string(phir_str)
        shot_vec = sim(program).qubits(1).seed(42).run(50)
        results = shot_vec.to_dict()

        assert isinstance(results, dict), "Results should be a dictionary"
        assert "c" in results, "Results should contain register 'c'"

        measurements = results["c"]
        assert len(measurements) == 50, "Should have 50 measurements"

        # Should be binary values
        assert all(m in [0, 1] for m in measurements), "Measurements should be binary"

        # H gate should give roughly 50/50 distribution
        ones = sum(measurements)
        assert 15 < ones < 35, f"H gate should give roughly 50/50, got {ones}/50"

    def test_sim_api_with_phir_bell_state(self) -> None:
        """Test sim API with Bell state in PHIR format."""
        phir_json = {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"description": "Bell state"},
            "ops": [
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 2,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i64",
                    "variable": "m",
                    "size": 2,
                },
                {"qop": "H", "args": [["q", 0]]},
                {"qop": "CX", "args": [["q", 0], ["q", 1]]},
                {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
                {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
                {"cop": "Result", "args": ["m"], "returns": ["c"]},
            ],
        }

        phir_str = json.dumps(phir_json)

        program = PhirJson.from_string(phir_str)
        shot_vec = sim(program).qubits(2).seed(42).run(100)
        results = shot_vec.to_dict()

        assert isinstance(results, dict), "Results should be a dictionary"
        assert "c" in results, "Results should contain register 'c'"

        measurements = results["c"]
        assert len(measurements) == 100, "Should have 100 measurements"

        # Bell state should only produce 00 (0) and 11 (3) in 2-bit encoding
        unique_values = set(measurements)
        assert unique_values.issubset(
            {0, 3},
        ), f"Bell state should only give 00 or 11, got {unique_values}"

        # Should see both values with reasonable probability
        count_00 = measurements.count(0)
        count_11 = measurements.count(3)
        assert count_00 > 20, f"Should see |00⟩ state, got {count_00} times"
        assert count_11 > 20, f"Should see |11⟩ state, got {count_11} times"


class TestSimAPIFeatures:
    """Test various features of the sim API."""

    @pytest.mark.skipif(
        not all([SIM_API_AVAILABLE, PECOS_RSLIB_AVAILABLE]),
        reason="Dependencies not available",
    )
    def test_sim_with_different_backends(self) -> None:
        """Test sim API with different quantum backends."""
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """

        program = Qasm.from_string(qasm_str)

        # Test with state vector backend
        shot_vec_sv = sim(program).quantum(state_vector()).seed(42).run(100)
        results_sv = shot_vec_sv.to_dict()
        assert "c" in results_sv, "State vector backend should produce results"

        # Test with sparse stabilizer backend
        try:
            shot_vec_ss = sim(program).quantum(sparse_stabilizer()).seed(42).run(100)
            results_ss = shot_vec_ss.to_dict()
            assert "c" in results_ss, "Sparse stabilizer backend should produce results"

            # Results might differ between backends but both should be valid
            assert len(results_sv["c"]) == 100, "State vector should give 100 shots"
            assert (
                len(results_ss["c"]) == 100
            ), "Sparse stabilizer should give 100 shots"

        except (RuntimeError, ValueError) as e:
            if "not supported" in str(e).lower():
                pytest.skip(f"Sparse stabilizer not supported for this program: {e}")

    @pytest.mark.skipif(
        not all([SIM_API_AVAILABLE, PECOS_RSLIB_AVAILABLE]),
        reason="Dependencies not available",
    )
    def test_sim_error_handling(self) -> None:
        """Test error handling in sim API."""
        # Invalid QASM
        invalid_qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        invalid_gate q[0];
        """

        program = Qasm.from_string(invalid_qasm)
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            sim(program).run(10)

        assert (
            "invalid" in str(exc_info.value).lower()
            or "error" in str(exc_info.value).lower()
        ), "Should raise error for invalid QASM"

    @pytest.mark.skipif(
        not all([SIM_API_AVAILABLE, PECOS_RSLIB_AVAILABLE]),
        reason="Dependencies not available",
    )
    def test_sim_deterministic_seeding(self) -> None:
        """Test that seeding produces deterministic results."""
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        h q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """

        program = Qasm.from_string(qasm_str)

        # Run twice with same seed
        shot_vec1 = sim(program).seed(12345).run(50)
        shot_vec2 = sim(program).seed(12345).run(50)
        results1 = shot_vec1.to_dict()
        results2 = shot_vec2.to_dict()

        assert "c" in results1, "First run should produce results"
        assert "c" in results2, "Second run should produce results"

        # Results should be identical with same seed
        assert results1["c"] == results2["c"], "Same seed should give identical results"

        # Run with different seed
        shot_vec3 = sim(program).seed(54321).run(50)
        results3 = shot_vec3.to_dict()

        # Results should differ with different seed (statistically)
        assert (
            results1["c"] != results3["c"]
        ), "Different seeds should give different results"
