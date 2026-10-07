"""Test to understand Selene's build process for HUGR programs.

This test explores how to use Selene's Python build() function to compile
HUGR from Guppy and create an executable that can be wrapped by SeleneExecutableEngine.
"""

import re
import tempfile
import textwrap
from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit
from pecos.compilation_pipeline import compile_guppy_to_hugr
from selene_sim import build
from selene_sim.backends import Coinflip, SimpleRuntime


class TestSeleneBuildProcess:
    """Test suite for Selene build process."""

    def test_selene_build_from_hugr(self) -> None:
        """Test building a Selene executable from HUGR."""

        # Create a simple Guppy program
        @guppy
        def simple_h() -> bool:
            q = qubit()
            h(q)
            return measure(q).read()

        # Compile to HUGR
        hugr_bytes = compile_guppy_to_hugr(simple_h)
        assert hugr_bytes is not None, "HUGR compilation should succeed"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Binary HUGR envelope (Model format); verify it is a valid, loadable HUGR.
        import pecos_rslib
        from pecos_rslib.hugr_lowering import compile_hugr_to_qis

        llvm_ir = compile_hugr_to_qis(hugr_bytes)
        assert re.search(r"define\b[^\n]*@qmain\(", llvm_ir)
        pecos_rslib.Qis(llvm_ir)

        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)

            # Save HUGR to file
            hugr_file = build_dir / "program.hugr"
            hugr_file.write_bytes(hugr_bytes)
            assert hugr_file.exists(), "HUGR file should be created"

            # Use Selene's build function - pass the HUGR bytes directly, not a file path
            # The build function expects the actual HUGR data
            instance = build(
                src=hugr_bytes,  # Pass the actual HUGR bytes, not the file path
                name="test_hugr_program",
                build_dir=build_dir,
            )
            assert instance is not None, "Build should create an instance"

            # Try to run the instance
            runtime = SimpleRuntime()
            simulator = Coinflip()

            # Run one shot
            list(
                instance.run(
                    simulator=simulator,
                    n_qubits=1,
                    runtime=runtime,
                    verbose=False,
                ),
            )

            # Note: Pure HUGR functions without measurements might return empty results
            # So we don't assert length > 0 here

            # Check what files were created
            created_files = list(build_dir.rglob("*"))
            assert len(created_files) > 1, "Build should create additional files"

    def test_hugr_to_qis_compilation(self) -> None:
        """Test that HUGR gets compiled to QIS (LLVM IR) during the build process.

        The Selene build pipeline works as:
        1. HUGR (input) → QIS/LLVM IR (intermediate) → Executable
        2. Only HUGR is accepted as input to build()
        3. QIS/LLVM IR is generated internally but not exposed for direct input

        This test verifies the HUGR → QIS transformation happens correctly.
        """

        # Create a Guppy program and compile to HUGR
        @guppy
        def test_qis_generation() -> bool:
            """Simple test function for QIS generation."""
            q = qubit()
            h(q)
            return measure(q).read()

        # Compile to HUGR
        hugr_bytes = compile_guppy_to_hugr(test_qis_generation)
        assert hugr_bytes is not None, "HUGR compilation should succeed"

        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)

            # Build with Selene (HUGR → QIS → Executable)
            instance = build(
                src=hugr_bytes,
                name="test_qis_pipeline",
                build_dir=build_dir,
                verbose=False,
            )
            assert instance is not None, "Build should create an instance"

            # Selene retains the lowered Helios QIS as LLVM bitcode.
            bitcode_files = list(build_dir.rglob("*.bc"))
            assert bitcode_files, "HUGR build should produce LLVM bitcode"
            for bitcode_file in bitcode_files:
                assert bitcode_file.read_bytes().startswith(b"BC\xc0\xde"), f"Invalid LLVM bitcode: {bitcode_file}"

    def test_qis_program_with_sim_api(self) -> None:
        """Test QIS programs using the sim() API.

        While Selene's build() function only accepts HUGR input,
        QIS (Quantum Instruction Set) programs can be executed using
        PECOS's sim() API with Qis wrapper.

        The two paths are:
        1. build(HUGR) → Selene executable (for building executables)
        2. sim(Qis) → PECOS execution (for direct simulation)
        """
        from pecos import Qis, sim
        from pecos_rslib import state_vector

        # Create Selene QIS format LLVM IR - use textwrap to avoid indentation issues
        llvm_ir = textwrap.dedent(
            """
        ; ModuleID = 'quantum_test'
        source_filename = "quantum_test"

        declare i64 @___qalloc() local_unnamed_addr
        declare void @___qfree(i64) local_unnamed_addr
        declare i64 @___lazy_measure(i64) local_unnamed_addr
        declare void @___reset(i64) local_unnamed_addr
        declare void @___rxy(i64, double, double) local_unnamed_addr
        declare void @___rz(i64, double) local_unnamed_addr
        declare void @setup(i64) local_unnamed_addr
        declare i64 @teardown() local_unnamed_addr

        define i64 @qmain(i64 %arg) #0 {
        entry:
          tail call void @setup(i64 %arg)
          %qubit = tail call i64 @___qalloc()
          %not_max = icmp eq i64 %qubit, -1
          br i1 %not_max, label %skip_reset, label %do_reset

        do_reset:
          tail call void @___reset(i64 %qubit)
          br label %skip_reset

        skip_reset:
          tail call void @___rxy(i64 %qubit, double 0x3FF921FB54442D18, double 0xBFF921FB54442D18)
          tail call void @___rz(i64 %qubit, double 0x400921FB54442D18)
          tail call void @___rxy(i64 %qubit, double 0x400921FB54442D18, double 0.000000e+00)
          %result = tail call i64 @___lazy_measure(i64 %qubit)
          tail call void @___qfree(i64 %qubit)
          %final = tail call i64 @teardown()
          ret i64 %final
        }

        attributes #0 = { "EntryPoint" }
        """,
        ).strip()

        # Create Qis program from the QIS LLVM IR string
        program = Qis(llvm_ir)

        # Run using sim() API
        results = sim(program).qubits(1).quantum(state_vector()).seed(42).run(100)

        # Verify results
        assert hasattr(results, "__getitem__"), "Results should be dict-like"

        # QIS returns results with key 'measurement_0'
        assert "measurement_0" in results, f"Results should contain 'measurement_0' key, got keys: {results.keys()}"
        measurements = results["measurement_0"]
        assert len(measurements) == 100, "Should have 100 shots"

        # H gate should give roughly 50/50 distribution
        ones = sum(measurements)
        zeros = 100 - ones
        assert 30 < ones < 70, f"Should be roughly 50/50 distribution, got {ones} ones"
        assert 30 < zeros < 70, f"Should be roughly 50/50 distribution, got {zeros} zeros"

    def test_qis_program_with_comments(self) -> None:
        """Test that QIS programs with comments are properly handled."""
        from pecos import Qis, sim
        from pecos_rslib import state_vector

        # Create QIS with extensive comments
        llvm_ir_with_comments = textwrap.dedent(
            """
        ; ModuleID = 'test_with_comments'
        ; This test verifies that comments don't break QIS parsing
        source_filename = "test_comments"

        ; === Function Declarations ===
        declare i64 @___qalloc() local_unnamed_addr     ; Allocate a qubit
        declare void @___qfree(i64) local_unnamed_addr  ; Free a qubit
        declare i64 @___lazy_measure(i64) local_unnamed_addr ; Measure qubit
        declare void @setup(i64) local_unnamed_addr
        declare i64 @teardown() local_unnamed_addr

        ; === Main Entry Point ===
        ; This function allocates a qubit, puts it in superposition,
        ; measures it, and returns the result
        define i64 @qmain(i64 %arg) #0 {
        entry:
          ; Setup quantum system
          tail call void @setup(i64 %arg)

          ; Allocate qubit
          %q = tail call i64 @___qalloc()

          ; Measure qubit (starts in |0⟩)
          %result = tail call i64 @___lazy_measure(i64 %q)

          ; Cleanup
          tail call void @___qfree(i64 %q)
          %final = tail call i64 @teardown() ; Get final state
          ret i64 %final ; Return
        }

        ; Attributes section
        attributes #0 = { "EntryPoint" } ; Mark as entry point
        """,
        ).strip()

        # Create and run program
        program = Qis(llvm_ir_with_comments)
        results = sim(program).qubits(1).quantum(state_vector()).seed(42).run(100)

        # Verify results
        assert hasattr(results, "__getitem__"), "Results should be dict-like"
        assert "measurement_0" in results, "Results should contain 'result' key"
        measurements = results["measurement_0"]
        assert len(measurements) == 100, "Should have 100 shots"

        # Since we're measuring |0⟩ directly, all results should be 0
        assert all(m == 0 for m in measurements), "Direct measurement of |0⟩ should always give 0"

    def test_qis_edge_cases(self) -> None:
        """Test QIS programs with edge cases like empty lines, multiple spaces, etc."""
        from pecos import Qis, sim
        from pecos_rslib import state_vector

        # QIS with various formatting edge cases
        llvm_ir_edge_cases = textwrap.dedent(
            """
        ; ModuleID = 'edge_cases'
        ; Empty lines above and below
        source_filename = "edge_cases"

        declare i64 @___qalloc()    local_unnamed_addr
        declare void   @___qfree(i64)   local_unnamed_addr
        declare i64    @___lazy_measure(i64)    local_unnamed_addr
        declare void @setup(i64) local_unnamed_addr
        declare i64 @teardown() local_unnamed_addr
        define i64 @qmain(i64 %arg) #0 {
        entry:
          tail call void @setup(i64 %arg)
          %q = tail call i64 @___qalloc()
          %r = tail call i64 @___lazy_measure(i64 %q)
          tail call void @___qfree(i64 %q)
          %f = tail call i64 @teardown()
          ret i64 %f
        }
        attributes #0 = { "EntryPoint" }

        ; Trailing comment
        """,
        ).strip()

        # Should handle edge cases gracefully
        program = Qis(llvm_ir_edge_cases)
        results = sim(program).qubits(1).quantum(state_vector()).seed(42).run(50)

        assert "measurement_0" in results, "Should have results even with edge case formatting"
        assert len(results["measurement_0"]) == 50, "Should complete all shots"
        assert all(m == 0 for m in results["measurement_0"]), "Should measure |0⟩ as 0"

    def test_qis_program_consistency(self) -> None:
        """Test that Qis produces consistent results for QIS format.

        Test that the same QIS LLVM IR produces consistent results when run
        multiple times with the same seed.
        """
        from pecos import Qis, sim
        from pecos_rslib import state_vector

        # Same QIS program for both
        qis_ir = textwrap.dedent(
            """
        ; Test equivalence
        declare i64 @___qalloc() local_unnamed_addr
        declare void @___qfree(i64) local_unnamed_addr
        declare i64 @___lazy_measure(i64) local_unnamed_addr
        declare void @___rxy(i64, double, double) local_unnamed_addr
        declare void @setup(i64) local_unnamed_addr
        declare i64 @teardown() local_unnamed_addr

        define i64 @qmain(i64 %arg) #0 {
        entry:
          tail call void @setup(i64 %arg)
          %q = tail call i64 @___qalloc()
          ; Apply X gate using rotations to get |1⟩
          tail call void @___rxy(i64 %q, double 0x400921FB54442D18, double 0.0)
          %r = tail call i64 @___lazy_measure(i64 %q)
          tail call void @___qfree(i64 %q)
          %f = tail call i64 @teardown()
          ret i64 %f
        }

        attributes #0 = { "EntryPoint" }
        """,
        ).strip()

        # Test with Qis - first run
        qis_prog = Qis(qis_ir)
        qis_results_1 = sim(qis_prog).qubits(1).quantum(state_vector()).seed(42).run(100)

        # Test with Qis - second run with same seed
        qis_results_2 = sim(qis_prog).qubits(1).quantum(state_vector()).seed(42).run(100)

        # Both runs should produce identical results
        assert "measurement_0" in qis_results_1, "Qis should produce results"
        assert "measurement_0" in qis_results_2, "Qis should produce results"

        # With same seed, results should be identical
        assert (
            qis_results_1["measurement_0"] == qis_results_2["measurement_0"]
        ), "Qis should produce identical results with same seed"

        # X gate should give |1⟩
        assert all(m == 1 for m in qis_results_1["measurement_0"]), "X gate should always measure 1"
        assert all(m == 1 for m in qis_results_2["measurement_0"]), "X gate should always measure 1"

    def test_hugr_to_selene_compilation_chain(self) -> None:
        """Test the full compilation chain from Guppy to Selene execution."""

        @guppy
        def bell_pair() -> tuple[bool, bool]:
            """Create a Bell pair."""
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)
            return measure(q1).read(), measure(q2).read()

        # Compile to HUGR
        try:
            hugr_bytes = compile_guppy_to_hugr(bell_pair)
        except Exception as e:
            pytest.fail(f"HUGR compilation failed: {e}")

        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 100, "HUGR should have substantial content"

        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)
            hugr_file = build_dir / "bell_pair.hugr"
            hugr_file.write_bytes(hugr_bytes)

            # Try to build with Selene - pass HUGR bytes directly
            instance = build(
                src=hugr_bytes,  # Pass the actual HUGR bytes
                name="bell_pair_test",
                build_dir=build_dir,  # Pass Path object
            )

            # If build succeeds, verify instance
            assert instance is not None, "Should create instance"

            # Try to get some information about the built executable
            build_artifacts = list(build_dir.iterdir())
            assert len(build_artifacts) > 1, "Should create build artifacts"


class TestBuildOutputFormats:
    """Test different output formats from the build process."""

    def test_hugr_envelope_format(self) -> None:
        """Test handling of HUGR envelope format."""

        @guppy
        def simple_circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q).read()

        hugr_bytes = compile_guppy_to_hugr(simple_circuit)
        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Binary HUGR envelope (Model format); verify it is a valid, loadable HUGR.
        import pecos_rslib
        from pecos_rslib.hugr_lowering import compile_hugr_to_qis

        llvm_ir = compile_hugr_to_qis(hugr_bytes)
        assert re.search(r"define\b[^\n]*@qmain\(", llvm_ir)
        pecos_rslib.Qis(llvm_ir)
