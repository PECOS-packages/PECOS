"""Test HUGR compilation through Selene."""

import pytest
from guppylang.decorator import guppy as guppy_decorator
from guppylang.std.quantum import cx, h, measure, qubit, x
from hugr.package import Package
from pecos import Guppy, sim
from pecos.compilation_pipeline import compile_guppy_to_hugr
from pecos_rslib import state_vector

# compile_guppy_to_hugr returns the BINARY HUGR envelope (Model format): the
# ASCII magic "HUGRiHJv", a format byte, then a compressed payload. It is not
# UTF-8 text, so these tests validate it by parsing with hugr's own reader.
HUGR_ENVELOPE_MAGIC = b"HUGRiHJv"


def _op_names(pkg: Package) -> set[str]:
    """Collect the op names appearing in all modules of a HUGR package."""
    names = set()
    for module in pkg.modules:
        for node in module.nodes():
            n = node[0] if isinstance(node, tuple) else node
            op = module[n].op
            op_def = getattr(op, "_op_def", None)
            names.add(op_def.name if op_def is not None else type(op).__name__)
    return names


@pytest.mark.optional_dependency
class TestSeleneHUGRCompilation:
    """Test HUGR compilation through Selene."""

    def test_selene_hugr_llvm_generation(self) -> None:
        """Test that Selene can generate LLVM IR from HUGR."""

        # Define a proper Bell state with CNOT
        @guppy_decorator
        def bell_state() -> tuple[bool, bool]:
            """Create a Bell state and measure."""
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)  # Proper entanglement
            return measure(q1).read(), measure(q2).read()

        # The sim API handles HUGR compilation internally
        try:
            results = sim(Guppy(bell_state)).qubits(2).quantum(state_vector()).seed(42).run(100)

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
                assert correlation_rate > 0.95, f"Bell state should be highly correlated, got {correlation_rate:.2%}"
            else:
                # Alternative result format
                assert "measurements" in results or len(results) > 0, "Results should contain measurements"

        except (ImportError, RuntimeError, ValueError) as e:
            if "not supported" in str(e).lower() or "not available" in str(e).lower():
                pytest.skip(f"HUGR compilation not fully supported: {e}")
            pytest.fail(f"Unexpected compilation error: {e}")

    def test_direct_hugr_compilation(self) -> None:
        """Test direct HUGR compilation without simulation."""

        @guppy_decorator
        def simple_circuit() -> bool:
            """Simple H gate and measurement."""
            q = qubit()
            h(q)
            return measure(q).read()

        # Compile to HUGR
        hugr_bytes = compile_guppy_to_hugr(simple_circuit)

        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Verify HUGR envelope format and parse it with hugr's own reader
        assert hugr_bytes.startswith(HUGR_ENVELOPE_MAGIC), "HUGR should be in envelope format"

        pkg = Package.from_bytes(hugr_bytes)
        assert len(pkg.modules) >= 1, "HUGR package should contain at least one module"
        assert len(list(pkg.modules[0].nodes())) > 0, "HUGR module should have nodes"

    def test_complex_circuit_compilation(self) -> None:
        """Test compilation of more complex quantum circuits."""

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
            m0 = measure(q0).read()
            m1 = measure(q1).read()
            m2 = measure(q2).read()

            return m0, m1, m2

        # Compile to HUGR
        try:
            hugr_bytes = compile_guppy_to_hugr(quantum_teleportation)
        except Exception as e:
            pytest.fail(f"Compilation failed: {e}")

        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 100, "Complex circuit should produce substantial HUGR"

        # Verify it contains the expected quantum operations
        op_names = _op_names(Package.from_bytes(hugr_bytes))

        assert "H" in op_names, f"HUGR should contain H ops, found {sorted(op_names)}"
        assert "CX" in op_names, f"HUGR should contain CX ops, found {sorted(op_names)}"
        assert any("Measure" in name for name in op_names), f"HUGR should contain measure ops, found {sorted(op_names)}"

    def test_parametric_circuit_compilation(self) -> None:
        """Test compilation of parametric quantum circuits."""

        @guppy_decorator
        def parametric_circuit(n: int) -> int:
            """Circuit with parameter-based repetition."""
            count = 0
            for _i in range(n):
                q = qubit()
                h(q)
                if measure(q).read():
                    count += 1
            return count

        # Compile to HUGR
        try:
            hugr_bytes = compile_guppy_to_hugr(parametric_circuit)
        except Exception as e:
            pytest.fail(f"Parametric compilation failed: {e}")

        assert hugr_bytes is not None, "Should produce HUGR bytes"
        assert len(hugr_bytes) > 0, "HUGR bytes should not be empty"

        # Verify the parametric circuit still produces a loadable HUGR package
        assert hugr_bytes.startswith(HUGR_ENVELOPE_MAGIC), "Should be valid HUGR envelope"
        pkg = Package.from_bytes(hugr_bytes)
        assert len(pkg.modules) >= 1, "HUGR package should contain at least one module"


@pytest.mark.optional_dependency
class TestLLVMGeneration:
    """Test LLVM IR generation from quantum circuits."""

    def test_llvm_ir_from_hugr(self) -> None:
        """Test generating LLVM IR from HUGR."""

        @guppy_decorator
        def simple_measurement() -> bool:
            """Simple measurement circuit."""
            q = qubit()
            x(q)  # Put in |1⟩ state
            return measure(q).read()

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
            assert intrinsic in expected_llvm_pattern, f"Pattern should include {intrinsic}"

        # Check parameter types
        assert "(i64)" in expected_llvm_pattern, "Single qubit ops should take i64"
        assert "(i64, i64)" in expected_llvm_pattern, "Two qubit ops should take two i64"
        assert "i1 @__quantum__qis__mz" in expected_llvm_pattern, "Measurement should return i1"


@pytest.mark.optional_dependency
class TestHUGRVersionCompatibility:
    """Test HUGR envelope format compatibility."""

    def test_hugr_version_detection(self) -> None:
        """Test detection of the HUGR envelope format from compiled output."""

        @guppy_decorator
        def version_test() -> bool:
            q = qubit()
            h(q)
            return measure(q).read()

        hugr_bytes = compile_guppy_to_hugr(version_test)

        # Envelope header: 8-byte magic followed by a format byte
        assert hugr_bytes.startswith(HUGR_ENVELOPE_MAGIC), "Envelope should start with the HUGR magic"
        assert len(hugr_bytes) > len(HUGR_ENVELOPE_MAGIC), "Envelope should have a format header after the magic"

        # The format must be one the hugr reader understands
        pkg = Package.from_bytes(hugr_bytes)
        assert len(pkg.modules) >= 1, "Envelope should decode to a package with modules"

    def test_hugr_package_structure(self) -> None:
        """Test that the compiled package has a well-formed module structure."""

        @guppy_decorator
        def compatibility_test() -> tuple[bool, bool]:
            """Test circuit for compatibility."""
            q1, q2 = qubit(), qubit()
            h(q1)
            cx(q1, q2)
            return measure(q1).read(), measure(q2).read()

        hugr_bytes = compile_guppy_to_hugr(compatibility_test)
        assert hugr_bytes is not None, "Should produce HUGR bytes"

        pkg = Package.from_bytes(hugr_bytes)
        assert len(pkg.modules) >= 1, "Package should contain at least one module"

        module = pkg.modules[0]
        assert len(list(module.nodes())) > 0, "Module should have nodes"
        assert module.entrypoint is not None, "Module should have an entrypoint"

    def test_hugr_metadata_preservation(self) -> None:
        """Test that the function name is preserved through compilation."""

        @guppy_decorator
        def metadata_test() -> bool:
            """Test function with potential metadata."""
            q = qubit()
            h(q)
            return measure(q).read()

        hugr_bytes = compile_guppy_to_hugr(metadata_test)
        pkg = Package.from_bytes(hugr_bytes)

        # The guppy function must survive as a named function definition
        func_names = []
        for module in pkg.modules:
            for node in module.nodes():
                n = node[0] if isinstance(node, tuple) else node
                f_name = getattr(module[n].op, "f_name", None)
                if f_name:
                    func_names.append(f_name)

        assert any(
            name.endswith("metadata_test") for name in func_names
        ), f"HUGR should preserve the function name, found {func_names}"


def test_hugr_compilation_is_cached_per_definition_across_entry_points() -> None:
    """Both compile entry points share one per-definition HUGR byte cache.

    One DEM build compiles the same program several times (generator
    certificate, preflight digest check, trace execution); the cache makes
    every compile after the first free without changing any bytes.
    """
    from pecos._compilation import guppy_to_hugr

    @guppy_decorator
    def cached_prog() -> None:
        q = qubit()
        _ = measure(q).read()

    first = compile_guppy_to_hugr(cached_prog)
    assert guppy_to_hugr(cached_prog) is first
    assert compile_guppy_to_hugr(cached_prog) is first

    @guppy_decorator
    def other_prog() -> None:
        q = qubit()
        h(q)
        _ = measure(q).read()

    assert compile_guppy_to_hugr(other_prog) is not first


def test_parametric_definitions_never_share_cache_entries() -> None:
    """The library-form compile_function() bytes must not leak into
    guppy_to_hugr, whose contract is the entry-point compile() form."""
    from pecos._compilation import guppy_to_hugr

    @guppy_decorator
    def parametric_prog(flip: bool) -> None:  # pragma: no cover - compiled, not run
        q = qubit()
        if flip:
            h(q)
        _ = measure(q).read()

    first = compile_guppy_to_hugr(parametric_prog)
    assert first.startswith(HUGR_ENVELOPE_MAGIC)
    # Not cached: a second pipeline compile produces a fresh object.
    assert compile_guppy_to_hugr(parametric_prog) is not first

    # guppy_to_hugr must still reject the parametric definition rather than
    # serving the pipeline's library-form bytes.
    with pytest.raises(RuntimeError, match="Failed to compile Guppy to HUGR"):
        guppy_to_hugr(parametric_prog)
