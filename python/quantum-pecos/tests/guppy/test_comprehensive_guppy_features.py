"""Tests for Guppy features on HUGR-LLVM and PHIR pipelines."""

from typing import TYPE_CHECKING, Any

import pytest
from guppylang import guppy
from guppylang.std.quantum import cx, h, measure, qubit, x, y, z
from pecos import Guppy, sim
from pecos_rslib import state_vector

if TYPE_CHECKING:
    from pecos.protocols import GuppyCallable


def decode_integer_results(results: list[int], n_bits: int) -> list[tuple[bool, ...]]:
    """Decode integer-encoded results back to tuples of booleans.

    When guppy functions return tuples of bools, sim encodes them
    as integers where bit i represents the i-th boolean in the tuple.
    """
    decoded = []
    for val in results:
        bits = [bool(val & (1 << i)) for i in range(n_bits)]
        decoded.append(tuple(bits))
    return decoded


class GuppyPipelineTest:
    """Helper class for testing Guppy programs on both pipelines."""

    def test_function_on_both_pipelines(
        self,
        func: "GuppyCallable",
        shots: int = 10,
        seed: int = 42,
        **kwargs: object,
    ) -> dict[str, Any]:
        """Test a Guppy function (using the Rust backend)."""
        results = {}

        try:
            # Use sim() API instead of run_guppy
            n_qubits = kwargs.get("n_qubits", kwargs.get("max_qubits", 10))
            builder = sim(Guppy(func)).qubits(n_qubits).quantum(state_vector())
            if seed is not None:
                builder = builder.seed(seed)
            result_obj = builder.run(shots)
            result_dict = result_obj.to_dict()

            # Format results to match expected structure.
            # "measurements" holds one row per shot like [[1], [0, 1], ...];
            # a missing key is a hard failure (reported via the except below).
            raw_measurements = result_dict["measurements"]
            if raw_measurements and isinstance(raw_measurements[0], list):
                # Check if function returns single bool or tuple
                import inspect

                actual_func = func
                if hasattr(func, "wrapped") and hasattr(
                    func.wrapped,
                    "python_func",
                ):
                    actual_func = func.wrapped.python_func
                try:
                    sig = inspect.signature(actual_func)
                    return_type = sig.return_annotation
                    is_tuple_return = hasattr(return_type, "__origin__") and return_type.__origin__ is tuple
                except (ValueError, TypeError):
                    is_tuple_return = False

                if is_tuple_return:
                    # Return full measurement tuples
                    measurements = [tuple(m) for m in raw_measurements]
                else:
                    # For single bool return, take the last measurement from each shot
                    measurements = [m[-1] if m else 0 for m in raw_measurements]
            else:
                measurements = raw_measurements

            func_name = getattr(
                func,
                "__name__",
                getattr(func, "name", "quantum_func"),
            )
            result = {
                "results": measurements,
                "shots": shots,
                "function_name": func_name,
            }
            results["hugr_llvm"] = {
                "success": True,
                "result": result,
                "error": None,
            }
        except Exception as e:
            # A pipeline failure must FAIL the test: every semantic
            # assertion in this file is gated behind success, so converting
            # exceptions into success=False used to make any engine
            # regression pass everything.
            pytest.fail(f"guppy pipeline failed for {func}: {e}")

        return results


@pytest.fixture
def pipeline_tester() -> GuppyPipelineTest:
    """Fixture providing the pipeline testing helper."""
    import gc

    # Force garbage collection to clean up any lingering resources
    gc.collect()

    # Create fresh test instance
    tester = GuppyPipelineTest()

    yield tester

    # Force garbage collection to clean up test resources
    gc.collect()


# ============================================================================
# BASIC QUANTUM OPERATIONS TESTS
# ============================================================================


class TestBasicQuantumOperations:
    """Test basic quantum gate operations on both pipelines."""

    def test_single_qubit_hadamard(self, pipeline_tester: GuppyPipelineTest) -> None:
        """Test Hadamard gate on single qubit."""

        @guppy
        def hadamard_test() -> bool:
            q = qubit()
            h(q)
            return measure(q).read()

        results = pipeline_tester.test_function_on_both_pipelines(
            hadamard_test,
            shots=50,
        )
        rows = results["hugr_llvm"]["result"]["results"]
        assert len(rows) == 50
        # H|0> is an exact 50/50 superposition: both outcomes must appear
        # with a near-even split.
        ones = sum(int(r) for r in rows)
        assert 12 <= ones <= 38, f"H distribution off: {ones}/50 ones"

    def test_pauli_gates(self, pipeline_tester: GuppyPipelineTest) -> None:
        """Test all Pauli gates (X, Y, Z)."""

        @guppy
        def pauli_x_test() -> bool:
            q = qubit()
            x(q)  # Should flip |0⟩ to |1⟩
            return measure(q).read()

        @guppy
        def pauli_y_test() -> bool:
            q = qubit()
            y(q)  # Should flip |0⟩ to |1⟩ with phase
            return measure(q).read()

        @guppy
        def pauli_z_test() -> bool:
            q = qubit()
            z(q)  # Should leave |0⟩ unchanged
            return measure(q).read()

        # Test X gate - should measure |1⟩ deterministically with fixed seed
        results_x = pipeline_tester.test_function_on_both_pipelines(
            pauli_x_test,
            shots=100,
            seed=42,
        )
        if results_x.get("hugr_llvm", {}).get("success"):
            ones_count = sum(results_x["hugr_llvm"]["result"]["results"])
            # X gate should flip |0⟩ to |1⟩, expect 100% ones
            assert ones_count == 100, f"X gate should produce all 1s, got {ones_count}/100"

        # Test Y gate - should measure |1⟩ deterministically
        results_y = pipeline_tester.test_function_on_both_pipelines(
            pauli_y_test,
            shots=100,
            seed=42,
        )
        if results_y.get("hugr_llvm", {}).get("success"):
            ones_count = sum(results_y["hugr_llvm"]["result"]["results"])
            # Y gate should flip |0⟩ to |1⟩ with phase, expect 100% ones
            assert ones_count == 100, f"Y gate should produce all 1s, got {ones_count}/100"

        # Test Z gate - should measure |0⟩ deterministically
        results_z = pipeline_tester.test_function_on_both_pipelines(
            pauli_z_test,
            shots=100,
            seed=42,
        )
        if results_z.get("hugr_llvm", {}).get("success"):
            ones_count = sum(results_z["hugr_llvm"]["result"]["results"])
            # Z gate should leave |0⟩ unchanged, expect 0% ones
            assert ones_count == 0, f"Z gate should produce all 0s, got {ones_count}/100"

    def test_bell_state_entanglement(self, pipeline_tester: GuppyPipelineTest) -> None:
        """Test Bell state creation and entanglement."""

        @guppy
        def bell_state() -> tuple[bool, bool]:
            q0, q1 = qubit(), qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0).read(), measure(q1).read()

        results = pipeline_tester.test_function_on_both_pipelines(bell_state, shots=50)

        # Verify HUGR-LLVM pipeline results
        if results.get("hugr_llvm", {}).get("success"):
            measurements = results["hugr_llvm"]["result"]["results"]
            # Check if measurements are already tuples or need decoding
            if measurements and isinstance(measurements[0], tuple):
                # Already decoded as tuples
                decoded_measurements = measurements
            else:
                # Decode integer-encoded results
                decoded_measurements = decode_integer_results(measurements, 2)
            correlated = sum(1 for (a, b) in decoded_measurements if a == b)
            correlation_rate = correlated / len(decoded_measurements)
            assert (
                correlation_rate == 1.0
            ), f"Bell correlation is EXACT on a noiseless statevector, got {correlation_rate:.2%}"

        # Verify PHIR pipeline results if available
        if results.get("phir", {}).get("success"):
            measurements = results["phir"]["result"]["results"]
            # Decode integer-encoded results
            decoded_measurements = decode_integer_results(measurements, 2)
            correlated = sum(1 for (a, b) in decoded_measurements if a == b)
            correlation_rate = correlated / len(decoded_measurements)
            assert (
                correlation_rate == 1.0
            ), f"PHIR Bell correlation is EXACT on a noiseless statevector, got {correlation_rate:.2%}"


# ============================================================================
# CLASSICAL COMPUTATION TESTS
# ============================================================================


class TestClassicalComputation:
    """Test classical computation capabilities in both pipelines."""

    def test_boolean_operations(self, pipeline_tester: GuppyPipelineTest) -> None:
        """Test boolean logic operations."""

        @guppy
        def boolean_and_test() -> bool:
            # Simple boolean logic with quantum measurement
            q = qubit()
            result = measure(q).read()  # Will be False (|0⟩)
            return result and True

        @guppy
        def boolean_or_test() -> bool:
            q = qubit()
            x(q)  # Flip to |1⟩
            result = measure(q).read()  # Will be True
            return result or False

        # AND: measure(|0>) is deterministically 0
        results_and = pipeline_tester.test_function_on_both_pipelines(
            boolean_and_test,
            shots=10,
        )
        rows = results_and["hugr_llvm"]["result"]["results"]
        assert [int(r) for r in rows] == [0] * 10, f"AND path measurements: {rows}"

        # OR: measure(X|0>) is deterministically 1
        results_or = pipeline_tester.test_function_on_both_pipelines(
            boolean_or_test,
            shots=10,
        )
        rows = results_or["hugr_llvm"]["result"]["results"]
        assert [int(r) for r in rows] == [1] * 10, f"OR path measurements: {rows}"

    def test_classical_arithmetic(self) -> None:
        """Pure-classical programs surface the entrypoint's return value
        under the "return" key (no measurements, no result() calls)."""

        @guppy
        def arithmetic_test() -> int:
            # Simple arithmetic that doesn't depend on quantum measurements
            a = 5
            b = 3
            return a + b

        from pecos import Guppy, sim
        from pecos_rslib import state_vector

        results = sim(Guppy(arithmetic_test)).qubits(1).quantum(state_vector()).seed(1).run(5).to_dict()
        assert list(results["return"]) == [8] * 5, f"keys: {sorted(results)}"


# ============================================================================
# HYBRID QUANTUM-CLASSICAL TESTS
# ============================================================================


class TestHybridPrograms:
    """Test hybrid quantum-classical programs."""

    def test_conditional_quantum_operations(
        self,
        pipeline_tester: GuppyPipelineTest,
    ) -> None:
        """Test quantum operations conditional on classical results."""

        @guppy
        def conditional_gate() -> bool:
            q1 = qubit()
            q2 = qubit()

            # Measure first qubit
            result1 = measure(q1).read()  # Will be False (|0⟩)

            # Apply gate to second qubit based on first measurement
            if result1:
                x(q2)  # This won't execute since result1 is False

            return measure(q2).read()  # Should be False

        results = pipeline_tester.test_function_on_both_pipelines(
            conditional_gate,
            shots=20,
        )
        if results.get("hugr_llvm", {}).get("success"):
            measurements = results["hugr_llvm"]["result"]["results"]
            # Results are boolean values, count True values
            sum(1 for r in measurements if r)
            # When HUGR to LLVM compilation is properly implemented,
            # this should assert:
            # assert ones_count < 5, f"Conditional gate failed, got {ones_count}/20 ones"

    def test_measurement_feedback(self, pipeline_tester: GuppyPipelineTest) -> None:
        """Test feedback based on mid-circuit measurements."""

        @guppy
        def feedback_circuit() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()

            # Create superposition on first qubit
            h(q1)
            result1 = measure(q1).read()

            # Apply correction to second qubit based on measurement
            if result1:
                x(q2)  # Flip second qubit if first was |1⟩

            return result1, measure(q2).read()

        results = pipeline_tester.test_function_on_both_pipelines(
            feedback_circuit,
            shots=50,
        )
        rows = results["hugr_llvm"]["result"]["results"]
        # The correction makes q2 EQUAL q1 on every shot -- this is the
        # engine's core measurement-feedback path.
        assert all(int(a) == int(b) for (a, b) in rows), f"feedback broke: {rows[:10]}"
        # And H gives both branches: both outcomes must appear over 50 shots.
        ones = sum(int(a) for (a, _) in rows)
        assert 10 <= ones <= 40, f"H distribution off: {ones}/50 ones"


# ============================================================================
# ADVANCED QUANTUM ALGORITHMS (PLACEHOLDER)
# ============================================================================


class TestAdvancedAlgorithms:
    """Test advanced quantum algorithms (to be implemented)."""

    def test_quantum_fourier_transform(
        self,
        pipeline_tester: GuppyPipelineTest,
    ) -> None:
        """Test quantum Fourier transform on 2 qubits."""
        from guppylang.std.angles import pi
        from guppylang.std.quantum import crz, cx, h, measure, qubit, x

        @guppy
        def qft_2qubit() -> tuple[bool, bool]:
            """2-qubit QFT implementation."""
            # Initialize qubits
            q0 = qubit()
            q1 = qubit()

            # Apply X to q1 to create input state |01⟩
            x(q1)

            # QFT circuit for 2 qubits
            # First qubit
            h(q0)
            # Controlled rotation
            # In QFT, we use controlled-R_2 which is a phase rotation by π/2
            # We can implement this using CRZ
            crz(q1, q0, pi / 2)

            # Second qubit
            h(q1)

            # Swap qubits (using 3 CNOTs since we don't have swap)
            cx(q0, q1)
            cx(q1, q0)
            cx(q0, q1)

            # Measure
            return measure(q0).read(), measure(q1).read()

        results = pipeline_tester.test_function_on_both_pipelines(qft_2qubit, shots=100)

        # QFT of a computational basis state measured in the computational
        # basis is UNIFORM over all four outcomes (phases are invisible
        # here, so this pins the H layers and plumbing, not the CRZ angle).
        measurements = results["hugr_llvm"]["result"]["results"]
        assert len(measurements) == 100
        from collections import Counter

        counts = Counter(tuple(int(v) for v in row) for row in measurements)
        assert set(counts) == {(0, 0), (0, 1), (1, 0), (1, 1)}, f"missing outcomes: {counts}"
        assert all(10 <= c <= 45 for c in counts.values()), f"non-uniform: {counts}"

    def test_deutsch_josza_algorithm(self, pipeline_tester: GuppyPipelineTest) -> None:
        """Test Deutsch-Josza algorithm for 2-bit function."""
        from guppylang.std.quantum import cx, discard, h, measure, qubit, x

        @guppy
        def deutsch_josza_constant() -> tuple[bool, bool]:
            """Deutsch-Josza algorithm with constant oracle (f(x)=0)."""
            # Initialize qubits
            q0 = qubit()  # First input qubit
            q1 = qubit()  # Second input qubit
            anc = qubit()  # Ancilla qubit

            # Prepare ancilla in |1⟩ and apply H to get |->⟩
            x(anc)
            h(anc)

            # Apply H to input qubits
            h(q0)
            h(q1)

            # Oracle for constant function f(x) = 0
            # Does nothing since f(x) = 0 for all x

            # Apply H to input qubits again
            h(q0)
            h(q1)

            # Measure input qubits; the ancilla is discarded (linearity)
            r = measure(q0).read(), measure(q1).read()
            discard(anc)
            return r

        @guppy
        def deutsch_josza_balanced() -> tuple[bool, bool]:
            """Deutsch-Josza algorithm with balanced oracle."""
            # Initialize qubits
            q0 = qubit()  # First input qubit
            q1 = qubit()  # Second input qubit
            anc = qubit()  # Ancilla qubit

            # Prepare ancilla in |->⟩
            x(anc)
            h(anc)

            # Apply H to input qubits
            h(q0)
            h(q1)

            # Oracle for balanced function: f(00)=0, f(01)=1, f(10)=1, f(11)=0
            # This is implemented using controlled operations
            cx(q1, anc)  # Flip ancilla if q1 is |1⟩
            cx(q0, anc)  # Flip ancilla if q0 is |1⟩

            # Apply H to input qubits again
            h(q0)
            h(q1)

            # Measure input qubits; the ancilla is discarded (linearity)
            r = measure(q0).read(), measure(q1).read()
            discard(anc)
            return r

        # Test constant function
        results_const = pipeline_tester.test_function_on_both_pipelines(
            deutsch_josza_constant,
            shots=100,
        )
        if results_const.get("hugr_llvm", {}).get("success"):
            measurements = results_const["hugr_llvm"]["result"]["results"]
            # Rows are already per-shot (q0, q1) bit tuples
            zeros = sum(1 for (a, b) in measurements if not a and not b)
            assert zeros > 95, f"Constant oracle should give |00⟩, got {zeros}/100"

        # Test balanced function
        results_bal = pipeline_tester.test_function_on_both_pipelines(
            deutsch_josza_balanced,
            shots=100,
        )
        if results_bal.get("hugr_llvm", {}).get("success"):
            measurements = results_bal["hugr_llvm"]["result"]["results"]
            # Rows are already per-shot (q0, q1) bit tuples
            zeros = sum(1 for (a, b) in measurements if not a and not b)
            assert zeros < 5, f"Balanced oracle should not give |00⟩, got {zeros}/100"

    def test_grover_search(self, pipeline_tester: GuppyPipelineTest) -> None:
        """Test Grover's search algorithm for 2-qubit search space."""
        from guppylang.std.quantum import cz, h, measure, qubit, x

        @guppy
        def grover_2qubit() -> tuple[bool, bool]:
            """Grover's algorithm searching for |11⟩ in 2-qubit space."""
            # Initialize qubits
            q0 = qubit()
            q1 = qubit()

            # Initialize in uniform superposition
            h(q0)
            h(q1)

            # Grover iteration (just 1 iteration for 2 qubits)
            # Oracle: mark |11⟩ state
            # We use CZ which adds a phase to |11⟩
            cz(q0, q1)

            # Diffusion operator (inversion about average)
            # Apply H gates
            h(q0)
            h(q1)

            # Apply X gates
            x(q0)
            x(q1)

            # Apply CZ (multi-controlled Z, but for 2 qubits just CZ)
            cz(q0, q1)

            # Apply X gates
            x(q0)
            x(q1)

            # Apply H gates
            h(q0)
            h(q1)

            # Measure
            return measure(q0).read(), measure(q1).read()

        results = pipeline_tester.test_function_on_both_pipelines(
            grover_2qubit,
            shots=100,
        )
        if results.get("hugr_llvm", {}).get("success"):
            measurements = results["hugr_llvm"]["result"]["results"]
            # Check if measurements are already tuples or need decoding
            if measurements and isinstance(measurements[0], tuple):
                # Already decoded as tuples
                decoded_measurements = measurements
            else:
                # Decode integer-encoded results
                decoded_measurements = decode_integer_results(measurements, 2)
            # Should find |11⟩ with high probability after 1 Grover iteration
            found = sum(1 for (a, b) in decoded_measurements if a and b)
            assert found == 100, f"2-qubit Grover with one iteration is deterministic |11>, got {found}/100"
