"""Test parity between direct HUGR interpreter and Selene/LLVM execution path.

This test suite validates that both HUGR execution paths produce equivalent results
for quantum circuits. The two paths are:

1. Direct HUGR Interpreter (pecos-hugr): Interprets HUGR graphs directly without
   LLVM compilation. This is the reference implementation that handles all HUGR
   features including control flow (while loops, conditionals).

2. Selene/LLVM Path: Compiles HUGR to LLVM IR using Selene's hugr-qis compiler,
   then JIT compiles and executes. This path has a known limitation with loops
   during operation collection mode (see KNOWN_LIMITATIONS below).

KNOWN LIMITATIONS:

1. Selene/LLVM Loop Limitation:
   - Selene execution path does not correctly handle while loops during operation
     collection mode. The `___read_future_bool` FFI function returns `false` by
     default, causing `while not result:` patterns to loop infinitely.
   - For circuits with loops, use the direct HUGR interpreter via `sim(Guppy(...))`.

2. selene-sim Reference Tests:
   - Tests using selene-sim directly as a reference are not yet properly configured
     to capture measurement results. The selene-sim output format may need special
     event hooks or result handling to extract measurements.
"""

import contextlib

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import ch, cx, discard, h, measure, qubit, x, y, z
from pecos import Guppy, sim
from pecos.compilation_pipeline import compile_guppy_to_hugr
from pecos_rslib import Qis, compile_hugr_to_qis, state_vector
from selene_sim import build
from selene_sim.backends import IdealErrorModel as IdealNoiseModel
from selene_sim.backends import Quest, SimpleRuntime


def run_with_selene_reference(
    guppy_func: object,
    num_qubits: int,
    shots: int = 100,
    seed: int = 42,
) -> dict:
    """Run a Guppy function using the true Selene reference implementation.

    This uses selene-sim directly to build and run the program, which is
    the authoritative reference for how Guppy programs should behave.
    """
    hugr_bytes = compile_guppy_to_hugr(guppy_func)
    instance = build(hugr_bytes)

    runtime = SimpleRuntime(random_seed=seed)
    simulator = Quest(random_seed=seed)
    noise_model = IdealNoiseModel()

    results = []
    for shot_results in instance.run_shots(
        simulator=simulator,
        n_qubits=num_qubits,
        runtime=runtime,
        error_model=noise_model,
        n_shots=shots,
        random_seed=seed,
    ):
        shot_data = dict(shot_results)
        results.append(shot_data)

    # Clean up the instance files
    with contextlib.suppress(Exception):
        instance.delete_files()

    return {"shots": results}


def run_with_direct_hugr(
    guppy_func: object,
    num_qubits: int,
    shots: int = 100,
    seed: int = 42,
) -> dict:
    """Run a Guppy function using the direct HUGR interpreter.

    This uses the pecos-hugr Rust crate to directly interpret the HUGR graph
    without going through LLVM compilation.
    """
    results = sim(Guppy(guppy_func)).qubits(num_qubits).quantum(state_vector()).seed(seed).run(shots)
    return results.to_dict()


def run_with_selene_llvm(
    guppy_func: object,
    num_qubits: int,
    shots: int = 100,
    seed: int = 42,
) -> dict:
    """Run a Guppy function using the PECOS Selene/LLVM execution path.

    This compiles the HUGR to LLVM IR using Selene's hugr-qis compiler,
    then executes via PECOS's JIT compilation infrastructure.
    """
    hugr_package = guppy_func.compile()
    hugr_bytes = hugr_package.to_bytes()
    qis_string = compile_hugr_to_qis(hugr_bytes, None)
    qis_program = Qis.from_string(qis_string)
    results = sim(qis_program).qubits(num_qubits).quantum(state_vector()).seed(seed).run(shots)
    return results.to_dict()


def extract_measurements(results: dict) -> list:
    """Extract measurement values from results dictionary.

    Handles multiple formats:
    1. Legacy Direct HUGR: {'measurements': [[1, 0], [1, 0], ...]}
       - Returns list of shots, each shot is a list of measurement values
    2. Selene/LLVM: {'measurement_0': [1, 1, ...], 'measurement_1': [0, 0, ...]}
       - Returns columnar format, transpose to row format
    3. result() format: {'m0': [1, 1, ...], 'm1': [0, 0, ...]}
       - Columnar format from result() calls, transpose to row format
    """
    # Format 1: Legacy Direct HUGR format with "measurements" key
    if "measurements" in results:
        return results["measurements"]

    # Format 2: Selene/LLVM format with measurement_N keys
    measurement_keys = sorted([k for k in results if k.startswith("measurement_")])
    if measurement_keys:
        # Transpose from columnar to row format
        num_shots = len(results[measurement_keys[0]])
        return [[results[key][shot_idx] for key in measurement_keys] for shot_idx in range(num_shots)]

    # Format 3: result() format with m0, m1, etc. or other named keys
    # Find all keys that look like measurement results (exclude metadata)
    result_keys = sorted(
        [k for k in results if k.startswith("m") and k not in ("measurements",)],
    )
    if result_keys:
        # Transpose from columnar to row format
        first_key = result_keys[0]
        if first_key in results and isinstance(results[first_key], list) and len(results[first_key]) > 0:
            num_shots = len(results[first_key])
            return [[int(results[key][shot_idx]) for key in result_keys] for shot_idx in range(num_shots)]

    # Fallback: single measurement register
    for key in sorted(results.keys()):
        if key.startswith(("q", "measurement")):
            values = results[key]
            # Wrap single values in lists if needed
            if values and not isinstance(values[0], list):
                return [[int(v)] for v in values]
            return values
    return []


def extract_selene_measurements(results: dict) -> list:
    """Extract measurement values from selene-sim results format.

    When Guppy programs use result("name", value), selene-sim returns:
    {"shots": [{"name": value}, {"name": value}, ...]}

    This function extracts all result values from each shot.
    """
    if "shots" not in results:
        return []

    measurements = []
    for shot in results["shots"]:
        # Each shot is a dict with result names as keys
        shot_measurements = []
        for key in sorted(shot.keys()):
            value = shot[key]
            if isinstance(value, (list, tuple)):
                shot_measurements.extend(int(v) for v in value)
            else:
                shot_measurements.append(int(value))
        measurements.append(shot_measurements)
    return measurements


def count_ones(measurements: list) -> int:
    """Count the number of ones/True values in measurements."""
    if not measurements:
        return 0
    if isinstance(measurements[0], list):
        # Flatten nested measurements
        return sum(sum(1 for v in m if v) for m in measurements)
    return sum(1 for m in measurements if m)


class TestSimpleCircuitParity:
    """Test that both interpreters produce equivalent results for simple circuits."""

    def test_single_hadamard_parity(self) -> None:
        """Test single Hadamard gate produces similar distributions on both paths."""

        @guppy
        def hadamard_test() -> None:
            q = qubit()
            h(q)
            result("m0", measure(q))

        # Run on both paths with same seed
        direct_results = run_with_direct_hugr(hadamard_test, num_qubits=1, shots=1000)
        selene_results = run_with_selene_llvm(hadamard_test, num_qubits=1, shots=1000)

        # Both should produce approximately 50/50 distribution
        direct_ones = count_ones(extract_measurements(direct_results))
        selene_ones = count_ones(extract_measurements(selene_results))

        # Allow for statistical variation (expect ~500 ones out of 1000)
        assert 400 < direct_ones < 600, f"Direct path: unexpected distribution {direct_ones}/1000"
        assert 400 < selene_ones < 600, f"Selene path: unexpected distribution {selene_ones}/1000"

    def test_deterministic_zero_state(self) -> None:
        """Test that measuring |0> gives consistent results on both paths."""

        @guppy
        def measure_zero() -> None:
            q = qubit()
            result("m0", measure(q))

        direct_results = run_with_direct_hugr(measure_zero, num_qubits=1, shots=100)
        selene_results = run_with_selene_llvm(measure_zero, num_qubits=1, shots=100)

        # Both should give all zeros (False)
        direct_ones = count_ones(extract_measurements(direct_results))
        selene_ones = count_ones(extract_measurements(selene_results))

        assert direct_ones == 0, f"Direct path: expected all zeros, got {direct_ones}/100 ones"
        assert selene_ones == 0, f"Selene path: expected all zeros, got {selene_ones}/100 ones"

    def test_deterministic_one_state(self) -> None:
        """Test that measuring |1> gives consistent results on both paths."""

        @guppy
        def measure_one() -> None:
            q = qubit()
            x(q)
            result("m0", measure(q))

        direct_results = run_with_direct_hugr(measure_one, num_qubits=1, shots=100)
        selene_results = run_with_selene_llvm(measure_one, num_qubits=1, shots=100)

        # Both should give all ones (True)
        direct_ones = count_ones(extract_measurements(direct_results))
        selene_ones = count_ones(extract_measurements(selene_results))

        assert direct_ones == 100, f"Direct path: expected all ones, got {direct_ones}/100"
        assert selene_ones == 100, f"Selene path: expected all ones, got {selene_ones}/100"

    def test_bell_state_correlation(self) -> None:
        """Test Bell state produces correlated measurements on both paths."""

        @guppy
        def bell_state() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            result("m0", measure(q0))
            result("m1", measure(q1))

        direct_results = run_with_direct_hugr(bell_state, num_qubits=2, shots=100)
        selene_results = run_with_selene_llvm(bell_state, num_qubits=2, shots=100)

        # Both measurements in each shot should be equal (perfect correlation)
        direct_meas = extract_measurements(direct_results)
        selene_meas = extract_measurements(selene_results)

        if direct_meas:
            direct_mismatches = sum(1 for m in direct_meas if len(m) >= 2 and m[0] != m[1])
            assert direct_mismatches == 0, "Direct path: Bell state correlation broken"

        if selene_meas:
            selene_mismatches = sum(1 for m in selene_meas if len(m) >= 2 and m[0] != m[1])
            assert selene_mismatches == 0, "Selene path: Bell state correlation broken"


class TestConditionalCircuitParity:
    """Test circuits with conditionals (mid-circuit measurement and feedforward)."""

    def test_conditional_x_from_zero(self) -> None:
        """Test conditional X gate when control qubit is |0>."""

        @guppy
        def conditional_x_zero() -> None:
            q1 = qubit()
            q2 = qubit()
            r1 = measure(q1)  # Always False
            if r1:
                x(q2)
            result("m0", r1)
            result("m1", measure(q2))  # Should be False

        direct_results = run_with_direct_hugr(
            conditional_x_zero,
            num_qubits=2,
            shots=100,
        )
        selene_results = run_with_selene_llvm(
            conditional_x_zero,
            num_qubits=2,
            shots=100,
        )

        # Check that all measurements (both q1 and q2) are 0
        # Direct path: q1=0 (no gate), q2=0 (conditional X not triggered) = 2 zeros per shot
        direct_measurements = extract_measurements(direct_results)
        for shot in direct_measurements:
            assert shot == [0, 0], f"Expected [0, 0], got {shot}"

        selene_measurements = extract_measurements(selene_results)
        for shot in selene_measurements:
            assert shot == [0, 0], f"Selene path: Expected [0, 0], got {shot}"

    def test_conditional_x_from_one(self) -> None:
        """Test conditional X gate when control qubit is |1>."""

        @guppy
        def conditional_x_one() -> None:
            q1 = qubit()
            q2 = qubit()
            x(q1)
            r1 = measure(q1)  # Always True
            if r1:
                x(q2)
            result("m0", r1)
            result("m1", measure(q2))  # Should be True

        direct_results = run_with_direct_hugr(
            conditional_x_one,
            num_qubits=2,
            shots=100,
        )
        selene_results = run_with_selene_llvm(
            conditional_x_one,
            num_qubits=2,
            shots=100,
        )

        # Check that all measurements (both q1 and q2) are 1
        # Direct path: q1=1 (from X), q2=1 (from conditional X) = 2 ones per shot
        direct_measurements = extract_measurements(direct_results)
        for shot in direct_measurements:
            assert shot == [1, 1], f"Expected [1, 1], got {shot}"

        selene_measurements = extract_measurements(selene_results)
        for shot in selene_measurements:
            assert shot == [1, 1], f"Selene path: Expected [1, 1], got {shot}"


class TestLoopCircuits:
    """Test circuits with while loops.

    Both the direct HUGR interpreter and Selene/LLVM path support while loops
    with measurement-dependent conditions. Both correctly handle loops by
    interpreting the CFG (Control Flow Graph) nodes directly.

    NOTE: Loop tests use extra qubits to account for the CFG qubit allocation
    behavior where each loop iteration may allocate a fresh qubit ID.
    """

    def test_repeat_until_one_direct_hugr(self) -> None:
        """Test repeat-until-one pattern using direct HUGR interpreter.

        This tests a loop that repeats until a measurement returns 1.
        The loop should always exit with a final measurement of 1.
        """

        @guppy
        def repeat_until_one() -> None:
            r: bool = False
            while not r:
                q = qubit()
                h(q)
                r = measure(q)
            result("m0", r)

        # Use more qubits since each loop iteration allocates a new qubit
        # Average iterations is 2 (geometric distribution), use 20 for safety
        results = run_with_direct_hugr(repeat_until_one, num_qubits=20, shots=100)

        # All final results should be True (that's what breaks the loop)
        ones = count_ones(extract_measurements(results))
        assert ones == 100, f"repeat_until_one should always return True, got {ones}/100"

    def test_repeat_until_one_selene(self) -> None:
        """Test repeat-until-one pattern using Selene/LLVM path.

        Tests that the Selene/LLVM path correctly handles while loops
        with measurement-dependent conditions via dynamic execution mode.
        """

        @guppy
        def repeat_until_one() -> None:
            r: bool = False
            while not r:
                q = qubit()
                h(q)
                r = measure(q)
            result("m0", r)

        results = run_with_selene_llvm(repeat_until_one, num_qubits=20, shots=100)
        # All final results should be True (that's what breaks the loop)
        ones = count_ones(extract_measurements(results))
        assert ones == 100, f"repeat_until_one should always return True, got {ones}/100"

    def test_bounded_loop_direct_hugr(self) -> None:
        """Test a bounded loop using direct HUGR interpreter."""

        @guppy
        def bounded_loop() -> None:
            count: int = 0
            for _i in range(5):
                q = qubit()
                h(q)
                if measure(q):
                    count = count + 1
            result("m0", count)

        # Use extra qubits: 5 iterations * potential CFG overhead
        results = run_with_direct_hugr(bounded_loop, num_qubits=20, shots=100)

        # The count should vary between 0 and 5 across shots
        # We just verify it runs without hanging
        assert results is not None


class TestGHZStates:
    """Test GHZ state preparation on both paths."""

    def test_ghz_3_qubit(self) -> None:
        """Test 3-qubit GHZ state produces expected correlations."""

        @guppy
        def ghz_3() -> None:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)
            cx(q0, q1)
            cx(q1, q2)
            result("m0", measure(q0))
            result("m1", measure(q1))
            result("m2", measure(q2))

        direct_results = run_with_direct_hugr(ghz_3, num_qubits=3, shots=100)
        selene_results = run_with_selene_llvm(ghz_3, num_qubits=3, shots=100)

        # All three qubits should have same measurement value in each shot
        direct_meas = extract_measurements(direct_results)
        selene_meas = extract_measurements(selene_results)

        if direct_meas:
            for m in direct_meas:
                if len(m) >= 3:
                    assert m[0] == m[1] == m[2], f"Direct: GHZ correlation broken: {m}"

        if selene_meas:
            for m in selene_meas:
                if len(m) >= 3:
                    assert m[0] == m[1] == m[2], f"Selene: GHZ correlation broken: {m}"


class TestControlledGatesParity:
    """Test controlled gates produce equivalent results on both paths.

    These tests specifically cover edge cases in gate decomposition,
    including the CH gate which decomposes to Ry(-pi/4) + CZ + Ry(pi/4).
    """

    def test_ch_control_zero(self) -> None:
        """Test CH gate with control=0 (should not apply H to target)."""

        @guppy
        def ch_control_zero() -> None:
            control = qubit()  # |0>
            target = qubit()  # |0>
            ch(control, target)  # CH: target unchanged when control=0
            result("m0", measure(control))
            result("m1", measure(target))

        direct_results = run_with_direct_hugr(ch_control_zero, num_qubits=2, shots=100)
        selene_results = run_with_selene_llvm(ch_control_zero, num_qubits=2, shots=100)

        # With control=0, both should be 0
        direct_meas = extract_measurements(direct_results)
        selene_meas = extract_measurements(selene_results)

        for shot in direct_meas:
            assert shot == [
                0,
                0,
            ], f"Direct path: CH control=0 should give [0, 0], got {shot}"
        for shot in selene_meas:
            assert shot == [
                0,
                0,
            ], f"Selene path: CH control=0 should give [0, 0], got {shot}"

    def test_ch_control_one(self) -> None:
        """Test CH gate with control=1 (should apply H to target)."""

        @guppy
        def ch_control_one() -> None:
            control = qubit()
            x(control)  # |1>
            target = qubit()  # |0>
            ch(control, target)  # CH: H applied to target when control=1
            result("m0", measure(control))
            result("m1", measure(target))

        direct_results = run_with_direct_hugr(ch_control_one, num_qubits=2, shots=1000)
        selene_results = run_with_selene_llvm(ch_control_one, num_qubits=2, shots=1000)

        direct_meas = extract_measurements(direct_results)
        selene_meas = extract_measurements(selene_results)

        # Control should always be 1
        for shot in direct_meas:
            assert shot[0] == 1, f"Direct path: control should be 1, got {shot[0]}"
        for shot in selene_meas:
            assert shot[0] == 1, f"Selene path: control should be 1, got {shot[0]}"

        # Target should be ~50/50 (H applied)
        direct_target_ones = sum(1 for shot in direct_meas if shot[1] == 1)
        selene_target_ones = sum(1 for shot in selene_meas if shot[1] == 1)

        assert (
            400 < direct_target_ones < 600
        ), f"Direct path: CH control=1 target should be ~50%, got {direct_target_ones}/1000"
        assert (
            400 < selene_target_ones < 600
        ), f"Selene path: CH control=1 target should be ~50%, got {selene_target_ones}/1000"


class TestQubitReuseParity:
    """Test qubit reuse after discard/measure produces equivalent results.

    These tests verify that when a qubit is freed and reallocated,
    the new qubit is properly initialized to |0>.
    """

    def test_discard_and_reuse(self) -> None:
        """Test that a qubit reused after discard is in |0> state."""

        @guppy
        def discard_and_reuse() -> None:
            q1 = qubit()
            h(q1)  # Put in superposition
            discard(q1)  # Discard (may be reused)
            q2 = qubit()  # Get new qubit (might be same physical qubit)
            x(q2)  # X on |0> should give |1>
            result("m0", measure(q2))  # Should always be 1

        direct_results = run_with_direct_hugr(
            discard_and_reuse,
            num_qubits=2,
            shots=100,
        )
        selene_results = run_with_selene_llvm(
            discard_and_reuse,
            num_qubits=2,
            shots=100,
        )

        direct_ones = count_ones(extract_measurements(direct_results))
        selene_ones = count_ones(extract_measurements(selene_results))

        # New qubit should be |0>, so X gives |1>
        assert direct_ones == 100, f"Direct path: reused qubit should be reset, got {direct_ones}/100 ones"
        assert selene_ones == 100, f"Selene path: reused qubit should be reset, got {selene_ones}/100 ones"

    def test_measure_and_reuse(self) -> None:
        """Test that a qubit reused after MeasureFree is in |0> state."""

        @guppy
        def measure_and_reuse() -> None:
            q1 = qubit()
            x(q1)  # |1>
            r1 = measure(q1)  # Measure and free

            q2 = qubit()  # Get new qubit
            r2 = measure(q2)  # Should be 0 (fresh |0>)

            result("m0", r1)
            result("m1", r2)

        direct_results = run_with_direct_hugr(
            measure_and_reuse,
            num_qubits=2,
            shots=100,
        )
        selene_results = run_with_selene_llvm(
            measure_and_reuse,
            num_qubits=2,
            shots=100,
        )

        # Check measurements are [1, 0] for each shot
        direct_meas = extract_measurements(direct_results)
        for shot in direct_meas:
            assert shot == [1, 0], f"Direct path: expected [1, 0], got {shot}"

        selene_meas = extract_measurements(selene_results)
        for shot in selene_meas:
            assert shot == [1, 0], f"Selene path: expected [1, 0], got {shot}"


class TestSequentialMeasurementsParity:
    """Test sequential qubit operations with multiple intermediate measurements.

    These tests verify that multiple measurements are captured correctly,
    especially when qubits are reused between measurements.
    """

    def test_two_sequential_measurements(self) -> None:
        """Test two sequential measure operations."""

        @guppy
        def two_measures() -> None:
            q1 = qubit()
            x(q1)  # |1>
            r1 = measure(q1)

            q2 = qubit()  # |0>
            r2 = measure(q2)

            result("m0", r1)
            result("m1", r2)

        direct_results = run_with_direct_hugr(two_measures, num_qubits=2, shots=100)
        selene_results = run_with_selene_llvm(two_measures, num_qubits=2, shots=100)

        # Should get [1, 0] for each shot
        direct_meas = extract_measurements(direct_results)
        assert len(direct_meas) == 100, f"Direct path: expected 100 shots, got {len(direct_meas)}"
        assert len(direct_meas[0]) == 2, f"Direct path: expected 2 measurements per shot, got {len(direct_meas[0])}"

        selene_meas = extract_measurements(selene_results)
        assert len(selene_meas) == 100, f"Selene path: expected 100 shots, got {len(selene_meas)}"
        assert len(selene_meas[0]) == 2, f"Selene path: expected 2 measurements per shot, got {len(selene_meas[0])}"

        for shot in direct_meas:
            assert shot == [1, 0], f"Direct path: expected [1, 0], got {shot}"
        for shot in selene_meas:
            assert shot == [1, 0], f"Selene path: expected [1, 0], got {shot}"

    def test_four_sequential_measurements(self) -> None:
        """Test four sequential measure operations with different gates."""

        @guppy
        def four_measures() -> None:
            q1 = qubit()
            h(q1)
            x(q1)
            r1 = measure(q1)  # H+X on |0> -> |-> = 50/50

            q2 = qubit()
            y(q2)
            r2 = measure(q2)  # Y|0> = i|1> -> always 1

            q3 = qubit()
            z(q3)
            r3 = measure(q3)  # Z|0> = |0> -> always 0

            q4 = qubit()
            x(q4)
            z(q4)
            r4 = measure(q4)  # X then Z on |0> = -|1> -> always 1

            result("m0", r1)
            result("m1", r2)
            result("m2", r3)
            result("m3", r4)

        direct_results = run_with_direct_hugr(four_measures, num_qubits=4, shots=100)
        selene_results = run_with_selene_llvm(four_measures, num_qubits=4, shots=100)

        # Should have 4 measurements per shot
        direct_meas = extract_measurements(direct_results)
        selene_meas = extract_measurements(selene_results)

        assert len(direct_meas[0]) == 4, f"Direct path: expected 4 measurements per shot, got {len(direct_meas[0])}"
        assert len(selene_meas[0]) == 4, f"Selene path: expected 4 measurements per shot, got {len(selene_meas[0])}"

        # Check deterministic results (indices 1, 2, 3 should be 1, 0, 1)
        for shot in direct_meas:
            assert shot[1] == 1, f"Direct path: Y|0> should give 1, got {shot[1]}"
            assert shot[2] == 0, f"Direct path: Z|0> should give 0, got {shot[2]}"
            assert shot[3] == 1, f"Direct path: XZ|0> should give 1, got {shot[3]}"

        for shot in selene_meas:
            assert shot[1] == 1, f"Selene path: Y|0> should give 1, got {shot[1]}"
            assert shot[2] == 0, f"Selene path: Z|0> should give 0, got {shot[2]}"
            assert shot[3] == 1, f"Selene path: XZ|0> should give 1, got {shot[3]}"


class TestSeleneReferenceValidation:
    """Validate PECOS implementations against the true Selene reference (selene-sim).

    These tests use selene-sim directly to establish ground truth for how Guppy
    programs should behave, then compare our PECOS implementations against it.

    NOTE: To get results from selene-sim, programs must use the result() function
    from guppylang.std.builtins to output values to the result stream. Return values
    are not automatically captured by selene-sim.
    """

    def test_simple_hadamard_against_reference(self) -> None:
        """Test that simple Hadamard produces expected distribution per Selene reference."""

        @guppy
        def hadamard_test() -> None:
            q = qubit()
            h(q)
            result("m", measure(q))

        # Run on true Selene reference
        reference_results = run_with_selene_reference(
            hadamard_test,
            num_qubits=1,
            shots=1000,
        )
        reference_meas = extract_selene_measurements(reference_results)
        reference_ones = count_ones(reference_meas)

        # Verify reference produces expected distribution (~50%)
        assert 400 < reference_ones < 600, f"Selene reference: unexpected {reference_ones}/1000"

    def test_deterministic_x_gate_against_reference(self) -> None:
        """Test X gate produces deterministic |1> per Selene reference."""

        @guppy
        def x_gate_test() -> None:
            q = qubit()
            x(q)
            result("m", measure(q))

        # Run on true Selene reference
        reference_results = run_with_selene_reference(
            x_gate_test,
            num_qubits=1,
            shots=100,
        )
        reference_meas = extract_selene_measurements(reference_results)
        reference_ones = count_ones(reference_meas)

        # Should be all ones
        assert reference_ones == 100, f"Selene reference: X gate should give all ones, got {reference_ones}/100"

    def test_bell_state_against_reference(self) -> None:
        """Test Bell state produces correlated measurements per Selene reference."""

        @guppy
        def bell_test() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            result("m0", measure(q0))
            result("m1", measure(q1))

        # Run on true Selene reference
        reference_results = run_with_selene_reference(
            bell_test,
            num_qubits=2,
            shots=100,
        )
        reference_meas = extract_selene_measurements(reference_results)

        # Bell state should have correlated measurements (both 0 or both 1)
        for shot in reference_meas:
            assert len(shot) == 2, f"Expected 2 measurements per shot, got {len(shot)}"
            assert shot[0] == shot[1], f"Bell state correlation broken: {shot}"

    def test_while_loop_against_reference(self) -> None:
        """Test while loop works correctly per Selene reference."""

        @guppy
        def repeat_until_one() -> None:
            r: bool = False
            while not r:
                q = qubit()
                h(q)
                r = measure(q)
            result("final", r)

        # Run on true Selene reference - it should handle loops correctly
        reference_results = run_with_selene_reference(
            repeat_until_one,
            num_qubits=20,
            shots=100,
        )
        reference_meas = extract_selene_measurements(reference_results)
        reference_ones = count_ones(reference_meas)

        # Should always be True (that's what breaks the loop)
        assert reference_ones == 100, f"Selene reference: expected all ones, got {reference_ones}/100"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
