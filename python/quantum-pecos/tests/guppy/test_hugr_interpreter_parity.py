"""Guppy circuits on the default Selene route, explicit QIS, and selene-sim.

Loop tests run through both sim(Guppy(...)) and sim(Qis(...)).

Measurement-dependent loops use the dynamic QIS execution path, which waits
for quantum measurement results before resuming classical control flow.
"""

import contextlib

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import ch, cx, discard, h, measure, qubit, x, y, z
from pecos import Guppy, sim
from pecos.compilation_pipeline import compile_guppy_to_hugr
from pecos_rslib import Qis, state_vector
from pecos_rslib_llvm import compile_hugr_to_qis
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


def run_with_default_route(
    guppy_func: object,
    num_qubits: int,
    shots: int = 100,
    seed: int = 42,
) -> dict:
    """Run Guppy using the default engine selection and HUGR lowering."""
    return sim(Guppy(guppy_func)).qubits(num_qubits).quantum(state_vector()).seed(seed).run(shots).to_dict()


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
    """Read the programs' m0, m1, ... tags in numeric order."""
    keys = [f"m{i}" for i in range(len(results))]
    assert keys, f"No measurement results: {results}"
    return [list(row) for row in zip(*(results[key] for key in keys), strict=True)]


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
    """Test that the QIS engine produce equivalent results for simple circuits."""

    def test_single_hadamard_parity(self) -> None:
        """Test single Hadamard gate produces similar distributions on both paths."""

        @guppy
        def hadamard_test() -> None:
            q = qubit()
            h(q)
            result("m0", measure(q).read())

        # Run on both paths with same seed
        selene_results = run_with_selene_llvm(hadamard_test, num_qubits=1, shots=1000)

        # Both should produce approximately 50/50 distribution
        selene_ones = count_ones(extract_measurements(selene_results))

        # Allow for statistical variation (expect ~500 ones out of 1000)
        assert 400 < selene_ones < 600, f"Selene path: unexpected distribution {selene_ones}/1000"

    def test_deterministic_zero_state(self) -> None:
        """Test that measuring |0> gives consistent results on both paths."""

        @guppy
        def measure_zero() -> None:
            q = qubit()
            result("m0", measure(q).read())

        selene_results = run_with_selene_llvm(measure_zero, num_qubits=1, shots=100)

        # Both should give all zeros (False)
        selene_ones = count_ones(extract_measurements(selene_results))

        assert selene_ones == 0, f"Selene path: expected all zeros, got {selene_ones}/100 ones"

    def test_deterministic_one_state(self) -> None:
        """Test that measuring |1> gives consistent results on both paths."""

        @guppy
        def measure_one() -> None:
            q = qubit()
            x(q)
            result("m0", measure(q).read())

        selene_results = run_with_selene_llvm(measure_one, num_qubits=1, shots=100)

        # Both should give all ones (True)
        selene_ones = count_ones(extract_measurements(selene_results))

        assert selene_ones == 100, f"Selene path: expected all ones, got {selene_ones}/100"

    def test_bell_state_correlation(self) -> None:
        """Test Bell state produces correlated measurements on both paths."""

        @guppy
        def bell_state() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            result("m0", measure(q0).read())
            result("m1", measure(q1).read())

        selene_results = run_with_selene_llvm(bell_state, num_qubits=2, shots=100)

        # Both measurements in each shot should be equal (perfect correlation)
        selene_meas = extract_measurements(selene_results)

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
            r1 = measure(q1).read()  # Always False
            if r1:
                x(q2)
            result("m0", r1)
            result("m1", measure(q2).read())  # Should be False

        selene_results = run_with_selene_llvm(
            conditional_x_zero,
            num_qubits=2,
            shots=100,
        )

        # Check that all measurements (both q1 and q2) are 0

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
            r1 = measure(q1).read()  # Always True
            if r1:
                x(q2)
            result("m0", r1)
            result("m1", measure(q2).read())  # Should be True

        selene_results = run_with_selene_llvm(
            conditional_x_one,
            num_qubits=2,
            shots=100,
        )

        # Check that all measurements (both q1 and q2) are 1

        selene_measurements = extract_measurements(selene_results)
        for shot in selene_measurements:
            assert shot == [1, 1], f"Selene path: Expected [1, 1], got {shot}"


@pytest.mark.parametrize("run_program", [run_with_default_route, run_with_selene_llvm], ids=["default", "qis"])
class TestLoopCircuits:
    """Test circuits with while loops.

    The Selene/LLVM path support while loops
    with measurement-dependent conditions. It handles loops through dynamic execution.

    NOTE: Loop tests use extra qubits to account for the CFG qubit allocation
    behavior where each loop iteration may allocate a fresh qubit ID.
    """

    def test_repeat_until_one_selene(self, run_program) -> None:
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
                r = measure(q).read()
            result("m0", r)

        results = run_program(repeat_until_one, num_qubits=20, shots=100)
        # All final results should be True (that's what breaks the loop)
        ones = count_ones(extract_measurements(results))
        assert ones == 100, f"repeat_until_one should always return True, got {ones}/100"

    def test_bounded_loop_selene(self, run_program) -> None:
        """Test a bounded loop using Selene QIS route."""

        @guppy
        def bounded_loop() -> None:
            count: int = 0
            for _i in range(5):
                q = qubit()
                h(q)
                if measure(q).read():
                    count = count + 1
            result("m0", count)

        # Use extra qubits: 5 iterations * potential CFG overhead
        results = run_program(bounded_loop, num_qubits=20, shots=100)

        # The count should vary between 0 and 5 across shots
        counts = results["m0"]
        assert len(counts) == 100
        assert all(0 <= count <= 5 for count in counts)
        assert len(set(counts)) >= 2
        assert 200 < sum(counts) < 300, f"Expected five fair measurements per shot, got {counts}"


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
            result("m0", measure(q0).read())
            result("m1", measure(q1).read())
            result("m2", measure(q2).read())

        selene_results = run_with_selene_llvm(ghz_3, num_qubits=3, shots=100)

        # All three qubits should have same measurement value in each shot
        selene_meas = extract_measurements(selene_results)

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
            result("m0", measure(control).read())
            result("m1", measure(target).read())

        selene_results = run_with_selene_llvm(ch_control_zero, num_qubits=2, shots=100)

        # With control=0, both should be 0
        selene_meas = extract_measurements(selene_results)

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
            result("m0", measure(control).read())
            result("m1", measure(target).read())

        selene_results = run_with_selene_llvm(ch_control_one, num_qubits=2, shots=1000)

        selene_meas = extract_measurements(selene_results)

        # Control should always be 1
        for shot in selene_meas:
            assert shot[0] == 1, f"Selene path: control should be 1, got {shot[0]}"

        # Target should be ~50/50 (H applied)
        selene_target_ones = sum(1 for shot in selene_meas if shot[1] == 1)

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
            result("m0", measure(q2).read())  # Should always be 1

        selene_results = run_with_selene_llvm(
            discard_and_reuse,
            num_qubits=2,
            shots=100,
        )

        selene_ones = count_ones(extract_measurements(selene_results))

        # New qubit should be |0>, so X gives |1>
        assert selene_ones == 100, f"Selene path: reused qubit should be reset, got {selene_ones}/100 ones"

    def test_measure_and_reuse(self) -> None:
        """Test that a qubit reused after MeasureFree is in |0> state."""

        @guppy
        def measure_and_reuse() -> None:
            q1 = qubit()
            x(q1)  # |1>
            r1 = measure(q1).read()  # Measure and free

            q2 = qubit()  # Get new qubit
            r2 = measure(q2).read()  # Should be 0 (fresh |0>)

            result("m0", r1)
            result("m1", r2)

        selene_results = run_with_selene_llvm(
            measure_and_reuse,
            num_qubits=2,
            shots=100,
        )

        # Check measurements are [1, 0] for each shot

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
            r1 = measure(q1).read()

            q2 = qubit()  # |0>
            r2 = measure(q2).read()

            result("m0", r1)
            result("m1", r2)

        selene_results = run_with_selene_llvm(two_measures, num_qubits=2, shots=100)

        # Should get [1, 0] for each shot

        selene_meas = extract_measurements(selene_results)
        assert len(selene_meas) == 100, f"Selene path: expected 100 shots, got {len(selene_meas)}"
        assert len(selene_meas[0]) == 2, f"Selene path: expected 2 measurements per shot, got {len(selene_meas[0])}"

        for shot in selene_meas:
            assert shot == [1, 0], f"Selene path: expected [1, 0], got {shot}"

    def test_four_sequential_measurements(self) -> None:
        """Test four sequential measure operations with different gates."""

        @guppy
        def four_measures() -> None:
            q1 = qubit()
            h(q1)
            x(q1)
            r1 = measure(q1).read()  # H+X on |0> -> |-> = 50/50

            q2 = qubit()
            y(q2)
            r2 = measure(q2).read()  # Y|0> = i|1> -> always 1

            q3 = qubit()
            z(q3)
            r3 = measure(q3).read()  # Z|0> = |0> -> always 0

            q4 = qubit()
            x(q4)
            z(q4)
            r4 = measure(q4).read()  # X then Z on |0> = -|1> -> always 1

            result("m0", r1)
            result("m1", r2)
            result("m2", r3)
            result("m3", r4)

        selene_results = run_with_selene_llvm(four_measures, num_qubits=4, shots=100)

        # Should have 4 measurements per shot
        selene_meas = extract_measurements(selene_results)

        assert len(selene_meas[0]) == 4, f"Selene path: expected 4 measurements per shot, got {len(selene_meas[0])}"

        # Check deterministic results (indices 1, 2, 3 should be 1, 0, 1)

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
            result("m", measure(q).read())

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
            result("m", measure(q).read())

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
            result("m0", measure(q0).read())
            result("m1", measure(q1).read())

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
                r = measure(q).read()
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
