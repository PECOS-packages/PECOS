"""Test arithmetic and boolean type support in Guppy->Selene pipeline."""

from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit
from pecos import Guppy, sim
from pecos_rslib import state_vector


def test_integer_arithmetic() -> None:
    """Test integer arithmetic operations."""

    @guppy
    def quantum_add() -> bool:
        q = qubit()
        x = 3
        y = 2
        result = x + y  # result = 5

        if result > 3:  # 5 > 3, so H gate applied
            h(q)

        return measure(q)

    import logging

    logging.basicConfig(level=logging.INFO)

    sim_builder = sim(Guppy(quantum_add)).qubits(1).quantum(state_vector()).seed(42)
    print(f"SimBuilder type: {type(sim_builder)}")

    results = sim_builder.run(10)
    print(f"Results: {results}")
    print(f"Results type: {type(results)}")

    if hasattr(results, "to_binary_dict"):
        binary_dict = results.to_binary_dict()
        print(f"Binary dict: {binary_dict}")
        results = binary_dict

    print(f"Final results: {results}")

    assert "measurement_0" in results
    measurements = results["measurement_0"]
    assert len(measurements) == 10
    # H gate should give mix of 0s and 1s
    assert 0 in measurements
    assert 1 in measurements


def test_boolean_operations() -> None:
    """Test boolean logic operations."""

    @guppy
    def quantum_bool_logic() -> bool:
        q1 = qubit()
        q2 = qubit()
        h(q1)
        m1 = measure(q1)
        m2 = measure(q2)
        return m1 and not m2

    results = (
        sim(Guppy(quantum_bool_logic))
        .qubits(2)
        .quantum(state_vector())
        .seed(42)
        .run(10)
    )

    assert "measurement_0" in results
    assert len(results["measurement_0"]) == 10


def test_integer_comparisons() -> None:
    """Test integer comparison operations."""

    @guppy
    def quantum_compare() -> bool:
        q = qubit()
        threshold = 42
        value = 50

        if value > threshold:
            h(q)

        return measure(q)

    results = (
        sim(Guppy(quantum_compare)).qubits(1).quantum(state_vector()).seed(42).run(10)
    )

    assert "measurement_0" in results
    measurements = results["measurement_0"]
    assert len(measurements) == 10
    assert 0 in measurements
    assert 1 in measurements


def test_arithmetic_in_loop() -> None:
    """Test arithmetic in loop control."""

    @guppy
    def quantum_loop() -> bool:
        q = qubit()
        count = 0
        max_count = 3

        while count < max_count:
            if count == 1:  # Only apply H on second iteration
                h(q)
            count = count + 1

        return measure(q)

    results = (
        sim(Guppy(quantum_loop)).qubits(1).quantum(state_vector()).seed(42).run(10)
    )

    assert "measurement_0" in results
    measurements = results["measurement_0"]
    assert len(measurements) == 10
    assert 0 in measurements
    assert 1 in measurements


def test_chained_comparisons() -> None:
    """Test multiple chained comparisons."""

    @guppy
    def quantum_chain() -> bool:
        q = qubit()
        a = 10
        b = 20
        c = 15

        if a < c and c < b:  # 10 < 15 < 20 is True
            h(q)

        return measure(q)

    results = (
        sim(Guppy(quantum_chain)).qubits(1).quantum(state_vector()).seed(42).run(10)
    )

    assert "measurement_0" in results
    measurements = results["measurement_0"]
    assert len(measurements) == 10
    assert 0 in measurements
    assert 1 in measurements


def test_arithmetic_with_measurements() -> None:
    """Test using measurement results in arithmetic."""

    @guppy
    def quantum_measure_math() -> bool:
        q1 = qubit()
        q2 = qubit()
        h(q1)
        h(q2)

        m1 = measure(q1)
        m2 = measure(q2)

        # Use measurements in arithmetic (bools as ints)
        q3 = qubit()
        if m1 or m2:  # At least one is True
            h(q3)

        return measure(q3)

    results = (
        sim(Guppy(quantum_measure_math))
        .qubits(3)
        .quantum(state_vector())
        .seed(42)
        .run(20)
    )

    assert "measurement_0" in results
    measurements = results["measurement_0"]
    assert len(measurements) == 20
    # Should have mix unless both m1 and m2 are 0 (25% chance)
