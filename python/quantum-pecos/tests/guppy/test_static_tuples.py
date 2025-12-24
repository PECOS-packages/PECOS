"""Test different tuple sizes with static functions."""

from guppylang import guppy
from guppylang.std.quantum import measure, qubit, x
from pecos import Guppy, sim
from pecos_rslib import state_vector


@guppy
def circuit_1_tuple() -> bool:
    """Test circuit returning a single boolean."""
    q = qubit()
    x(q)
    return measure(q)


@guppy
def circuit_2_tuple() -> tuple[bool, bool]:
    """Test circuit returning a 2-tuple."""
    q1 = qubit()
    x(q1)
    r1 = measure(q1)

    q2 = qubit()
    r2 = measure(q2)

    return r1, r2


@guppy
def circuit_3_tuple() -> tuple[bool, bool, bool]:
    """Test circuit returning a 3-tuple."""
    q1 = qubit()
    x(q1)
    r1 = measure(q1)

    q2 = qubit()
    r2 = measure(q2)

    q3 = qubit()
    x(q3)
    r3 = measure(q3)

    return r1, r2, r3


@guppy
def circuit_4_tuple() -> tuple[bool, bool, bool, bool]:
    """Test circuit returning a 4-tuple."""
    q1 = qubit()
    x(q1)
    r1 = measure(q1)

    q2 = qubit()
    r2 = measure(q2)

    q3 = qubit()
    x(q3)
    r3 = measure(q3)

    q4 = qubit()
    r4 = measure(q4)

    return r1, r2, r3, r4


@guppy
def circuit_5_tuple() -> tuple[bool, bool, bool, bool, bool]:
    """Test circuit returning a 5-tuple."""
    q1 = qubit()
    x(q1)
    r1 = measure(q1)

    q2 = qubit()
    r2 = measure(q2)

    q3 = qubit()
    x(q3)
    r3 = measure(q3)

    q4 = qubit()
    r4 = measure(q4)

    q5 = qubit()
    x(q5)
    r5 = measure(q5)

    return r1, r2, r3, r4, r5


def test_1_tuple_return() -> None:
    """Test that 1-tuple (bool) returns work correctly."""
    results = sim(Guppy(circuit_1_tuple)).qubits(1).quantum(state_vector()).run(5)
    assert "measurement_0" in results
    measurements = results["measurement_0"]
    assert len(measurements) == 5
    assert all(m == 1 for m in measurements)  # X gate applied


def test_2_tuple_return() -> None:
    """Test that 2-tuple returns work correctly."""
    results = sim(Guppy(circuit_2_tuple)).qubits(2).quantum(state_vector()).run(5)
    assert "measurement_0" in results
    assert "measurement_1" in results
    # First qubit has X, second doesn't
    assert all(results["measurement_0"][i] == 1 for i in range(5))
    assert all(results["measurement_1"][i] == 0 for i in range(5))


def test_3_tuple_return() -> None:
    """Test that 3-tuple returns work correctly."""
    results = sim(Guppy(circuit_3_tuple)).qubits(3).quantum(state_vector()).run(5)
    assert "measurement_0" in results
    assert "measurement_1" in results
    assert "measurement_2" in results
    # Pattern: X, no X, X
    assert all(results["measurement_0"][i] == 1 for i in range(5))
    assert all(results["measurement_1"][i] == 0 for i in range(5))
    assert all(results["measurement_2"][i] == 1 for i in range(5))


def test_4_tuple_return() -> None:
    """Test that 4-tuple returns work correctly."""
    results = sim(Guppy(circuit_4_tuple)).qubits(4).quantum(state_vector()).run(5)
    assert "measurement_0" in results
    assert "measurement_1" in results
    assert "measurement_2" in results
    assert "measurement_3" in results
    # Pattern: X, no X, X, no X
    assert all(results["measurement_0"][i] == 1 for i in range(5))
    assert all(results["measurement_1"][i] == 0 for i in range(5))
    assert all(results["measurement_2"][i] == 1 for i in range(5))
    assert all(results["measurement_3"][i] == 0 for i in range(5))


def test_5_tuple_return() -> None:
    """Test that 5-tuple returns work correctly."""
    results = sim(Guppy(circuit_5_tuple)).qubits(5).quantum(state_vector()).run(5)
    assert "measurement_0" in results
    assert "measurement_1" in results
    assert "measurement_2" in results
    assert "measurement_3" in results
    assert "measurement_4" in results
    # Pattern: X, no X, X, no X, X
    assert all(results["measurement_0"][i] == 1 for i in range(5))
    assert all(results["measurement_1"][i] == 0 for i in range(5))
    assert all(results["measurement_2"][i] == 1 for i in range(5))
    assert all(results["measurement_3"][i] == 0 for i in range(5))
    assert all(results["measurement_4"][i] == 1 for i in range(5))
