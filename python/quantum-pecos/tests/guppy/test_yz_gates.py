"""Test Y and Z gates specifically."""

from guppylang import guppy
from guppylang.std.quantum import measure, qubit, x, y, z
from pecos import Guppy, sim
from pecos_rslib import state_vector


def test_y_gate_only() -> None:
    """Test Y gate by itself."""

    @guppy
    def y_only() -> bool:
        q = qubit()
        y(q)
        return measure(q)

    results = sim(Guppy(y_only)).qubits(1).quantum(state_vector()).run(5)
    measurements = results.get("measurements", results.get("measurement_0", []))
    assert all(val == 1 for val in measurements)  # Y|0⟩ should give |1⟩


def test_z_gate_only() -> None:
    """Test Z gate by itself."""

    @guppy
    def z_only() -> bool:
        q = qubit()
        z(q)
        return measure(q)

    results = sim(Guppy(z_only)).qubits(1).quantum(state_vector()).run(5)
    measurements = results.get("measurements", results.get("measurement_0", []))
    assert all(val == 0 for val in measurements)  # Z|0⟩ should give |0⟩


def test_y_and_z_tuple() -> None:
    """Test Y and Z gates returning a tuple."""

    @guppy
    def yz_tuple() -> tuple[bool, bool]:
        q1 = qubit()
        y(q1)  # Y|0⟩ = i|1⟩
        r1 = measure(q1)

        q2 = qubit()
        z(q2)  # Z|0⟩ = |0⟩
        r2 = measure(q2)

        return r1, r2

    results = sim(Guppy(yz_tuple)).qubits(2).quantum(state_vector()).run(5)
    m1 = results.get("measurement_0", [])
    m2 = results.get("measurement_1", [])

    for i in range(5):
        assert m1[i] == 1  # Y|0⟩ should give |1⟩
        assert m2[i] == 0  # Z|0⟩ should give |0⟩


def test_xyz_tuple() -> None:
    """Test X, Y, Z gates returning a tuple."""

    @guppy
    def xyz_tuple() -> tuple[bool, bool, bool]:
        q1 = qubit()
        x(q1)  # X|0⟩ = |1⟩
        r1 = measure(q1)

        q2 = qubit()
        y(q2)  # Y|0⟩ = i|1⟩
        r2 = measure(q2)

        q3 = qubit()
        z(q3)  # Z|0⟩ = |0⟩
        r3 = measure(q3)

        return r1, r2, r3

    results = sim(Guppy(xyz_tuple)).qubits(3).quantum(state_vector()).run(5)
    m1 = results.get("measurement_0", [])
    m2 = results.get("measurement_1", [])
    m3 = results.get("measurement_2", [])

    for i in range(5):
        assert m1[i] == 1  # X|0⟩ should give |1⟩
        assert m2[i] == 1  # Y|0⟩ should give |1⟩
        assert m3[i] == 0  # Z|0⟩ should give |0⟩
