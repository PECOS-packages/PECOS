"""Test real quantum circuits through the Guppy->HUGR->Selene->ByteMessage pipeline."""

import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.quantum import cx, h, measure, qubit, ry, rz, x, z
from pecos import Guppy, sim
from pecos_rslib import state_vector

pytestmark = pytest.mark.optional_dependency


def test_bell_state_preparation() -> None:
    """Test Bell state preparation and measurement."""

    @guppy
    def prepare_bell_state() -> tuple[bool, bool]:
        """Prepare a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        q1 = qubit()
        q2 = qubit()

        # Create Bell state
        h(q1)
        cx(q1, q2)

        # Measure both qubits
        m1 = measure(q1)
        m2 = measure(q2)

        return (m1, m2)

    # Run simulation with state_vector backend
    # Use seed for reproducibility
    shot_vec = sim(Guppy(prepare_bell_state)).qubits(2).quantum(state_vector()).seed(42).run(1000)
    assert shot_vec is not None, "Should get results"
    # "measurements" holds one row per shot, ordered like the returned tuple.
    shots = shot_vec.to_dict()["measurements"]
    assert len(shots) == 1000, "Should have one measurement row per shot"

    both_zero = sum(1 for m1, m2 in shots if (m1, m2) == (0, 0))
    both_one = sum(1 for m1, m2 in shots if (m1, m2) == (1, 1))
    anti_correlated = len(shots) - both_zero - both_one

    # Bell state should only produce correlated outcomes
    assert anti_correlated == 0, f"Bell state should not produce anti-correlated outcomes, got {anti_correlated}"
    assert both_zero > 0, "Should see |00⟩ outcomes"
    assert both_one > 0, "Should see |11⟩ outcomes"

    # Should be roughly 50/50 split
    total = both_zero + both_one
    assert 0.4 < both_zero / total < 0.6, f"Should be ~50% |00⟩, got {both_zero / total}"
    assert 0.4 < both_one / total < 0.6, f"Should be ~50% |11⟩, got {both_one / total}"


def test_ghz_state() -> None:
    """Test 3-qubit GHZ state preparation."""

    @guppy
    def prepare_ghz_state() -> tuple[bool, bool, bool]:
        """Prepare a GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2."""
        q1 = qubit()
        q2 = qubit()
        q3 = qubit()

        # Create GHZ state
        h(q1)
        cx(q1, q2)
        cx(q1, q3)

        # Measure all qubits
        m1 = measure(q1)
        m2 = measure(q2)
        m3 = measure(q3)

        return (m1, m2, m3)

    # Run simulation with state_vector backend
    shot_vec = sim(Guppy(prepare_ghz_state)).qubits(3).quantum(state_vector()).seed(42).run(1000)
    assert shot_vec is not None, "Should get results"
    # "measurements" holds one row per shot, ordered like the returned tuple.
    shots = shot_vec.to_dict()["measurements"]
    assert len(shots) == 1000, "Should have one measurement row per shot"

    # GHZ state should give either all 0s or all 1s
    all_zero = sum(1 for m1, m2, m3 in shots if (m1, m2, m3) == (0, 0, 0))
    all_one = sum(1 for m1, m2, m3 in shots if (m1, m2, m3) == (1, 1, 1))
    other = len(shots) - all_zero - all_one

    # GHZ state should only produce |000⟩ or |111⟩
    assert other == 0, f"GHZ state should not produce mixed outcomes, got {other}"
    assert all_zero > 0, "Should see |000⟩ outcomes"
    assert all_one > 0, "Should see |111⟩ outcomes"


def test_quantum_phase_kickback() -> None:
    """Test quantum phase kickback circuit."""

    @guppy
    def phase_kickback_circuit() -> tuple[bool, bool]:
        """Demonstrate phase kickback with controlled-Z gate."""
        control = qubit()
        target = qubit()

        # Put control in superposition
        h(control)

        # Put target in |1⟩ state
        x(target)

        # Apply controlled-Z (phase kickback occurs)
        # Since we don't have cz directly, use the equivalence: CZ = H·CX·H
        h(target)
        cx(control, target)
        h(target)

        # Measure in X basis for control (apply H before measure)
        h(control)
        m1 = measure(control)

        # Measure target in Z basis
        m2 = measure(target)

        return (m1, m2)

    # Run simulation with state_vector backend
    shot_vec = sim(Guppy(phase_kickback_circuit)).qubits(2).quantum(state_vector()).seed(42).run(1000)
    assert shot_vec is not None, "Should get results"
    # "measurements" holds one row per shot, ordered like the returned tuple.
    shots = shot_vec.to_dict()["measurements"]
    assert len(shots) == 1000, "Should have one measurement row per shot"

    # The control qubit should measure |1⟩ in X basis (due to phase kickback)
    # The target should remain in |1⟩
    control_one_count = sum(1 for m1, _ in shots if m1 == 1)
    target_one_count = sum(1 for _, m2 in shots if m2 == 1)
    total = len(shots)

    # Control should be predominantly |1⟩ due to phase kickback
    assert (
        control_one_count / total > 0.9
    ), f"Control should be ~100% |1⟩ after phase kickback, got {control_one_count / total}"
    # Target should remain |1⟩
    assert target_one_count / total > 0.9, f"Target should remain |1⟩, got {target_one_count / total}"


def test_quantum_interference() -> None:
    """Test quantum interference in a simple interferometer."""

    @guppy
    def quantum_interferometer() -> bool:
        """Create quantum interference using H gates."""
        q = qubit()

        # First H gate - creates superposition
        h(q)

        # Phase shift of π
        z(q)

        # Second H gate - creates interference
        h(q)

        # Should measure |1⟩ due to destructive interference
        return measure(q)

    # Run simulation with state_vector backend
    shot_vec = sim(Guppy(quantum_interferometer)).qubits(1).quantum(state_vector()).seed(42).run(1000)
    assert shot_vec is not None, "Should get results"
    # "measurements" holds one single-element row per shot.
    shots = shot_vec.to_dict()["measurements"]
    assert len(shots) == 1000, "Should have one measurement row per shot"

    # Due to interference, should measure |1⟩ ~100% of the time
    one_count = sum(1 for (m,) in shots if m == 1)
    total = len(shots)

    assert one_count / total > 0.95, f"Should measure |1⟩ due to interference, got {one_count / total}"


def test_rotation_gates() -> None:
    """Test rotation gates with specific angles."""

    @guppy
    def rotation_circuit() -> bool:
        """Test Y and Z rotations."""
        q = qubit()

        # Rotate around Y axis by π/2 (creates equal superposition)
        # angle takes halfturns, so 0.5 halfturns = π/2
        ry(q, angle(0.5))  # π/2

        # Rotate around Z axis by π/4 (adds phase)
        # 0.25 halfturns = π/4
        rz(q, angle(0.25))  # π/4

        # Measure
        return measure(q)

    # Run simulation with state_vector backend
    shot_vec = sim(Guppy(rotation_circuit)).qubits(1).quantum(state_vector()).seed(42).run(1000)

    assert shot_vec is not None, "Should get results"
    # "measurements" holds one single-element row per shot.
    shots = shot_vec.to_dict()["measurements"]
    assert len(shots) == 1000, "Should have one measurement row per shot"

    # After Ry(π/2), should be in equal superposition
    # Rz just adds phase, doesn't change measurement probabilities
    zero_count = sum(1 for (m,) in shots if m == 0)
    one_count = len(shots) - zero_count

    total = len(shots)
    # Should be roughly 50/50 after Ry(π/2)
    assert 0.4 < zero_count / total < 0.6, f"Should be ~50% |0⟩ after Ry(π/2), got {zero_count / total}"
    assert 0.4 < one_count / total < 0.6, f"Should be ~50% |1⟩ after Ry(π/2), got {one_count / total}"
