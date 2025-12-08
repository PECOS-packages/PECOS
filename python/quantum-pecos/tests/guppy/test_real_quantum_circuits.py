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
    shot_vec = (
        sim(Guppy(prepare_bell_state))
        .qubits(2)
        .quantum(state_vector())
        .seed(42)
        .run(1000)
    )
    assert shot_vec is not None, "Should get results"
    results = shot_vec.to_dict()
    # Count outcomes
    both_zero = 0
    both_one = 0
    anti_correlated = 0

    # Results come as a dict with measurement keys
    m1_list = results.get("measurement_0", [])
    m2_list = results.get("measurement_1", [])

    for m1, m2 in zip(m1_list, m2_list, strict=False):
        if m1 == 0 and m2 == 0:
            both_zero += 1
        elif m1 == 1 and m2 == 1:
            both_one += 1
        else:
            anti_correlated += 1

    # Bell state should only produce correlated outcomes
    assert (
        anti_correlated == 0
    ), f"Bell state should not produce anti-correlated outcomes, got {anti_correlated}"
    assert both_zero > 0, "Should see |00⟩ outcomes"
    assert both_one > 0, "Should see |11⟩ outcomes"

    # Should be roughly 50/50 split
    total = both_zero + both_one
    assert (
        0.4 < both_zero / total < 0.6
    ), f"Should be ~50% |00⟩, got {both_zero / total}"
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
    shot_vec = (
        sim(Guppy(prepare_ghz_state))
        .qubits(3)
        .quantum(state_vector())
        .seed(42)
        .run(1000)
    )
    assert shot_vec is not None, "Should get results"
    results = shot_vec.to_dict()

    # GHZ state should give either all 0s or all 1s
    all_zero = 0
    all_one = 0
    other = 0

    m1_list = results.get("measurement_0", [])
    m2_list = results.get("measurement_1", [])
    m3_list = results.get("measurement_2", [])

    for m1, m2, m3 in zip(m1_list, m2_list, m3_list, strict=False):
        if m1 == 0 and m2 == 0 and m3 == 0:
            all_zero += 1
        elif m1 == 1 and m2 == 1 and m3 == 1:
            all_one += 1
        else:
            other += 1

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
    results = (
        sim(Guppy(phase_kickback_circuit))
        .qubits(2)
        .quantum(state_vector())
        .seed(42)
        .run(1000)
    )
    assert results is not None, "Should get results"

    # The control qubit should measure |1⟩ in X basis (due to phase kickback)
    # The target should remain in |1⟩
    control_one_count = 0
    target_one_count = 0
    total = 0

    if hasattr(results, "__getitem__"):
        m1_list = results.get("measurement_0", [])
        m2_list = results.get("measurement_1", [])

        for m1, m2 in zip(m1_list, m2_list, strict=False):
            total += 1
            if m1 == 1:
                control_one_count += 1
            if m2 == 1:
                target_one_count += 1

    # Control should be predominantly |1⟩ due to phase kickback
    assert (
        control_one_count / total > 0.9
    ), f"Control should be ~100% |1⟩ after phase kickback, got {control_one_count / total}"
    # Target should remain |1⟩
    assert (
        target_one_count / total > 0.9
    ), f"Target should remain |1⟩, got {target_one_count / total}"


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
    results = (
        sim(Guppy(quantum_interferometer))
        .qubits(1)
        .quantum(state_vector())
        .seed(42)
        .run(1000)
    )
    assert results is not None, "Should get results"

    # Due to interference, should measure |1⟩ ~100% of the time
    one_count = 0
    total = 0

    if hasattr(results, "__getitem__"):
        measurements = results.get("measurement_0", [])
        for m in measurements:
            total += 1
            if m == 1:
                one_count += 1

    assert (
        one_count / total > 0.95
    ), f"Should measure |1⟩ due to interference, got {one_count / total}"


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
    results = (
        sim(Guppy(rotation_circuit))
        .qubits(1)
        .quantum(state_vector())
        .seed(42)
        .run(1000)
    )

    assert results is not None, "Should get results"

    # After Ry(π/2), should be in equal superposition
    # Rz just adds phase, doesn't change measurement probabilities
    zero_count = 0
    one_count = 0

    if hasattr(results, "__getitem__"):
        measurements = results.get("measurement_0", [])
        for m in measurements:
            if m == 0:
                zero_count += 1
            else:
                one_count += 1

    total = zero_count + one_count
    # Should be roughly 50/50 after Ry(π/2)
    assert (
        0.4 < zero_count / total < 0.6
    ), f"Should be ~50% |0⟩ after Ry(π/2), got {zero_count / total}"
    assert (
        0.4 < one_count / total < 0.6
    ), f"Should be ~50% |1⟩ after Ry(π/2), got {one_count / total}"
