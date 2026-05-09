# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for surface code geometry across all dimensions.

Verifies that the stabilizer generators, code parameters, CNOT scheduling,
and circuit generation work correctly for:
- Standard odd square codes (d=3,5,7)
- Even square codes (d=2,4)
- Asymmetric codes (dx != dz)
- Repetition codes (dx=1 or dz=1)
- Single qubit (dx=dz=1)
"""

import pytest
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.schedule import compute_cnot_schedule

# ============================================================
# Code parameters: n, k, d
# ============================================================


@pytest.mark.parametrize(
    ("dx", "dz", "expected_n", "expected_k", "expected_d"),
    [
        # Single qubit
        (1, 1, 1, 1, 1),
        # Repetition codes (X checks)
        (1, 3, 3, 1, 1),
        (1, 5, 5, 1, 1),
        (1, 7, 7, 1, 1),
        # Repetition codes (Z checks)
        (3, 1, 3, 1, 1),
        (5, 1, 5, 1, 1),
        # Even square
        (2, 2, 4, 1, 2),
        (4, 4, 16, 1, 4),
        # Odd square (standard surface code)
        (3, 3, 9, 1, 3),
        (5, 5, 25, 1, 5),
        (7, 7, 49, 1, 7),
        # Asymmetric
        (2, 3, 6, 1, 2),
        (3, 2, 6, 1, 2),
        (3, 5, 15, 1, 3),
        (5, 3, 15, 1, 3),
        (2, 5, 10, 1, 2),
    ],
)
def test_code_parameters(dx, dz, expected_n, expected_k, expected_d):
    """Code parameters [[n,k,d]] should be correct for all dimensions."""
    patch = SurfacePatch.create(dx=dx, dz=dz)
    n = patch.num_data
    k = n - patch.geometry.num_x_stab - patch.geometry.num_z_stab
    d = patch.distance

    assert n == expected_n
    assert k == expected_k
    assert d == expected_d


# ============================================================
# Stabilizer structure
# ============================================================


@pytest.mark.parametrize(
    ("dx", "dz", "expected_x", "expected_z"),
    [
        (1, 1, 0, 0),
        (1, 3, 2, 0),
        (3, 1, 0, 2),
        (1, 5, 4, 0),
        (5, 1, 0, 4),
        (2, 2, 2, 1),
        (3, 3, 4, 4),
        (2, 3, 3, 2),
        (3, 2, 2, 3),
        (4, 4, 8, 7),
        (5, 5, 12, 12),
    ],
)
def test_stabilizer_counts(dx, dz, expected_x, expected_z):
    """Number of X and Z stabilizers should match expected values."""
    patch = SurfacePatch.create(dx=dx, dz=dz)
    assert patch.geometry.num_x_stab == expected_x
    assert patch.geometry.num_z_stab == expected_z


def test_repetition_code_x_checks():
    """dx=1 repetition code should have only X stabilizers (adjacent XX pairs)."""
    patch = SurfacePatch.create(dx=1, dz=5)
    assert patch.geometry.num_z_stab == 0
    qubits = [s.data_qubits for s in patch.geometry.x_stabilizers]
    # All should be adjacent pairs covering 0..4
    for q1, q2 in qubits:
        assert q2 == q1 + 1


def test_repetition_code_z_checks():
    """dz=1 repetition code should have only Z stabilizers (adjacent ZZ pairs)."""
    patch = SurfacePatch.create(dx=5, dz=1)
    assert patch.geometry.num_x_stab == 0
    qubits = [s.data_qubits for s in patch.geometry.z_stabilizers]
    for q1, q2 in qubits:
        assert q2 == q1 + 1


def test_all_data_qubits_in_stabilizer_support():
    """Every data qubit (except logical operator edges) should appear in at least one stabilizer."""
    for dx, dz in [(3, 3), (2, 3), (3, 2), (2, 2), (4, 4)]:
        patch = SurfacePatch.create(dx=dx, dz=dz)
        touched = set()
        for s in patch.geometry.x_stabilizers:
            touched.update(s.data_qubits)
        for s in patch.geometry.z_stabilizers:
            touched.update(s.data_qubits)
        assert touched == set(range(patch.num_data)), f"Untouched qubits in {dx}x{dz}"


def test_stabilizer_weights():
    """Bulk stabilizers should be weight 4, boundary weight 2."""
    for dx, dz in [(3, 3), (5, 5), (2, 3), (3, 5)]:
        patch = SurfacePatch.create(dx=dx, dz=dz)
        for s in list(patch.geometry.x_stabilizers) + list(patch.geometry.z_stabilizers):
            if s.is_boundary:
                assert len(s.data_qubits) == 2, f"Boundary stab has weight {len(s.data_qubits)}"
            else:
                assert len(s.data_qubits) == 4, f"Bulk stab has weight {len(s.data_qubits)}"


# ============================================================
# Logical operators
# ============================================================


def test_logical_x_weight_equals_dx():
    """Logical X should have weight dx (left edge of the grid)."""
    for dx, dz in [(3, 3), (3, 5), (5, 3), (2, 3), (1, 5)]:
        patch = SurfacePatch.create(dx=dx, dz=dz)
        assert len(patch.geometry.logical_x.data_qubits) == dx


def test_logical_z_weight_equals_dz():
    """Logical Z should have weight dz (top edge of the grid)."""
    for dx, dz in [(3, 3), (3, 5), (5, 3), (2, 3), (1, 5)]:
        patch = SurfacePatch.create(dx=dx, dz=dz)
        assert len(patch.geometry.logical_z.data_qubits) == dz


def test_logical_x_is_left_edge():
    """Logical X qubits should be column 0 of the dx x dz grid."""
    for dx, dz in [(3, 3), (3, 5), (2, 3)]:
        patch = SurfacePatch.create(dx=dx, dz=dz)
        expected = tuple(i * dz for i in range(dx))
        assert patch.geometry.logical_x.data_qubits == expected


def test_logical_z_is_top_edge():
    """Logical Z qubits should be row 0 of the dx x dz grid."""
    for dx, dz in [(3, 3), (3, 5), (2, 3)]:
        patch = SurfacePatch.create(dx=dx, dz=dz)
        expected = tuple(range(dz))
        assert patch.geometry.logical_z.data_qubits == expected


# ============================================================
# CNOT schedule: no conflicts
# ============================================================


@pytest.mark.parametrize(
    ("dx", "dz"),
    [
        (1, 3),
        (3, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (3, 5),
        (5, 3),
        (4, 4),
        (5, 5),
    ],
)
def test_cnot_schedule_no_conflicts(dx, dz):
    """No data qubit should be touched twice in the same CNOT round."""
    patch = SurfacePatch.create(dx=dx, dz=dz)
    schedule = compute_cnot_schedule(patch)
    for rnd_idx, rnd in enumerate(schedule):
        data_qubits = [dq for _, _, dq in rnd]
        assert len(data_qubits) == len(
            set(data_qubits),
        ), f"{dx}x{dz} round {rnd_idx}: data qubit collision {data_qubits}"


# ============================================================
# Backward compatibility: square odd codes unchanged
# ============================================================


def test_square_odd_codes_match_original():
    """The generalized generators should produce identical results to the
    original single-d generators for square odd codes.
    """
    from pecos.qec.surface.layouts.rotated_lattice import (
        compute_rotated_x_stabilizers,
        compute_rotated_z_stabilizers,
        get_rotated_logical_x,
        get_rotated_logical_z,
    )

    for d in [3, 5, 7]:
        x_single = compute_rotated_x_stabilizers(d)
        x_pair = compute_rotated_x_stabilizers(d, d)
        z_single = compute_rotated_z_stabilizers(d)
        z_pair = compute_rotated_z_stabilizers(d, d)

        assert len(x_single) == len(x_pair)
        assert len(z_single) == len(z_pair)

        for a, b in zip(x_single, x_pair, strict=False):
            assert a.data_qubits == b.data_qubits
            assert a.is_boundary == b.is_boundary

        for a, b in zip(z_single, z_pair, strict=False):
            assert a.data_qubits == b.data_qubits
            assert a.is_boundary == b.is_boundary

        assert get_rotated_logical_x(d) == get_rotated_logical_x(d, d)
        assert get_rotated_logical_z(d) == get_rotated_logical_z(d, d)


# ============================================================
# Transposition symmetry
# ============================================================


def test_transpose_swaps_x_and_z_counts():
    """Swapping dx and dz should swap the number of X and Z stabilizers."""
    for dx, dz in [(2, 3), (3, 5), (2, 5), (1, 3), (1, 5)]:
        p1 = SurfacePatch.create(dx=dx, dz=dz)
        p2 = SurfacePatch.create(dx=dz, dz=dx)
        assert p1.geometry.num_x_stab == p2.geometry.num_z_stab
        assert p1.geometry.num_z_stab == p2.geometry.num_x_stab


# ============================================================
# Circuit generation
# ============================================================


@pytest.mark.parametrize(
    ("dx", "dz"),
    [
        (1, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ],
)
def test_circuit_generation(dx, dz):
    """LogicalCircuitBuilder should produce a valid TickCircuit for all dimensions."""
    from pecos.qec.surface import LogicalCircuitBuilder

    patch = SurfacePatch.create(dx=dx, dz=dz)
    lcb = LogicalCircuitBuilder()
    lcb.add_patch(patch, "A")
    basis = "Z" if dx >= dz else "X"
    lcb.add_memory("A", rounds=2, basis=basis)
    tc = lcb.to_tick_circuit()

    assert tc.num_ticks() > 0
    assert tc.gate_count() > 0


# ============================================================
# Transversal gate square check
# ============================================================


def test_transversal_h_rejects_nonsquare():
    """Transversal H should reject non-square patches."""
    from pecos.qec.surface import LogicalCircuitBuilder

    patch = SurfacePatch.create(dx=2, dz=3)
    lcb = LogicalCircuitBuilder()
    lcb.add_patch(patch, "A")
    with pytest.raises(ValueError, match="square"):
        lcb.add_transversal_h("A")


def test_transversal_h_accepts_square():
    """Transversal H should accept square patches of any distance."""
    from pecos.qec.surface import LogicalCircuitBuilder

    for d in [2, 3, 4, 5]:
        patch = SurfacePatch.create(distance=d)
        lcb = LogicalCircuitBuilder()
        lcb.add_patch(patch, "A")
        lcb.add_memory("A", rounds=1, basis="Z")
        lcb.add_transversal_h("A")
        lcb.add_memory("A", rounds=1, basis="X")


# ============================================================
# Validation
# ============================================================


def test_distance_zero_rejected():
    """Distance 0 should raise ValueError."""
    with pytest.raises(ValueError, match=r"Distance must be >= 1"):
        SurfacePatch.create(distance=0)


def test_negative_distance_rejected():
    """Negative distance should raise ValueError."""
    with pytest.raises(ValueError, match=r"dx must be >= 1"):
        SurfacePatch.create(dx=-1, dz=3)
