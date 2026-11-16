# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for pecos.qec.surface layout functions."""

import pytest
from pecos.qec.surface.layouts import (
    StabilizerSupport,
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    generate_nonrotated_surface_layout,
    generate_surface_layout,
    get_boundary_stabilizer_indices,
    get_boundary_stabilizers,
    get_bulk_stabilizer_indices,
    get_bulk_stabilizers,
    get_stabilizer_counts,
)


class TestGenerateNonrotatedSurfaceLayout:
    """Tests for non-rotated surface code layout generation."""

    def test_d3_data_qubit_count(self) -> None:
        """Distance-3 non-rotated code should have 13 data qubits."""
        data, _ancilla, _polygons = generate_nonrotated_surface_layout(3, 3)
        assert len(data) == 13

    def test_d5_data_qubit_count(self) -> None:
        """Distance-5 non-rotated code should have 41 data qubits."""
        data, _ancilla, _polygons = generate_nonrotated_surface_layout(5, 5)
        assert len(data) == 41

    def test_d3_total_positions(self) -> None:
        """Check total positions for d=3."""
        data, ancilla, _polygons = generate_nonrotated_surface_layout(3, 3)
        # 5x5 lattice = 25 total positions
        assert len(data) + len(ancilla) == 25

    def test_asymmetric_layout(self) -> None:
        """Test asymmetric width/height."""
        data, ancilla, _polygons = generate_nonrotated_surface_layout(3, 5)
        # 5x9 lattice
        assert len(data) + len(ancilla) == 45

    def test_data_positions_are_unique(self) -> None:
        """All data positions should be unique."""
        data, _, _ = generate_nonrotated_surface_layout(5, 5)
        assert len(data) == len(set(data))

    def test_ancilla_positions_are_unique(self) -> None:
        """All ancilla positions should be unique."""
        _, ancilla, _ = generate_nonrotated_surface_layout(5, 5)
        assert len(ancilla) == len(set(ancilla))

    def test_no_overlap_data_ancilla(self) -> None:
        """Data and ancilla positions should not overlap."""
        data, ancilla, _ = generate_nonrotated_surface_layout(5, 5)
        data_set = set(data)
        ancilla_set = set(ancilla)
        assert data_set.isdisjoint(ancilla_set)


class TestGenerateSurfaceLayout:
    """Tests for rotated surface code layout generation (default)."""

    def test_d3_data_qubit_count(self) -> None:
        """Distance-3 rotated code should have 9 data qubits."""
        data, _ancilla = generate_surface_layout(3, 3)
        assert len(data) == 9

    def test_d5_data_qubit_count(self) -> None:
        """Distance-5 rotated code should have 25 data qubits."""
        data, _ancilla = generate_surface_layout(5, 5)
        assert len(data) == 25

    def test_d3_ancilla_count(self) -> None:
        """Distance-3 rotated code should have 8 ancilla qubits."""
        _data, ancilla = generate_surface_layout(3, 3)
        assert len(ancilla) == 8

    def test_data_positions_are_unique(self) -> None:
        """All data positions should be unique."""
        data, _ = generate_surface_layout(5, 5)
        assert len(data) == len(set(data))

    def test_ancilla_positions_are_unique(self) -> None:
        """All ancilla positions should be unique."""
        _, ancilla = generate_surface_layout(5, 5)
        assert len(ancilla) == len(set(ancilla))

    def test_no_overlap_data_ancilla(self) -> None:
        """Data and ancilla positions should not overlap."""
        data, ancilla = generate_surface_layout(5, 5)
        data_set = set(data)
        ancilla_set = set(ancilla)
        assert data_set.isdisjoint(ancilla_set)

    def test_data_at_odd_odd_positions(self) -> None:
        """In interior, data qubits should be at odd-odd positions."""
        data, _ = generate_surface_layout(5, 5)
        for x, y in data:
            # Interior data qubits are at odd-odd positions
            if 0 < x < 10 and 0 < y < 10:
                assert x % 2 == 1
                assert y % 2 == 1


class TestComputeStabilizerSupports:
    """Tests for stabilizer support computation."""

    def test_x_stabilizer_count_d3(self) -> None:
        """Distance-3 should have 4 X stabilizers."""
        stabs = compute_x_stabilizer_supports(3)
        assert len(stabs) == 4

    def test_z_stabilizer_count_d3(self) -> None:
        """Distance-3 should have 4 Z stabilizers."""
        stabs = compute_z_stabilizer_supports(3)
        assert len(stabs) == 4

    def test_x_stabilizer_count_d5(self) -> None:
        """Distance-5 should have 12 X stabilizers."""
        stabs = compute_x_stabilizer_supports(5)
        assert len(stabs) == 12

    def test_z_stabilizer_count_d5(self) -> None:
        """Distance-5 should have 12 Z stabilizers."""
        stabs = compute_z_stabilizer_supports(5)
        assert len(stabs) == 12

    def test_stabilizer_indices_are_sequential(self) -> None:
        """Stabilizer indices should be 0 to n-1."""
        stabs = compute_x_stabilizer_supports(5)
        indices = sorted(s.index for s in stabs)
        assert indices == list(range(len(stabs)))

    def test_bulk_stabilizers_have_weight_4(self) -> None:
        """Non-boundary stabilizers should have weight 4."""
        stabs = compute_x_stabilizer_supports(5)
        for s in stabs:
            if not s.is_boundary:
                assert s.weight == 4

    def test_boundary_stabilizers_have_weight_2(self) -> None:
        """Boundary stabilizers should have weight 2."""
        stabs = compute_x_stabilizer_supports(5)
        for s in stabs:
            if s.is_boundary:
                assert s.weight == 2

    def test_stabilizer_support_is_dataclass(self) -> None:
        """StabilizerSupport should be a frozen dataclass."""
        stabs = compute_x_stabilizer_supports(3)
        s = stabs[0]
        assert isinstance(s, StabilizerSupport)
        # Frozen dataclass should be hashable
        assert hash(s) is not None


class TestStabilizerCategories:
    """Tests for bulk/boundary stabilizer categorization."""

    def test_stabilizer_counts_d3(self) -> None:
        """Test stabilizer counts for d=3."""
        total, n_bulk, n_boundary = get_stabilizer_counts(3)
        assert total == 4
        assert n_bulk == 2
        assert n_boundary == 2

    def test_stabilizer_counts_d5(self) -> None:
        """Test stabilizer counts for d=5."""
        total, n_bulk, n_boundary = get_stabilizer_counts(5)
        assert total == 12
        assert n_bulk == 8
        assert n_boundary == 4

    def test_stabilizer_counts_d7(self) -> None:
        """Test stabilizer counts for d=7."""
        total, n_bulk, n_boundary = get_stabilizer_counts(7)
        assert total == 24
        assert n_bulk == 18
        assert n_boundary == 6

    def test_bulk_indices_d3(self) -> None:
        """Bulk indices for d=3 should be [1, 2]."""
        indices = get_bulk_stabilizer_indices(3)
        assert indices == [1, 2]

    def test_boundary_indices_d3(self) -> None:
        """Boundary indices for d=3 should be [0, 3]."""
        indices = get_boundary_stabilizer_indices(3)
        assert indices == [0, 3]

    def test_bulk_indices_d5(self) -> None:
        """Bulk indices for d=5."""
        indices = get_bulk_stabilizer_indices(5)
        assert len(indices) == 8
        assert indices == list(range(2, 10))

    def test_boundary_indices_d5(self) -> None:
        """Boundary indices for d=5."""
        indices = get_boundary_stabilizer_indices(5)
        assert len(indices) == 4
        assert indices == [0, 1, 10, 11]

    def test_bulk_plus_boundary_equals_total(self) -> None:
        """Bulk + boundary indices should cover all stabilizers."""
        for d in [3, 5, 7]:
            bulk = get_bulk_stabilizer_indices(d)
            boundary = get_boundary_stabilizer_indices(d)
            total, _, _ = get_stabilizer_counts(d)
            assert len(bulk) + len(boundary) == total
            assert set(bulk).isdisjoint(set(boundary))

    def test_get_bulk_stabilizers_x(self) -> None:
        """Get bulk X stabilizers."""
        bulk = get_bulk_stabilizers(5, "X")
        assert len(bulk) == 8
        for s in bulk:
            assert not s.is_boundary
            assert s.weight == 4

    def test_get_boundary_stabilizers_x(self) -> None:
        """Get boundary X stabilizers."""
        boundary = get_boundary_stabilizers(5, "X")
        assert len(boundary) == 4
        for s in boundary:
            assert s.is_boundary
            assert s.weight == 2

    def test_get_bulk_stabilizers_z(self) -> None:
        """Get bulk Z stabilizers."""
        bulk = get_bulk_stabilizers(5, "Z")
        assert len(bulk) == 8
        for s in bulk:
            assert not s.is_boundary
            assert s.weight == 4

    def test_get_boundary_stabilizers_z(self) -> None:
        """Get boundary Z stabilizers."""
        boundary = get_boundary_stabilizers(5, "Z")
        assert len(boundary) == 4
        for s in boundary:
            assert s.is_boundary
            assert s.weight == 2
