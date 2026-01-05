# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for pecos.qec.protocols module."""

from dataclasses import FrozenInstanceError

import pytest
from pecos.qec.protocols import (
    InnerCodeGeometry,
    MSDProtocol,
    OuterCodeGeometry,
    create_msd_protocol,
)


class TestInnerCodeGeometry:
    """Tests for inner code (distance-2) geometry."""

    def test_default_data_qubits(self) -> None:
        """Inner code should use qubits 0, 1, 3, 4."""
        inner = InnerCodeGeometry()
        assert inner.data_qubits == (0, 1, 3, 4)

    def test_num_data(self) -> None:
        """Inner code should have 4 data qubits."""
        inner = InnerCodeGeometry()
        assert inner.num_data == 4

    def test_z_stabilizer(self) -> None:
        """Z stabilizer should act on all 4 qubits."""
        inner = InnerCodeGeometry()
        assert inner.z_stabilizer == (0, 3, 1, 4)  # column-major order

    def test_x_stabilizers(self) -> None:
        """Should have 2 X stabilizers (top and bottom rows)."""
        inner = InnerCodeGeometry()
        assert inner.num_x_stabilizers == 2
        assert inner.x_stabilizers == ((0, 1), (3, 4))

    def test_num_z_stabilizers(self) -> None:
        """Should have 1 Z stabilizer."""
        inner = InnerCodeGeometry()
        assert inner.num_z_stabilizers == 1

    def test_num_syndromes(self) -> None:
        """Total syndromes per round should be 3 (2 X + 1 Z)."""
        inner = InnerCodeGeometry()
        assert inner.num_syndromes == 3

    def test_is_frozen(self) -> None:
        """InnerCodeGeometry should be immutable."""
        inner = InnerCodeGeometry()
        with pytest.raises(FrozenInstanceError):
            inner.data_qubits = (0, 1, 2, 3)


class TestOuterCodeGeometry:
    """Tests for outer code (distance-3) geometry."""

    def test_default_data_qubits(self) -> None:
        """Outer code should use all 9 qubits."""
        outer = OuterCodeGeometry()
        assert outer.data_qubits == (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def test_num_data(self) -> None:
        """Outer code should have 9 data qubits."""
        outer = OuterCodeGeometry()
        assert outer.num_data == 9

    def test_inner_qubits(self) -> None:
        """Inner qubits should be 0, 1, 3, 4."""
        outer = OuterCodeGeometry()
        assert outer.inner_qubits == (0, 1, 3, 4)

    def test_expansion_qubits(self) -> None:
        """Expansion qubits should be 2, 5, 6, 7, 8."""
        outer = OuterCodeGeometry()
        assert outer.expansion_qubits == (2, 5, 6, 7, 8)

    def test_num_expansion(self) -> None:
        """Should add 5 qubits during expansion."""
        outer = OuterCodeGeometry()
        assert outer.num_expansion == 5

    def test_x_stabilizers_count(self) -> None:
        """Should have 4 X stabilizers."""
        outer = OuterCodeGeometry()
        assert outer.num_x_stabilizers == 4

    def test_z_stabilizers_count(self) -> None:
        """Should have 4 Z stabilizers."""
        outer = OuterCodeGeometry()
        assert outer.num_z_stabilizers == 4

    def test_num_syndromes(self) -> None:
        """Total syndromes per round should be 8 (4 X + 4 Z)."""
        outer = OuterCodeGeometry()
        assert outer.num_syndromes == 8

    def test_bulk_x_stabilizers(self) -> None:
        """Should have 2 bulk X stabilizers (weight 4)."""
        outer = OuterCodeGeometry()
        bulk = outer.get_bulk_x_stabilizers()
        assert len(bulk) == 2
        for stab in bulk:
            assert len(stab) == 4

    def test_boundary_x_stabilizers(self) -> None:
        """Should have 2 boundary X stabilizers (weight 2)."""
        outer = OuterCodeGeometry()
        boundary = outer.get_boundary_x_stabilizers()
        assert len(boundary) == 2
        for stab in boundary:
            assert len(stab) == 2

    def test_bulk_z_stabilizers(self) -> None:
        """Should have 2 bulk Z stabilizers (weight 4)."""
        outer = OuterCodeGeometry()
        bulk = outer.get_bulk_z_stabilizers()
        assert len(bulk) == 2
        for stab in bulk:
            assert len(stab) == 4

    def test_boundary_z_stabilizers(self) -> None:
        """Should have 2 boundary Z stabilizers (weight 2)."""
        outer = OuterCodeGeometry()
        boundary = outer.get_boundary_z_stabilizers()
        assert len(boundary) == 2
        for stab in boundary:
            assert len(stab) == 2

    def test_is_frozen(self) -> None:
        """OuterCodeGeometry should be immutable."""
        outer = OuterCodeGeometry()
        with pytest.raises(FrozenInstanceError):
            outer.data_qubits = (0, 1, 2)


class TestMSDProtocol:
    """Tests for MSD protocol structure."""

    def test_default_inner_rounds(self) -> None:
        """Default inner rounds should be 2."""
        msd = MSDProtocol()
        assert msd.inner_rounds == 2

    def test_default_outer_rounds(self) -> None:
        """Default outer rounds should be 1."""
        msd = MSDProtocol()
        assert msd.outer_rounds == 1

    def test_total_data_qubits(self) -> None:
        """Total data qubits should be 9 (same as outer code)."""
        msd = MSDProtocol()
        assert msd.total_data_qubits == 9

    def test_inner_syndrome_bits(self) -> None:
        """Inner syndrome bits per round should be 3."""
        msd = MSDProtocol()
        assert msd.inner_syndrome_bits == 3

    def test_outer_syndrome_bits(self) -> None:
        """Outer syndrome bits per round should be 8."""
        msd = MSDProtocol()
        assert msd.outer_syndrome_bits == 8

    def test_total_inner_syndromes(self) -> None:
        """Total inner syndromes should be 3 * 2 = 6."""
        msd = MSDProtocol()
        assert msd.total_inner_syndromes == 6

    def test_total_outer_syndromes(self) -> None:
        """Total outer syndromes should be 8 * 1 = 8."""
        msd = MSDProtocol()
        assert msd.total_outer_syndromes == 8

    def test_expansion_prep_states(self) -> None:
        """Expansion prep states should be correct."""
        msd = MSDProtocol()
        prep = msd.get_expansion_prep_states()
        assert prep[2] == "0"
        assert prep[5] == "0"
        assert prep[6] == "+"
        assert prep[7] == "+"
        assert prep[8] == "+"

    def test_inner_init_states(self) -> None:
        """Inner init states should be correct."""
        msd = MSDProtocol()
        init = msd.get_inner_init_states()
        assert init[0] == "T+"
        assert init[1] == "0"
        assert init[3] == "+"
        assert init[4] == "+"


class TestCreateMSDProtocol:
    """Tests for create_msd_protocol factory function."""

    def test_default_creation(self) -> None:
        """Default creation should work."""
        msd = create_msd_protocol()
        assert msd.inner_rounds == 2
        assert msd.outer_rounds == 1

    def test_custom_inner_rounds(self) -> None:
        """Custom inner rounds should be respected."""
        msd = create_msd_protocol(inner_rounds=3)
        assert msd.inner_rounds == 3

    def test_custom_outer_rounds(self) -> None:
        """Custom outer rounds should be respected."""
        msd = create_msd_protocol(outer_rounds=2)
        assert msd.outer_rounds == 2

    def test_both_custom(self) -> None:
        """Both custom parameters should be respected."""
        msd = create_msd_protocol(inner_rounds=4, outer_rounds=3)
        assert msd.inner_rounds == 4
        assert msd.outer_rounds == 3
        assert msd.total_inner_syndromes == 12
        assert msd.total_outer_syndromes == 24
