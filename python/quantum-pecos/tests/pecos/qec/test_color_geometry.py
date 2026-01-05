# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for pecos.qec.color geometry functions."""

from dataclasses import FrozenInstanceError

import pytest
from pecos.qec.color import (
    ColorCode488,
    ColorCode488Builder,
    ColorCode488Geometry,
    ColorCodeStabilizer,
    generate_488_layout,
)


class TestGenerate488Layout:
    """Tests for 4.8.8 color code layout generation."""

    def test_d3_qubit_count(self) -> None:
        """Distance-3 color code should have 7 data qubits."""
        nodeid2pos, _polygons = generate_488_layout(3)
        assert len(nodeid2pos) == 7

    def test_d5_qubit_count(self) -> None:
        """Distance-5 color code should have 17 data qubits."""
        nodeid2pos, _polygons = generate_488_layout(5)
        assert len(nodeid2pos) == 17

    def test_d7_qubit_count(self) -> None:
        """Distance-7 color code should have 31 data qubits."""
        nodeid2pos, _polygons = generate_488_layout(7)
        assert len(nodeid2pos) == 31

    def test_d3_polygon_count(self) -> None:
        """Distance-3 color code should have 3 polygons (stabilizers)."""
        _nodeid2pos, polygons = generate_488_layout(3)
        assert len(polygons) == 3

    def test_invalid_distance_even(self) -> None:
        """Even distances should raise ValueError."""
        with pytest.raises(ValueError, match="odd"):
            generate_488_layout(4)

    def test_invalid_distance_too_small(self) -> None:
        """Distance < 3 should raise ValueError."""
        with pytest.raises(ValueError, match="odd"):
            generate_488_layout(1)

    def test_positions_are_unique(self) -> None:
        """All qubit positions should be unique."""
        nodeid2pos, _ = generate_488_layout(5)
        positions = list(nodeid2pos.values())
        assert len(positions) == len(set(positions))

    def test_polygon_contains_color(self) -> None:
        """Each polygon should end with a color string."""
        _, polygons = generate_488_layout(5)
        for poly in polygons:
            color = poly[-1]
            assert color in ("red", "green", "blue")


class TestColorCode488:
    """Tests for ColorCode488 class."""

    def test_create_d3(self) -> None:
        """Create distance-3 color code."""
        code = ColorCode488.create(distance=3)
        assert code.distance == 3
        assert code.num_data == 7

    def test_create_d5(self) -> None:
        """Create distance-5 color code."""
        code = ColorCode488.create(distance=5)
        assert code.distance == 5
        assert code.num_data == 17

    def test_num_stabilizers(self) -> None:
        """Check stabilizer count."""
        code = ColorCode488.create(distance=3)
        assert code.num_stabilizers == 3

    def test_stabilizers_have_colors(self) -> None:
        """Each stabilizer should have a color."""
        code = ColorCode488.create(distance=5)
        for stab in code.stabilizers:
            assert stab.color in ("red", "green", "blue")

    def test_get_stabilizers_by_color(self) -> None:
        """Filter stabilizers by color."""
        code = ColorCode488.create(distance=5)
        red = code.get_stabilizers_by_color("red")
        green = code.get_stabilizers_by_color("green")
        blue = code.get_stabilizers_by_color("blue")
        assert len(red) + len(green) + len(blue) == code.num_stabilizers

    def test_logical_x(self) -> None:
        """Logical X should be on bottom boundary."""
        code = ColorCode488.create(distance=3)
        logical_x = code.get_logical_x()
        assert len(logical_x) == 3  # d qubits for logical operator

    def test_logical_z(self) -> None:
        """Logical Z should be on same support (self-dual code)."""
        code = ColorCode488.create(distance=3)
        logical_z = code.get_logical_z()
        assert logical_z == code.get_logical_x()  # Self-dual

    def test_parity_matrix_shape(self) -> None:
        """Parity matrix should have shape (num_stab, num_data)."""
        code = ColorCode488.create(distance=3)
        parity_matrix = code.get_parity_matrix()
        assert parity_matrix.shape == (code.num_stabilizers, code.num_data)


class TestColorCode488Builder:
    """Tests for ColorCode488Builder."""

    def test_builder_pattern(self) -> None:
        """Builder should create valid code."""
        code = ColorCode488Builder().with_distance(5).build()
        assert code.distance == 5

    def test_builder_requires_distance(self) -> None:
        """Builder should raise if distance not set."""
        with pytest.raises(ValueError, match="Distance"):
            ColorCode488Builder().build()


class TestColorCodeStabilizer:
    """Tests for ColorCodeStabilizer dataclass."""

    def test_stabilizer_is_frozen(self) -> None:
        """ColorCodeStabilizer should be immutable."""
        stab = ColorCodeStabilizer(index=0, qubits=(0, 1, 2), color="red")
        with pytest.raises(FrozenInstanceError):
            stab.index = 1

    def test_weight_property(self) -> None:
        """Weight should be number of qubits."""
        stab = ColorCodeStabilizer(index=0, qubits=(0, 1, 2, 3), color="red")
        assert stab.weight == 4

    def test_boundary_default_false(self) -> None:
        """is_boundary should default to False."""
        stab = ColorCodeStabilizer(index=0, qubits=(0, 1), color="red")
        assert stab.is_boundary is False
