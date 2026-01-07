# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for pecos.qec.analysis utilities."""

import pytest
from pecos.qec.analysis import (
    logical_error_rate,
    logical_fidelity,
    logical_from_data,
    logical_x_from_data,
    logical_z_from_data,
    lower_bound_fidelity,
    syndrome_difference,
    syndrome_to_detection_events,
)


class TestLogicalExtraction:
    """Tests for logical operator extraction from measurement data."""

    def test_logical_x_all_zeros(self) -> None:
        """All-zero data should give logical X = 0."""
        data = [0] * 9  # d=3
        assert logical_x_from_data(3, data) == 0

    def test_logical_z_all_zeros(self) -> None:
        """All-zero data should give logical Z = 0."""
        data = [0] * 9  # d=3
        assert logical_z_from_data(3, data) == 0

    def test_logical_x_left_column_ones(self) -> None:
        """Ones in left column should give logical X = 1."""
        # Left column indices for d=3: 0, 3, 6
        data = [1, 0, 0, 1, 0, 0, 1, 0, 0]
        assert logical_x_from_data(3, data) == 1

    def test_logical_z_top_row_ones(self) -> None:
        """Ones in top row should give logical Z = 1."""
        # Top row indices for d=3: 0, 1, 2
        data = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        assert logical_z_from_data(3, data) == 1

    def test_logical_x_even_parity(self) -> None:
        """Even number of ones in left column gives 0."""
        data = [1, 0, 0, 1, 0, 0, 0, 0, 0]  # 2 ones in left column
        assert logical_x_from_data(3, data) == 0

    def test_logical_z_even_parity(self) -> None:
        """Even number of ones in top row gives 0."""
        data = [1, 1, 0, 0, 0, 0, 0, 0, 0]  # 2 ones in top row
        assert logical_z_from_data(3, data) == 0

    def test_logical_from_data_both(self) -> None:
        """Test combined extraction."""
        data = [1, 1, 1, 1, 0, 0, 1, 0, 0]
        x, z = logical_from_data(3, data)
        assert x == 1  # 3 ones in left column (0, 3, 6)
        assert z == 1  # 3 ones in top row (0, 1, 2)

    def test_logical_x_d5(self) -> None:
        """Test logical X for d=5."""
        # Left column indices: 0, 5, 10, 15, 20
        data = [0] * 25
        data[0] = 1
        data[5] = 1
        data[10] = 1
        assert logical_x_from_data(5, data) == 1

    def test_logical_z_d5(self) -> None:
        """Test logical Z for d=5."""
        # Top row indices: 0, 1, 2, 3, 4
        data = [0] * 25
        data[0] = 1
        data[1] = 1
        data[2] = 1
        data[3] = 1
        data[4] = 1
        assert logical_z_from_data(5, data) == 1

    def test_invalid_data_length(self) -> None:
        """Wrong data length should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 9"):
            logical_x_from_data(3, [0] * 10)


class TestLogicalFidelity:
    """Tests for fidelity calculation."""

    def test_perfect_fidelity(self) -> None:
        """All correct outcomes should give fidelity 1.0."""
        outcomes = [[0] * 9 for _ in range(100)]
        fid, err = logical_fidelity(outcomes, d=3, basis=0, expected=0)
        assert fid == 1.0
        assert err == 0.0

    def test_zero_fidelity(self) -> None:
        """All wrong outcomes should give fidelity 0.0."""
        # All ones in left column for X basis
        outcomes = [[1, 0, 0, 1, 0, 0, 1, 0, 0] for _ in range(100)]
        fid, err = logical_fidelity(outcomes, d=3, basis=0, expected=0)
        assert fid == 0.0
        assert err == 0.0

    def test_half_fidelity(self) -> None:
        """Half correct outcomes should give fidelity 0.5."""
        correct = [0] * 9
        wrong = [1, 0, 0, 1, 0, 0, 1, 0, 0]  # logical X = 1
        outcomes = [correct] * 50 + [wrong] * 50
        fid, err = logical_fidelity(outcomes, d=3, basis=0, expected=0)
        assert fid == 0.5
        assert err == pytest.approx(0.05, abs=0.01)

    def test_z_basis(self) -> None:
        """Test Z basis measurement."""
        correct = [0] * 9
        wrong = [1, 1, 1, 0, 0, 0, 0, 0, 0]  # logical Z = 1
        outcomes = [correct] * 80 + [wrong] * 20
        fid, err = logical_fidelity(outcomes, d=3, basis=1, expected=0)
        assert fid == 0.8
        assert err == pytest.approx(0.04, abs=0.01)

    def test_empty_outcomes(self) -> None:
        """Empty outcomes should raise ValueError."""
        with pytest.raises(ValueError, match="No outcomes"):
            logical_fidelity([], d=3, basis=0)


class TestLogicalErrorRate:
    """Tests for error rate calculation."""

    def test_error_rate_is_one_minus_fidelity(self) -> None:
        """Error rate should be 1 - fidelity."""
        correct = [0] * 9
        wrong = [1, 0, 0, 1, 0, 0, 1, 0, 0]
        outcomes = [correct] * 80 + [wrong] * 20
        err_rate, _err_bar = logical_error_rate(outcomes, d=3, basis=0)
        fid, _ = logical_fidelity(outcomes, d=3, basis=0)
        assert err_rate == pytest.approx(1 - fid)


class TestSyndromeDifference:
    """Tests for syndrome difference computation."""

    def test_empty_syndromes(self) -> None:
        """Empty input should return empty output."""
        assert syndrome_difference([]) == []

    def test_single_round(self) -> None:
        """Single round should return the syndrome itself."""
        syn = [1, 0, 1, 0]
        result = syndrome_difference([syn])
        assert result == [[1, 0, 1, 0]]

    def test_two_identical_rounds(self) -> None:
        """Identical rounds should give zero difference."""
        syn = [1, 0, 1, 0]
        result = syndrome_difference([syn, syn])
        assert result[0] == [1, 0, 1, 0]  # First round vs zeros
        assert result[1] == [0, 0, 0, 0]  # Second round vs first

    def test_alternating_syndromes(self) -> None:
        """Alternating syndromes should show changes."""
        syn1 = [1, 0, 1, 0]
        syn2 = [0, 1, 0, 1]
        result = syndrome_difference([syn1, syn2])
        assert result[0] == [1, 0, 1, 0]  # syn1 XOR 0
        assert result[1] == [1, 1, 1, 1]  # syn2 XOR syn1

    def test_three_rounds(self) -> None:
        """Test three-round case."""
        syns = [[1, 1], [1, 0], [0, 0]]
        result = syndrome_difference(syns)
        assert result[0] == [1, 1]  # Round 0 vs zeros
        assert result[1] == [0, 1]  # Round 1 vs round 0
        assert result[2] == [1, 0]  # Round 2 vs round 1


class TestSyndromeToDetectionEvents:
    """Tests for detection event extraction."""

    def test_no_events(self) -> None:
        """All-zero syndromes should give no events."""
        syns = [[0, 0], [0, 0], [0, 0]]
        events = syndrome_to_detection_events(syns)
        assert events == []

    def test_first_round_events(self) -> None:
        """Events in first round (difference from zero)."""
        syns = [[1, 0, 1, 0]]
        events = syndrome_to_detection_events(syns)
        assert (0, 0) in events
        assert (2, 0) in events
        assert len(events) == 2

    def test_later_round_events(self) -> None:
        """Events from syndrome changes."""
        syns = [[0, 0], [1, 0], [1, 0]]
        events = syndrome_to_detection_events(syns)
        # Round 0: no events (all zeros)
        # Round 1: stabilizer 0 changed
        # Round 2: no change from round 1
        assert events == [(0, 1)]


class TestLowerBoundFidelity:
    """Tests for fidelity lower bound."""

    def test_perfect_fidelities(self) -> None:
        """Perfect fidelities should give bound of 1.0."""
        bound = lower_bound_fidelity(1.0, 1.0)
        assert bound == pytest.approx(1.0)

    def test_formula(self) -> None:
        """Check the formula is applied correctly."""
        # bound = (4/5) * (f1 + f2) - (3/5)
        bound = lower_bound_fidelity(0.9, 0.8)
        expected = (4 / 5) * (0.9 + 0.8) - (3 / 5)
        assert bound == pytest.approx(expected)
