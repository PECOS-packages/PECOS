# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fault-count masses, conditional sampling, and count-only estimation."""

import math
import sys

import pytest
from pecos.quantum import TickCircuit
from pecos_rslib_exp import FaultCountPmf, StratifiedEstimate, fault_catalog


def catalog():
    """Several active probability groups and inactive preparation locations."""
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().h([0])
    circuit.tick().cx([(0, 1)])
    circuit.tick().mz([0, 1])
    circuit.set_meta("num_measurements", "2")
    circuit.set_meta("detectors", "[]")
    circuit.set_meta("observables", "[]")
    return fault_catalog(circuit, p1=0.09, p2=0.17, p_meas=0.12, p_prep=0)


def signature(config):
    return (
        config.location_indices,
        config.alternative_indices,
        config.measurements,
        config.detectors,
        config.observables,
        config.tracked_paulis,
        config.selected_probability,
        config.configuration_probability,
    )


def test_pmf_matches_enumeration():
    cat = catalog()
    pmf = cat.fault_count_pmf(3)
    assert isinstance(pmf, FaultCountPmf)
    for k, mass in enumerate(pmf.masses):
        expected = sum(c.configuration_probability for c in cat.fault_configurations(k))
        assert mass == pytest.approx(expected, rel=1e-12, abs=0)
        assert math.exp(pmf.log_masses[k]) == pytest.approx(expected, rel=1e-12, abs=0)
    assert 0 < pmf.tail_bound < 1


def test_conditional_samples_and_seed():
    cat = catalog()
    for k in (0, 1, 2, 3):
        expected = {
            (tuple(c.location_indices), tuple(c.alternative_indices)): signature(c) for c in cat.fault_configurations(k)
        }
        first = cat.sample_fault_configurations(k, 200, 12)
        second = cat.sample_fault_configurations(k, 200, 12)
        assert [signature(c) for c in first] == [signature(c) for c in second]
        for config in first:
            assert len(config.location_indices) == k
            assert len(set(config.location_indices)) == k
            assert signature(config) == expected[(tuple(config.location_indices), tuple(config.alternative_indices))]
            assert config.locations == [cat.locations[i] for i in config.location_indices]
            assert config.faults == [
                cat.locations[i].faults[a]
                for i, a in zip(config.location_indices, config.alternative_indices, strict=True)
            ]
    assert [signature(c) for c in cat.sample_fault_configurations(2, 50, 1)] != [
        signature(c) for c in cat.sample_fault_configurations(2, 50, 2)
    ]


def test_estimator_arithmetic():
    pmf = catalog().fault_count_pmf(3)
    counts = [(0, 100, 80, 10), (2, 200, 120, 30), (3, 50, 25, 0)]
    result = pmf.stratified_estimate(counts)
    assert isinstance(result, StratifiedEstimate)
    a = sum(pmf.masses[k] * f / n for k, n, s, f in counts)
    b = sum(pmf.masses[k] * s / n for k, n, s, f in counts)
    r = a / b
    va = sum(pmf.masses[k] ** 2 * (f / n) * (1 - f / n) / n for k, n, s, f in counts)
    vb = sum(pmf.masses[k] ** 2 * (s / n) * (1 - s / n) / n for k, n, s, f in counts)
    vr = sum(
        pmf.masses[k] ** 2 * ((f * (1 - r) ** 2 + (s - f) * r**2) / n - (f / n - r * s / n) ** 2) / n / b**2
        for k, n, s, f in counts
    )
    assert result.failure_probability == pytest.approx(a)
    assert result.survival_probability == pytest.approx(b)
    assert result.failure_given_survival == pytest.approx(r)
    assert result.failure_standard_error == pytest.approx(math.sqrt(va))
    assert result.survival_standard_error == pytest.approx(math.sqrt(vb))
    assert result.ratio_standard_error == pytest.approx(math.sqrt(vr))
    assert result.unsampled_mass == pytest.approx(pmf.masses[1])
    assert result.tail_bound == pmf.tail_bound
    empty = pmf.stratified_estimate([(0, 10, 0, 0)])
    assert empty.failure_given_survival is None
    assert empty.ratio_standard_error is None


@pytest.mark.parametrize(
    ("probability", "message"),
    [
        (-0.1, "finite and non-negative"),
        (float("nan"), "finite and non-negative"),
        (float("inf"), "finite and non-negative"),
        (1.0, ">= 1"),
    ],
)
def test_catalog_probability_errors(probability, message):
    cat = catalog().parameterized(p1=probability)
    with pytest.raises(ValueError, match=message):
        cat.fault_count_pmf(0)
    with pytest.raises(ValueError, match=message):
        cat.sample_fault_configurations(0, 0, 1)


def test_count_errors():
    cat = catalog()
    with pytest.raises(ValueError, match="active locations"):
        cat.fault_count_pmf(100)
    with pytest.raises(ValueError, match="active locations"):
        cat.sample_fault_configurations(100, 10, 0)
    with pytest.raises(ValueError, match="allocate"):
        cat.sample_fault_configurations(1, sys.maxsize, 0)
    pmf = cat.fault_count_pmf(2)
    for counts, message in (
        ([(0, 0, 0, 0)], "invalid counts"),
        ([(0, 10, 5, 6)], "invalid counts"),
        ([(0, 10, 11, 0)], "invalid counts"),
        ([(3, 10, 10, 1)], "outside the supplied PMF"),
        ([(0, 10, 10, 1), (0, 10, 10, 1)], "duplicate counts"),
    ):
        with pytest.raises(ValueError, match=message):
            pmf.stratified_estimate(counts)
