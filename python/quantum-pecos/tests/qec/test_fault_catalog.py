# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for the fault_catalog() public API."""

import pytest
from pecos.quantum import PauliString, TickCircuit
from pecos_rslib_exp import (
    FaultAlternative,
    FaultCatalog,
    FaultLocation,
    depolarizing,
    fault_catalog,
)


def build_h_mz():
    """H(0) MZ(0): single-qubit depolarizing."""
    tc = TickCircuit()
    tc.tick().h([0])
    tick = tc.tick()
    tick.mz([0])
    tc.set_meta("num_measurements", "1")
    tc.set_meta("detectors", "[]")
    tc.set_meta("observables", "[]")
    return tc


def build_cx_mz():
    """CX(0,1) MZ(0) MZ(1): two-qubit depolarizing."""
    tc = TickCircuit()
    tc.tick().cx([(0, 1)])
    tick = tc.tick()
    tick.mz([0])
    tick = tc.tick()
    tick.mz([1])
    tc.set_meta("num_measurements", "2")
    tc.set_meta("detectors", "[]")
    tc.set_meta("observables", "[]")
    return tc


def pauli_terms(pauli):
    """Return a {qubit: label} map from a PECOS PauliString."""
    return {q: str(p).split(".")[-1] for p, q in pauli.get_paulis()}


class TestFaultCatalogStructure:
    def test_returns_fault_catalog(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        assert isinstance(catalog, FaultCatalog)
        assert isinstance(catalog.locations, list)
        assert len(catalog) > 0
        assert isinstance(catalog[0], FaultLocation)
        assert catalog[0] is catalog.locations[0]

    def test_fault_catalog_is_sequence_like(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        assert len(list(catalog)) == len(catalog.locations)
        assert catalog[-1] is catalog.locations[-1]

    def test_location_attributes(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        loc = catalog[0]
        assert hasattr(loc, "tick")
        assert hasattr(loc, "gate_index")
        assert hasattr(loc, "gate_type")
        assert hasattr(loc, "qubits")
        assert hasattr(loc, "channel_probability")
        assert hasattr(loc, "faults")

    def test_fault_alternative_attributes(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        fault = catalog[0].faults[0]
        assert isinstance(fault, FaultAlternative)
        assert hasattr(fault, "kind")
        assert hasattr(fault, "pauli")
        assert hasattr(fault, "detectors")
        assert hasattr(fault, "observables")
        assert hasattr(fault, "tracked_paulis")
        assert hasattr(fault, "measurements")
        assert hasattr(fault, "conditional_probability")
        assert hasattr(fault, "absolute_probability")
        assert hasattr(fault, "channel_probability")


class TestPauliStringOutput:
    def test_pauli_alternatives_are_pauli_string(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        for loc in catalog:
            for fault in loc.faults:
                if fault.kind == "pauli":
                    assert isinstance(fault.pauli, PauliString), f"Expected PauliString, got {type(fault.pauli)}"

    def test_meas_prep_faults_have_none_pauli(self):
        tc = TickCircuit()
        tc.tick().pz([0])
        tick = tc.tick()
        tick.mz([0])
        tc.set_meta("num_measurements", "1")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        noise = depolarizing().p1(0).p2(0).p_meas(0.01).p_prep(0.01)
        catalog = fault_catalog(tc, noise)

        for loc in catalog:
            for fault in loc.faults:
                if fault.kind in ("measurement_flip", "prep_flip"):
                    assert fault.pauli is None

    def test_two_qubit_pauli_has_two_terms(self):
        tc = build_cx_mz()
        noise = depolarizing().p1(0).p2(0.15).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        cx_loc = next(loc for loc in catalog if loc.gate_type == "CX")
        # At least some alternatives should have two Pauli terms (XX, XY, etc.)
        two_term = [f for f in cx_loc.faults if len(f.pauli.get_paulis()) == 2]
        assert len(two_term) == 9, f"Expected 9 two-qubit Paulis, got {len(two_term)}"

    def test_new_gate_pauli_labels_and_measurement_effects(self):
        tc = TickCircuit()
        tc.tick().sx([0])
        tc.tick().szz([(0, 1)])
        tick = tc.tick()
        tick.mz([0])
        tick.mz([1])
        tc.set_meta("num_measurements", "2")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        noise = depolarizing().p1(0.03).p2(0.15).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        sx_loc = next(loc for loc in catalog if loc.gate_type == "SX")
        assert [pauli_terms(f.pauli) for f in sx_loc.faults] == [
            {0: "X"},
            {0: "Y"},
            {0: "Z"},
        ]
        assert any(f.measurements for f in sx_loc.faults)

        szz_loc = next(loc for loc in catalog if loc.gate_type == "SZZ")
        assert len(szz_loc.faults) == 15
        observed = {(terms.get(0, "I"), terms.get(1, "I")) for terms in (pauli_terms(f.pauli) for f in szz_loc.faults)}
        expected = {
            ("X", "I"),
            ("Y", "I"),
            ("Z", "I"),
            ("I", "X"),
            ("I", "Y"),
            ("I", "Z"),
            ("X", "X"),
            ("X", "Y"),
            ("X", "Z"),
            ("Y", "X"),
            ("Y", "Y"),
            ("Y", "Z"),
            ("Z", "X"),
            ("Z", "Y"),
            ("Z", "Z"),
        }
        assert observed == expected
        assert any(f.measurements for f in szz_loc.faults)


class TestNoEffectLocationsIncluded:
    def test_no_downstream_measurement_location_included(self):
        """A gate with p1>0 but no MZ after it still appears in the catalog."""
        tc = TickCircuit()
        tc.tick().h([0])  # No MZ follows — no measurement effect
        tc.set_meta("num_measurements", "0")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        noise = depolarizing().p1(0.01).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        h_locs = [loc for loc in catalog if loc.gate_type == "H"]
        assert len(h_locs) == 1, "H with no downstream MZ should still appear"
        assert len(h_locs[0].faults) == 3
        # All alternatives should have empty effects
        for fault in h_locs[0].faults:
            assert fault.measurements == []
            assert fault.detectors == []
            assert fault.observables == []
            assert abs(fault.absolute_probability - 0.01 / 3) < 1e-10

    def test_prep_fault_with_no_effect_included(self):
        """PZ followed by H then MZ: prep X → H → Z → no flip. Still in catalog."""
        tc = TickCircuit()
        tc.tick().pz([0])
        tc.tick().h([0])
        tick = tc.tick()
        tick.mz([0])
        tc.set_meta("num_measurements", "1")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        noise = depolarizing().p1(0).p2(0).p_meas(0).p_prep(0.005)
        catalog = fault_catalog(tc, noise)

        prep_locs = [loc for loc in catalog if any(f.kind == "prep_flip" for f in loc.faults)]
        assert len(prep_locs) == 1
        fault = prep_locs[0].faults[0]
        assert fault.kind == "prep_flip"
        assert fault.pauli is None
        # Prep X through H becomes Z which doesn't flip MZ → empty
        assert fault.measurements == []


class TestProbabilities:
    def test_structural_catalog_without_noise(self):
        tc = build_h_mz()
        catalog = fault_catalog(tc)

        assert len(catalog.locations) == 2
        assert {loc.channel for loc in catalog.locations} == {"p1", "p_meas"}
        assert all(loc.channel_probability == 0.0 for loc in catalog.locations)
        assert all(loc.no_fault_probability == 1.0 for loc in catalog.locations)
        assert all(fault.absolute_probability == 0.0 for loc in catalog.locations for fault in loc.faults)

    def test_with_noise_updates_existing_python_references(self):
        tc = build_h_mz()
        catalog = fault_catalog(tc)
        h_loc = next(loc for loc in catalog if loc.channel == "p1")
        h_fault = h_loc.faults[0]
        mz_loc = next(loc for loc in catalog if loc.channel == "p_meas")

        catalog.with_noise(p1=0.06, p_meas=0.02)

        assert abs(h_loc.channel_probability - 0.06) < 1e-12
        assert abs(h_loc.no_fault_probability - 0.94) < 1e-12
        assert abs(h_fault.absolute_probability - 0.02) < 1e-12
        assert abs(h_fault.channel_probability - 0.06) < 1e-12
        assert abs(mz_loc.channel_probability - 0.02) < 1e-12
        assert abs(mz_loc.faults[0].absolute_probability - 0.02) < 1e-12

    def test_parameterized_returns_independent_catalog(self):
        tc = build_h_mz()
        catalog = fault_catalog(tc, p1=0.03, p_meas=0.01)
        clone = catalog.parameterized(p1=0.09, p_meas=0.04)

        original_h = next(loc for loc in catalog if loc.channel == "p1")
        clone_h = next(loc for loc in clone if loc.channel == "p1")
        assert abs(original_h.channel_probability - 0.03) < 1e-12
        assert abs(clone_h.channel_probability - 0.09) < 1e-12
        assert original_h is not clone_h

    def test_sparse_channel_keeps_zero_probability_structure(self):
        tc = build_h_mz()
        catalog = fault_catalog(tc, p1=0.0, p_meas=0.02)

        h_loc = next(loc for loc in catalog if loc.channel == "p1")
        mz_loc = next(loc for loc in catalog if loc.channel == "p_meas")
        assert h_loc.channel_probability == 0.0
        assert all(f.absolute_probability == 0.0 for f in h_loc.faults)
        assert abs(mz_loc.channel_probability - 0.02) < 1e-12
        assert abs(mz_loc.faults[0].absolute_probability - 0.02) < 1e-12

    def test_single_qubit_location_fields(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        h_loc = next(loc for loc in catalog if loc.gate_type == "H")
        assert h_loc.channel == "p1"
        assert abs(h_loc.channel_probability - 0.03) < 1e-10
        assert abs(h_loc.no_fault_probability - 0.97) < 1e-10
        assert h_loc.num_alternatives == 3
        for fault in h_loc.faults:
            assert abs(fault.conditional_probability - 1.0 / 3) < 1e-10
            assert abs(fault.absolute_probability - 0.01) < 1e-10

    def test_two_qubit_location_fields(self):
        tc = build_cx_mz()
        noise = depolarizing().p1(0).p2(0.15).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        cx_loc = next(loc for loc in catalog if loc.gate_type == "CX")
        assert cx_loc.channel == "p2"
        assert abs(cx_loc.channel_probability - 0.15) < 1e-10
        assert abs(cx_loc.no_fault_probability - 0.85) < 1e-10
        assert cx_loc.num_alternatives == 15
        for fault in cx_loc.faults:
            assert abs(fault.conditional_probability - 1.0 / 15) < 1e-10
            assert abs(fault.absolute_probability - 0.01) < 1e-10

    def test_full_configuration_probability(self):
        """Compute one full-circuit event probability from catalog fields."""
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0.01).p_prep(0)
        catalog = fault_catalog(tc, noise)

        # Pick first alternative at H location, no fault at MZ location
        h_loc = next(loc for loc in catalog if loc.channel == "p1")
        mz_loc = next(loc for loc in catalog if loc.channel == "p_meas")

        # P(alt 0 at H, no fault at MZ) = (p1/3) * (1 - p_meas)
        config_prob = h_loc.faults[0].absolute_probability * mz_loc.no_fault_probability
        expected = (0.03 / 3) * (1 - 0.01)  # 0.01 * 0.99 = 0.0099
        assert abs(config_prob - expected) < 1e-10


class TestDetectorObservableMapping:
    def test_detectors_are_lists(self):
        tc = TickCircuit()
        tc.tick().h([0])
        tc.tick().cx([(0, 1)])
        tc.tick().h([0])
        tick = tc.tick()
        tick.mz([0])
        tick = tc.tick()
        tick.mz([1])
        tc.set_meta("num_measurements", "2")
        tc.set_meta("detectors", '[{"records": [-2, -1]}]')
        tc.set_meta("observables", "[]")

        noise = depolarizing().p1(0.01).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        has_det = any(f.detectors for loc in catalog for f in loc.faults)
        assert has_det, "Some faults should fire detectors"

        for loc in catalog:
            for fault in loc.faults:
                assert isinstance(fault.detectors, list)
                assert isinstance(fault.observables, list)


class TestFaultConfigurations:
    def test_k0_one_no_fault_event(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0.01).p_prep(0)
        catalog = fault_catalog(tc, noise)

        configs = list(catalog.fault_configurations(0))
        assert len(configs) == 1
        c = configs[0]
        assert c.location_indices == []
        assert c.alternative_indices == []
        assert c.measurements == []
        assert c.detectors == []
        assert c.observables == []
        assert c.selected_probability == 1.0
        # config_prob = product of all no_fault_probability
        expected = 1.0
        for loc in catalog.locations:
            expected *= loc.no_fault_probability
        assert abs(c.configuration_probability - expected) < 1e-12

    def test_k1_exposes_single_fault(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0.01).p_prep(0)
        catalog = fault_catalog(tc, noise)

        configs = list(catalog.fault_configurations(1))
        # Total = sum of num_alternatives
        expected_count = sum(loc.num_alternatives for loc in catalog.locations)
        assert len(configs) == expected_count

        # First config: location 0, alternative 0
        c = configs[0]
        assert c.location_indices == [0]
        assert c.alternative_indices == [0]
        assert c.selected_probability > 0

    def test_k1_skips_zero_probability_structural_locations(self):
        tc = build_h_mz()
        catalog = fault_catalog(tc, p1=0.03, p_meas=0.0)

        assert len(catalog.locations) == 2
        h_idx = next(i for i, loc in enumerate(catalog) if loc.gate_type == "H")
        mz_idx = next(i for i, loc in enumerate(catalog) if loc.gate_type == "MZ")
        assert catalog.locations[mz_idx].channel_probability == 0.0

        configs = list(catalog.fault_configurations(1))
        assert len(configs) == 3
        assert all(c.location_indices == [h_idx] for c in configs)
        assert all(c.selected_probability > 0 for c in configs)
        assert all(mz_idx not in c.location_indices for c in configs)
        assert list(catalog.fault_configurations(2)) == []

    def test_all_zero_noise_only_yields_k0(self):
        tc = build_h_mz()
        catalog = fault_catalog(tc, p1=0.0, p_meas=0.0)

        k0 = list(catalog.fault_configurations(0))
        assert len(k0) == 1
        assert k0[0].configuration_probability == 1.0
        assert list(catalog.fault_configurations(1)) == []

    def test_nonzero_silent_faults_are_yielded(self):
        tc = TickCircuit()
        tc.tick().h([0])
        tc.set_meta("num_measurements", "0")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        catalog = fault_catalog(tc, p1=0.03, p_meas=0.0)
        configs = list(catalog.fault_configurations(1))

        assert len(configs) == 3
        assert all(c.measurements == [] for c in configs)
        assert all(c.detectors == [] for c in configs)
        assert all(c.observables == [] for c in configs)
        assert all(c.selected_probability > 0 for c in configs)

    def test_with_noise_zeroes_channel_for_new_iterators(self):
        tc = build_h_mz()
        catalog = fault_catalog(tc, p1=0.03, p_meas=0.01)

        catalog.with_noise(p1=0.0, p_meas=0.02)

        mz_idx = next(i for i, loc in enumerate(catalog) if loc.gate_type == "MZ")
        configs = list(catalog.fault_configurations(1))
        assert len(configs) == 1
        assert configs[0].location_indices == [mz_idx]
        assert configs[0].selected_probability == pytest.approx(0.02)

    def test_k2_xor_cancels_effects(self):
        """Two faults flipping the same detector XOR-cancel."""
        tc = TickCircuit()
        tc.tick().h([0])
        tc.tick().h([0])
        tick = tc.tick()
        tick.mz([0])
        tc.set_meta("num_measurements", "1")
        tc.set_meta("detectors", '[{"records":[-1]}]')
        tc.set_meta("observables", "[]")

        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        configs = list(catalog.fault_configurations(2))
        # Some configs should have empty detectors (XOR cancel)
        cancelled = [c for c in configs if c.detectors == []]
        assert len(cancelled) > 0

    def test_k2_probability_hand_calc(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0.01).p_prep(0)
        catalog = fault_catalog(tc, noise)
        # 2 locations: H(3 alts, p=0.03) and MZ(1 alt, p=0.01)
        # k=2: both fire. selected = (0.03/3) * (0.01/1) = 0.0001
        # config = 0.0001 (no unselected locations)

        configs = list(catalog.fault_configurations(2))
        assert len(configs) == 3  # 3 H alternatives x 1 MZ alternative
        for c in configs:
            assert abs(c.selected_probability - 0.0001) < 1e-12
            assert abs(c.configuration_probability - 0.0001) < 1e-12

    def test_returns_lazy_iterator_not_list(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0.01).p_prep(0)
        catalog = fault_catalog(tc, noise)

        it = catalog.fault_configurations(1)
        assert not isinstance(it, list), "Should be a lazy iterator, not a list"
        assert hasattr(it, "__next__"), "Should have __next__"
        first = next(it)
        assert hasattr(first, "location_indices")
        assert hasattr(first, "locations")
        assert hasattr(first, "faults")
        assert hasattr(first, "tracked_paulis")

    def test_yielded_locations_and_faults(self):
        tc = build_h_mz()
        noise = depolarizing().p1(0.03).p2(0).p_meas(0.01).p_prep(0)
        catalog = fault_catalog(tc, noise)

        first = next(catalog.fault_configurations(1))
        # .locations should be the FaultLocation objects for selected indices
        assert len(first.locations) == 1
        assert first.locations[0] is catalog.locations[first.location_indices[0]]
        # .faults should be the FaultAlternative objects
        assert len(first.faults) == 1
        loc = catalog.locations[first.location_indices[0]]
        assert first.faults[0] is loc.faults[first.alternative_indices[0]]

    def test_tracked_paulis_are_distinct_from_observables(self):
        tc = TickCircuit()
        tc.tick().h([0])
        tc.tracked_pauli(PauliString.from_str("Z"), label="tracked_z")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        noise = depolarizing().p1(0.03).p2(0).p_meas(0).p_prep(0)
        catalog = fault_catalog(tc, noise)

        h_loc = next(loc for loc in catalog if loc.gate_type == "H")
        tracked = [fault.tracked_paulis for fault in h_loc.faults]
        assert tracked.count([0]) == 2
        assert tracked.count([]) == 1
        assert all(fault.observables == [] for fault in h_loc.faults)

        configs = list(catalog.fault_configurations(1))
        assert any(c.tracked_paulis == [0] and c.observables == [] for c in configs)
