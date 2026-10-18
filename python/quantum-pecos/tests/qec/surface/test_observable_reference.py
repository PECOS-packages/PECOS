# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Noiseless observable flips, with independent signed and sampled oracles."""

import json
import re
from itertools import product
from types import SimpleNamespace

import pytest
import stim
from pecos.qec import MissingObservableReferenceError, empirical_correlation_table, fit_dem_from_simulation
from pecos.qec._replay import _replay_tick_circuit
from pecos.qec.surface import (
    LogicalCircuitBuilder,
    SurfacePatch,
    build_memory_circuit,
    extract_detection_events_and_observables,
    generate_tick_circuit_from_patch,
    get_observable_descriptors_from_tick_circuit,
)
from pecos.qec.surface._twirl_config import TwirlConfig
from pecos.qec.surface.circuit_builder import (
    CircuitStep,
    OpType,
    TickCircuitRenderer,
    build_surface_code_circuit,
    tick_circuit_to_stim,
)
from pecos.qec.surface.logical_circuit import LogicalGateType
from pecos.testing import simulate_tick_circuit
from pecos_rslib.quantum import TickCircuit
from pecos_rslib_exp import depolarizing


def definition_probe(observables):
    """Exercise the helper's validation independently of the reader's JSON schema."""
    return SimpleNamespace(
        circuit_definitions=lambda: {"observables": observables, "detectors": [], "num_measurements": 1},
    )


def single_measurement_circuit():
    tc = TickCircuit()
    tc.tick().mz_with_ids([0], [0])
    return tc


class ReferenceProbe(LogicalCircuitBuilder):
    """Expose lowered operations for the independent single-patch oracle."""

    def lowered_operations(self):
        return self._lowered_operations()


GATE_WORDS = [word for length in range(4) for word in product(("SZ", "SZdg", "H"), repeat=length)]
NAMED_CASES = [
    ("X", ("SZ", "SZ"), "X", 1),
    ("X", ("SZ", "SZdg"), "X", 0),
    ("X", ("SZ", "H", "SZdg"), "X", 1),
    ("Z", ("SZdg",), "Z", 0),
    ("X", ("SZdg",), "Y", 1),
    ("X", ("SZ",), "Y", 0),
]


def reference_builder(preparation, word, readout, distance=3, patch_count=1):
    """Build one or two patches; the two-patch sweep always crosses a CX."""
    builder = ReferenceProbe()
    patch = SurfacePatch.create(distance)
    labels = ["A", "B"][:patch_count]
    for index, label in enumerate(labels):
        builder.add_patch(patch, label, qubit_offset=index * patch.num_qubits)
    builder.add_memory(labels, 1, preparation)
    if patch_count == 2:
        builder.add_transversal_cx("A", "B")
    for gate in word:
        if gate == "H":
            builder.add_transversal_h("A")
        else:
            builder.add_logical_sz("A", dagger=gate == "SZdg")
    builder.add_memory(labels, 1, {"A": readout, "B": preparation})
    return builder


def signed_reference(builder):
    """Conjugate the prepared Pauli forward, including the lowered Y-readout fold once."""
    operations = builder.lowered_operations()
    axis = operations[0].basis
    sign = 0
    images = {
        "SZ": {"X": ("Y", 0), "Y": ("X", 1), "Z": ("Z", 0)},
        "SZdg": {"X": ("Y", 1), "Y": ("X", 0), "Z": ("Z", 0)},
        "H": {"X": ("Z", 0), "Y": ("Y", 1), "Z": ("X", 0)},
    }
    for operation in operations[1:-1]:
        gate = "H" if operation.gate_type == LogicalGateType.TRANSVERSAL_H else operation.fold
        axis, phase = images[gate][axis]
        sign ^= phase
    readout = operations[-1].per_patch_basis.get("A", operations[-1].basis)
    return sign if axis == readout else None


def assert_clean_flips(tc):
    """Check replay and independently sampled rows against Stim's relative sampler."""
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    rows = circuit.compile_sampler(seed=123).sample(16)
    replay, fired, raw = simulate_tick_circuit(tc, seed=19)
    observables = json.loads(tc.get_meta("observables"))
    assert raw == {obs["id"]: obs["reference"] for obs in observables}
    assert fired == 0
    events, flips = extract_detection_events_and_observables(tc, [replay, *rows])
    assert events == [[] for _ in range(17)]
    assert flips == [[] for _ in range(17)]
    detectors, relative = circuit.compile_detector_sampler(seed=456).sample(16, separate_observables=True)
    assert not detectors.any()
    assert not relative.any()


@pytest.mark.parametrize("patch_count", [1, 2])
@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("preparation", ["X", "Z"])
@pytest.mark.parametrize("word", GATE_WORDS, ids=lambda word: "-".join(word) or "memory")
@pytest.mark.parametrize("readout", ["X", "Y", "Z"])
def test_noiseless_reference_sweep(preparation, word, readout, distance, patch_count):
    builder = reference_builder(preparation, word, readout, distance, patch_count)
    tc = builder.to_tick_circuit()
    if patch_count == 1:
        expected = signed_reference(builder)
        observables = json.loads(tc.get_meta("observables"))
        # Assert determinism in both directions, including suppressed readouts.
        assert bool(observables) == (expected is not None)
        if expected is not None:
            assert observables[0]["reference"] == expected
    assert_clean_flips(tc)


@pytest.mark.parametrize(("preparation", "word", "readout", "reference"), NAMED_CASES)
def test_named_references(preparation, word, readout, reference):
    builder = reference_builder(preparation, word, readout)
    assert signed_reference(builder) == reference
    tc = builder.to_tick_circuit()
    assert json.loads(tc.get_meta("observables"))[0]["reference"] == reference
    assert_clean_flips(tc)


def test_distinct_references_stay_with_their_observables():
    tc = reference_builder("X", ("SZ", "SZ"), "X", patch_count=2).to_tick_circuit()
    observables = json.loads(tc.get_meta("observables"))
    assert [(obs["id"], obs["reference"]) for obs in observables] == [(0, 1), (1, 0)]
    row, _, _ = simulate_tick_circuit(tc)
    assert extract_detection_events_and_observables(tc, [row])[1] == [[]]
    for obs in observables:
        changed = row.copy()
        changed[obs["records"][0]] ^= 1
        assert extract_detection_events_and_observables(tc, [changed])[1] == [[obs["id"]]]


def sparse_observable_circuit():
    """Keep observable 1 while the non-deterministic observable 0 is suppressed."""
    builder = ReferenceProbe()
    patch = SurfacePatch.create(3)
    builder.add_patch(patch, "A")
    builder.add_patch(patch, "B", qubit_offset=patch.num_qubits)
    builder.add_memory(["A", "B"], 1, {"A": "Z", "B": "X"})
    builder.add_logical_sz("B")
    builder.add_logical_sz("B")
    builder.add_memory(["A", "B"], 1, "X")
    return builder.to_tick_circuit()


def test_sparse_observable_id():
    tc = sparse_observable_circuit()
    observables = json.loads(tc.get_meta("observables"))
    assert [(obs["id"], obs["reference"]) for obs in observables] == [(1, 1)]
    row, _, _ = simulate_tick_circuit(tc)
    assert extract_detection_events_and_observables(tc, [row])[1] == [[]]
    row[observables[0]["records"][0]] ^= 1
    assert extract_detection_events_and_observables(tc, [row])[1] == [[1]]


@pytest.mark.parametrize("rows", [[], [[0]]])
def test_missing_reference_names_entry(rows):
    tc = single_measurement_circuit()
    tc.set_meta("observables", json.dumps([{"id": 7, "records": [-1]}]))
    with pytest.raises(
        MissingObservableReferenceError,
        match=(
            r"entry 0 \(id=7\).*has no 'reference'.*flip cannot be derived from its raw parity.*"
            r"LogicalCircuitBuilder.to_tick_circuit.*TickCircuitRenderer.*emit it"
        ),
    ):
        list(extract_detection_events_and_observables(tc, rows)[1])


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_memory_reference(basis, distance):
    tc = build_memory_circuit(distance=distance, rounds=2, basis=basis)
    assert json.loads(tc.get_meta("observables"))[0]["reference"] == 0
    assert_clean_flips(tc)


@pytest.mark.parametrize("reference", [None, 0, 1])
@pytest.mark.parametrize("cached", [False, True])
def test_descriptors_do_not_project_reference(reference, cached):
    patch = SurfacePatch.create(3)
    tc = reference_builder("X", ("SZ", "SZ"), "X").to_tick_circuit()
    observables = json.loads(tc.get_meta("observables"))
    if reference is None:
        del observables[0]["reference"]
    else:
        observables[0]["reference"] = reference
    tc.set_meta("observables", json.dumps(observables))
    if cached:
        get_observable_descriptors_from_tick_circuit(tc, patch)
    descriptor = get_observable_descriptors_from_tick_circuit(tc, patch)[0]
    assert "reference" not in descriptor


@pytest.mark.parametrize("field", ["records", "meas_ids"])
def test_extraction_metadata_reference_formats(field):
    tc = single_measurement_circuit()
    tc.set_meta("detectors", "[]")
    tc.set_meta("observables", json.dumps([{"id": 4, field: [-1] if field == "records" else [0], "reference": 1}]))
    assert extract_detection_events_and_observables(tc, [[1], [0]])[1] == [[], [4]]


def test_memory_renderer_computes_reference_from_gates():
    """A deterministic physical X before terminal readout rules out a hardcoded zero."""
    patch = SurfacePatch.create(3)
    steps, allocation = build_surface_code_circuit(patch, 1, "Z")
    final = next(i for i, step in enumerate(steps) if step.label == "final[0]")
    steps.insert(final, CircuitStep(OpType.X, [allocation.data_qubits[0]]))
    tc = TickCircuitRenderer().render(steps, allocation, patch, 1, "Z")
    assert json.loads(tc.get_meta("observables"))[0]["reference"] == 1
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    rows = circuit.compile_sampler(seed=123).sample(16)
    assert extract_detection_events_and_observables(tc, rows)[1] == [[] for _ in range(16)]


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"ancilla_budget": 1}, id="reused-ancilla"),
        pytest.param({"interaction_basis": "szz"}, id="szz"),
        pytest.param({"interaction_basis": "szz", "szz_physical_prefixes": True}, id="physical-prefixes"),
        pytest.param({"twirl": TwirlConfig()}, id="round-twirl"),
        pytest.param({"twirl": TwirlConfig(site_schedule="before_two_qubit_gate")}, id="gate-twirl"),
    ],
)
def test_memory_reference_variants(basis, options):
    tc = generate_tick_circuit_from_patch(SurfacePatch.create(3), 2, basis, **options)
    assert json.loads(tc.get_meta("observables"))[0]["reference"] == 0
    assert_clean_flips(tc)


@pytest.mark.parametrize("consumer", ["logical", "memory", "signed"])
@pytest.mark.parametrize("count_delta", [-1, 1], ids=["short", "long"])
def test_replay_measurement_count_mismatch(monkeypatch, consumer, count_delta):
    """Both producers and the signed accessor reject a truncated or extended replay."""
    builder = reference_builder("X", ("SZ", "SZ"), "X")
    if consumer == "signed":
        tc = builder.to_tick_circuit()

    def replay_wrong_count(circuit, num_ticks, seed):
        sim, measurements = _replay_tick_circuit(circuit, num_ticks, seed)
        if count_delta < 0:
            measurements.pop()
        else:
            measurements.append(0)
        return sim, measurements

    build = {
        "logical": builder.to_tick_circuit,
        "memory": lambda: build_memory_circuit(distance=3, rounds=1),
        "signed": lambda: simulate_tick_circuit(tc),
    }[consumer]

    target = "pecos.testing" if consumer == "signed" else "pecos.qec._replay"
    monkeypatch.setattr(f"{target}._replay_tick_circuit", replay_wrong_count)
    source = "definitions" if consumer == "signed" else "metadata"
    with pytest.raises(ValueError, match=rf"^Replay measurement count disagrees with circuit {source}$"):
        build()


@pytest.mark.parametrize("rows", [[], [[0]]])
def test_missing_reference_and_id_names_entry(rows):
    tc = definition_probe([{"id": 0, "measurements": [0], "reference": 0}, {"measurements": [0]}])
    with pytest.raises(
        MissingObservableReferenceError,
        match=r"^observable entry 1 has no 'reference'.*flip cannot be derived from its raw parity",
    ):
        list(extract_detection_events_and_observables(tc, rows)[1])


@pytest.mark.parametrize("backend", ["stabilizer", "meas_sampling"])
def test_analysis_reference_one_is_clean(backend):
    tc = reference_builder("X", ("SZ", "SZ"), "X").to_tick_circuit()
    table = dict(empirical_correlation_table(tc, depolarizing(), shots=64, backend=backend))
    assert table.get(("L0",), 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("field", ["records", "meas_ids"])
@pytest.mark.parametrize("order", [1, 2], ids=["marginal", "joint"])
def test_analysis_sparse_observable_id(field, order):
    tc = sparse_observable_circuit()
    # Keep each supported metadata format independently, for both kinds of entry.
    for key in ("detectors", "observables"):
        entries = json.loads(tc.get_meta(key))
        for entry in entries:
            del entry["meas_ids" if field == "records" else "records"]
        tc.set_meta(key, json.dumps(entries))
    table = dict(empirical_correlation_table(tc, depolarizing().p_meas(0.1), shots=256))
    assert {key[-1] for key in table if len(key) == order and key[-1].startswith("L")} == {"L1"}


@pytest.mark.parametrize("rows", [[], [[0]], [[1]]])
def test_missing_id_names_entry(rows):
    tc = definition_probe([{"measurements": [0], "reference": 0}])
    with pytest.raises(MissingObservableReferenceError, match=r"observable entry 0 is missing 'id'"):
        list(extract_detection_events_and_observables(tc, rows)[1])


@pytest.mark.parametrize("reference", [2, "0", 0.0, True])
def test_invalid_reference_names_value(reference):
    tc = definition_probe([{"id": 7, "measurements": [0], "reference": reference}])
    pattern = rf"observable entry 0 \(id=7\).*invalid 'reference' {re.escape(repr(reference))};"
    with pytest.raises(MissingObservableReferenceError, match=pattern):
        list(extract_detection_events_and_observables(tc, [])[1])


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("rounds", [0, 2])
@pytest.mark.parametrize(
    "options",
    [
        *[
            pytest.param(
                {"check_plan": plan, **({"ancilla_budget": 2} if "round_order" in plan else {})},
                id=plan,
            )
            for plan in (
                "cx_standard_v1",
                "cx_balanced_data_v1",
                "szz_current_v1",
                "szz_boundary_first_v1",
                "szz_balanced_data_v1",
                "szz_boundary_first_balanced_data_v1",
                "szz_balanced_data_round_order_1032_v1",
                "szz_balanced_data_round_order_3102_v1",
            )
        ],
        *[
            pytest.param({"interaction_basis": "szz", "clifford_frame_policy": policy}, id=policy)
            for policy in ("global_axis_cycle_f", "checkerboard_xzzx", "checkerboard_zxxz")
        ],
        *[
            pytest.param({"interaction_basis": interaction, "ancilla_budget": 1}, id=f"{interaction}-reused")
            for interaction in ("cx", "szz")
        ],
        *[
            pytest.param(
                {"twirl": TwirlConfig(site_schedule=schedule)},
                id=f"cx-{schedule}",
            )
            for schedule in ("between_rounds", "before_two_qubit_gate")
        ],
        pytest.param({"interaction_basis": "szz", "szz_physical_prefixes": True}, id="physical-prefixes"),
    ],
)
def test_memory_observable_determinism_stim_oracle(basis, distance, rounds, options):
    """Stim's exact backward analysis proves the memory readout is determined."""
    tc = generate_tick_circuit_from_patch(SurfacePatch.create(distance), rounds, basis, **options)
    stim.Circuit(tick_circuit_to_stim(tc)).detector_error_model(allow_gauge_detectors=False)


@pytest.mark.parametrize("consumer", ["logical", "memory"])
@pytest.mark.parametrize("has_observables", [False, True])
def test_reference_replay_count(monkeypatch, consumer, has_observables):
    calls = []

    def replay(circuit, num_ticks, seed):
        calls.append(seed)
        return _replay_tick_circuit(circuit, num_ticks, seed)

    monkeypatch.setattr("pecos.qec._replay._replay_tick_circuit", replay)
    if consumer == "logical":
        reference_builder("X", (), "X" if has_observables else "Z").to_tick_circuit()
    else:
        generate_tick_circuit_from_patch(SurfacePatch.create(3), 1, add_detectors=has_observables)
    assert len(calls) == int(has_observables)


@pytest.mark.parametrize("observable_id", ["a", None, 1.5, True])
@pytest.mark.parametrize("rows", [[], [[1]]])
def test_invalid_observable_id(observable_id, rows):
    tc = definition_probe([{"id": observable_id, "measurements": [0], "reference": 0}])
    pattern = rf"observable entry 0.*invalid 'id' {re.escape(repr(observable_id))}; expected an integer"
    with pytest.raises(MissingObservableReferenceError, match=pattern):
        list(extract_detection_events_and_observables(tc, rows)[1])


@pytest.mark.parametrize("rows", [[], [[1]]])
def test_duplicate_observable_id(rows):
    tc = definition_probe([{"id": 7, "measurements": [0], "reference": 0}] * 2)
    with pytest.raises(MissingObservableReferenceError, match=r"observable entry 1 has duplicate 'id' 7"):
        list(extract_detection_events_and_observables(tc, rows)[1])


@pytest.mark.parametrize("reference", [{}, {"reference": None}])
@pytest.mark.parametrize("rows", [[], [[0]]])
def test_helper_missing_or_null_reference(reference, rows):
    tc = definition_probe([{"id": 7, "measurements": [0], **reference}])
    with pytest.raises(MissingObservableReferenceError, match=r"entry 0 \(id=7\) has no 'reference'"):
        list(extract_detection_events_and_observables(tc, rows)[1])


@pytest.mark.parametrize("field", ["records", "meas_ids"])
def test_signed_oracle_record_formats(field):
    tc = sparse_observable_circuit()
    row, _, _ = simulate_tick_circuit(tc)
    # Use nonzero detector and signed observable parity so skipping either loop fails.
    record = row.index(1)
    offsets = [record - len(row)] if field == "records" else [record]
    tc.set_meta("detectors", json.dumps([{"id": 0, field: offsets}]))
    tc.set_meta("observables", json.dumps([{"id": 7, field: offsets, "reference": 1}]))
    assert simulate_tick_circuit(tc) == (row, 1, {7: 1})


@pytest.mark.parametrize("backend", ["stabilizer", "meas_sampling"])
def test_fit_dem_record_formats(backend):
    tc = generate_tick_circuit_from_patch(SurfacePatch.create(3), 1, add_typed_annotations=False)
    noise = depolarizing().p_meas(0.1)
    expected = fit_dem_from_simulation(tc, noise, shots=256, backend=backend)
    num_meas = int(tc.get_meta("num_measurements"))
    for key in ("detectors", "observables"):
        entries = json.loads(tc.get_meta(key))
        for entry in entries:
            entry["meas_ids"] = [num_meas + rec for rec in entry.pop("records")]
        tc.set_meta(key, json.dumps(entries))
    assert fit_dem_from_simulation(tc, noise, shots=256, backend=backend) == expected


@pytest.mark.parametrize("producer", ["logical", "memory"])
@pytest.mark.parametrize("reference", [0, 1])
def test_reader_round_trip_reference(producer, reference):
    if producer == "logical":
        word = ("SZ", "SZ") if reference else ("SZ", "SZdg")
        tc = reference_builder("X", word, "X").to_tick_circuit()
    else:
        patch = SurfacePatch.create(3)
        steps, allocation = build_surface_code_circuit(patch, 1, "Z")
        if reference:
            final = next(i for i, step in enumerate(steps) if step.label == "final[0]")
            steps.insert(final, CircuitStep(OpType.X, [allocation.data_qubits[0]]))
        tc = TickCircuitRenderer().render(steps, allocation, patch, 1, "Z")
    metadata = json.loads(tc.get_meta("observables"))
    resolved = tc.circuit_definitions()["observables"]
    assert {obs["id"]: obs["reference"] for obs in metadata} == {0: reference}
    assert {obs["id"]: obs["reference"] for obs in resolved} == {0: reference}


@pytest.mark.parametrize("metadata_reference", [{}, {"reference": None}])
def test_reader_absent_reference(metadata_reference):
    tc = TickCircuit()
    tc.tick().mz([0])
    tc.set_meta("observables", json.dumps([{"id": 7, "records": [-1], **metadata_reference}]))
    assert tc.circuit_definitions()["observables"][0]["reference"] is None
    with pytest.raises(MissingObservableReferenceError, match=r"entry 0 \(id=7\) has no 'reference'"):
        list(extract_detection_events_and_observables(tc, [])[1])


@pytest.mark.parametrize("reference", [2, -1, 1.5, "1", True])
def test_reader_rejects_invalid_reference(reference):
    tc = TickCircuit()
    tc.tick().mz([0])
    tc.set_meta("observables", json.dumps([{"id": 7, "records": [-1], "reference": reference}]))
    with pytest.raises(ValueError, match=rf"observable 7.*invalid 'reference' {re.escape(json.dumps(reference))}"):
        tc.circuit_definitions()


@pytest.mark.parametrize("consumer", ["dem", "sampler"])
def test_rust_fault_effects_ignore_reference(consumer):
    from pecos_rslib.qec import DemSampler, DetectorErrorModel

    tc = reference_builder("X", ("SZ", "SZ"), "X").to_tick_circuit()
    metadata = json.loads(tc.get_meta("observables"))
    assert metadata[0]["reference"] == 1
    noise = {"p1": 0.01, "p2": 0.02, "p_meas": 0.03}

    def effects():
        if consumer == "dem":
            return DetectorErrorModel.from_circuit(tc, **noise).to_string()
        sampler = DemSampler.from_circuit(tc, **noise)
        return [sampler.sample(seed=seed) for seed in range(32)]

    expected = effects()
    metadata[0]["reference"] = 0
    tc.set_meta("observables", json.dumps(metadata))
    assert effects() == expected


@pytest.mark.parametrize("reference", [None, 0, 1])
@pytest.mark.parametrize("consumer", ["json", "reader"])
def test_native_topology_reference_round_trip(monkeypatch, reference, consumer):
    from pecos.qec.surface import NoiseParameters, decode
    from pecos.qec.surface.decode import (
        _build_native_sampler_from_cached_surface_topology,
        _surface_native_topology,
        _surface_patch_cache_key,
    )

    tc = single_measurement_circuit()
    entry = {"id": 0, "records": [-1], "label": "logical"}
    if reference is not None:
        entry["reference"] = reference
    tc.set_meta("observables", json.dumps([entry]))
    monkeypatch.setattr(decode, "_build_surface_tick_circuit_for_native_model", lambda *_args, **_kwargs: tc)
    topology = _surface_native_topology(
        _surface_patch_cache_key(SurfacePatch.create(3)),
        1,
        "Z",
        None,
        "abstract",
        False,
    )
    sampler = _build_native_sampler_from_cached_surface_topology(topology, NoiseParameters(p_meas=0.1))
    if consumer == "json":
        assert json.loads(sampler.observables_json) == [entry]
    else:
        tc.set_meta("observables", sampler.observables_json)
        assert tc.circuit_definitions()["observables"][0]["reference"] == reference


@pytest.mark.parametrize("reference", [None, 0, 1])
def test_typed_observable_reference_round_trip(reference):
    tc = single_measurement_circuit()
    tc.add_observable([-1], observable_id=7, reference=reference)
    assert tc.circuit_definitions()["observables"][0]["reference"] == reference


@pytest.mark.parametrize("reference", [-1, 2, 256, 0.0, 1.5, "1", True, False])
def test_typed_observable_rejects_invalid_reference(reference):
    tc = single_measurement_circuit()
    with pytest.raises(ValueError, match="reference must be integer 0 or 1 or absent"):
        tc.add_observable([-1], reference=reference)
    assert tc.get_meta("observables") is None


@pytest.mark.parametrize("backend", ["stabilizer", "meas_sampling"])
def test_detector_only_analysis_without_reference(backend):
    tc = generate_tick_circuit_from_patch(SurfacePatch.create(3), 1, add_typed_annotations=False)
    noise = depolarizing().p_meas(0.1)
    expected = fit_dem_from_simulation(tc, noise, shots=64, backend=backend)
    entries = json.loads(tc.get_meta("observables"))
    for entry in entries:
        del entry["reference"]
    tc.set_meta("observables", json.dumps(entries))
    assert fit_dem_from_simulation(tc, noise, shots=64, backend=backend) == expected
    with pytest.raises(MissingObservableReferenceError, match="has no 'reference'"):
        empirical_correlation_table(tc, noise, shots=64, backend=backend)


def test_lazy_flips_snapshot_and_reuse():
    tc = single_measurement_circuit()
    tc.set_meta("observables", '[{"id":7,"records":[-1],"reference":1}]')

    def rows():
        row = [1]
        yield row
        row[0] = 0
        yield row

    _, flips = extract_detection_events_and_observables(tc, rows())
    assert list(flips) == [[], [7]]
    assert list(flips) == [[], [7]]
    assert flips[1:] == [[7]]


@pytest.mark.parametrize("producer", ["bb", "coloration"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_css_memory_reference_round_trip(producer, basis):
    from pecos.qec import bb_memory_circuit, coloration_memory_circuit
    from pecos.quantum import ParityCheckMatrix

    if producer == "bb":
        tc = bb_memory_circuit(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 2, basis)
    else:
        checks = ParityCheckMatrix(
            [
                [1, 0, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1],
            ],
        )
        tc = coloration_memory_circuit(checks, checks, 2, basis)
    definitions = tc.circuit_definitions()["observables"]
    assert {obs["reference"] for obs in definitions} == {0}
    # Independent physical sampling checks the positive-eigenvalue derivation,
    # including both bases and the random opposite-basis syndrome projections.
    circuit = stim.Circuit()
    names = {"PX": "RX", "PZ": "R", "CX": "CX", "MX": "MX", "MZ": "M", "Idle": "I", "X": "X", "Z": "Z"}
    for _, gate in tc.gate_batches():
        circuit.append(names[gate.gate_type.name], [int(q) for q in gate.qubits])
    rows = circuit.compile_sampler(seed=123).sample(32)
    for obs in definitions:
        assert all(sum(int(row[position]) for position in obs["measurements"]) % 2 == 0 for row in rows)
    assert extract_detection_events_and_observables(tc, rows)[1] == [[] for _ in rows]
