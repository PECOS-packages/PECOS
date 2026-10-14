# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""EEG metadata keeps declared ids and resolves both reference forms."""

import json
import re

import pytest
from pecos_rslib.quantum import TickCircuit
from pecos_rslib_exp import (
    correlation_matching_dem,
    exact_correlation_table,
    exact_detection_rates,
    noise_characterization,
)


def _circuit(ids=(5, 2, 3), refs="records", stamped=(0, 1)):
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    for qubits in ([0, 1], [0, 1], [1], [1]):
        circuit.tick().h(qubits)
    circuit.tick().mz_with_ids([0], [stamped[0]])
    circuit.tick().mz_with_ids([1], [stamped[1]])
    circuit.set_meta("num_measurements", "2")
    values = (-2, -1) if refs == "records" else stamped
    circuit.set_meta(
        "detectors",
        json.dumps(
            [
                {"id": ids[0], refs: [values[0]]},
                {"id": ids[1], refs: [values[1]]},
            ],
        ),
    )
    circuit.set_meta("observables", json.dumps([{"id": ids[2], refs: [values[1]]}]))
    return circuit


def _dem_events(dem, *, relabel=False):
    events = {}
    labels = {"D0": "D5", "D1": "D2", "L0": "L3"}
    for line in dem.splitlines():
        match = re.fullmatch(r"error\(([^)]+)\) (.*)", line)
        if match:
            targets = match[2].split()
            if relabel:
                targets = [labels.get(target, target) for target in targets]
            key = tuple(sorted(targets))
            assert key not in events
            events[key] = float(match[1])
    return events


# MUST FAIL BEFORE: definitions used list positions.
def test_noise_definitions_keep_declared_ids():
    data, _, _ = noise_characterization(_circuit(), p1=0.05, prune=0.0)
    definitions = json.loads(data)["definitions"]
    assert [item["label"] for item in definitions] == ["D2", "D5", "L3"]
    assert {item["label"]: item["records"] for item in definitions} == {
        "D5": [-2],
        "D2": [-1],
        "L3": [-1],
    }


# MUST FAIL BEFORE: sparse ids dropped exact marginals from the fit.
@pytest.mark.parametrize("compress", [False, True])
def test_noise_dem_probabilities_keep_declared_ids(compress):
    _, raw, decomposed = noise_characterization(_circuit(), p1=0.05, prune=0.0, compress=compress)
    _, baseline, baseline_decomposed = noise_characterization(
        _circuit((0, 1, 0)),
        p1=0.05,
        prune=0.0,
        compress=compress,
    )
    assert _dem_events(baseline)
    assert _dem_events(raw) == pytest.approx(_dem_events(baseline, relabel=True))
    assert _dem_events(decomposed) == pytest.approx(_dem_events(baseline_decomposed, relabel=True))


# MUST FAIL BEFORE: matching DEM iterated counts instead of ids.
def test_matching_dem_keeps_declared_ids():
    baseline = correlation_matching_dem(_circuit((0, 1, 0)), p1=0.05, prune=0.0)
    actual = correlation_matching_dem(_circuit(), p1=0.05, prune=0.0)
    assert _dem_events(baseline)
    assert _dem_events(actual) == pytest.approx(_dem_events(baseline, relabel=True))


# MUST FAIL BEFORE: stamped meas_ids were lost and yielded identity stabilizers.
@pytest.mark.parametrize("stamped", [(0, 1), (17, 9)])
def test_meas_ids_resolve_like_records(stamped):
    actual = _circuit(refs="meas_ids", stamped=stamped)
    expected = _circuit()
    assert dict(exact_detection_rates(actual, p1=0.05, prune=0.0)) == pytest.approx(
        dict(exact_detection_rates(expected, p1=0.05, prune=0.0)),
    )
    _, raw, _ = noise_characterization(actual, p1=0.05, prune=0.0)
    _, baseline, _ = noise_characterization(expected, p1=0.05, prune=0.0)
    assert _dem_events(raw) == pytest.approx(_dem_events(baseline))


# MUST FAIL BEFORE: invalid metadata was silently ignored or position-labelled.
@pytest.mark.parametrize("key", ["detectors", "observables"])
@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("[", "{key} JSON is malformed"),
        ("{}", "{key}_json must be a JSON list"),
        ("[1]", "{kind} entry must be an object"),
        ('[{"records":[-1]}]', "missing {kind} id"),
        ('[{"id":3}]', "{kind} entry has neither 'records' nor 'meas_ids'"),
        ('[{"id":3,"records":[-1]},{"id":3,"records":[-2]}]', "{kind} id 3 is repeated in metadata"),
        (
            '[{"id":3,"records":[-2],"meas_ids":[1]}]',
            "{kind} id 3: records resolve to [0], meas_ids to [1]",
        ),
        ([{"id": 3, "records": [-1]}], "{key} metadata must be a JSON string; pass a JSON string"),
        ('[{"id":3,"kind":"tracked_pauli","records":[-1]}]', '{kind} entry uses kind="tracked_pauli"'),
        ('[{"id":3,"records":[1.5]}]', "{kind} record offsets must be integers"),
        ('[{"id":3,"meas_ids":[99]}]', "{kind} id 3: meas_id 99 is not present in the circuit's measurements"),
    ],
)
@pytest.mark.parametrize("entrypoint", [noise_characterization, exact_correlation_table])
def test_invalid_metadata_raises(key, metadata, message, entrypoint):
    circuit = _circuit()
    circuit.set_meta(key, metadata)
    with pytest.raises(ValueError, match=re.escape(message.format(key=key, kind=key[:-1]))):
        entrypoint(circuit, p1=0.05, prune=0.0)


# REGRESSION GUARD: exact APIs already honour integer ids on records.
def test_exact_rates_keep_declared_ids():
    rates = dict(exact_detection_rates(_circuit(), p1=0.05, prune=0.0))
    assert set(rates) == {2, 5}
    assert rates == pytest.approx(
        {
            5: (1.0 - (1.0 - 4.0 * 0.05 / 3.0) ** 2) / 2.0,
            2: (1.0 - (1.0 - 4.0 * 0.05 / 3.0) ** 4) / 2.0,
        },
        abs=1e-12,
    )
    table = {tuple(nodes): p for nodes, p in exact_correlation_table(_circuit(), p1=0.05, prune=0.0)}
    assert table[("D5",)] == pytest.approx(rates[5])
    assert table[("D2",)] == pytest.approx(rates[2])
    assert table[("L3",)] == pytest.approx(rates[2])


# REGRESSION GUARD: only None means metadata is absent.
def test_absent_metadata_is_empty():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    assert exact_detection_rates(circuit) == []
    assert exact_correlation_table(circuit) == []
    data, raw, decomposed = noise_characterization(circuit)
    assert json.loads(data)["definitions"] == []
    assert _dem_events(raw) == _dem_events(decomposed) == {}


# MUST FAIL BEFORE: canonical aliases and prefixed ids were ignored.
def test_canonical_aliases_and_prefixed_ids():
    circuit = _circuit()
    circuit.set_meta(
        "detectors",
        '[{"detector_id":"D5","records":[-2]},{"detector_id":"D2","records":[-1],"extra":{"nested":true}}]',
    )
    circuit.set_meta("observables", '[{"observable_id":"L3","records":[-1],"label":"logical"}]')
    data, raw, _ = noise_characterization(circuit, p1=0.05, prune=0.0)
    assert [item["label"] for item in json.loads(data)["definitions"]] == ["D2", "D5", "L3"]
    _, baseline, _ = noise_characterization(_circuit(), p1=0.05, prune=0.0)
    assert _dem_events(raw) == pytest.approx(_dem_events(baseline))


# MUST FAIL BEFORE: the canonical schema rejects invalid coordinates and labels.
@pytest.mark.parametrize(
    ("key", "metadata"),
    [
        ("detectors", '[{"id":5,"records":[-2],"coords":[1,2]}]'),
        ("detectors", '[{"id":5,"records":[-2],"coords":[1,2,"x"]}]'),
        ("observables", '[{"id":3,"records":[-1],"label":42}]'),
    ],
)
def test_canonical_optional_fields_are_validated(key, metadata):
    circuit = _circuit()
    circuit.set_meta(key, metadata)
    with pytest.raises(ValueError, match=r"coords|label"):
        exact_correlation_table(circuit, p1=0.05, prune=0.0)


# MUST FAIL BEFORE: an empty form must not reject the other non-empty form.
@pytest.mark.parametrize("key", ["detectors", "observables"])
@pytest.mark.parametrize(
    "metadata",
    [
        '[{"id":5,"records":[],"meas_ids":[0]}]',
        '[{"id":5,"records":[-2],"meas_ids":[]}]',
    ],
)
def test_empty_copresent_reference_uses_nonempty_form(key, metadata):
    circuit = _circuit()
    circuit.set_meta(key, metadata)
    expected = _circuit()
    expected.set_meta(key, '[{"id":5,"records":[-2]}]')
    actual_table = {tuple(nodes): p for nodes, p in exact_correlation_table(circuit, p1=0.05, prune=0.0)}
    expected_table = {tuple(nodes): p for nodes, p in exact_correlation_table(expected, p1=0.05, prune=0.0)}
    assert actual_table == pytest.approx(expected_table)


# REGRESSION GUARD: equal multisets may differ in order and retain duplicate XOR parity.
# The stamps follow emission order. EEG numbers records in emission order and
# DemBuilder in MeasId order, so out-of-order stamps would make a record and a
# meas_id name different measurements depending on the component; that
# convention is not decided here.
@pytest.mark.parametrize("key", ["detectors", "observables"])
@pytest.mark.parametrize("records", [[-2], [-2, -1], [-2, -2, -1]])
def test_agreeing_reference_forms_are_resolved_once(key, records):
    circuit = _circuit(stamped=(0, 1))
    stamps = {-2: 0, -1: 1}
    meas_ids = [stamps[record] for record in reversed(records)]
    circuit.set_meta(key, json.dumps([{"id": 5, "records": records, "meas_ids": meas_ids}]))
    expected = _circuit(stamped=(0, 1))
    expected.set_meta(key, json.dumps([{"id": 5, "records": records}]))
    actual_table = {tuple(nodes): p for nodes, p in exact_correlation_table(circuit, p1=0.05, prune=0.0)}
    expected_table = {tuple(nodes): p for nodes, p in exact_correlation_table(expected, p1=0.05, prune=0.0)}
    assert actual_table == pytest.approx(expected_table)


# MUST FAIL BEFORE: set comparison discarded reference multiplicity.
@pytest.mark.parametrize("key", ["detectors", "observables"])
@pytest.mark.parametrize(("records", "meas_ids"), [([-2, -2], [0]), ([-2], [0, 0])])
def test_reference_multiplicity_disagreement_raises(key, records, meas_ids):
    circuit = _circuit()
    circuit.set_meta(key, json.dumps([{"id": 1, "records": records, "meas_ids": meas_ids}]))
    message = f"{key[:-1]} id 1: records resolve to {[r + 2 for r in records]}, meas_ids to {meas_ids}"
    with pytest.raises(ValueError, match=re.escape(message)):
        exact_correlation_table(circuit, p1=0.05, prune=0.0)


# REGRESSION GUARD: batched stamps pair with qubit positions, not sorted qubits or ids.
# Only meas_ids are used: with out-of-order stamps, what a record offset names
# depends on whether records count emission order (EEG) or MeasId order
# (DemBuilder), a convention not decided here.
def test_batched_meas_ids_resolve_like_records():
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    for qubits in ([0, 1], [0, 1], [1], [1]):
        circuit.tick().h(qubits)
    # Qubit 1 is stamped 17 and qubit 0 is stamped 9, in one batched call.
    circuit.tick().mz_with_ids([1, 0], [17, 9])
    circuit.set_meta("num_measurements", "2")
    circuit.set_meta("detectors", json.dumps([{"id": 5, "meas_ids": [9]}, {"id": 2, "meas_ids": [17]}]))
    circuit.set_meta("observables", json.dumps([{"id": 3, "meas_ids": [17]}]))
    expected = _circuit()
    rates = dict(exact_detection_rates(circuit, p1=0.05, prune=0.0))
    assert rates == pytest.approx(dict(exact_detection_rates(expected, p1=0.05, prune=0.0)))
    table = {tuple(nodes): p for nodes, p in exact_correlation_table(circuit, p1=0.05, prune=0.0)}
    expected_table = {tuple(nodes): p for nodes, p in exact_correlation_table(expected, p1=0.05, prune=0.0)}
    assert table == pytest.approx(expected_table)


class _MetadataProxy:
    def __init__(self, circuit):
        self.circuit = circuit
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.circuit, name)

    def get_meta(self, key):
        self.calls.append(key)
        return self.circuit.get_meta(key)


# MUST FAIL BEFORE: noise characterization read each metadata key twice.
def test_metadata_is_read_once():
    circuit = _MetadataProxy(_circuit())
    noise_characterization(circuit, p1=0.05, prune=0.0)
    assert circuit.calls.count("detectors") == circuit.calls.count("observables") == 1


# MUST FAIL BEFORE: get_meta exceptions were silently treated as absence.
def test_metadata_exception_propagates():
    class RaisingMetadata(_MetadataProxy):
        def get_meta(self, key):
            message = f"cannot read {key}"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="cannot read detectors"):
        exact_detection_rates(RaisingMetadata(_circuit()))


# MUST FAIL BEFORE: non-string metadata needs an actionable error message.
def test_non_string_metadata_requests_json_string():
    circuit = _circuit()
    circuit.set_meta("observables", [{"id": 3, "records": [-1]}])
    with pytest.raises(ValueError, match="pass a JSON string"):
        noise_characterization(circuit)


# MUST FAIL BEFORE: ids are bounded by the canonical parser's u32 type.
@pytest.mark.parametrize("key", ["detectors", "observables"])
def test_ids_above_u32_are_rejected(key):
    circuit = _circuit()
    circuit.set_meta(key, '[{"id":4294967296,"records":[-1]}]')
    with pytest.raises(ValueError, match="id out of range"):
        exact_correlation_table(circuit, p1=0.05, prune=0.0)


# REGRESSION GUARD: records outside the measurement range already raise.
@pytest.mark.parametrize("key", ["detectors", "observables"])
def test_unresolvable_records_raise(key):
    circuit = _circuit()
    circuit.set_meta(key, '[{"id":3,"records":[-3]}]')
    with pytest.raises(ValueError, match="record offset -3"):
        exact_correlation_table(circuit, p1=0.05, prune=0.0)


def _annotation_circuit():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().h([0])
    circuit.tick().h([0])
    m = circuit.tick().mz([0])
    circuit.detector(m)
    return circuit


# MUST FAIL BEFORE: annotation-only detectors were ignored.
def test_annotation_detector_matches_metadata():
    annotated = _annotation_circuit()
    metadata = TickCircuit()
    metadata.tick().pz([0])
    metadata.tick().h([0])
    metadata.tick().h([0])
    metadata.tick().mz([0])
    metadata.add_detector([-1])
    expected = dict(exact_detection_rates(metadata, p1=0.1))
    assert expected[0] == pytest.approx((1 - (1 - 4 * 0.1 / 3) ** 2) / 2)
    assert dict(exact_detection_rates(annotated, p1=0.1)) == pytest.approx(expected)


# MUST FAIL BEFORE: empty metadata must retain nonempty annotation definitions.
@pytest.mark.parametrize("metadata", ["[]", "", '[{"id":0,"records":[-1]}]'])
def test_annotations_with_empty_or_agreeing_metadata(metadata):
    circuit = _annotation_circuit()
    circuit.set_meta("detectors", metadata)
    expected = dict(exact_detection_rates(_annotation_circuit(), p1=0.1))
    assert expected
    assert dict(exact_detection_rates(circuit, p1=0.1)) == pytest.approx(expected)


# MUST FAIL BEFORE: disagreeing metadata and annotations were accepted.
def test_annotations_disagree_with_metadata():
    circuit = _annotation_circuit()
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
    with pytest.raises(ValueError, match=r"metadata positions.*differ from annotation positions"):
        exact_detection_rates(circuit)


# MUST FAIL BEFORE: the declared measurement count was not checked.
@pytest.mark.parametrize("count", ["2", 1])
def test_measurement_count_is_validated(count):
    circuit = _annotation_circuit()
    circuit.set_meta("num_measurements", count)
    with pytest.raises(ValueError, match="num_measurements"):
        exact_detection_rates(circuit)


# MUST FAIL BEFORE: malformed annotations were never read.
@pytest.mark.parametrize("annotations", [[{}], [{"kind": "bad"}], [{"kind": "detector"}], [1]])
def test_malformed_annotations_raise(annotations):
    class Malformed(_MetadataProxy):
        def annotations(self):
            return annotations

    with pytest.raises((ValueError, TypeError)):
        exact_detection_rates(Malformed(_circuit()))


# MUST FAIL BEFORE: an absent annotations method was ignored.
def test_missing_annotations_method_raises():
    class Missing(_MetadataProxy):
        def __getattr__(self, name):
            if name == "annotations":
                raise AttributeError(name)
            return super().__getattr__(name)

    with pytest.raises(AttributeError, match="annotations"):
        exact_detection_rates(Missing(_circuit()))


# MUST FAIL BEFORE: exports retained only the original reference form.
@pytest.mark.parametrize("refs", ["records", "meas_ids"])
def test_noise_exports_resolved_references(refs):
    data, _, _ = noise_characterization(_circuit(refs=refs, stamped=(17, 9)))
    definitions = json.loads(data)["definitions"]
    assert [(d["id"], d["records"], d["meas_ids"]) for d in definitions] == [
        (2, [-1], [9]),
        (5, [-2], [17]),
        (3, [-1], [9]),
    ]


# MUST FAIL BEFORE: unknown ids were exported as an empty list.
def test_noise_omits_ids_for_idless_references():
    class GateProxy:
        def __init__(self, gate):
            self.gate = gate
            self.meas_ids = []

        def __getattr__(self, name):
            return getattr(self.gate, name)

    class TickProxy:
        def __init__(self, tick):
            self.tick = tick

        def gate_batches(self):
            return [GateProxy(gate) for gate in self.tick.gate_batches()]

    class Idless(_MetadataProxy):
        def get_tick(self, index):
            return TickProxy(self.circuit.get_tick(index))

    data, _, _ = noise_characterization(Idless(_circuit()))
    definitions = json.loads(data)["definitions"]
    assert all("meas_ids" not in definition for definition in definitions)
    assert [definition["records"] for definition in definitions] == [[-1], [-2], [-1]]


# MUST FAIL BEFORE: annotation kinds and duplicate references were ignored.
def test_annotation_kinds_and_duplicate_parity():
    circuit = _circuit((0, 1, 0), stamped=(17, 9))

    class Annotated(_MetadataProxy):
        def annotations(self):
            return [
                {"kind": "tracked_pauli", "label": None},
                {"kind": "observable", "measurement_ids": [9], "label": None},
                {"kind": "detector", "measurement_ids": [17, 9, 9], "label": None},
                {"kind": "detector", "measurement_ids": [9], "label": None},
            ]

    circuit.set_meta("detectors", "[]")
    actual = {tuple(nodes): p for nodes, p in exact_correlation_table(Annotated(circuit), p1=0.05)}
    expected = {tuple(nodes): p for nodes, p in exact_correlation_table(_circuit((0, 1, 0)), p1=0.05)}
    assert actual == pytest.approx(expected)


# MUST FAIL BEFORE: a count string was incorrectly described as JSON.
def test_non_string_measurement_count_requests_decimal_string():
    circuit = _annotation_circuit()
    circuit.set_meta("num_measurements", 1)
    with pytest.raises(ValueError, match="num_measurements metadata must be a decimal count string"):
        exact_detection_rates(circuit)
