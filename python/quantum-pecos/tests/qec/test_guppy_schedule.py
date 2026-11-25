"""Independent clock arithmetic and conservative boundaries for schedule receipts."""

from copy import deepcopy
from types import SimpleNamespace

import pecos
import pytest
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import GuppyOrdering, ScheduledCircuit, summarize_runtime_batches
from selene_simple_runtime_plugin import SimpleRuntimePlugin


def _batch(index, start, duration, *ops):
    return {
        "shot_id": 1,
        "batch_index": index,
        "num_qubits": 2,
        "start_time_nanos": start,
        "duration_nanos": duration,
        "operations": list(ops),
    }


def _op(name, q):
    return {name: {"qubit_id": q}}


def test_distinct_site_intervals_and_reuse_gaps_are_not_live_idle_exposure():
    report = summarize_runtime_batches(
        [
            _batch(0, 100, 10, _op("Reset", 0)),
            _batch(1, 90, 5, _op("Reset", 1)),  # disjoint sites need not be emitted in clock order
            _batch(2, 120, 20, {"Rzz": {"qubit_id_1": 0, "qubit_id_2": 1}}),
            _batch(3, 140, 30, _op("Measure", 0)),
            _batch(4, 200, 10, _op("Reset", 0)),  # gap after measurement can cross lifetimes
        ],
    )
    assert report["first_batch_start_ns"] == 90
    assert report["last_batch_end_ns"] == 210
    assert report["batch_span_ns"] == 120
    assert report["sites"] == [
        {"runtime_site": 0, "batch_assigned_operation_ns": 70, "between_operation_gap_ns": 40},
        {"runtime_site": 1, "batch_assigned_operation_ns": 25, "between_operation_gap_ns": 25},
    ]
    assert not report["live_idle_exposure_verified"]


def test_multiple_normalized_operations_share_one_batch_interval():
    report = summarize_runtime_batches([_batch(0, 5, 20, _op("Rz", 0), _op("Rxy", 0))])
    assert report["operation_counts"] == {"Rxy": 1, "Rz": 1}
    assert report["sites"] == [{"runtime_site": 0, "batch_assigned_operation_ns": 20, "between_operation_gap_ns": 0}]


def test_zero_duration_and_empty_batches_are_distinct():
    assert summarize_runtime_batches([])["timing_status"] == "no_batches"
    report = summarize_runtime_batches([_batch(0, 7, 0, _op("Rz", 0))])
    assert report["timing_status"] == "zero_durations"
    assert report["batch_span_ns"] == 0
    custom = summarize_runtime_batches([_batch(0, 7, 5, {"Custom": {"tag": 17, "data": []}})])
    assert custom["batch_span_ns"] == 5
    assert custom["sites"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_time_nanos", -1),
        ("duration_nanos", True),
        ("duration_nanos", 1.5),
        ("duration_nanos", 1 << 64),
        ("batch_index", 1),
        ("batch_index", False),
        ("num_qubits", -1),
    ],
)
def test_invalid_clock_or_identity_fails(field, value):
    batch = _batch(0, 0, 1, _op("Reset", 0))
    batch[field] = value
    with pytest.raises(ValueError, match="Runtime batch"):
        summarize_runtime_batches([batch])


@pytest.mark.parametrize(("field", "value"), [("shot_id", 2), ("num_qubits", 3), ("start_time_nanos", 9)])
def test_mixed_shots_capacities_and_overlaps_fail(field, value):
    second = _batch(1, 10, 10, _op("Rxy", 0))
    second[field] = value
    with pytest.raises(ValueError, match="Runtime batches"):
        summarize_runtime_batches([_batch(0, 0, 10, _op("Reset", 0)), second])


@pytest.mark.parametrize(
    "op",
    [_op("Unknown", 0), _op("Reset", 2), _op("Reset", True), {"Rzz": {"qubit_id_1": 0, "qubit_id_2": 0}}],
)
def test_unknown_operations_and_bad_site_ids_fail(op):
    with pytest.raises(ValueError, match=r"Unsupported runtime|Invalid runtime"):
        summarize_runtime_batches([_batch(0, 0, 5, op)])


def _fake_capture(monkeypatch, batch):
    def capture(_program, _n, **options):
        options["batch_adapter_factory"]().normalize(deepcopy(batch))
        return [{"operations": [], "lowered_quantum_ops_complete": True, "runtime_events_complete": True}]

    monkeypatch.setattr(pecos, "capture_qis_operation_trace", capture)
    monkeypatch.setattr(ScheduledCircuit, "guppy_program", lambda *_args, **_kwargs: None)


def test_audit_preserves_delegate_and_copies_original_and_normalized_evidence(monkeypatch):
    original = _batch(0, 10, 20, {"Custom": {"tag": 17, "data": []}}, _op("Rxy", 0))
    _fake_capture(monkeypatch, original)
    created = []

    class Adapter:
        def __init__(self):
            created.append(self)

        def normalize(self, batch):
            batch["operations"][0] = _op("Rz", 0)
            batch["duration_nanos"] = 999  # callback mutation must not rewrite the captured native clock
            return batch["operations"]

    plan = ScheduledCircuit(1, (), (), ())
    plugin = SimpleNamespace(batch_adapter_factory=Adapter, library_file="unused", get_init_args=list)
    audit = plan.audit_guppy_schedule(runtime=plugin)
    report, evidence = audit.to_dict(), audit.evidence()
    assert len(created) == 1
    assert report["event_adapter"] == "delegated"
    assert report["source_runtime"]["operation_counts"] == {"Custom": 1, "Rxy": 1}
    assert report["normalized_runtime"]["operation_counts"] == {"Rxy": 1, "Rz": 1}
    assert evidence["runtime_batches"][0]["duration_nanos"] == 20
    assert evidence["runtime_batches"][0]["operations"] == original["operations"]
    assert report["normalized_runtime"]["sites"][0]["batch_assigned_operation_ns"] == 20
    evidence["runtime_batches"].clear()
    assert audit.evidence()["runtime_batches"]
    assert not report["source_moment_timing_verified"]
    assert not report["calibrated_durations_verified"]
    assert not report["noise_equivalence_verified"]
    assert not report["measurement_binding_verified"]


def test_uninterpreted_custom_events_are_rejected(monkeypatch):
    _fake_capture(monkeypatch, _batch(0, 0, 5, {"Custom": {"tag": 1, "data": []}}))
    with pytest.raises(ValueError, match="unhandled custom"):
        ScheduledCircuit(1, (), (), ()).audit_guppy_schedule(runtime=SimpleRuntimePlugin())


def test_missing_barriers_fail_without_a_receipt(monkeypatch):
    _fake_capture(monkeypatch, _batch(0, 0, 0, _op("Reset", 0)))
    plan = ScheduledCircuit(1, ((Op("R", (0,)),), (Op("M", (0,)),)), (), ())
    with pytest.raises(ValueError, match="expected 1"):
        plan.audit_guppy_schedule(runtime=SimpleRuntimePlugin())
    # Functional captures explicitly have no authored-moment segmentation.
    audit = plan.audit_guppy_schedule(runtime=SimpleRuntimePlugin(), ordering=GuppyOrdering.FUNCTIONAL)
    assert audit.to_dict()["qis_moments"] is None


@pytest.mark.parametrize("runtime", [None, "selene_simple_runtime", "runtime.dylib", object()])
def test_implicit_runtime_selectors_fail_before_compilation(runtime):
    with pytest.raises(TypeError, match="explicit runtime plugin"):
        ScheduledCircuit(1, (), (), ()).audit_guppy_schedule(runtime=runtime)


def test_explicit_adapter_override_takes_precedence_and_can_drop_known_metadata(monkeypatch):
    _fake_capture(monkeypatch, _batch(0, 5, 10, {"Custom": {"tag": 17, "data": []}}))

    def unwanted_factory():
        pytest.fail("Plugin's default adapter must not override the explicit factory")

    class MetadataAdapter:
        def normalize(self, batch):
            assert batch["operations"] == [{"Custom": {"tag": 17, "data": []}}]
            return []

    plugin = SimpleNamespace(library_file="unused", get_init_args=list, batch_adapter_factory=unwanted_factory)
    audit = ScheduledCircuit(1, (), (), ()).audit_guppy_schedule(runtime=plugin, batch_adapter_factory=MetadataAdapter)
    assert audit.to_dict()["normalized_runtime"]["sites"] == []
    assert audit.to_dict()["normalized_runtime"]["batch_span_ns"] == 10


def test_noncallable_adapter_fails_before_compilation():
    with pytest.raises(TypeError, match="batch_adapter_factory must be callable"):
        ScheduledCircuit(1, (), (), ()).audit_guppy_schedule(runtime=SimpleRuntimePlugin(), batch_adapter_factory=7)
