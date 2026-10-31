"""Runtime-emitted prefixes reach providers before subsequent live feedback."""

import json
from dataclasses import FrozenInstanceError, replace

import pecos_rslib
import pytest
from pecos import Qis, selene_engine, sim
from pecos.qec import RuntimeDecodeBatchPrefix, RuntimeDecodeBatchRequest, RuntimeDecoder, RuntimeReadSource

from qec.test_runtime_decode_context import context
from qec.test_runtime_observed_reads import checked_window, observed_program


def build(provider, *, runtime="selene_simple_runtime", source=None, workers=1):
    return (
        sim(Qis(observed_program() if source is None else source))
        .classical(selene_engine(runtime).runtime_decoder(provider, capture_batches=True))
        .qubits(1)
        .workers(workers)
        .build()
    )


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_live_batches_select_supplied_sources_and_feedback_with_worker_reuse(runtime):
    sessions = []

    def factory():
        seen = []
        sessions.append(seen)

        def provide(request):
            assert type(request) is RuntimeDecodeBatchRequest
            assert request.context == context(request.instance_id)
            prefix = request.batch_prefix
            assert prefix.first_batch_index == (seen[-1].next_batch_index if seen else 0)
            if seen:
                assert prefix.runtime_shot_id == seen[-1].runtime_shot_id
            seen.append(prefix)
            ops = [op for batch in prefix.batches for op in batch.operations]
            measured = [op.source_id for op in ops if op.kind == "Measure"]
            # Independent known schedule: the first interval has the detector
            # measurement; later ones also contain the previous after-readout.
            assert measured == (
                [0] if request.instance_id == 0 else [2 * request.instance_id - 1, 2 * request.instance_id]
            )
            assert all(op.qubits == (0,) for op in ops)
            assert any(op.kind == "RXY" for op in ops)
            with pytest.raises(FrozenInstanceError):
                prefix.next_batch_index = 0
            definition = checked_window(request.instance_id)
            return replace(
                definition,
                read_sources=(RuntimeReadSource(record=definition.read_sources[0].record, source_id=measured[-1]),),
                output_map=((0,) if request.instance_id != 1 else (),),
            )

        return provide

    runner = build(RuntimeDecoder(provider_factory=factory), runtime=runtime, workers=2)
    for _ in range(2):
        assert runner.run(4).to_dict()["after"] == [[0, 1, 0]] * 4
    assert len(sessions) == 8
    assert all(len(seen) == 3 for seen in sessions)


def test_provider_failure_discards_interval_and_restarts_shot():
    fail = True
    seen = []

    def factory():
        local = []
        seen.append(local)

        def provide(request):
            nonlocal fail
            local.append(request.batch_prefix)
            if request.instance_id == 1 and fail:
                fail = False
                msg = "synthetic provider failure"
                raise ValueError(msg)
            return checked_window(request.instance_id)

        return provide

    runner = build(RuntimeDecoder(provider_factory=factory))
    with pytest.raises(Exception, match="synthetic provider failure"):
        runner.run(1)
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert [len(items) for items in seen] == [2, 3]
    assert [items[0].first_batch_index for items in seen] == [0, 0]
    # Built-simulation runs may create fresh engine clones, so engine-local
    # shot IDs can repeat. Session isolation, not a globally unique ID, is key.
    assert seen[0][0] is not seen[1][0]


def test_opt_in_never_falls_back_to_old_session_method_or_plain_import():
    class OldSession:
        def decode(self, *_):
            pytest.fail("legacy fallback")

        def decode_observed_v1(self, *_):
            pytest.fail("observed fallback")

    with pytest.raises(Exception, match="decode_observed_with_batches_v1"):
        build(OldSession).run(1)
    source = observed_program().replace("pecos_runtime_decode_observed_v1", "pecos_runtime_decode_v1")
    with pytest.raises(Exception, match="requires observed decode calls"):
        build(RuntimeDecoder({7: checked_window()}), source=source).run(1)


def test_scheduled_transport_rejected_before_execution():
    with pytest.raises(Exception, match="ordinary dynamic native execution"):
        (
            sim(Qis(observed_program()))
            .classical(
                pecos_rslib.selene_engine()
                .runtime_decoder(RuntimeDecoder({7: checked_window()}), capture_batches=True)
                .scheduled_batches(),
            )
            .qubits(1)
            .build()
        )


def payload():
    return {
        "version": 1,
        "runtime_shot_id": 12,
        "first_batch_index": 0,
        "next_batch_index": 1,
        "batches": [
            {
                "batch_index": 0,
                "start_time_nanos": 20,
                "duration_nanos": 5,
                "normalization_applied": False,
                "operations": [{"Idle": [20e-9, 0]}, {"RXY": [1.0, 0.5, 0]}, {"Measure": [0, 7]}],
            },
        ],
    }


def test_prefix_retains_units_order_and_source_identity():
    prefix = RuntimeDecodeBatchPrefix.from_json(json.dumps(payload()).encode())
    batch = prefix.batches[0]
    assert (batch.start_time_nanos, batch.duration_nanos) == (20, 5)
    assert [op.kind for op in batch.operations] == ["Idle", "RXY", "Measure"]
    assert batch.operations[0].parameters == (20e-9,)
    assert batch.operations[-1].source_id == 7
    empty = payload() | {"batches": [], "first_batch_index": 4, "next_batch_index": 4}
    assert RuntimeDecodeBatchPrefix.from_json(json.dumps(empty).encode()).batches == ()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(version=2),
        lambda p: p.update(version=True),
        lambda p: p.update(next_batch_index=2),
        lambda p: p.update(runtime_shot_id=True),
        lambda p: p["batches"][0].update(batch_index=1),
        lambda p: p["batches"][0].update(start_time_nanos=2**64 - 1),
        lambda p: p["batches"][0].update(normalization_applied=1),
        lambda p: p["batches"][0].update(operations=[{"Custom": [1, [2]]}]),
        lambda p: p["batches"][0].update(operations=[{"Measure": [0, 2**64 - 2]}]),
        lambda p: p["batches"][0].update(operations=[{"RXY": [float("inf"), 0, 0]}]),
    ],
)
def test_invalid_prefix_rejected(mutation):
    raw = payload()
    mutation(raw)
    with pytest.raises(ValueError, match="invalid decoder batch prefix"):
        RuntimeDecodeBatchPrefix.from_json(json.dumps(raw).encode())


def test_session_rejects_repeated_prefix_before_provider_and_latches():
    seen = []
    session = RuntimeDecoder(
        provider_factory=lambda: lambda request: (seen.append(request), checked_window(request.instance_id))[1],
    )()
    raw = payload()
    assert (
        session.decode_observed_with_batches_v1(
            7,
            0,
            [1],
            context(0).to_json(),
            [("readout", 0, 0, True)],
            json.dumps(raw).encode(),
        )
        == 1
    )
    with pytest.raises(ValueError, match="skipped/repeated"):
        session.decode_observed_with_batches_v1(
            7,
            1,
            [1],
            context(1).to_json(),
            [("readout", 0, 2, True)],
            json.dumps(raw).encode(),
        )
    with pytest.raises(RuntimeError, match="failed"):
        session.decode_observed_with_batches_v1(7, 1, [1], context(1).to_json(), [], json.dumps(raw).encode())
    assert len(seen) == 1


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_empty_live_prefix_does_not_force_an_unread_future(runtime):
    from qec.test_runtime_decode_context import binding

    definition = checked_window()
    schema = replace(binding().schema, detectors=(replace(binding().schema.detectors[0], records=(), reference=True),))
    bound = replace(binding(), schema=schema)
    definition = replace(definition, binding=bound, read_sources=())
    envelope = replace(context(0), schema_id=schema.schema_id, branches=(), loops=(), inputs=()).to_json()
    escaped = "".join(f"\\{byte:02X}" for byte in envelope)
    source = f"""@ctx = private constant [{len(envelope)} x i8] c"{escaped}"
declare i64 @___qalloc()
declare void @___reset(i64)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___dec_future_refcount(i64)
declare void @___qfree(i64)
declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)
define i64 @qmain(i64 %shot) #0 {{
entry:
  %q = call i64 @___qalloc()
  call void @___reset(i64 %q)
  %m = call i64 @___lazy_measure(i64 %q)
  %data = alloca i8
  store i8 1, ptr %data
  %f0 = call i64 @pecos_runtime_decode_observed_v1(i64 7, i64 0, ptr %data, i64 1, ptr @ctx, i64 {len(envelope)})
  %f1 = call i64 @pecos_runtime_decode_observed_v1(i64 7, i64 1, ptr %data, i64 1, ptr @ctx, i64 {len(envelope)})
  %value = call i1 @___read_future_bool(i64 %m)
  call void @___dec_future_refcount(i64 %m)
  call void @___qfree(i64 %q)
  ret i64 0
}}
attributes #0 = {{ "EntryPoint" }}
"""
    seen = []

    def provide(request):
        seen.append(request.batch_prefix)
        return definition

    def trace(capture):
        return (
            sim(Qis(source))
            .classical(
                selene_engine(runtime).runtime_decoder(
                    RuntimeDecoder(provider_factory=lambda: provide) if capture else RuntimeDecoder({7: definition}),
                    capture_batches=capture,
                ),
            )
            .qubits(1)
            .capture_operation_trace()
        )

    captured, baseline = trace(True), trace(False)
    # The entire emitted gate stream must agree, including idle timing. Capture
    # must not turn a decode into a barrier or force extra ready work.
    assert [gate for chunk in captured for gate in chunk["lowered_quantum_ops"]] == [
        gate for chunk in baseline for gate in chunk["lowered_quantum_ops"]
    ]
    assert len(seen) == 2
    measured = [
        op.source_id for prefix in seen for batch in prefix.batches for op in batch.operations if op.kind == "Measure"
    ]
    # These two schedulers intentionally differ in when unread work is ready.
    assert measured == ([0] if runtime == "selene_simple_runtime" else [])
    assert seen[1].batches == ()
    assert seen[1].first_batch_index == seen[0].next_batch_index == seen[1].next_batch_index
