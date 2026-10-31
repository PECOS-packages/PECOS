"""Compiler-generated control intervals join live batches without changing forcing."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from pecos import Qis, selene_engine, sim
from pecos.qec import RuntimeControlSnapshot, RuntimeDecodeBatchPrefix, RuntimeDecodeControlRequest, RuntimeDecoder

from qec.test_runtime_decode_context import context
from qec.test_runtime_decoder_batches import payload
from qec.test_runtime_decoder_transport import NATIVE_PROGRAM
from qec.test_runtime_observed_reads import checked_window


def source():
    envelope = replace(context(), branches=(), loops=(), inputs=()).to_json()
    escaped = "".join(f"\\{byte:02X}" for byte in envelope)
    declarations = (
        f'@ctx = private constant [{len(envelope)} x i8] c"{escaped}"\n'
        '@read_name = private constant [7 x i8] c"readout"\n'
        "declare i1 @pecos_runtime_read_bool_v1(ptr, i64, i64)\n"
    )
    return declarations + NATIVE_PROGRAM.replace(
        "declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)",
        "declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)",
    ).replace(
        "@pecos_runtime_decode(i64 7, i64 %instance, ptr %data, i64 1)",
        f"@pecos_runtime_decode_observed_v1(i64 7, i64 %instance, ptr %data, i64 1, ptr @ctx, i64 {len(envelope)})",
    ).replace(
        "%observed = call i1 @___read_future_bool(i64 %m)",
        "%observed = call i1 @pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %m)",
    )


def builder(decoder, *, runtime="selene_simple_runtime", program=None, capture=True):
    return (
        sim(Qis(source() if program is None else program))
        .classical(selene_engine(runtime).runtime_decoder(decoder, capture_batches=True, capture_control_flow=capture))
        .qubits(1)
    )


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize("worker_context", [False, True])
def test_live_control_drives_feedback_and_restarts_on_worker_and_shot_reuse(runtime, worker_context):
    sessions = []

    def factory():
        seen = []
        sessions.append(seen)

        def provide(request):
            assert type(request) is RuntimeDecodeControlRequest
            assert request.context.branches == request.context.loops == request.context.inputs == ()
            control = request.control
            sites = control.manifest.sites
            assert control.events[-1].site_id == control.decode_site
            assert sites[control.decode_site].kind == "decode"
            assert control.runtime_shot_id == request.batch_prefix.runtime_shot_id
            assert control.first_batch_index == request.batch_prefix.first_batch_index
            assert control.next_batch_index == request.batch_prefix.next_batch_index
            assert [(loop.occurrence, loop.iteration) for loop in control.active_loops] == [(0, request.instance_id)]
            assert [e.value for e in control.events if sites[e.site_id].kind == "branch"] == (
                [] if request.instance_id == 0 else [int(request.instance_id == 1), 1]
            )
            assert [e.value for e in control.events if sites[e.site_id].kind == "decoder_frame"] == (
                [] if request.instance_id == 0 else [int(request.instance_id == 1)]
            )
            assert [e.value for e in control.events if sites[e.site_id].kind == "read_bool"] == (
                [1] if request.instance_id == 0 else [int(request.instance_id == 2), 1]
            )
            assert len([e for e in control.events if sites[e.site_id].kind == "shot"]) == int(not seen)
            if seen:
                assert control.manifest == seen[0].manifest
                assert control.first_batch_index == seen[-1].next_batch_index
            with pytest.raises(FrozenInstanceError):
                control.decode_site = 100
            seen.append(control)
            return checked_window(request.instance_id, output_map=((0,) if request.instance_id != 1 else (),))

        return provide

    runner = builder(RuntimeDecoder(provider_factory=factory), runtime=runtime).workers(2)
    runner = runner.python_worker_context(worker_context).build()
    for _ in range(2):
        assert runner.run(4).to_dict()["after"] == [[0, 1, 0]] * 4
    assert len(sessions) == 8
    assert all(len(items) == 3 for items in sessions)
    assert len({items[0].manifest.module_sha256 for items in sessions}) == 1
    assert all(items[0].events[0].occurrence == 0 for items in sessions)


def test_provider_failure_discards_control_and_recovers():
    fail = True
    shots = []

    def factory():
        seen = []
        shots.append(seen)

        def provide(request):
            nonlocal fail
            seen.append(request.control)
            if fail and request.instance_id == 1:
                fail = False
                msg = "synthetic control provider failure"
                raise ValueError(msg)
            return checked_window(request.instance_id)

        return provide

    runner = builder(RuntimeDecoder(provider_factory=factory)).build()
    with pytest.raises(Exception, match="synthetic control provider failure"):
        runner.run(1)
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert [len(items) for items in shots] == [2, 3]
    assert all(items[0].events[0].occurrence == 0 for items in shots)


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_control_capture_preserves_emitted_operations_and_timing(runtime):
    decoder = RuntimeDecoder(provider_factory=lambda: lambda r: checked_window(r.instance_id))
    captured = builder(decoder, runtime=runtime).capture_operation_trace()
    baseline = builder(decoder, runtime=runtime, capture=False).capture_operation_trace()
    assert [gate for chunk in captured for gate in chunk["lowered_quantum_ops"]] == [
        gate for chunk in baseline for gate in chunk["lowered_quantum_ops"]
    ]


def test_control_opt_in_requires_batches_and_never_falls_back():
    class OldSession:
        def decode(self, *_):
            pytest.fail("legacy fallback")

        def decode_observed_with_batches_v1(self, *_):
            pytest.fail("batch-only fallback")

    with pytest.raises(Exception, match="decode_observed_with_control_v1"):
        builder(OldSession).run(1)
    with pytest.raises(Exception, match="requires batch capture"):
        sim(Qis(source())).classical(
            selene_engine().runtime_decoder(OldSession, capture_control_flow=True),
        ).qubits(1).build()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("store i8 %bit, ptr %data", "; missing initialization"),
        lambda s: s.replace("ptr %data, i64 1, ptr @ctx", "ptr %data, i64 2, ptr @ctx"),
        lambda s: s.replace("ptr @read_name, i64 7", "ptr @read_name, i64 8"),
        lambda s: s.replace("%next = add i64", "%next = udiv i64"),
        lambda s: "declare void @pecos_runtime_control_v1(i64, i64)\n" + s,
    ],
)
def test_unsupported_control_program_rejected_before_provider(mutate):
    def unexpected():
        pytest.fail("provider reached unsupported program")

    with pytest.raises(Exception, match="compiler control provenance unsupported"):
        builder(unexpected, program=mutate(source())).build()


def control_payload():
    return {
        "version": 1,
        "runtime_shot_id": 12,
        "first_batch_index": 0,
        "next_batch_index": 1,
        "decode_site": 2,
        "manifest": {
            "module_sha256": "a" * 64,
            "instrumenter": "pecos-qis-control-v1",
            "sites": [
                {"site_id": i, "block": 0, "kind": kind, "scalar_type": "u64", "input_dependencies": []}
                for i, kind in enumerate(("block", "shot", "decode"))
            ],
            "successors": [[]],
            "loop_headers": [],
            "loop_members": [],
        },
        "events": [{"site_id": i, "occurrence": 0, "value": 0} for i in range(3)],
        "active_loops": [],
    }


def test_control_payload_is_immutable_and_session_rejects_repeated_occurrences():
    prefix = RuntimeDecodeBatchPrefix.from_json(json.dumps(payload()).encode())
    raw = control_payload()
    snapshot = RuntimeControlSnapshot.from_json(json.dumps(raw).encode(), prefix)
    events = list(snapshot.events)
    frozen = replace(snapshot, events=events)
    events.clear()
    assert frozen == snapshot
    seen = []
    session = RuntimeDecoder(provider_factory=lambda: lambda r: (seen.append(r), checked_window(r.instance_id))[1])()
    args = (7, 0, [1], context().to_json(), [("readout", 0, 0, True)])
    assert session.decode_observed_with_control_v1(*args, json.dumps(payload()).encode(), json.dumps(raw).encode()) == 1
    second = payload() | {"first_batch_index": 1, "next_batch_index": 1, "batches": []}
    raw.update(first_batch_index=1, next_batch_index=1)
    with pytest.raises(ValueError, match="occurrence skipped or repeated"):
        session.decode_observed_with_control_v1(*args, json.dumps(second).encode(), json.dumps(raw).encode())
    with pytest.raises(RuntimeError, match="failed"):
        session.decode_observed_with_control_v1(*args, json.dumps(second).encode(), json.dumps(raw).encode())
    assert len(seen) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(version=True),
        lambda p: p.update(version=2),
        lambda p: p.update(runtime_shot_id=13),
        lambda p: p.update(decode_site=0),
        lambda p: p.update(events=p["events"][:-1]),
        lambda p: p["manifest"].update(module_sha256="caller-label"),
        lambda p: p["manifest"]["sites"][0].update(input_dependencies=[2]),
        lambda p: p["events"][0].update(value=1),
        lambda p: p.update(active_loops=[{"header": 0, "occurrence": 0, "iteration": 0}]),
    ],
)
def test_invalid_control_envelope_rejected(mutate):
    raw = control_payload()
    mutate(raw)
    prefix = RuntimeDecodeBatchPrefix.from_json(json.dumps(payload()).encode())
    with pytest.raises(ValueError, match="invalid compiler control snapshot"):
        RuntimeControlSnapshot.from_json(json.dumps(raw).encode(), prefix)


@pytest.mark.parametrize("position", ["before_first", "after_last"])
def test_pure_classical_loop_aborts_at_pending_event_bound(position):
    spin = """br label %spin
spin:
    %spin_index = phi i64 [0, %PREVIOUS], [%spin_next, %spin]
    %spin_next = add i64 %spin_index, 1
    %spin_more = icmp ult i64 %spin_next, 2200
    br i1 %spin_more, label %spin, label %NEXT
NEXT:"""
    if position == "before_first":
        program = (
            source()
            .replace(
                "br label %loop\nloop:",
                spin.replace("PREVIOUS", "entry").replace("NEXT", "loop"),
            )
            .replace("%instance = phi i64 [0, %entry]", "%instance = phi i64 [0, %spin]")
        )
    else:
        program = source().replace(
            "done:",
            "done:\n    " + spin.replace("PREVIOUS", "done").replace("NEXT", "finish"),
        )
    decoder = RuntimeDecoder(provider_factory=lambda: lambda r: checked_window(r.instance_id))
    with pytest.raises(Exception, match="control event pending queue limit exceeded"):
        builder(decoder, program=program).run(1)
    assert builder(decoder).run(1).to_dict()["after"] == [[0, 0, 0]]
