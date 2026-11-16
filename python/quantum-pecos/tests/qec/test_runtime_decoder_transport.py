"""Live callback boundaries and native-read trace provenance for decoder calls."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pecos import Qis, qis_operation_trace_to_tick_circuit, selene_engine, sim
from pecos.qec import RuntimeDecodeContext, RuntimeDecoder

from qec.test_runtime_decode_context import binding, context, program
from qec.test_runtime_decoder import window

NATIVE_PROGRAM = (
    Path(__file__).resolve().parents[4] / "crates/pecos-qis/tests/fixtures/runtime_decode_native.ll"
).read_text()


def test_live_mixed_calls_share_counters_and_isolate_shots_workers():
    source = "declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)\n" + program(
        [context(i).to_json() for i in range(3)],
    )
    source = source.replace(
        "%frame = call i64 @pecos_runtime_decode_v1(i64 7, i64 %instance, ptr %data, i64 1, ptr %context, i64 %length)",
        """%old = icmp eq i64 %instance, 1
    br i1 %old, label %legacy, label %versioned
legacy:
    %old_frame = call i64 @pecos_runtime_decode(i64 8, i64 %instance, ptr %data, i64 1)
    br label %decoded
versioned:
    %new_frame = call i64 @pecos_runtime_decode_v1(i64 7, i64 %instance, ptr %data, i64 1, ptr %context, i64 %length)
    br label %decoded
decoded:
    %frame = phi i64 [%old_frame, %legacy], [%new_frame, %versioned]""",
    )
    windows = {7: replace(window(), binding=binding()), 8: replace(window(), window_id=8)}
    sessions = []

    def factory():
        seen = []
        sessions.append(seen)

        def provide(request):
            seen.append(request)
            return windows[request.window_id]

        return provide

    runner = (
        sim(Qis(source))
        .classical(selene_engine().runtime_decoder(RuntimeDecoder(provider_factory=factory)))
        .qubits(1)
        .workers(2)
        .build()
    )
    for _ in range(2):
        assert runner.run(4).to_dict()["after"] == [[0] * 3] * 4
    assert len(sessions) == 8
    for seen in sessions:
        assert [(r.window_id, r.instance_id) for r in seen] == [(7, 0), (8, 1), (7, 2)]
        assert [r.context for r in seen] == [context(0), None, context(2)]


def test_live_exact_context_limit_preserves_payload_and_feedback():
    expected = [context(i) for i in range(3)]
    # JSON permits trailing whitespace. Exercise the native byte bound without
    # changing the semantic declaration or manufacturing thousands of branches.
    payloads = [item.to_json().ljust(65536, b" ") for item in expected]
    seen = []
    config = RuntimeDecoder({7: replace(window(), binding=binding())})

    class Session:
        def __init__(self):
            self.checked = config()

        def decode(self, *_):
            pytest.fail("unexpected legacy fallback")

        def decode_v1(self, window_id, instance_id, syndrome, payload):
            assert payload == payloads[instance_id]
            seen.append(RuntimeDecodeContext.from_json(payload))
            return self.checked.decode_v1(window_id, instance_id, syndrome, payload)

    result = sim(Qis(program(payloads))).classical(selene_engine().runtime_decoder(Session)).qubits(1).run(1)
    assert result.to_dict()["after"] == [[0, 0, 0]]
    assert seen == expected


@pytest.mark.parametrize("invalid", [-1, 2**64, "1"])
def test_live_invalid_v1_frame_aborts_and_next_run_has_fresh_session(invalid):
    sessions = []
    fail = True

    class Session:
        def __init__(self):
            self.seen = []
            sessions.append(self.seen)

        def decode(self, *_):
            pytest.fail("unexpected legacy fallback")

        def decode_v1(self, window_id, instance_id, syndrome, payload):
            nonlocal fail
            assert window_id == 7
            assert syndrome == b"\x01"
            assert RuntimeDecodeContext.from_json(payload) == context(instance_id)
            self.seen.append(instance_id)
            if instance_id == 1 and fail:
                fail = False
                return invalid
            return 1

    runner = (
        sim(Qis(program([context(i).to_json() for i in range(3)])))
        .classical(selene_engine().runtime_decoder(Session))
        .qubits(1)
        .build()
    )
    with pytest.raises(Exception, match="Python runtime decoder"):
        runner.run(1)
    assert sessions == [[0, 1]]
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert sessions == [[0, 1], [0, 1, 2]]


def test_custom_python_session_still_requires_legacy_protocol_method():
    class VersionedOnly:
        def decode_v1(self, *_):
            pytest.fail("factory must reject a missing decode method before dispatch")

    with pytest.raises(Exception, match="has no attribute 'decode'"):
        (
            sim(Qis(program([context(i).to_json() for i in range(3)])))
            .classical(selene_engine().runtime_decoder(VersionedOnly))
            .qubits(1)
            .run(1)
        )


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("route", [(), (0,)])
def test_native_read_decoder_trace_json_round_trip_and_replay(runtime, versioned, route):
    declared = replace(binding(), program_revision="synthetic-native-reset-loop-v1")
    contexts = [
        replace(context(i, corrected=bool(route)), program_revision=declared.program_revision) for i in range(3)
    ]
    source = program([item.to_json() for item in contexts], source=NATIVE_PROGRAM) if versioned else NATIVE_PROGRAM
    definition = replace(window(), binding=declared if versioned else None, output_map=(route,))
    trace = (
        sim(Qis(source))
        .classical(selene_engine(runtime).runtime_decoder(RuntimeDecoder({7: definition})))
        .qubits(1)
        .capture_operation_trace()
    )
    restored = json.loads(json.dumps(trace))
    assert restored == trace
    assert restored[-1]["stage"] == "trace_complete"
    requests = [
        op["RuntimeRequest"]["request"]["Decode"]
        for chunk in restored
        for op in chunk["operations"]
        if "Decode" in op.get("RuntimeRequest", {}).get("request", {})
    ]
    assert [(r["window_id"], r["instance_id"], r["syndrome"]) for r in requests] == [(7, i, [1]) for i in range(3)]
    assert [RuntimeDecodeContext.from_json(bytes(r["context"])) if r["context"] else None for r in requests] == (
        contexts if versioned else [None] * 3
    )
    named = [item for chunk in restored for item in chunk.get("named_result_traces", [])]
    # The native getter's observed reads feed both the decoder buffer and these
    # named outputs. Projection is the independent control for live feedback.
    assert [(item["name"], item["values"]) for item in named] == [
        (name, value) for _ in range(3) for name, value in [("before", [True]), ("after", [not bool(route)])]
    ]
    ids = [result_id for item in named for result_id in item["result_ids"]]
    assert len(ids) == len(set(ids)) == 6
    circuit = qis_operation_trace_to_tick_circuit(restored)
    assert circuit.num_measurements() == 6
