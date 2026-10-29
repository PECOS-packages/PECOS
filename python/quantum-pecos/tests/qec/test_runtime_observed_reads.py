"""Checked source bindings use native getter results, including live feedback."""

from dataclasses import replace

import pytest
from pecos import Qis, selene_engine, sim
from pecos.qec import RuntimeDecoder, RuntimeReadSource, RuntimeRecordRef

from qec.test_runtime_decode_context import binding, context, program
from qec.test_runtime_decoder import window
from qec.test_runtime_decoder_transport import NATIVE_PROGRAM


def checked_window(instance=0, **updates):
    # This explicit arithmetic is correct for this fixed synthetic schedule only:
    # before/after measurements alternate, and each read interval has one read.
    return replace(
        window(),
        binding=binding(),
        read_sources=(
            RuntimeReadSource(record=RuntimeRecordRef(name="readout", occurrence=0), source_id=2 * instance),
        ),
        **updates,
    )


def observed_program(source=NATIVE_PROGRAM):
    source = program([context(i).to_json() for i in range(3)], source=source)
    return (
        '@read_name = private constant [7 x i8] c"readout"\n'
        "declare i1 @pecos_runtime_read_bool_v1(ptr, i64, i64)\n"
        + source.replace("pecos_runtime_decode_v1", "pecos_runtime_decode_observed_v1").replace(
            "%observed = call i1 @___read_future_bool(i64 %m)",
            "%observed = call i1 @pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %m)",
        )
    )


def config():
    return RuntimeDecoder(provider_factory=lambda: lambda request: checked_window(request.instance_id))


def run(source, decoder=None):
    return sim(Qis(source)).classical(selene_engine().runtime_decoder(decoder or config())).qubits(1).run(1)


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize("route", [(), (0,)])
def test_live_checked_reads_control_feedback_loops_and_shot_worker_reuse(runtime, route):
    sessions = []

    def factory():
        seen = []
        sessions.append(seen)

        def provide(request):
            assert vars(request).keys() == {"window_id", "instance_id", "context"}
            seen.append(request.instance_id)
            return checked_window(request.instance_id, output_map=(route,))

        return provide

    runner = (
        sim(Qis(observed_program()))
        .classical(selene_engine(runtime).runtime_decoder(RuntimeDecoder(provider_factory=factory)))
        .qubits(1)
        .workers(2)
        .build()
    )
    for _ in range(2):
        assert runner.run(4).to_dict()["after"] == [[0 if route else 1] * 3] * 4
    assert sessions == [[0, 1, 2]] * 8


def test_equal_valued_wrong_measurement_source_is_rejected_before_feedback():
    source = NATIVE_PROGRAM.replace(
        "%observed = call i1 @___read_future_bool(i64 %m)",
        "%other = call i64 @___lazy_measure(i64 %q)\n"
        "    %observed = call i1 @___read_future_bool(i64 %m)\n"
        "    call void @___dec_future_refcount(i64 %other)",
    )
    source = observed_program(source).replace(
        "@pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %m)",
        "@pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %other)",
    )
    # Both measurements are deterministically one, so parity alone would pass.
    with pytest.raises(Exception, match="measurement identities"):
        run(source)


def test_live_failure_restarts_sources_and_occurrences_and_bad_parity_is_rejected():
    failed = False
    seen = []

    def factory():
        def provide(request):
            nonlocal failed
            seen.append(request.instance_id)
            definition = checked_window(request.instance_id)
            if request.instance_id == 1 and not failed:
                failed = True
                return replace(definition, read_sources=(replace(definition.read_sources[0], source_id=99),))
            return definition

        return provide

    runner = (
        sim(Qis(observed_program()))
        .classical(selene_engine().runtime_decoder(RuntimeDecoder(provider_factory=factory)))
        .qubits(1)
        .build()
    )
    with pytest.raises(Exception, match="measurement identities"):
        runner.run(1)
    assert seen == [0, 1]
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert seen == [0, 1, 0, 1, 2]
    with pytest.raises(Exception, match="parity disagrees"):
        run(observed_program().replace("store i8 %bit, ptr %data", "store i8 0, ptr %data"))


@pytest.mark.parametrize("mode", ["legacy", "v1"])
def test_every_decode_boundary_consumes_reads_even_without_observed_dispatch(mode):
    source = observed_program()
    if mode == "v1":
        declaration = "declare i64 @pecos_runtime_decode_v1(i64, i64, ptr, i64, ptr, i64)"
        call = "@pecos_runtime_decode_v1(i64 8, i64 %instance, ptr %data, i64 1, ptr %context, i64 %length)"
    else:
        declaration = "declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)"
        call = "@pecos_runtime_decode(i64 8, i64 %instance, ptr %data, i64 1)"
    source = (
        declaration
        + "\n"
        + source.replace(
            "%frame = call i64 @pecos_runtime_decode_observed_v1"
            "(i64 7, i64 %instance, ptr %data, i64 1, ptr %context, i64 %length)",
            f"""%old = icmp eq i64 %instance, 1
    br i1 %old, label %plain, label %checked
plain:
    %plain_frame = call i64 {call}
    br label %decoded
checked:
    %checked_frame = call i64 @pecos_runtime_decode_observed_v1(
        i64 7, i64 %instance, ptr %data, i64 1, ptr %context, i64 %length)
    br label %decoded
decoded:
    %frame = phi i64 [%plain_frame, %plain], [%checked_frame, %checked]""",
        )
    )
    plain = replace(window(), window_id=8, binding=binding() if mode == "v1" else None)
    decoder = RuntimeDecoder(
        provider_factory=lambda: lambda request: (
            plain if request.window_id == 8 else checked_window(request.instance_id)
        ),
    )
    assert run(source, decoder).to_dict()["after"] == [[0, 0, 0]]


def test_live_repeated_reads_get_distinct_occurrences_and_empty_interval_is_not_absent():
    source = observed_program().replace(
        "call void @print_bool_selene(ptr @before, i64 6, i1 %observed)",
        "%again = call i1 @pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %m)",
    )
    captures = []

    class Session:
        def decode(self, *_):
            pytest.fail("unexpected fallback")

        def decode_observed_v1(self, _window_id, _instance, _syndrome, _payload, observations):
            captures.append(observations)
            return 1

    assert run(source, Session).to_dict()["after"] == [[0, 0, 0]]
    assert captures == [[("readout", 0, 2 * i, True), ("readout", 1, 2 * i, True)] for i in range(3)]
    captures.clear()
    source = source.replace(
        "@pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %m)",
        "@___read_future_bool(i64 %m)",
    )
    assert run(source, Session).to_dict()["after"] == [[0, 0, 0]]
    assert captures == [[], [], []]


def test_observed_calls_do_not_fall_back_to_declared_only_sessions():
    class Session:
        def decode(self, *_):
            pytest.fail("legacy fallback")

        def decode_v1(self, *_):
            pytest.fail("declared-only fallback")

    with pytest.raises(Exception, match="decode_observed_v1"):
        run(observed_program(), Session)
    with pytest.raises(Exception, match="checked windows require"):
        run(observed_program().replace("decode_observed_v1", "decode_v1"))
    with pytest.raises(Exception, match="require read_sources"):
        run(observed_program(), RuntimeDecoder({7: replace(window(), binding=binding())}))


@pytest.mark.parametrize(
    ("pointer", "length", "message"),
    [("null", "7", "null"), ("@read_name", "0", "name length"), ("@read_name", "257", "name length")],
)
def test_invalid_read_names_fail_before_getter(pointer, length, message):
    source = observed_program().replace("ptr @read_name, i64 7, i64 %m", f"ptr {pointer}, i64 {length}, i64 %m")
    with pytest.raises(Exception, match=message):
        run(source)


@pytest.mark.parametrize(
    "observations",
    [
        [],
        None,
        [("wrong", 0, 0, True)],
        [("readout", 1, 0, True)],
        [("readout", 0, 1, True)],
        [("readout", 0, 0, 1)],
        [("readout", 0, 0, False)],
        [("readout", 0, True, True)],
        [("readout", 0, 2**64 - 2, True)],
        [("readout", 0, 0, True)] * 2,
    ],
)
def test_invalid_observations_latch_session_failure(observations):
    session = RuntimeDecoder({7: checked_window()})()
    with pytest.raises((ValueError, TypeError)):
        session.decode_observed_v1(7, 0, [1], context().to_json(), observations)
    with pytest.raises(RuntimeError, match="session failed"):
        session.decode_observed_v1(7, 0, [1], context().to_json(), [("readout", 0, 0, True)])


def test_constants_reference_signs_and_exact_expected_source_coverage():
    definition = checked_window()
    ref = definition.read_sources[0]
    with pytest.raises(ValueError, match="exactly"):
        replace(definition, read_sources=())
    with pytest.raises(ValueError, match="exactly"):
        replace(definition, read_sources=(ref, ref))
    with pytest.raises(ValueError, match="schema binding"):
        replace(definition, binding=None)
    detector = replace(binding().schema.detectors[0], records=(), reference=True)
    schema = replace(binding().schema, detectors=(detector,))
    constant = replace(definition, binding=replace(binding(), schema=schema), read_sources=())
    payload = replace(context(), schema_id=schema.schema_id).to_json()
    assert RuntimeDecoder({7: constant})().decode_observed_v1(7, 0, [1], payload, []) == 1
    with pytest.raises(ValueError, match="parity disagrees"):
        RuntimeDecoder({7: constant})().decode_observed_v1(7, 0, [0], payload, [])
    with pytest.raises(ValueError, match="require"):
        RuntimeDecoder({7: constant})().decode_v1(7, 0, [1], payload)


def test_multiple_read_parity_uses_reference_sign_and_runtime_occurrences():
    first = RuntimeRecordRef(name="readout", occurrence=0)
    second = RuntimeRecordRef(name="readout", occurrence=1)
    detector = replace(binding().schema.detectors[0], records=(first, second), reference=True)
    schema = replace(binding().schema, detectors=(detector,))
    definition = replace(
        checked_window(),
        binding=replace(binding(), schema=schema),
        read_sources=(RuntimeReadSource(record=first, source_id=0), RuntimeReadSource(record=second, source_id=1)),
    )
    payload = replace(context(), schema_id=schema.schema_id).to_json()
    observations = [("readout", 0, 0, True), ("readout", 1, 1, False)]
    assert RuntimeDecoder({7: definition})().decode_observed_v1(7, 0, [0], payload, observations) == 0
    with pytest.raises(ValueError, match="parity disagrees"):
        RuntimeDecoder({7: definition})().decode_observed_v1(7, 0, [1], payload, observations)
    with pytest.raises(ValueError, match="contiguous"):
        replace(
            definition,
            binding=replace(binding(), schema=replace(schema, detectors=(replace(detector, records=(second,)),))),
            read_sources=(definition.read_sources[1],),
        )


def test_observed_trace_preserves_sources_and_live_feedback_through_json_and_replay():
    import json

    from pecos import qis_operation_trace_to_tick_circuit

    trace = (
        sim(Qis(observed_program()))
        .classical(selene_engine().runtime_decoder(config()))
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
    assert [request["observed_reads"] for request in requests] == [
        [{"name": "readout", "occurrence": 0, "source_id": 2 * i, "value": True}] for i in range(3)
    ]
    named = [item for chunk in restored for item in chunk.get("named_result_traces", [])]
    assert [(item["name"], item["values"]) for item in named] == [
        (name, value) for _ in range(3) for name, value in [("before", [True]), ("after", [False])]
    ]
    assert [item["result_ids"] for item in named] == [[i] for i in range(6)]
    assert qis_operation_trace_to_tick_circuit(restored).num_measurements() == 6
