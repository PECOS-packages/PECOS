"""Versioned declarations are checked before live decoder feedback is returned."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from pecos import Qis, selene_engine, sim
from pecos.qec import (
    RuntimeBranchRecord,
    RuntimeClassicalInput,
    RuntimeDecodeBinding,
    RuntimeDecodeContext,
    RuntimeDecoder,
    RuntimeDecodeSchema,
    RuntimeDetectorBinding,
    RuntimeInputDependency,
    RuntimeLoopRecord,
    RuntimeRecordRef,
)

from qec.test_runtime_decoder import PROGRAM, window


def binding():
    return RuntimeDecodeBinding(
        program_revision="synthetic-reset-loop-v1",
        compiler_revision="synthetic-llvm-v1",
        decode_site="qis:qmain/decode",
        schema=RuntimeDecodeSchema(
            detectors=(
                RuntimeDetectorBinding(detector_id="parity", records=(RuntimeRecordRef(name="readout", occurrence=0),)),
            ),
            frame_ids=("frame_x",),
        ),
    )


def context(instance=0, *, corrected=True):
    expected = binding()
    return RuntimeDecodeContext(
        version=1,
        program_revision=expected.program_revision,
        compiler_revision=expected.compiler_revision,
        decode_site=expected.decode_site,
        schema_id=expected.schema.schema_id,
        branches=(
            ()
            if instance == 0
            else (
                RuntimeBranchRecord(site_id="qis:qmain/correct", occurrence=0, outcome=corrected),
                RuntimeBranchRecord(site_id="qis:qmain/more", occurrence=0, outcome=True),
            )
        ),
        loops=(RuntimeLoopRecord(site_id="qis:qmain/loop", occurrence=0, iteration=instance),),
        inputs=(RuntimeClassicalInput(name="round", kind="u64", value=instance, dependencies=()),),
    )


def program(payloads, *, source=PROGRAM):
    """Explicit synthetic frontend instrumentation; no trace-to-model inference."""
    constants = []
    for i, data in enumerate(payloads):
        escaped = "".join(f"\\{byte:02X}" for byte in data)
        constants.append(f'@context{i} = private constant [{len(data)} x i8] c"{escaped}"')
    constants.append("@contexts = private constant [3 x ptr] [ptr @context0, ptr @context1, ptr @context2]")
    constants.append("@lengths = private constant [3 x i64] [" + ", ".join(f"i64 {len(p)}" for p in payloads) + "]")
    source = source.replace(
        "declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)",
        "declare i64 @pecos_runtime_decode_v1(i64, i64, ptr, i64, ptr, i64)",
    ).replace(
        "%frame = call i64 @pecos_runtime_decode(i64 7, i64 %instance, ptr %data, i64 1)",
        """%context_slot = getelementptr [3 x ptr], ptr @contexts, i64 0, i64 %instance
    %context = load ptr, ptr %context_slot
    %length_slot = getelementptr [3 x i64], ptr @lengths, i64 0, i64 %instance
    %length = load i64, ptr %length_slot
    %frame = call i64 @pecos_runtime_decode_v1(i64 7, i64 %instance, ptr %data, i64 1, ptr %context, i64 %length)""",
    )
    return "\n".join(constants) + "\n" + source


def test_schema_fingerprint_has_a_fixed_cross_language_vector():
    # Hand-written wire schema, independent of dataclass serialization.
    import hashlib

    canonical = (
        b'{"detectors":[{"detector_id":"parity","records":[{"name":"readout","occurrence":0}],'
        b'"reference":false}],"frame_ids":["frame_x"],"version":1}'
    )
    assert binding().schema.schema_id == hashlib.sha256(canonical).hexdigest()
    detector = binding().schema.detectors[0]
    a = RuntimeRecordRef(name="a", occurrence=0)
    b = RuntimeRecordRef(name="a", occurrence=1)
    first = replace(binding().schema, detectors=(replace(detector, records=(a, b)),))
    second = replace(binding().schema, detectors=(replace(detector, records=(b, a)),))
    assert first.schema_id == second.schema_id
    assert first.schema_id != binding().schema.schema_id
    assert replace(first, detectors=(replace(first.detectors[0], reference=True),)).schema_id != first.schema_id
    with pytest.raises(ValueError, match="unique"):
        replace(detector, records=(a, a))


def test_context_round_trip_is_deeply_immutable():
    original = context(1)
    restored = RuntimeDecodeContext.from_json(original.to_json())
    assert restored == original
    assert hash(restored) == hash(original)
    with pytest.raises(FrozenInstanceError):
        restored.loops[0].iteration = 2
    mutable = [RuntimeRecordRef(name="read", occurrence=0)]
    detector = RuntimeDetectorBinding(detector_id="d", records=mutable)
    mutable.clear()
    assert len(detector.records) == 1


@pytest.mark.parametrize(
    "update",
    [
        {"version": 2},
        {"version": True},
        {"extra": "unknown"},
        {"schema_id": "0" * 63},
        {"schema_id": "A" * 64},
        {"decode_site": "unqualified"},
        {"program_revision": ""},
        {"compiler_revision": "\ud800"},
        {"branches": {}},
        {"loops": None},
        {"inputs": [[1, 2]]},
        {"branches": [{"site_id": "qis:b", "occurrence": 0, "outcome": 1}]},
        {"branches": [{"site_id": "qis:b", "occurrence": 1, "outcome": True}]},
        {"loops": [{"site_id": "hugr:l", "occurrence": 0, "iteration": -1}]},
        {"loops": [{"site_id": "hugr:l", "occurrence": True, "iteration": 0}]},
        {"inputs": [{"name": "x", "kind": "u64", "value": True, "dependencies": []}]},
        {"inputs": [{"name": "x", "kind": "i64", "value": 2**63, "dependencies": []}]},
        {"inputs": [{"name": "x", "kind": "u64", "value": 2**64, "dependencies": []}]},
        {"inputs": [{"name": "x", "kind": "f64", "value": 1.0, "dependencies": []}]},
        {"inputs": [{"name": "x", "kind": "bool", "value": 1, "dependencies": []}]},
        {"inputs": [{"name": "x", "kind": "u64", "value": 0, "dependencies": {}}]},
    ],
)
def test_strict_json_rejects_ambiguous_or_unsupported_declarations(update):
    raw = json.loads(context().to_json()) | update
    with pytest.raises(ValueError, match="invalid runtime decode context"):
        RuntimeDecodeContext.from_json(json.dumps(raw).encode())


@pytest.mark.parametrize("payload", [b"{}", b"[]", b"null", b"", b"\xff", b" " * 65537, b'{"a":NaN}'])
def test_invalid_envelopes(payload):
    with pytest.raises(ValueError, match="context"):
        RuntimeDecodeContext.from_json(payload)


def test_duplicate_keys_and_missing_fields_fail_and_latch_before_provider():
    for payload in [context().to_json()[:-1] + b',"version":1}', b"{}"]:
        calls = []
        session = RuntimeDecoder(provider_factory=lambda calls=calls: calls.append)()
        with pytest.raises(ValueError, match="context"):
            session.decode_v1(7, 0, b"\x01", payload)
        assert calls == []
        with pytest.raises(RuntimeError, match="session failed"):
            session.decode(7, 0, [1])


def test_schema_binds_detector_order_and_frame_destinations():
    definition = window()
    declared = binding()
    with pytest.raises(ValueError, match="detector identities"):
        replace(
            definition,
            binding=replace(
                declared,
                schema=replace(
                    declared.schema,
                    detectors=(replace(declared.schema.detectors[0], detector_id="different-read"),),
                ),
            ),
        )
    with pytest.raises(ValueError, match="undeclared schema frame"):
        replace(definition, binding=declared, output_map=((1,),))
    # Width is unchanged. A different named read changes identity and is rejected.
    wrong_schema = replace(
        declared.schema,
        detectors=(
            replace(declared.schema.detectors[0], records=(RuntimeRecordRef(name="other-read", occurrence=0),)),
        ),
    )
    session = RuntimeDecoder({7: replace(definition, binding=declared)})()
    with pytest.raises(ValueError, match="does not match"):
        session.decode_v1(7, 0, [1], replace(context(), schema_id=wrong_schema.schema_id).to_json())


@pytest.mark.parametrize("field", ["program_revision", "compiler_revision", "decode_site", "schema_id"])
def test_mismatch_of_any_binding_identity_rejected(field):
    value = "qis:other" if field == "decode_site" else "0" * 64
    session = RuntimeDecoder({7: replace(window(), binding=binding())})()
    with pytest.raises(ValueError, match="does not match"):
        session.decode_v1(7, 0, [1], replace(context(), **{field: value}).to_json())
    with pytest.raises(RuntimeError, match="session failed"):
        session.decode_v1(7, 0, [1], context().to_json())


def test_bound_and_legacy_calls_share_counter_but_never_silently_downgrade():
    bound = replace(window(), binding=binding())
    legacy = replace(window(), window_id=8)
    config = RuntimeDecoder({7: bound, 8: legacy})
    session = config()
    assert session.decode(8, 0, [1]) == 1
    assert session.decode_v1(7, 1, [1], context(1).to_json()) == 1
    assert session.decode(8, 2, [0]) == 0
    with pytest.raises(ValueError, match="requires a versioned"):
        config().decode(7, 0, [1])
    with pytest.raises(ValueError, match="does not match"):
        config().decode_v1(8, 0, [1], context().to_json())
    with pytest.raises(ValueError, match="contiguous"):
        session.decode_v1(7, 2, [1], context(2).to_json())


def test_dependency_kind_and_prior_instance_are_checked():
    previous = RuntimeInputDependency(source="decoder", name="frame_x", occurrence=0)
    value = RuntimeClassicalInput(name="frame", kind="bool", value=True, dependencies=(previous,))
    session = RuntimeDecoder({7: replace(window(), binding=binding())})()
    with pytest.raises(ValueError, match="prior instance"):
        session.decode_v1(7, 0, [1], replace(context(), inputs=(value,)).to_json())
    fresh = RuntimeDecoder({7: replace(window(), binding=binding())})()
    assert fresh.decode_v1(7, 0, [1], context().to_json()) == 1
    assert fresh.decode_v1(7, 1, [1], replace(context(1), inputs=(value,)).to_json()) == 1
    with pytest.raises(ValueError, match="source must be"):
        replace(previous, source="hidden_fault")


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize("route", [(), (0,)])
def test_live_versioned_context_feedback_loops_and_workers(runtime, route):
    definition = replace(window(), binding=binding(), output_map=(route,))
    expected_contexts = [context(i, corrected=bool(route)) for i in range(3)]
    sessions = []

    def factory():
        seen = []
        sessions.append(seen)

        def provide(request):
            assert vars(request).keys() == {"window_id", "instance_id", "context"}
            assert request.context == expected_contexts[request.instance_id]
            seen.append(request.context)
            return definition

        return provide

    runner = (
        sim(Qis(program([c.to_json() for c in expected_contexts])))
        .classical(selene_engine(runtime).runtime_decoder(RuntimeDecoder(provider_factory=factory)))
        .qubits(1)
        .workers(2)
        .seed(5)
        .build()
    )
    for _ in range(2):
        assert runner.run(4).to_dict()["after"] == [[0 if route else 1] * 3] * 4
    assert sessions == [expected_contexts] * 8


def test_live_wrong_schema_fails_before_frame_and_next_run_starts_fresh():
    good = replace(window(), binding=binding())
    wrong = replace(good, binding=replace(binding(), compiler_revision="other-compiler"))
    seen = []
    failed = False

    def factory():
        def provide(request):
            nonlocal failed
            seen.append(request.instance_id)
            if request.instance_id == 1 and not failed:
                failed = True
                return wrong
            return good

        return provide

    runner = (
        sim(Qis(program([context(i).to_json() for i in range(3)])))
        .classical(selene_engine().runtime_decoder(RuntimeDecoder(provider_factory=factory)))
        .qubits(1)
        .build()
    )
    with pytest.raises(Exception, match="does not match"):
        runner.run(1)
    assert seen == [0, 1]
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert seen == [0, 1, 0, 1, 2]


def test_live_legacy_python_callback_cannot_ignore_v1_context():
    class Legacy:
        def decode(self, *_):
            pytest.fail("versioned request was silently downgraded")

    with pytest.raises(Exception, match="decode_v1"):
        (
            sim(Qis(program([context(i).to_json() for i in range(3)])))
            .classical(selene_engine().runtime_decoder(Legacy))
            .qubits(1)
            .run(1)
        )


@pytest.mark.parametrize(
    ("pointer", "length", "payload", "message"),
    [
        ("null", "1", b"{}", "null"),
        ("%context", "0", b"{}", "context length"),
        ("%context", "65537", b"{}", "context length"),
        ("%context", "%length", b"\xff", "UTF-8"),
        ("%context", "%length", b"{}", "invalid runtime decode context"),
    ],
)
def test_live_bad_context_envelope_fails_without_provider(pointer, length, payload, message):
    source = program([payload] * 3).replace("ptr %context, i64 %length)", f"ptr {pointer}, i64 {length})")
    seen = []
    config = RuntimeDecoder(provider_factory=lambda: seen.append)
    with pytest.raises(Exception, match=message):
        sim(Qis(source)).classical(selene_engine().runtime_decoder(config)).qubits(1).run(1)
    assert seen == []


def test_two_detector_binding_allows_only_the_named_permutation():
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = DagCircuit()
    circuit.pz([0, 1])
    circuit.mz([0])
    circuit.mz([1])
    model = (
        DemBuilder(DagFaultAnalyzer(circuit).build_influence_map())
        .with_noise(0.0, 0.0, 0.1, 0.0)
        .with_detectors_json('[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]')
        .with_observables_json('[{"id":0,"records":[-2]}]')
        .build()
    )
    schema = RuntimeDecodeSchema(
        detectors=tuple(
            RuntimeDetectorBinding(detector_id=name, records=(RuntimeRecordRef(name=name, occurrence=0),))
            for name in ("second", "first")
        ),
        frame_ids=("frame_x",),
    )
    bound = replace(binding(), schema=schema)
    definition = replace(
        window(),
        model=model,
        detector_ids=("first", "second"),
        commit=("first", "second"),
        detector_order=(1, 0),
        binding=bound,
    )
    with pytest.raises(ValueError, match="detector identities"):
        replace(definition, detector_order=(0, 1))
    session = RuntimeDecoder({7: definition})()
    payload = replace(context(), schema_id=schema.schema_id).to_json()
    # The semantic first read occupies program byte 1 and drives L0.
    assert session.decode_v1(7, 0, [0, 1], payload) == 1
    assert session.decode_v1(7, 1, [1, 0], payload) == 0
