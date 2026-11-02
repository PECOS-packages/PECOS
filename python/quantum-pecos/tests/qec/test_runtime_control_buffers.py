"""Bounded detector buffers use the existing observed-data and slice contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
from pecos.qec import RuntimeDecoder, RuntimeDecodeSchema, RuntimeDetectorBinding, RuntimeReadSource, RuntimeRecordRef

from qec.test_runtime_control import builder
from qec.test_runtime_decode_context import binding, context
from qec.test_runtime_decoder import composed_window


def buffer_binding():
    return replace(
        binding(),
        schema=RuntimeDecodeSchema(
            detectors=tuple(
                RuntimeDetectorBinding(
                    detector_id=name,
                    records=(RuntimeRecordRef(name="readout", occurrence=i),),
                )
                for i, name in enumerate(("first", "second"))
            ),
            frame_ids=("frame_x",),
        ),
    )


def buffer_source():
    envelope = replace(
        context(),
        schema_id=buffer_binding().schema.schema_id,
        branches=(),
        loops=(),
        inputs=(),
    ).to_json()
    escaped = "".join(f"\\{byte:02X}" for byte in envelope)
    fixture = Path(__file__).resolve().parents[4] / "crates/pecos-qis/tests/fixtures/runtime_decode_control_buffer.ll"
    return (
        f'@ctx = private constant [{len(envelope)} x i8] c"{escaped}"\n'
        '@read_name = private constant [7 x i8] c"readout"\n'
        + fixture.read_text().replace("CONTEXT_LENGTH", str(len(envelope)))
    )


def checked_buffer(definition, instance):
    return replace(
        definition,
        binding=buffer_binding(),
        read_sources=tuple(
            RuntimeReadSource(record=RuntimeRecordRef(name="readout", occurrence=i), source_id=3 * instance + i)
            for i in range(2)
        ),
    )


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize("worker_context", [False, True])
@pytest.mark.parametrize("route", [(), (0,)])
def test_composed_multi_detector_feedback_with_control_loops_and_shot_reuse(runtime, worker_context, route):
    definition = composed_window(route)
    sessions = []

    def factory():
        seen = []
        sessions.append(seen)

        def provide(request):
            control = request.control
            kinds = {site.site_id: site.kind for site in control.manifest.sites}
            reads = [event.value for event in control.events if kinds[event.site_id] == "read_bool"]
            # The tail read from the previous iteration precedes this pair.
            assert reads[-2:] == [1, int(request.instance_id == 1)]
            frames = [event.value for event in control.events if kinds[event.site_id] == "decoder_frame"]
            assert frames == ([] if request.instance_id == 0 else [int(bool(route) and request.instance_id == 1)])
            assert [(loop.occurrence, loop.iteration) for loop in control.active_loops] == [(0, request.instance_id)]
            assert control.first_batch_index == (seen[-1].next_batch_index if seen else 0)
            seen.append(control)
            return checked_buffer(definition, request.instance_id)

        return provide

    runner = builder(RuntimeDecoder(provider_factory=factory), runtime=runtime, program=buffer_source())
    runner = runner.workers(2).python_worker_context(worker_context).build()
    # Independent oracle: D=[1,0], [1,1], [1,0], and L0 = D0 XOR D1.
    expected = [0, 0, 0] if route else [1, 0, 1]
    for _ in range(2):
        assert runner.run(4).to_dict()["after"] == [expected] * 4
    assert len(sessions) == 8
    assert all(len(seen) == 3 and seen[0].events[0].occurrence == 0 for seen in sessions)


def test_buffer_source_failure_restarts_and_second_detector_parity_is_checked():
    definition = composed_window()
    failed = False
    seen = []

    def provide(request):
        nonlocal failed
        seen.append(request.instance_id)
        result = checked_buffer(definition, request.instance_id)
        if request.instance_id == 1 and not failed:
            failed = True
            return replace(result, read_sources=(result.read_sources[0], replace(result.read_sources[1], source_id=99)))
        return result

    runner = builder(RuntimeDecoder(provider_factory=lambda: provide), program=buffer_source()).build()
    with pytest.raises(Exception, match="measurement identities"):
        runner.run(1)
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert seen == [0, 1, 0, 1, 2]
    bad = buffer_source().replace("store i8 %b_byte, ptr %second", "store i8 1, ptr %second")
    with pytest.raises(Exception, match="parity disagrees with detector second"):
        builder(RuntimeDecoder(provider_factory=lambda: provide), program=bad).run(1)


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_buffer_control_capture_preserves_physical_operations_and_timing(runtime):
    definition = composed_window()
    decoder = RuntimeDecoder(provider_factory=lambda: lambda r: checked_buffer(definition, r.instance_id))
    options = {"runtime": runtime, "program": buffer_source()}
    captured = builder(decoder, **options).capture_operation_trace()
    baseline = builder(decoder, **options, capture=False).capture_operation_trace()
    assert [gate for chunk in captured for gate in chunk["lowered_quantum_ops"]] == [
        gate for chunk in baseline for gate in chunk["lowered_quantum_ops"]
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("store i8 %b_byte, ptr %second", ""),
        lambda s: s.replace("store i8 %b_byte, ptr %second", "store i8 %b_byte, ptr %data"),
        lambda s: s.replace("ptr %data, i64 2, ptr @ctx", "ptr %data, i64 3, ptr @ctx"),
        lambda s: s.replace("ptr %buffer, i64 0, i64 2", "ptr %buffer, i64 0, i64 %shot"),
    ],
)
def test_invalid_buffer_rejected_before_provider(mutate):
    def unexpected():
        pytest.fail("provider reached unsupported buffer")

    with pytest.raises(Exception, match="compiler control provenance unsupported"):
        builder(unexpected, program=mutate(buffer_source())).build()
