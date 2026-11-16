"""Schema-driven QIS lowering preserves observed-data checks and live feedback."""

import runpy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from pecos.qec import RuntimeDecodeQisSite, RuntimeDecoder, RuntimeDetectorBinding, RuntimeRecordRef

from qec.test_runtime_control import builder
from qec.test_runtime_control_buffers import buffer_source
from qec.test_runtime_decoder import composed_window

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[4] / "examples/runtime_decoder_qis.py"))
FIRST, SECOND = EXAMPLE["FIRST"], EXAMPLE["SECOND"]
site_definition, program = EXAMPLE["decode_site"], EXAMPLE["program"]


def bound(site, window, instance):
    return site.bind_window(window, source_ids={FIRST: 3 * instance, SECOND: 3 * instance + 1})


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize("worker_context", [False, True])
@pytest.mark.parametrize("route", [(), (0,)])
def test_lowered_live_feedback_permutation_and_reuse(runtime, worker_context, route):
    original = site_definition()
    # Reverse program byte order; the model retains its original column order.
    site = replace(
        original,
        binding=replace(
            original.binding,
            schema=replace(
                original.binding.schema,
                detectors=tuple(reversed(original.binding.schema.detectors)),
            ),
        ),
    )
    window = composed_window(route)
    seen = []

    def factory():
        requests = []
        seen.append(requests)

        def provide(request):
            assert request.control.manifest.instrumenter == "pecos-qis-control-v1"
            requests.append(request.instance_id)
            definition = bound(site, window, request.instance_id)
            assert definition.detector_order == (1, 0)
            return definition

        return provide

    runner = builder(RuntimeDecoder(provider_factory=factory), runtime=runtime, program=program(site))
    runner = runner.workers(2).python_worker_context(worker_context).build()
    for _ in range(2):
        assert runner.run(4).to_dict()["after"] == ([[0, 0, 0]] if route else [[1, 0, 1]]) * 4
    assert seen == [[0, 1, 2]] * 8


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_lowering_preserves_handwritten_physical_operations_and_timing(runtime):
    site, window = site_definition(), composed_window()
    decoder = RuntimeDecoder(provider_factory=lambda: lambda r: bound(site, window, r.instance_id))
    generated = builder(decoder, runtime=runtime, program=program(site)).capture_operation_trace()
    from qec.test_runtime_control_buffers import checked_buffer

    handwritten = builder(
        RuntimeDecoder(provider_factory=lambda: lambda r: checked_buffer(window, r.instance_id)),
        runtime=runtime,
        program=buffer_source(),
    ).capture_operation_trace()
    assert [g for c in generated for g in c["lowered_quantum_ops"]] == [
        g for c in handwritten for g in c["lowered_quantum_ops"]
    ]


def test_generated_shared_parity_reference_and_constant_detectors():
    original = site_definition()
    schema = replace(
        original.binding.schema,
        detectors=(
            RuntimeDetectorBinding(detector_id="odd", records=(FIRST, SECOND), reference=True),
            RuntimeDetectorBinding(detector_id="shared", records=(SECOND,)),
            RuntimeDetectorBinding(detector_id="constant", records=(), reference=True),
        ),
    )
    site = replace(original, binding=replace(original.binding, schema=schema))
    seen = []

    class Session:
        def decode(self, *_):
            pytest.fail("legacy fallback")

        def decode_observed_with_control_v1(self, _window, _instance, syndrome, _context, reads, *_):
            # The expected vectors below are derived directly, not from emitted IR.
            seen.append((list(syndrome), [r[3] for r in reads]))
            return 0

    builder(Session, program=program(site)).run(1)
    assert seen == [([0, 0, 1], [True, False]), ([1, 1, 1], [True, True]), ([0, 0, 1], [True, False])]


def test_source_failure_recovers_and_tampered_parity_still_rejects():
    site, window = site_definition(), composed_window()
    fail = True
    seen = []

    def provide(request):
        nonlocal fail
        seen.append(request.instance_id)
        if request.instance_id == 1 and fail:
            fail = False
            return site.bind_window(window, source_ids={FIRST: 3, SECOND: 99})
        return bound(site, window, request.instance_id)

    decoder = RuntimeDecoder(provider_factory=lambda: provide)
    runner = builder(decoder, program=program(site)).build()
    with pytest.raises(Exception, match="measurement identities"):
        runner.run(1)
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert seen == [0, 1, 0, 1, 2]
    tampered = program(site).replace("store i8 %pecos.decode.parity.byte1", "store i8 1")
    with pytest.raises(Exception, match="parity disagrees"):
        builder(decoder, program=tampered).run(1)


def test_unicode_names_have_exact_byte_lengths_and_safe_llvm_escaping():
    label = 'détect"\\'
    a, b = RuntimeRecordRef(name=label, occurrence=0), RuntimeRecordRef(name=label, occurrence=1)
    site = site_definition()
    schema = replace(
        site.binding.schema,
        detectors=(
            RuntimeDetectorBinding(detector_id="first", records=(a,)),
            RuntimeDetectorBinding(detector_id="second", records=(b,)),
        ),
    )
    changed = replace(site, binding=replace(site.binding, schema=schema, program_revision='revision"\\é'))
    # Preserve the example's physical body but substitute the two generated reads.
    source = (
        program(site)
        .replace(site.globals_ir(), changed.globals_ir())
        .replace(
            site.read_ir(FIRST, future="%m0"),
            changed.read_ir(a, future="%m0"),
        )
        .replace(site.read_ir(SECOND, future="%m1"), changed.read_ir(b, future="%m1"))
        .replace(
            site.decode_ir(instance="%instance"),
            changed.decode_ir(instance="%instance"),
        )
    )
    assert f"i64 {len(label.encode())}, i64 %m0" in source
    window = composed_window()
    decoder = RuntimeDecoder(
        provider_factory=lambda: lambda r: changed.bind_window(
            window,
            source_ids={a: 3 * r.instance_id, b: 3 * r.instance_id + 1},
        ),
    )
    assert builder(decoder, program=source).run(1).to_dict()["after"] == [[0, 0, 0]]


@pytest.mark.parametrize("name", ["", "0bad", "bad-name", "bad\nname", "a" * 65, "é"])
def test_invalid_site_names_reject(name):
    with pytest.raises(ValueError, match="site name"):
        replace(site_definition(), name=name)


@pytest.mark.parametrize("operand", ["%0", "0", "%x\nret i64 0", "@global", "true", ""])
def test_unsafe_or_unsupported_ssa_operands_reject(operand):
    site = site_definition()
    with pytest.raises(ValueError, match="named LLVM SSA"):
        site.read_ir(FIRST, future=operand)
    with pytest.raises(ValueError, match="named LLVM SSA"):
        site.decode_ir(instance=operand)


@pytest.mark.parametrize("value", [True, -1, 2**64, 0.5, None])
def test_invalid_unsigned_values_reject(value):
    with pytest.raises(ValueError, match="unsigned"):
        replace(site_definition(), window_id=value)
    with pytest.raises(ValueError, match="unsigned"):
        site_definition().decode_ir(instance=value)


def test_site_is_immutable_and_rejects_invalid_schemas():
    site = site_definition()
    with pytest.raises(FrozenInstanceError):
        site.name = "changed"
    bad_schemas = [
        replace(site.binding.schema, detectors=()),
        replace(
            site.binding.schema,
            detectors=tuple(RuntimeDetectorBinding(detector_id=str(i), records=()) for i in range(4097)),
        ),
        replace(
            site.binding.schema,
            detectors=(
                RuntimeDetectorBinding(
                    detector_id="gap",
                    records=(RuntimeRecordRef(name="readout", occurrence=1),),
                ),
            ),
        ),
        replace(
            site.binding.schema,
            detectors=(
                RuntimeDetectorBinding(
                    detector_id="long",
                    records=(RuntimeRecordRef(name="é" * 129, occurrence=0),),
                ),
            ),
        ),
    ]
    for schema in bad_schemas:
        with pytest.raises(ValueError, match=r"(detectors|contiguous|UTF-8)"):
            replace(site, binding=replace(site.binding, schema=schema))
    with pytest.raises(ValueError, match="not in the site"):
        site.read_ir(RuntimeRecordRef(name="absent", occurrence=0), future="%m")


def test_window_binding_derives_only_declared_mappings_and_rejects_conflicts():
    site, window = site_definition(), composed_window()
    result = bound(site, window, 0)
    assert result.model is window.model
    for name in ("output_map", "fault_ids", "commit", "independent", "decoder"):
        assert getattr(result, name) == getattr(window, name)
    assert bound(site, result, 0) == result
    assert bound(site, replace(result, read_sources=tuple(reversed(result.read_sources))), 0) == result
    for bad in [replace(window, window_id=8), replace(result, binding=replace(site.binding, program_revision="other"))]:
        with pytest.raises(ValueError, match=r"(window identity|different binding)"):
            bound(site, bad, 0)
    with pytest.raises(ValueError, match="different read sources"):
        bound(site, result, 1)
    with pytest.raises(ValueError, match="cover exactly"):
        site.bind_window(window, source_ids={FIRST: 0})


def test_duplicate_and_missing_read_fragments_reject_before_provider():
    site = site_definition()
    source = program(site)
    read = site.read_ir(FIRST, future="%m0")

    def unexpected():
        pytest.fail("invalid program reached provider")

    for bad in (source.replace(read, ""), source.replace(read, read + read)):
        with pytest.raises(Exception, match=r"(llvm-as|Invalid|invalid)"):
            builder(unexpected, program=bad).build()


def test_two_constant_sites_share_imports_and_accept_explicit_empty_sources():
    original = site_definition()
    schema = replace(
        original.binding.schema,
        detectors=(
            RuntimeDetectorBinding(detector_id="first", records=()),
            RuntimeDetectorBinding(detector_id="second", records=(), reference=True),
        ),
    )
    first = replace(original, binding=replace(original.binding, schema=schema))
    second = replace(first, name="second", window_id=8, binding=replace(first.binding, decode_site="qis:qmain/second"))
    source = first.declarations_ir() + first.globals_ir() + second.globals_ir() + f"""
@label = private constant [5 x i8] c"after"
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {{
entry:
{first.entry_ir()}
{second.entry_ir()}
{first.decode_ir(instance=0)}
{second.decode_ir(instance=1)}
%a = icmp ne i64 {first.frame_value}, 0
%b = icmp ne i64 {second.frame_value}, 0
call void @print_bool_selene(ptr @label, i64 5, i1 %a)
call void @print_bool_selene(ptr @label, i64 5, i1 %b)
ret i64 0
}}
attributes #0 = {{ "EntryPoint" }}
"""
    window = composed_window()
    windows = {
        7: first.bind_window(window, source_ids={}),
        8: second.bind_window(replace(window, window_id=8), source_ids={}),
    }
    assert windows[7].read_sources == ()
    assert builder(RuntimeDecoder(windows), program=source).run(2).to_dict()["after"] == [[1, 1]] * 2
    # Each fragment is bounded locally; the whole-module storage budget still applies.
    oversized = source.replace("alloca [2 x i8]", "alloca [4096 x i8]").replace(
        "getelementptr inbounds [2 x i8]",
        "getelementptr inbounds [4096 x i8]",
    )
    with pytest.raises(Exception, match="allocation byte limit"):
        builder(RuntimeDecoder(windows), program=oversized).build()


def test_incorrect_same_name_read_occurrence_order_is_not_silently_rebound():
    site, window = site_definition(), composed_window()
    source = (
        program(site)
        .replace(site.read_ir(FIRST, future="%m0"), site.read_ir(SECOND, future="%m0"))
        .replace(
            site.read_ir(SECOND, future="%m1"),
            site.read_ir(FIRST, future="%m1"),
        )
    )
    decoder = RuntimeDecoder(provider_factory=lambda: lambda r: bound(site, window, r.instance_id))
    with pytest.raises(Exception, match="parity disagrees"):
        builder(decoder, program=source).run(1)


def test_runnable_example():
    EXAMPLE["main"]()


def test_nonconstant_sites_reset_same_name_occurrences_in_live_loop():
    from qec.test_runtime_decoder import window

    original = site_definition()
    schema = replace(
        original.binding.schema,
        detectors=(RuntimeDetectorBinding(detector_id="parity", records=(FIRST,)),),
    )
    first = replace(original, binding=replace(original.binding, schema=schema))
    second = replace(first, name="second", window_id=8, binding=replace(first.binding, decode_site="qis:qmain/second"))
    source = first.declarations_ir() + first.globals_ir() + second.globals_ir() + f"""
@label = private constant [5 x i8] c"after"
@frame_label = private constant [5 x i8] c"frame"
declare i64 @___qalloc()
declare void @___qfree(i64)
declare void @___reset(i64)
declare void @___rxy(i64, double, double)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___dec_future_refcount(i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {{
entry:
%q = call i64 @___qalloc()
{first.entry_ir()}
{second.entry_ir()}
br label %loop
loop:
%round = phi i64 [0, %entry], [%next, %tail]
%i = mul i64 %round, 2
%j = add i64 %i, 1
call void @___reset(i64 %q)
call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
%m0 = call i64 @___lazy_measure(i64 %q)
{first.read_ir(FIRST, future="%m0")}
call void @___dec_future_refcount(i64 %m0)
{first.decode_ir(instance="%i")}
%first_frame = icmp ne i64 {first.frame_value}, 0
call void @print_bool_selene(ptr @frame_label, i64 5, i1 %first_frame)
call void @___reset(i64 %q)
call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
%m1 = call i64 @___lazy_measure(i64 %q)
{second.read_ir(FIRST, future="%m1")}
call void @___dec_future_refcount(i64 %m1)
{second.decode_ir(instance="%j")}
%correct = icmp ne i64 {second.frame_value}, 0
br i1 %correct, label %apply, label %tail
apply:
call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
br label %tail
tail:
%after = call i64 @___lazy_measure(i64 %q)
%value = call i1 @___read_future_bool(i64 %after)
call void @___dec_future_refcount(i64 %after)
call void @print_bool_selene(ptr @label, i64 5, i1 %value)
%next = add i64 %round, 1
%more = icmp ult i64 %next, 3
br i1 %more, label %loop, label %done
done:
call void @___qfree(i64 %q)
ret i64 0
}}
attributes #0 = {{ "EntryPoint" }}
"""
    windows = {7: window(), 8: replace(window(), window_id=8)}
    sites = {7: first, 8: second}
    calls = []

    def factory():
        seen = []
        calls.append(seen)

        def provide(request):
            seen.append((request.window_id, request.instance_id))
            return sites[request.window_id].bind_window(
                windows[request.window_id],
                source_ids={
                    FIRST: 3 * (request.instance_id // 2) + request.instance_id % 2,
                },
            )

        return provide

    runner = builder(RuntimeDecoder(provider_factory=factory), program=source).workers(2).build()
    for _ in range(2):
        result = runner.run(2).to_dict()
        assert result["frame"] == [[1, 1, 1]] * 2
        assert result["after"] == [[0, 0, 0]] * 2
    assert calls == [[(7 + i % 2, i) for i in range(6)]] * 4
