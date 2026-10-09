"""Checked local faults through the native scheduled sim() route."""

import math

import pytest


class PassThrough:
    def validate(self, batch):
        assert all(not isinstance(op, tuple) for op in batch.operations)

    def translate(self, batch):
        return batch.operations


def simulation(library, program, profile, events, *, idle_only=False, adapter_factory=None):
    import pecos
    import pecos_rslib as pr
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    classical = pr.qis_engine().selene_runtime_plugin(str(library), SimpleRuntimePlugin().get_init_args())
    if events:
        classical.scheduled_event_batches()
        factory = pr.scheduled_event_idle_noise if idle_only else pr.scheduled_event_local_noise
        profile = factory(profile, adapter_factory or (lambda _: PassThrough()))
    else:
        classical.scheduled_batches()
    return (
        pecos.sim(pecos.Qis(program))
        .classical(classical.interface(pr.qis_helios_interface()))
        .qubits(1)
        .quantum(pr.state_vector())
        .noise(profile)
        .seed(42)
        .workers(2)
    )


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize(
    ("faults", "expected"),
    [({"prep": 1.0}, 1), ({"meas0": 1.0}, 1), ({"prep": 1.0, "meas1": 1.0}, 0)],
)
def test_local_faults_and_repeated_worker_runs(tmp_path, scheduled_support, events, faults, expected):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 0)
    profile = pr.scheduled_local_noise(pr.scheduled_idle_noise(1), **faults)
    program = scheduled_support.ramsey.replace(
        "declare void @setup",
        "declare void @___reset(i64)\ndeclare void @setup",
    ).replace("  %q = call i64 @___qalloc()", "  %q = call i64 @___qalloc()\n  call void @___reset(i64 %q)")
    built = simulation(library, program, profile, events).build()
    for shots in (4, 6):
        assert built.run(shots).to_dict()["measurement_0"] == [expected] * shots


@pytest.mark.parametrize("events", [False, True])
def test_zero_local_faults_preserve_seeded_idle_output(tmp_path, scheduled_support, events):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 1_000_000_000)
    idle = pr.scheduled_idle_noise(1, linear=0.2, coherent=0.4)
    local = pr.scheduled_local_noise(idle)
    actual = simulation(library, scheduled_support.ramsey, local, events)
    expected = simulation(library, scheduled_support.ramsey, idle, events, idle_only=True)
    assert actual.run(32).to_dict() == expected.run(32).to_dict()


@pytest.mark.parametrize("parameter", ["p1", "p2", "prep", "meas0", "meas1"])
@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, math.nan])
def test_invalid_probability(parameter, value):
    import pecos_rslib as pr

    with pytest.raises(ValueError, match="probabilities"):
        pr.scheduled_local_noise(pr.scheduled_idle_noise(1), **{parameter: value})


@pytest.mark.parametrize("parameter", ["emission", "crosstalk", "idle_after_2q", "p2_angle_params"])
def test_factory_signature_rejects_unsupported_keywords(parameter):
    import pecos_rslib as pr

    with pytest.raises(TypeError):
        pr.scheduled_local_noise(pr.scheduled_idle_noise(1), **{parameter: 0.1})


@pytest.mark.parametrize("events", [False, True])
def test_noisy_measurement_drives_runtime_feedback(tmp_path, scheduled_support, events):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(tmp_path, 0)
    program = (
        scheduled_support.ramsey.replace(
            "declare void @setup",
            "declare void @___reset(i64)\ndeclare i1 @___read_future_bool(i64)\ndeclare void @setup",
        )
        .replace(
            "  %q = call i64 @___qalloc()",
            "  %q = call i64 @___qalloc()\n  call void @___reset(i64 %q)",
        )
        .replace(
            "  call void @___qfree(i64 %q)",
            """  %value = call i1 @___read_future_bool(i64 %r)
  br i1 %value, label %flip, label %done
flip:
  call void @___rxy(i64 %q, double 0x400921FB54442D18, double 0.0)
  br label %done
done:
  %second = call i64 @___lazy_measure(i64 %q)
  call void @___qfree(i64 %q)""",
        )
    )
    # The first measurement is 1 only because of the preparation fault. Its
    # returned value must resume the program and select the physical X branch.
    profile = pr.scheduled_local_noise(pr.scheduled_idle_noise(1), prep=1.0)
    result = simulation(library, program, profile, events).run(6).to_dict()
    assert result["measurement_0"] == [1] * 6
    assert result["measurement_1"] == [0] * 6


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize("idle_only", [False, True])
def test_native_multi_operation_batch_reuses_qubit_before_live_feedback(
    tmp_path,
    scheduled_support,
    events,
    idle_only,
):
    import pecos_rslib as pr

    library, _ = scheduled_support.build_runtime(
        tmp_path,
        0,
        prep_gap_nanos=1_000_000_000,
        coalesce_queued=True,
    )
    program = """
declare i64 @___qalloc()
declare void @___qfree(i64)
declare i64 @___lazy_measure(i64)
declare void @___reset(i64)
declare i1 @___read_future_bool(i64)
declare void @___rxy(i64, double, double)
declare void @setup(i64)
declare i64 @teardown()
define i64 @qmain(i64 %arg) #0 {
  call void @setup(i64 %arg)
  %q = call i64 @___qalloc()
  call void @___reset(i64 %q)
  %first = call i64 @___lazy_measure(i64 %q)
  %repeat = call i64 @___lazy_measure(i64 %q)
  call void @___reset(i64 %q)
  %after_reset = call i64 @___lazy_measure(i64 %q)
  %value = call i1 @___read_future_bool(i64 %first)
  br i1 %value, label %flip, label %done
flip:
  call void @___rxy(i64 %q, double 0x400921FB54442D18, double 0.0)
  br label %done
done:
  %last = call i64 @___lazy_measure(i64 %q)
  call void @___qfree(i64 %q)
  %end = call i64 @teardown()
  ret i64 %end
}
attributes #0 = { "EntryPoint" }
"""
    idle = pr.scheduled_idle_noise(1, linear=1.0, linear_model={"L": 1.0})
    # The initial prep has its own batch at time zero. One second of idle before
    # the readout batch leaks the qubit; the later reset clears that leakage.
    # Only the returned first result can make the live program enqueue its X branch.
    profile = idle if idle_only else pr.scheduled_local_noise(idle, prep=1.0)
    batches = []

    class Inspect(PassThrough):
        def __init__(self):
            self.batches = []
            batches.append(self.batches)

        def translate(self, batch):
            self.batches.append((batch.start_nanos, len(batch.operations), [entry[0] for entry in batch.measurements]))
            return super().translate(batch)

    built = simulation(
        library,
        program,
        profile,
        events,
        idle_only=idle_only,
        adapter_factory=lambda _: Inspect(),
    ).build()
    for shots in (4, 6):
        batches.clear()
        result = built.run(shots).to_dict()
        assert result["measurement_0"] == [1] * shots
        assert result["measurement_1"] == [1] * shots
        assert result["measurement_2"] == [0 if idle_only else 1] * shots
        assert result["measurement_3"] == [1 if idle_only else 0] * shots
        if events:
            # The initial PZ is separate; MZ, MZ, PZ, MZ arrive together after
            # the idle gap, then RXY, MZ arrive after the first result returns.
            assert batches == [[(0, 1, []), (1_000_000_000, 4, [0, 1, 3]), (1_000_000_000, 2, [1])]] * shots
