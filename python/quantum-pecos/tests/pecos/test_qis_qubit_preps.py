"""QIS lifetime prep regressions through the native Selene sim() route."""

import pytest
from pecos_rslib import Qis, depolarizing_noise, sim, sparse_stab, state_vector


@pytest.mark.parametrize("quantum_engine", [sparse_stab, state_vector])
def test_legacy_measurements_keep_static_handles_live(quantum_engine) -> None:
    """Red/green regression for #1016: measurements retain static handles."""
    program = Qis.from_string(
        """
        define i64 @qmain(i64 %unused) #0 {
            call void @__quantum__qis__x__body(i64 1)
            %r1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
            %r0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            %r2 = call i32 @__quantum__qis__m__body(i64 1, i64 2)
            ret i64 0
        }
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
        """,
    )
    results = sim(program).qubits(2).quantum(quantum_engine()).seed(1016).run(20).to_dict()
    assert results["measurement_0"] == [0] * 20
    assert results["measurement_1"] == [1] * 20
    assert results["measurement_2"] == [1] * 20


@pytest.mark.parametrize("quantum_engine", [sparse_stab, state_vector])
@pytest.mark.parametrize("reset_reused", [False, True])
def test_allocation_after_release_preserves_runtime_prep_policy(quantum_engine, reset_reused) -> None:
    """Simple reuses the physical state; only an explicit reset prepares it."""
    program = Qis.from_string(
        """
        define i64 @qmain(i64 %unused) #0 {
            %q = call i64 @__quantum__rt__qubit_allocate()
            call void @__quantum__qis__x__body(i64 %q)
            %r0 = call i32 @__quantum__qis__m__body(i64 %q, i64 0)
            call void @__quantum__rt__qubit_release(i64 %q)
            %q2 = call i64 @__quantum__rt__qubit_allocate()
            REUSED_PREPARATION
            %r1 = call i32 @__quantum__qis__m__body(i64 %q2, i64 1)
            ret i64 0
        }
        declare i64 @__quantum__rt__qubit_allocate()
        declare void @__quantum__rt__qubit_release(i64)
        declare void @__quantum__qis__reset__body(i64)
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
        """.replace("REUSED_PREPARATION", "call void @__quantum__qis__reset__body(i64 %q2)" if reset_reused else ""),
    )
    results = sim(program).qubits(1).quantum(quantum_engine()).seed(1016).run(20).to_dict()
    assert results["measurement_0"] == [1] * 20
    assert results["measurement_1"] == [0 if reset_reused else 1] * 20


@pytest.mark.parametrize("allocated", [False, True], ids=["static", "allocated"])
@pytest.mark.parametrize("explicit_reset", [False, True], ids=["no-prep", "explicit-reset"])
def test_scheduled_late_first_use_prep_timing(allocated: bool, explicit_reset: bool) -> None:
    """Only requested preparation occupies time in the runtime schedule."""
    import pecos
    import pecos_rslib as pr

    # Ships with selene-sim, a required dependency: a missing package must fail.
    import selene_soft_rz_runtime_plugin as plugin_package
    from pecos_rslib.quantum import GateType

    reset_nanos, rxy_nanos, measure_nanos = 7, 11, 13
    runtime = plugin_package.SoftRZRuntimePlugin(
        duration_ns_reset=reset_nanos,
        duration_ns_rxy=rxy_nanos,
        duration_ns_measure=measure_nanos,
        duration_ns_rzz=rxy_nanos,
        duration_ns_measure_leaked=measure_nanos,
        max_batch_size=1,
    )
    q0, q1 = ("%q0", "%q1") if allocated else ("0", "1")
    allocate_q0 = "%q0 = call i64 @__quantum__rt__qubit_allocate()" if allocated else ""
    allocate_q1 = "%q1 = call i64 @__quantum__rt__qubit_allocate()" if allocated else ""
    prep_q1 = f"call void @__quantum__qis__reset__body(i64 {q1})" if explicit_reset else ""
    program = f"""
        define i64 @qmain(i64 %unused) #0 {{
            {allocate_q0}
            {allocate_q1}
            call void @__quantum__qis__reset__body(i64 {q0})
            call void @__quantum__qis__x__body(i64 {q0})
            {prep_q1}
            call void @__quantum__qis__x__body(i64 {q1})
            %r = call i32 @__quantum__qis__m__body(i64 {q1}, i64 0)
            %value = call i1 @___read_future_bool(i64 0)
            ret i64 0
        }}
        declare i64 @__quantum__rt__qubit_allocate()
        declare void @__quantum__qis__reset__body(i64)
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        attributes #0 = {{ "EntryPoint" }}
    """
    batches = []

    class Inspect:
        def validate(self, batch):
            pass

        def translate(self, batch):
            batches.append(
                (
                    batch.start_nanos,
                    batch.duration_nanos,
                    [(op.gate_type, op.qubits) for op in batch.operations],
                ),
            )
            return batch.operations

    classical = (
        pr.qis_engine()
        .selene_runtime_plugin(str(runtime.library_file), runtime.get_init_args())
        .scheduled_event_batches()
        .interface(pr.qis_helios_interface())
    )
    result = (
        pecos.sim(pecos.Qis(program))
        .classical(classical)
        .qubits(2)
        .quantum(pr.state_vector())
        .noise(pr.scheduled_event_idle_noise(pr.scheduled_idle_noise(2), lambda _: Inspect()))
        .seed(1017)
        .workers(1)
        .run(1)
        .to_dict()
    )
    # The runtime emits no implicit reset. Read the future explicitly so the
    # lazy runtime must release its measurement before completion.
    late_start = reset_nanos + rxy_nanos
    expected_batches = [
        (0, reset_nanos, [(GateType.Prep, [0])]),
        (reset_nanos, rxy_nanos, [(GateType.RXY1Q, [0])]),
    ]
    if explicit_reset:
        expected_batches.append((late_start, reset_nanos, [(GateType.Prep, [1])]))
        late_start += reset_nanos
    expected_batches.extend(
        [
            (late_start, rxy_nanos, [(GateType.RXY1Q, [1])]),
            (late_start + rxy_nanos, measure_nanos, [(GateType.Measure, [1])]),
        ],
    )
    assert (batches, result["measurement_0"]) == (expected_batches, [1])


@pytest.mark.parametrize("explicit_reset", [False, True])
def test_only_emitted_prep_receives_prep_noise(explicit_reset) -> None:
    """Allocation alone introduces neither preparation nor preparation noise."""
    program = Qis.from_string(
        """
        define i64 @qmain(i64 %unused) #0 {
            %q = call i64 @__quantum__rt__qubit_allocate()
            PREPARATION
            %r0 = call i32 @__quantum__qis__m__body(i64 %q, i64 0)
            ret i64 0
        }
        declare i64 @__quantum__rt__qubit_allocate()
        declare void @__quantum__qis__reset__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
        """.replace("PREPARATION", "call void @__quantum__qis__reset__body(i64 %q)" if explicit_reset else ""),
    )
    noise = depolarizing_noise().with_p_prep(1.0).with_p_meas(0.0).with_p1(0.0).with_p2(0.0)
    results = sim(program).qubits(1).quantum(sparse_stab()).noise(noise).seed(1016).run(20).to_dict()
    assert results["measurement_0"] == [int(explicit_reset)] * 20
