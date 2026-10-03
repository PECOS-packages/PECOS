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
def test_allocation_after_release_preps_reused_slot(quantum_engine) -> None:
    """Red/green: the second allocation must not inherit the released slot's |1>."""
    program = Qis.from_string(
        """
        define i64 @qmain(i64 %unused) #0 {
            %q = call i64 @__quantum__rt__qubit_allocate()
            call void @__quantum__qis__x__body(i64 %q)
            %r0 = call i32 @__quantum__qis__m__body(i64 %q, i64 0)
            call void @__quantum__rt__qubit_release(i64 %q)
            %q2 = call i64 @__quantum__rt__qubit_allocate()
            %r1 = call i32 @__quantum__qis__m__body(i64 %q2, i64 1)
            ret i64 0
        }
        declare i64 @__quantum__rt__qubit_allocate()
        declare void @__quantum__rt__qubit_release(i64)
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
        """,
    )
    results = sim(program).qubits(1).quantum(quantum_engine()).seed(1016).run(20).to_dict()
    assert results["measurement_0"] == [1] * 20
    assert results["measurement_1"] == [0] * 20


def test_inserted_prep_receives_prep_noise() -> None:
    """Red/green: allocation without a program prep still receives normal prep noise."""
    program = Qis.from_string(
        """
        define i64 @qmain(i64 %unused) #0 {
            %q = call i64 @__quantum__rt__qubit_allocate()
            %r0 = call i32 @__quantum__qis__m__body(i64 %q, i64 0)
            ret i64 0
        }
        declare i64 @__quantum__rt__qubit_allocate()
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
        """,
    )
    noise = depolarizing_noise().with_p_prep(1.0).with_p_meas(0.0).with_p1(0.0).with_p2(0.0)
    results = sim(program).qubits(1).quantum(sparse_stab()).noise(noise).seed(1016).run(20).to_dict()
    assert results["measurement_0"] == [1] * 20
