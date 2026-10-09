"""Measurement feedback must use the outcome supplied by the quantum engine."""

import pecos
from pecos_rslib import state_vector


def test_result_get_one_uses_the_measured_value() -> None:
    program = """declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i32 @__quantum__rt__result_get_one(i64)
define void @main() #0 {
entry:
  call void @__quantum__qis__x__body(i64 0)
  %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
  %one = call i32 @__quantum__rt__result_get_one(i64 0)
  %is1 = icmp ne i32 %one, 0
  br i1 %is1, label %flip, label %done
flip:
  call void @__quantum__qis__x__body(i64 1)
  br label %done
done:
  %m1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
  ret void
}
attributes #0 = { "EntryPoint" }
"""
    results = pecos.sim(pecos.Qis(program)).qubits(2).quantum(state_vector()).run(3).to_dict()
    assert results["measurement_1"] == [1, 1, 1]
