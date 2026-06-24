; Adaptive program whose conditional branch IS taken (unlike qprog.ll, whose q0 is deterministic).
; h q0 -> measure q0 (random) -> if q0==1 { x q1 } -> measure q1.  Invariant: final_q1 == mid_q0.
declare void @__quantum__qis__h__body(i64)
declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare void @__quantum__rt__result_record_output(i64, i8*)

define i64 @qmain(i64 %arg) #0 {
    call void @__quantum__qis__h__body(i64 0)
    %mid = call i32 @__quantum__qis__m__body(i64 0, i64 2)
    %cond = icmp eq i32 %mid, 1
    br i1 %cond, label %apply_x, label %skip_x

apply_x:
    call void @__quantum__qis__x__body(i64 1)
    br label %final

skip_x:
    br label %final

final:
    %f1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
    call void @__quantum__rt__result_record_output(i64 2, i8* null)
    call void @__quantum__rt__result_record_output(i64 1, i8* null)
    ret i64 0
}

attributes #0 = { "EntryPoint" }
