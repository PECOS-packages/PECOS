; Straight-line program exercising the widened gate set (cz, swap, x) with a deterministic outcome.
;   h q0; x q1; cz q0,q1; h q0; swap q1,q2; measure all
; Walkthrough: q0=|+>; q1=|1>; cz with control q1=|1> applies Z to q0 (|+> -> |->); h q0 maps |-> -> |1>
; (so q0 measures 1); swap q1,q2 moves the |1> from q1 to q2 (q1 -> 0, q2 -> 1).
; Deterministic: r0 == 1, r1 == 0, r2 == 1.
declare void @__quantum__qis__h__body(i64)
declare void @__quantum__qis__x__body(i64)
declare void @__quantum__qis__cz__body(i64, i64)
declare void @__quantum__qis__swap__body(i64, i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare void @__quantum__rt__result_record_output(i64, i8*)

define i64 @qmain(i64 %arg) #0 {
    call void @__quantum__qis__h__body(i64 0)
    call void @__quantum__qis__x__body(i64 1)
    call void @__quantum__qis__cz__body(i64 0, i64 1)
    call void @__quantum__qis__h__body(i64 0)
    call void @__quantum__qis__swap__body(i64 1, i64 2)
    %r0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %r1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
    %r2 = call i32 @__quantum__qis__m__body(i64 2, i64 2)
    call void @__quantum__rt__result_record_output(i64 0, i8* null)
    call void @__quantum__rt__result_record_output(i64 1, i8* null)
    call void @__quantum__rt__result_record_output(i64 2, i8* null)
    ret i64 0
}

attributes #0 = { "EntryPoint" }
