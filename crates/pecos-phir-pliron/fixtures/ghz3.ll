; 3-qubit GHZ (straight-line, no conditional branch) -- exercises dynamic qubit count past the
; 2-qubit fixtures: h q0; cx q0,q1; cx q1,q2; measure all; record. State is (|000>+|111>)/sqrt(2),
; so the three recorded results are perfectly correlated: r0 == r1 == r2 in every shot.
declare void @__quantum__qis__h__body(i64)
declare void @__quantum__qis__cx__body(i64, i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare void @__quantum__rt__result_record_output(i64, i8*)

define i64 @qmain(i64 %arg) #0 {
    call void @__quantum__qis__h__body(i64 0)
    call void @__quantum__qis__cx__body(i64 0, i64 1)
    call void @__quantum__qis__cx__body(i64 1, i64 2)
    %r0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %r1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
    %r2 = call i32 @__quantum__qis__m__body(i64 2, i64 2)
    call void @__quantum__rt__result_record_output(i64 0, i8* null)
    call void @__quantum__rt__result_record_output(i64 1, i8* null)
    call void @__quantum__rt__result_record_output(i64 2, i8* null)
    ret i64 0
}

attributes #0 = { "EntryPoint" }
