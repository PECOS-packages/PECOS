; Bell State Circuit
; This demonstrates immediate measurement capability with integer-based parameters

declare void @__quantum__qis__h__body(i64)
declare void @__quantum__qis__cx__body(i64, i64)
declare i32 @__quantum__qis__m__body(i64, i64)  ; Returns result immediately
declare void @__quantum__rt__result_record_output(i64, i8*)

@.str.c = constant [2 x i8] c"c\00"

; Helios-compatible entry point: i64 qmain(i64)
define i64 @qmain(i64 %arg) #0 {
    ; Create Bell state: |00⟩ + |11⟩
    call void @__quantum__qis__h__body(i64 0)
    call void @__quantum__qis__cx__body(i64 0, i64 1)

    ; IMMEDIATE measurements - get results right away
    %result0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %result1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)

    ; Record both results to "c" register (just like the original Bell examples)
    call void @__quantum__rt__result_record_output(i64 0, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.c, i32 0, i32 0))
    call void @__quantum__rt__result_record_output(i64 1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.c, i32 0, i32 0))

    ; Note: %result0 and %result1 are available here for immediate classical logic
    ; but we're keeping this example simple

    ret i64 0
}

attributes #0 = { "EntryPoint" }
