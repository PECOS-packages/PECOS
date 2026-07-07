; Adaptive program with a measurement INSIDE the conditional branch. Coverage for the engine's
; outcome reconstruction when b2's length depends on which branch ran (taken: [m1, f0, f1];
; skipped: [f0, f1]) -- the j-index must walk the actually-emitted measures, not a fixed layout.
;   h q0 -> mid = m(q0) -> if mid==1 { x q1; m1 = m(q1, rid 3) } -> final m(q0, rid 0); m(q1, rid 1)
; The in-branch m1 is measured for side effect only and is NOT recorded: recording a value defined
; inside a qec.if region from the outer block is a cross-region SSA escape that needs block-args /
; yield (measurement-SSA Phase 2), out of scope here. Recorded outputs are the unconditional finals.
; Invariants: final_q0 (r0) == mid and final_q1 (r1) == mid in every shot (the in-branch x+measure
; on q1 leaves q1 == mid, so the later final measure still reads mid).
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
    %m1 = call i32 @__quantum__qis__m__body(i64 1, i64 3)
    br label %final

skip_x:
    br label %final

final:
    %f0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %f1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
    call void @__quantum__rt__result_record_output(i64 0, i8* null)
    call void @__quantum__rt__result_record_output(i64 1, i8* null)
    ret i64 0
}

attributes #0 = { "EntryPoint" }
