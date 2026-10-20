; ModuleID = 'hugr'
source_filename = "hugr"

@"res_raw measur.E917903A.0" = private constant [27 x i8] c"\1AUSER:BOOL:raw measurements"
@res_DETECTOR.4C935625.0 = private constant [19 x i8] c"\12USER:BOOL:DETECTOR"
@res_obs.2D65FFAD.0 = private constant [14 x i8] c"\0DUSER:BOOL:obs"
@"e_No more qu.3B2EEBF0.0" = private constant [47 x i8] c".EXIT:INT:No more qubits available to allocate."

define void @__hugr__.__main__.changing_schedule.1() local_unnamed_addr {
alloca_block:
  %qalloc.i = tail call i64 @___qalloc()
  %not_max.not.not.i = icmp eq i64 %qalloc.i, -1
  br i1 %not_max.not.not.i, label %cond_39_case_0.i, label %__hugr__.__tk2_helios_qalloc.35.exit

cond_39_case_0.i:                                 ; preds = %alloca_block
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.35.exit:             ; preds = %alloca_block
  tail call void @___reset(i64 %qalloc.i)
  %lazy_measure = tail call i64 @___lazy_measure(i64 %qalloc.i)
  tail call void @___qfree(i64 %qalloc.i)
  %qalloc.i51 = tail call i64 @___qalloc()
  %not_max.not.not.i52 = icmp eq i64 %qalloc.i51, -1
  br i1 %not_max.not.not.i52, label %cond_39_case_0.i53, label %__hugr__.__tk2_helios_qalloc.35.exit54

cond_39_case_0.i53:                               ; preds = %__hugr__.__tk2_helios_qalloc.35.exit
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.35.exit54:           ; preds = %__hugr__.__tk2_helios_qalloc.35.exit
  tail call void @___reset(i64 %qalloc.i51)
  %read_bool = tail call i1 @___read_future_bool(i64 %lazy_measure)
  tail call void @___dec_future_refcount(i64 %lazy_measure)
  br i1 %read_bool, label %0, label %1

0:                                                ; preds = %__hugr__.__tk2_helios_qalloc.35.exit54
  tail call void @___rxy(i64 %qalloc.i51, double 0x400921FB54442D18, double 0.000000e+00)
  br label %1

1:                                                ; preds = %__hugr__.__tk2_helios_qalloc.35.exit54, %0
  %lazy_measure29 = tail call i64 @___lazy_measure(i64 %qalloc.i51)
  tail call void @___qfree(i64 %qalloc.i51)
  tail call void @print_bool(ptr nonnull @"res_raw measur.E917903A.0", i64 26, i1 %read_bool)
  %read_bool33 = tail call i1 @___read_future_bool(i64 %lazy_measure29)
  tail call void @___dec_future_refcount(i64 %lazy_measure29)
  %2 = xor i1 %read_bool, %read_bool33
  tail call void @print_bool(ptr nonnull @"res_raw measur.E917903A.0", i64 26, i1 %read_bool33)
  tail call void @print_bool(ptr nonnull @res_DETECTOR.4C935625.0, i64 18, i1 %2)
  tail call void @print_bool(ptr nonnull @res_obs.2D65FFAD.0, i64 13, i1 %read_bool33)
  ret void
}

declare i64 @___lazy_measure(i64) local_unnamed_addr

declare void @___qfree(i64) local_unnamed_addr

declare i1 @___read_future_bool(i64) local_unnamed_addr

declare void @___dec_future_refcount(i64) local_unnamed_addr

declare void @print_bool(ptr, i64, i1) local_unnamed_addr

declare i64 @___qalloc() local_unnamed_addr

declare void @___reset(i64) local_unnamed_addr

; Function Attrs: noreturn
declare void @panic(i32, ptr) local_unnamed_addr #0

declare void @___rxy(i64, double, double) local_unnamed_addr

define i64 @qmain(i64 %0) local_unnamed_addr {
entry:
  tail call void @setup(i64 %0)
  tail call void @__hugr__.__main__.changing_schedule.1()
  %1 = tail call i64 @teardown()
  ret i64 %1
}

declare void @setup(i64) local_unnamed_addr

declare i64 @teardown() local_unnamed_addr

attributes #0 = { noreturn }

!name = !{!0}

!0 = !{!"mainlib"}
