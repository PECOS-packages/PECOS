; ModuleID = 'hugr'
source_filename = "hugr"

@"e_Some array.A77EF32E.0" = private constant [48 x i8] c"/EXIT:INT:Some array elements have been borrowed"
@res_DETECTOR.4C935625.0 = private constant [19 x i8] c"\12USER:BOOL:DETECTOR"
@"res_raw measur.6302A477.0" = private constant [30 x i8] c"\1DUSER:BOOLARR:raw measurements"
@res_obs.2D65FFAD.0 = private constant [14 x i8] c"\0DUSER:BOOL:obs"
@res_unused.7E919D03.0 = private constant [17 x i8] c"\10USER:BOOL:unused"
@"e_No more qu.3B2EEBF0.0" = private constant [47 x i8] c".EXIT:INT:No more qubits available to allocate."

define void @__hugr__.__main__.parities.1() local_unnamed_addr {
alloca_block:
  %qalloc.i = tail call i64 @___qalloc()
  %not_max.not.not.i = icmp eq i64 %qalloc.i, -1
  br i1 %not_max.not.not.i, label %cond_31_case_0.i, label %__hugr__.__tk2_helios_qalloc.27.exit

cond_31_case_0.i:                                 ; preds = %alloca_block
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.27.exit:             ; preds = %alloca_block
  tail call void @___reset(i64 %qalloc.i)
  %qalloc.i58 = tail call i64 @___qalloc()
  %not_max.not.not.i59 = icmp eq i64 %qalloc.i58, -1
  br i1 %not_max.not.not.i59, label %cond_31_case_0.i60, label %__hugr__.__tk2_helios_qalloc.27.exit61

cond_31_case_0.i60:                               ; preds = %__hugr__.__tk2_helios_qalloc.27.exit
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.27.exit61:           ; preds = %__hugr__.__tk2_helios_qalloc.27.exit
  tail call void @___reset(i64 %qalloc.i58)
  tail call void @___rxy(i64 %qalloc.i58, double 0xBFF921FB54442D18, double 0x3FF921FB54442D18)
  tail call void @___rzz(i64 %qalloc.i, i64 %qalloc.i58, double 0x3FF921FB54442D18)
  tail call void @___rz(i64 %qalloc.i, double 0xBFF921FB54442D18)
  tail call void @___rxy(i64 %qalloc.i58, double 0x3FF921FB54442D18, double 0x400921FB54442D18)
  tail call void @___rz(i64 %qalloc.i58, double 0xBFF921FB54442D18)
  %qalloc.i62 = tail call i64 @___qalloc()
  %not_max.not.not.i63 = icmp eq i64 %qalloc.i62, -1
  br i1 %not_max.not.not.i63, label %cond_31_case_0.i64, label %__hugr__.__tk2_helios_qalloc.27.exit65

cond_31_case_0.i64:                               ; preds = %__hugr__.__tk2_helios_qalloc.27.exit61
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.27.exit65:           ; preds = %__hugr__.__tk2_helios_qalloc.27.exit61
  tail call void @___reset(i64 %qalloc.i62)
  %lazy_measure = tail call i64 @___lazy_measure(i64 %qalloc.i)
  tail call void @___qfree(i64 %qalloc.i)
  %lazy_measure5 = tail call i64 @___lazy_measure(i64 %qalloc.i58)
  tail call void @___qfree(i64 %qalloc.i58)
  %lazy_measure7 = tail call i64 @___lazy_measure(i64 %qalloc.i62)
  tail call void @___qfree(i64 %qalloc.i62)
  %read_bool = tail call i1 @___read_future_bool(i64 %lazy_measure7)
  tail call void @___dec_future_refcount(i64 %lazy_measure7)
  %read_bool10 = tail call i1 @___read_future_bool(i64 %lazy_measure5)
  tail call void @___dec_future_refcount(i64 %lazy_measure5)
  %read_bool13 = tail call i1 @___read_future_bool(i64 %lazy_measure)
  tail call void @___dec_future_refcount(i64 %lazy_measure)
  %0 = tail call ptr @heap_alloc(i64 3)
  %1 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %1, align 1
  store i1 %read_bool, ptr %0, align 1
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 1
  store i1 %read_bool13, ptr %2, align 1
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 2
  store i1 %read_bool10, ptr %3, align 1
  %4 = load i64, ptr %1, align 4
  %5 = and i64 %4, 7
  store i64 %5, ptr %1, align 4
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %__barray_check_none_borrowed.exit, label %mask_block_err.i

mask_block_err.i:                                 ; preds = %__hugr__.__tk2_helios_qalloc.27.exit65
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit:                ; preds = %__hugr__.__tk2_helios_qalloc.27.exit65
  %7 = tail call ptr @heap_alloc(i64 3)
  %8 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %8, align 1
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %7, ptr noundef nonnull align 1 dereferenceable(3) %0, i64 3, i1 false)
  tail call void @heap_free(ptr nonnull %7)
  %9 = load i64, ptr %1, align 4
  %10 = and i64 %9, 7
  store i64 %10, ptr %1, align 4
  %11 = icmp eq i64 %10, 0
  br i1 %11, label %__barray_check_none_borrowed.exit67, label %mask_block_err.i66

mask_block_err.i66:                               ; preds = %__barray_check_none_borrowed.exit
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit67:              ; preds = %__barray_check_none_borrowed.exit
  %12 = and i1 %read_bool10, %read_bool13
  %13 = xor i1 %read_bool, %read_bool13
  tail call void @print_bool(ptr nonnull @res_DETECTOR.4C935625.0, i64 18, i1 %13)
  %out_arr_alloca = alloca <{ i32, i32, ptr, ptr }>, align 8
  %arr_ptr = getelementptr inbounds nuw i8, ptr %out_arr_alloca, i64 8
  %mask_ptr = getelementptr inbounds nuw i8, ptr %out_arr_alloca, i64 16
  %14 = alloca [3 x i1], align 1
  store i1 false, ptr %14, align 1
  %.repack56 = getelementptr inbounds nuw i8, ptr %14, i64 1
  store i1 false, ptr %.repack56, align 1
  %.repack57 = getelementptr inbounds nuw i8, ptr %14, i64 2
  store i1 false, ptr %.repack57, align 1
  store <2 x i32> <i32 3, i32 1>, ptr %out_arr_alloca, align 8
  store ptr %0, ptr %arr_ptr, align 8
  store ptr %14, ptr %mask_ptr, align 8
  call void @print_bool_arr(ptr nonnull @"res_raw measur.6302A477.0", i64 29, ptr nonnull %out_arr_alloca)
  call void @print_bool(ptr nonnull @res_obs.2D65FFAD.0, i64 13, i1 %read_bool10)
  call void @print_bool(ptr nonnull @res_unused.7E919D03.0, i64 16, i1 %12)
  ret void
}

declare i64 @___lazy_measure(i64) local_unnamed_addr

declare void @___qfree(i64) local_unnamed_addr

declare i1 @___read_future_bool(i64) local_unnamed_addr

declare void @___dec_future_refcount(i64) local_unnamed_addr

declare ptr @heap_alloc(i64) local_unnamed_addr

; Function Attrs: noreturn
declare void @panic(i32, ptr) local_unnamed_addr #0

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

declare void @heap_free(ptr) local_unnamed_addr

declare void @print_bool(ptr, i64, i1) local_unnamed_addr

declare void @print_bool_arr(ptr, i64, ptr) local_unnamed_addr

declare i64 @___qalloc() local_unnamed_addr

declare void @___reset(i64) local_unnamed_addr

declare void @___rxy(i64, double, double) local_unnamed_addr

declare void @___rzz(i64, i64, double) local_unnamed_addr

declare void @___rz(i64, double) local_unnamed_addr

define i64 @qmain(i64 %0) local_unnamed_addr {
entry:
  tail call void @setup(i64 %0)
  tail call void @__hugr__.__main__.parities.1()
  %1 = tail call i64 @teardown()
  ret i64 %1
}

declare void @setup(i64) local_unnamed_addr

declare i64 @teardown() local_unnamed_addr

attributes #0 = { noreturn }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!name = !{!0}

!0 = !{!"mainlib"}
