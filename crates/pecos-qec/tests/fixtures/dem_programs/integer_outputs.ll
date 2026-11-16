; ModuleID = 'hugr'
source_filename = "hugr"

@"e_Some array.A77EF32E.0" = private constant [48 x i8] c"/EXIT:INT:Some array elements have been borrowed"
@"res_raw measur.E917903A.0" = private constant [27 x i8] c"\1AUSER:BOOL:raw measurements"
@res_checks.EA8C04AB.0 = private constant [19 x i8] c"\12USER:INTARR:checks"
@res_checks.2D4BAD12.0 = private constant [16 x i8] c"\0FUSER:INT:checks"
@res_bits.ED7143A1.0 = private constant [17 x i8] c"\10USER:INTARR:bits"
@res_counts.95A27DF5.0 = private constant [16 x i8] c"\0FUSER:INT:counts"
@"e_No more qu.3B2EEBF0.0" = private constant [47 x i8] c".EXIT:INT:No more qubits available to allocate."

define void @__hugr__.__main__.integer_outputs.1() local_unnamed_addr {
alloca_block:
  %qalloc.i = tail call i64 @___qalloc()
  %not_max.not.not.i = icmp eq i64 %qalloc.i, -1
  br i1 %not_max.not.not.i, label %cond_138_case_0.i, label %__hugr__.__tk2_helios_qalloc.134.exit

cond_138_case_0.i:                                ; preds = %alloca_block
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.134.exit:            ; preds = %alloca_block
  tail call void @___reset(i64 %qalloc.i)
  %lazy_measure = tail call i64 @___lazy_measure(i64 %qalloc.i)
  tail call void @___qfree(i64 %qalloc.i)
  %qalloc.i163 = tail call i64 @___qalloc()
  %not_max.not.not.i164 = icmp eq i64 %qalloc.i163, -1
  br i1 %not_max.not.not.i164, label %cond_138_case_0.i165, label %__hugr__.__tk2_helios_qalloc.134.exit166

cond_138_case_0.i165:                             ; preds = %__hugr__.__tk2_helios_qalloc.134.exit
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.134.exit166:         ; preds = %__hugr__.__tk2_helios_qalloc.134.exit
  tail call void @___reset(i64 %qalloc.i163)
  %lazy_measure3 = tail call i64 @___lazy_measure(i64 %qalloc.i163)
  tail call void @___qfree(i64 %qalloc.i163)
  %qalloc.i167 = tail call i64 @___qalloc()
  %not_max.not.not.i168 = icmp eq i64 %qalloc.i167, -1
  br i1 %not_max.not.not.i168, label %cond_138_case_0.i169, label %__hugr__.__tk2_helios_qalloc.134.exit170

cond_138_case_0.i169:                             ; preds = %__hugr__.__tk2_helios_qalloc.134.exit166
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.134.exit170:         ; preds = %__hugr__.__tk2_helios_qalloc.134.exit166
  tail call void @___reset(i64 %qalloc.i167)
  %lazy_measure5 = tail call i64 @___lazy_measure(i64 %qalloc.i167)
  tail call void @___qfree(i64 %qalloc.i167)
  %read_bool = tail call i1 @___read_future_bool(i64 %lazy_measure5)
  tail call void @___dec_future_refcount(i64 %lazy_measure5)
  %. = zext i1 %read_bool to i64
  %read_bool35 = tail call i1 @___read_future_bool(i64 %lazy_measure3)
  tail call void @___dec_future_refcount(i64 %lazy_measure3)
  %"038.0" = zext i1 %read_bool35 to i64
  %0 = tail call ptr @heap_alloc(i64 16)
  %1 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %1, align 1
  store i64 %"038.0", ptr %0, align 4
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %., ptr %2, align 4
  %3 = load i64, ptr %1, align 4
  %4 = and i64 %3, 3
  store i64 %4, ptr %1, align 4
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %__barray_check_none_borrowed.exit, label %mask_block_err.i

mask_block_err.i:                                 ; preds = %__hugr__.__tk2_helios_qalloc.134.exit170
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit:                ; preds = %__hugr__.__tk2_helios_qalloc.134.exit170
  %6 = tail call ptr @heap_alloc(i64 16)
  %7 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %7, align 1
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(16) %6, ptr noundef nonnull align 1 dereferenceable(16) %0, i64 16, i1 false)
  tail call void @heap_free(ptr nonnull %6)
  %8 = load i64, ptr %1, align 4
  %9 = and i64 %8, 3
  store i64 %9, ptr %1, align 4
  %10 = icmp eq i64 %9, 0
  br i1 %10, label %__barray_check_none_borrowed.exit172, label %mask_block_err.i171

mask_block_err.i171:                              ; preds = %__barray_check_none_borrowed.exit
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit172:             ; preds = %__barray_check_none_borrowed.exit
  %read_bool76 = tail call i1 @___read_future_bool(i64 %lazy_measure)
  tail call void @___dec_future_refcount(i64 %lazy_measure)
  %"079.0" = zext i1 %read_bool76 to i64
  %11 = tail call ptr @heap_alloc(i64 8)
  %12 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %12, align 1
  store i64 %"079.0", ptr %11, align 4
  %13 = load i64, ptr %12, align 4
  %14 = and i64 %13, 1
  store i64 %14, ptr %12, align 4
  %15 = icmp eq i64 %14, 0
  br i1 %15, label %__barray_check_none_borrowed.exit174, label %mask_block_err.i173

mask_block_err.i173:                              ; preds = %__barray_check_none_borrowed.exit172
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit174:             ; preds = %__barray_check_none_borrowed.exit172
  %16 = tail call ptr @heap_alloc(i64 8)
  %17 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %17, align 1
  %18 = load i64, ptr %11, align 1
  store i64 %18, ptr %16, align 1
  tail call void @heap_free(ptr nonnull %16)
  %19 = load i64, ptr %12, align 4
  %20 = and i64 %19, 1
  store i64 %20, ptr %12, align 4
  %21 = icmp eq i64 %20, 0
  br i1 %21, label %__barray_check_none_borrowed.exit176, label %mask_block_err.i175

mask_block_err.i175:                              ; preds = %__barray_check_none_borrowed.exit174
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit176:             ; preds = %__barray_check_none_borrowed.exit174
  tail call void @print_bool(ptr nonnull @"res_raw measur.E917903A.0", i64 26, i1 %read_bool76)
  tail call void @print_bool(ptr nonnull @"res_raw measur.E917903A.0", i64 26, i1 %read_bool35)
  tail call void @print_bool(ptr nonnull @"res_raw measur.E917903A.0", i64 26, i1 %read_bool)
  %out_arr_alloca = alloca <{ i32, i32, ptr, ptr }>, align 8
  %y_ptr = getelementptr inbounds nuw i8, ptr %out_arr_alloca, i64 4
  %arr_ptr = getelementptr inbounds nuw i8, ptr %out_arr_alloca, i64 8
  %mask_ptr = getelementptr inbounds nuw i8, ptr %out_arr_alloca, i64 16
  %22 = alloca i1, align 1
  store i1 false, ptr %22, align 1
  store i32 1, ptr %out_arr_alloca, align 8
  store i32 1, ptr %y_ptr, align 4
  store ptr %11, ptr %arr_ptr, align 8
  store ptr %22, ptr %mask_ptr, align 8
  call void @print_int_arr(ptr nonnull @res_checks.EA8C04AB.0, i64 18, ptr nonnull %out_arr_alloca)
  call void @print_int(ptr nonnull @res_checks.2D4BAD12.0, i64 15, i64 %"038.0")
  %out_arr_alloca124 = alloca <{ i32, i32, ptr, ptr }>, align 8
  %y_ptr126 = getelementptr inbounds nuw i8, ptr %out_arr_alloca124, i64 4
  %arr_ptr127 = getelementptr inbounds nuw i8, ptr %out_arr_alloca124, i64 8
  %mask_ptr128 = getelementptr inbounds nuw i8, ptr %out_arr_alloca124, i64 16
  %23 = alloca [2 x i1], align 1
  store i1 false, ptr %23, align 1
  %.repack162 = getelementptr inbounds nuw i8, ptr %23, i64 1
  store i1 false, ptr %.repack162, align 1
  store i32 2, ptr %out_arr_alloca124, align 8
  store i32 1, ptr %y_ptr126, align 4
  store ptr %0, ptr %arr_ptr127, align 8
  store ptr %23, ptr %mask_ptr128, align 8
  call void @print_int_arr(ptr nonnull @res_bits.ED7143A1.0, i64 16, ptr nonnull %out_arr_alloca124)
  call void @print_int(ptr nonnull @res_counts.95A27DF5.0, i64 15, i64 2)
  call void @print_int(ptr nonnull @res_counts.95A27DF5.0, i64 15, i64 %.)
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

declare void @print_int_arr(ptr, i64, ptr) local_unnamed_addr

declare void @print_int(ptr, i64, i64) local_unnamed_addr

declare i64 @___qalloc() local_unnamed_addr

declare void @___reset(i64) local_unnamed_addr

define i64 @qmain(i64 %0) local_unnamed_addr {
entry:
  tail call void @setup(i64 %0)
  tail call void @__hugr__.__main__.integer_outputs.1()
  %1 = tail call i64 @teardown()
  ret i64 %1
}

declare void @setup(i64) local_unnamed_addr

declare i64 @teardown() local_unnamed_addr

attributes #0 = { noreturn }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!name = !{!0}

!0 = !{!"mainlib"}
