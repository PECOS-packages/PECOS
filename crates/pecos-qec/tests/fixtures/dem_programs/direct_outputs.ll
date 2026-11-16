; ModuleID = 'hugr'
source_filename = "hugr"

@"e_Array elem.E746B1A3.0" = private constant [43 x i8] c"*EXIT:INT:Array element is already borrowed"
@"e_Array cont.EFA5AC45.0" = private constant [70 x i8] c"EEXIT:INT:Array contains non-borrowed elements and cannot be discarded"
@"e_Array alre.5A300C2A.0" = private constant [57 x i8] c"8EXIT:INT:Array already contains an element at this index"
@"e_Some array.A77EF32E.0" = private constant [48 x i8] c"/EXIT:INT:Some array elements have been borrowed"
@res_physical.316B36C1.0 = private constant [22 x i8] c"\15USER:BOOLARR:physical"
@res_checks.C79F19B2.0 = private constant [17 x i8] c"\10USER:BOOL:checks"
@res_logical_z.F0F1F36B.0 = private constant [20 x i8] c"\13USER:BOOL:logical_z"
@res_logical_x.FFE63383.0 = private constant [23 x i8] c"\16USER:BOOLARR:logical_x"
@"e_No more qu.3B2EEBF0.0" = private constant [47 x i8] c".EXIT:INT:No more qubits available to allocate."

define void @__hugr__.__main__.direct_outputs.1() local_unnamed_addr {
alloca_block:
  %0 = tail call ptr @heap_alloc(i64 3)
  %1 = tail call ptr @heap_alloc(i64 8)
  store i64 -1, ptr %1, align 1
  %2 = tail call ptr @heap_alloc(i64 24)
  %3 = tail call ptr @heap_alloc(i64 8)
  store i64 -1, ptr %3, align 1
  %qalloc.i = tail call i64 @___qalloc()
  %not_max.not.not.i = icmp eq i64 %qalloc.i, -1
  br i1 %not_max.not.not.i, label %cond_218_case_0.i, label %__hugr__.__tk2_helios_qalloc.214.exit

cond_218_case_0.i:                                ; preds = %alloca_block
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.214.exit:            ; preds = %alloca_block
  tail call void @___reset(i64 %qalloc.i)
  %qalloc.i686 = tail call i64 @___qalloc()
  %not_max.not.not.i687 = icmp eq i64 %qalloc.i686, -1
  br i1 %not_max.not.not.i687, label %cond_218_case_0.i688, label %__hugr__.__tk2_helios_qalloc.214.exit689

cond_218_case_0.i688:                             ; preds = %__hugr__.__tk2_helios_qalloc.214.exit
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.214.exit689:         ; preds = %__hugr__.__tk2_helios_qalloc.214.exit
  tail call void @___reset(i64 %qalloc.i686)
  %qalloc.i690 = tail call i64 @___qalloc()
  %not_max.not.not.i691 = icmp eq i64 %qalloc.i690, -1
  br i1 %not_max.not.not.i691, label %cond_218_case_0.i692, label %__hugr__.__tk2_helios_qalloc.214.exit693

cond_218_case_0.i692:                             ; preds = %__hugr__.__tk2_helios_qalloc.214.exit689
  tail call void @panic(i32 1001, ptr nonnull @"e_No more qu.3B2EEBF0.0")
  unreachable

__hugr__.__tk2_helios_qalloc.214.exit693:         ; preds = %__hugr__.__tk2_helios_qalloc.214.exit689
  tail call void @___reset(i64 %qalloc.i690)
  %4 = tail call ptr @heap_alloc(i64 24)
  %5 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %5, align 1
  store i64 %qalloc.i, ptr %4, align 4
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %qalloc.i686, ptr %6, align 4
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 %qalloc.i690, ptr %7, align 4
  %8 = load i64, ptr %5, align 4
  %9 = trunc i64 %8 to i1
  br i1 %9, label %panic.i696, label %.thread

panic.i:                                          ; preds = %.thread.2, %.thread.1, %.thread
  tail call void @panic(i32 1002, ptr nonnull @"e_Array alre.5A300C2A.0")
  unreachable

mask_block_err.i:                                 ; preds = %cond_exit_62.2
  tail call void @panic(i32 1002, ptr nonnull @"e_Array cont.EFA5AC45.0")
  unreachable

panic.i696:                                       ; preds = %cond_exit_62.1, %cond_exit_62, %__hugr__.__tk2_helios_qalloc.214.exit693
  tail call void @panic(i32 1002, ptr nonnull @"e_Array elem.E746B1A3.0")
  unreachable

.thread:                                          ; preds = %__hugr__.__tk2_helios_qalloc.214.exit693
  %10 = or disjoint i64 %8, 1
  store i64 %10, ptr %5, align 4
  %11 = load i64, ptr %4, align 4
  %lazy_measure = tail call i64 @___lazy_measure(i64 %11)
  tail call void @___qfree(i64 %11)
  %12 = load i64, ptr %3, align 4
  %13 = trunc i64 %12 to i1
  br i1 %13, label %cond_exit_62, label %panic.i

cond_exit_62:                                     ; preds = %.thread
  %14 = and i64 %12, -2
  store i64 %14, ptr %3, align 4
  store i64 %lazy_measure, ptr %2, align 4
  %15 = load i64, ptr %5, align 4
  %16 = and i64 %15, 2
  %.not766 = icmp eq i64 %16, 0
  br i1 %.not766, label %.thread.1, label %panic.i696

.thread.1:                                        ; preds = %cond_exit_62
  %17 = or disjoint i64 %15, 2
  store i64 %17, ptr %5, align 4
  %18 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %19 = load i64, ptr %18, align 4
  %lazy_measure.1 = tail call i64 @___lazy_measure(i64 %19)
  tail call void @___qfree(i64 %19)
  %20 = load i64, ptr %3, align 4
  %21 = and i64 %20, 2
  %.not767 = icmp eq i64 %21, 0
  br i1 %.not767, label %panic.i, label %cond_exit_62.1

cond_exit_62.1:                                   ; preds = %.thread.1
  %22 = and i64 %20, -3
  store i64 %22, ptr %3, align 4
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i64 %lazy_measure.1, ptr %23, align 4
  %24 = load i64, ptr %5, align 4
  %25 = and i64 %24, 4
  %.not768 = icmp eq i64 %25, 0
  br i1 %.not768, label %.thread.2, label %panic.i696

.thread.2:                                        ; preds = %cond_exit_62.1
  %26 = or disjoint i64 %24, 4
  store i64 %26, ptr %5, align 4
  %27 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %28 = load i64, ptr %27, align 4
  %lazy_measure.2 = tail call i64 @___lazy_measure(i64 %28)
  tail call void @___qfree(i64 %28)
  %29 = load i64, ptr %3, align 4
  %30 = and i64 %29, 4
  %.not769 = icmp eq i64 %30, 0
  br i1 %.not769, label %panic.i, label %cond_exit_62.2

cond_exit_62.2:                                   ; preds = %.thread.2
  %31 = and i64 %29, -5
  store i64 %31, ptr %3, align 4
  %32 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i64 %lazy_measure.2, ptr %32, align 4
  %33 = load i64, ptr %5, align 4
  %34 = or i64 %33, -8
  store i64 %34, ptr %5, align 4
  %35 = icmp eq i64 %34, -1
  br i1 %35, label %loop_body113.preheader.preheader, label %mask_block_err.i

loop_body113.preheader.preheader:                 ; preds = %cond_exit_62.2
  tail call void @heap_free(ptr nonnull %4)
  tail call void @heap_free(ptr nonnull %5)
  %36 = load i64, ptr %3, align 4
  %37 = trunc i64 %36 to i1
  br i1 %37, label %panic.i699, label %__barray_check_bounds.exit702

panic.i699:                                       ; preds = %__barray_check_bounds.exit698.2, %cond_exit_128, %loop_body113.preheader.preheader
  tail call void @panic(i32 1002, ptr nonnull @"e_Array elem.E746B1A3.0")
  unreachable

__barray_check_bounds.exit702:                    ; preds = %loop_body113.preheader.preheader
  %38 = or disjoint i64 %36, 1
  store i64 %38, ptr %3, align 4
  %39 = load i64, ptr %2, align 4
  tail call void @___inc_future_refcount(i64 %39)
  %40 = load i64, ptr %3, align 4
  %41 = trunc i64 %40 to i1
  br i1 %41, label %cond_169_case_1.thread, label %panic.i703

panic.i703:                                       ; preds = %__barray_check_bounds.exit702.2, %__barray_check_bounds.exit702.1, %__barray_check_bounds.exit702
  tail call void @panic(i32 1002, ptr nonnull @"e_Array alre.5A300C2A.0")
  unreachable

panic.i707:                                       ; preds = %__barray_check_bounds.exit706.2, %cond_169_case_1.thread.1, %cond_169_case_1.thread
  tail call void @panic(i32 1002, ptr nonnull @"e_Array alre.5A300C2A.0")
  unreachable

cond_169_case_1.thread:                           ; preds = %__barray_check_bounds.exit702
  %42 = and i64 %40, -2
  store i64 %42, ptr %3, align 4
  store i64 %39, ptr %2, align 4
  %read_bool = tail call i1 @___read_future_bool(i64 %39)
  tail call void @___dec_future_refcount(i64 %39)
  %43 = load i64, ptr %1, align 4
  %44 = trunc i64 %43 to i1
  br i1 %44, label %cond_exit_128, label %panic.i707

mask_block_err.i712:                              ; preds = %cond_exit_265.2
  tail call void @panic(i32 1002, ptr nonnull @"e_Array cont.EFA5AC45.0")
  unreachable

__barray_mask_borrow.exit720:                     ; preds = %cond_exit_128.2
  %45 = or disjoint i64 %85, 1
  store i64 %45, ptr %3, align 4
  %46 = load i64, ptr %2, align 4
  tail call void @___dec_future_refcount(i64 %46)
  br label %cond_exit_265

cond_exit_265:                                    ; preds = %__barray_mask_borrow.exit720, %cond_exit_128.2
  %47 = load i64, ptr %3, align 4
  %48 = and i64 %47, 2
  %.not770 = icmp eq i64 %48, 0
  br i1 %.not770, label %__barray_mask_borrow.exit720.1, label %cond_exit_265.1

__barray_mask_borrow.exit720.1:                   ; preds = %cond_exit_265
  %49 = or disjoint i64 %47, 2
  store i64 %49, ptr %3, align 4
  %50 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %51 = load i64, ptr %50, align 4
  tail call void @___dec_future_refcount(i64 %51)
  br label %cond_exit_265.1

cond_exit_265.1:                                  ; preds = %__barray_mask_borrow.exit720.1, %cond_exit_265
  %52 = load i64, ptr %3, align 4
  %53 = and i64 %52, 4
  %.not771 = icmp eq i64 %53, 0
  br i1 %.not771, label %__barray_mask_borrow.exit720.2, label %cond_exit_265.2

__barray_mask_borrow.exit720.2:                   ; preds = %cond_exit_265.1
  %54 = or disjoint i64 %52, 4
  store i64 %54, ptr %3, align 4
  %55 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %56 = load i64, ptr %55, align 4
  tail call void @___dec_future_refcount(i64 %56)
  br label %cond_exit_265.2

cond_exit_265.2:                                  ; preds = %__barray_mask_borrow.exit720.2, %cond_exit_265.1
  %57 = load i64, ptr %3, align 4
  %58 = or i64 %57, -8
  store i64 %58, ptr %3, align 4
  %59 = icmp eq i64 %58, -1
  br i1 %59, label %loop_out112, label %mask_block_err.i712

cond_exit_128:                                    ; preds = %cond_169_case_1.thread
  %60 = and i64 %43, -2
  store i64 %60, ptr %1, align 4
  store i1 %read_bool, ptr %0, align 1
  %61 = load i64, ptr %3, align 4
  %62 = and i64 %61, 2
  %.not754 = icmp eq i64 %62, 0
  br i1 %.not754, label %__barray_check_bounds.exit702.1, label %panic.i699

__barray_check_bounds.exit702.1:                  ; preds = %cond_exit_128
  %63 = or disjoint i64 %61, 2
  store i64 %63, ptr %3, align 4
  %64 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %65 = load i64, ptr %64, align 4
  tail call void @___inc_future_refcount(i64 %65)
  %66 = load i64, ptr %3, align 4
  %67 = and i64 %66, 2
  %.not755 = icmp eq i64 %67, 0
  br i1 %.not755, label %panic.i703, label %cond_169_case_1.thread.1

cond_169_case_1.thread.1:                         ; preds = %__barray_check_bounds.exit702.1
  %68 = and i64 %66, -3
  store i64 %68, ptr %3, align 4
  store i64 %65, ptr %64, align 4
  %read_bool.1 = tail call i1 @___read_future_bool(i64 %65)
  tail call void @___dec_future_refcount(i64 %65)
  %69 = load i64, ptr %1, align 4
  %70 = and i64 %69, 2
  %.not756 = icmp eq i64 %70, 0
  br i1 %.not756, label %panic.i707, label %__barray_check_bounds.exit698.2

__barray_check_bounds.exit698.2:                  ; preds = %cond_169_case_1.thread.1
  %71 = and i64 %69, -3
  store i64 %71, ptr %1, align 4
  %72 = getelementptr inbounds nuw i8, ptr %0, i64 1
  store i1 %read_bool.1, ptr %72, align 1
  %73 = load i64, ptr %3, align 4
  %74 = and i64 %73, 4
  %.not757 = icmp eq i64 %74, 0
  br i1 %.not757, label %__barray_check_bounds.exit702.2, label %panic.i699

__barray_check_bounds.exit702.2:                  ; preds = %__barray_check_bounds.exit698.2
  %75 = or disjoint i64 %73, 4
  store i64 %75, ptr %3, align 4
  %76 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %77 = load i64, ptr %76, align 4
  tail call void @___inc_future_refcount(i64 %77)
  %78 = load i64, ptr %3, align 4
  %79 = and i64 %78, 4
  %.not758 = icmp eq i64 %79, 0
  br i1 %.not758, label %panic.i703, label %__barray_check_bounds.exit706.2

__barray_check_bounds.exit706.2:                  ; preds = %__barray_check_bounds.exit702.2
  %80 = and i64 %78, -5
  store i64 %80, ptr %3, align 4
  store i64 %77, ptr %76, align 4
  %read_bool.2 = tail call i1 @___read_future_bool(i64 %77)
  tail call void @___dec_future_refcount(i64 %77)
  %81 = load i64, ptr %1, align 4
  %82 = and i64 %81, 4
  %.not759 = icmp eq i64 %82, 0
  br i1 %.not759, label %panic.i707, label %cond_exit_128.2

cond_exit_128.2:                                  ; preds = %__barray_check_bounds.exit706.2
  %83 = and i64 %81, -5
  store i64 %83, ptr %1, align 4
  %84 = getelementptr inbounds nuw i8, ptr %0, i64 2
  store i1 %read_bool.2, ptr %84, align 1
  %85 = load i64, ptr %3, align 4
  %86 = trunc i64 %85 to i1
  br i1 %86, label %cond_exit_265, label %__barray_mask_borrow.exit720

loop_out112:                                      ; preds = %cond_exit_265.2
  tail call void @heap_free(ptr nonnull %2)
  tail call void @heap_free(ptr nonnull %3)
  %87 = load i64, ptr %1, align 4
  %88 = trunc i64 %87 to i1
  br i1 %88, label %panic.i721, label %__barray_mask_check_not_borrowed.exit

panic.i721:                                       ; preds = %loop_out112
  tail call void @panic(i32 1002, ptr nonnull @"e_Array elem.E746B1A3.0")
  unreachable

__barray_mask_check_not_borrowed.exit:            ; preds = %loop_out112
  %89 = load i1, ptr %0, align 1
  %90 = and i64 %87, 2
  %.not = icmp eq i64 %90, 0
  br i1 %.not, label %__barray_mask_check_not_borrowed.exit723, label %panic.i722

panic.i722:                                       ; preds = %__barray_mask_check_not_borrowed.exit
  tail call void @panic(i32 1002, ptr nonnull @"e_Array elem.E746B1A3.0")
  unreachable

__barray_mask_check_not_borrowed.exit723:         ; preds = %__barray_mask_check_not_borrowed.exit
  %91 = load i1, ptr %72, align 1
  %92 = and i64 %87, 4
  %.not751 = icmp eq i64 %92, 0
  br i1 %.not751, label %__barray_mask_check_not_borrowed.exit725, label %panic.i724

panic.i724:                                       ; preds = %__barray_mask_check_not_borrowed.exit723
  tail call void @panic(i32 1002, ptr nonnull @"e_Array elem.E746B1A3.0")
  unreachable

__barray_mask_check_not_borrowed.exit725:         ; preds = %__barray_mask_check_not_borrowed.exit723
  %93 = load i1, ptr %84, align 1
  store i64 0, ptr %1, align 4
  %94 = tail call ptr @heap_alloc(i64 3)
  %95 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %95, align 1
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %94, ptr noundef nonnull align 1 dereferenceable(3) %0, i64 3, i1 false)
  tail call void @heap_free(ptr nonnull %94)
  %96 = load i64, ptr %1, align 4
  %97 = and i64 %96, 7
  store i64 %97, ptr %1, align 4
  %98 = icmp eq i64 %97, 0
  br i1 %98, label %__barray_check_none_borrowed.exit730, label %mask_block_err.i728

mask_block_err.i728:                              ; preds = %__barray_mask_check_not_borrowed.exit725
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit730:             ; preds = %__barray_mask_check_not_borrowed.exit725
  %99 = tail call ptr @heap_alloc(i64 2)
  %100 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %100, align 1
  store i1 %91, ptr %99, align 1
  %101 = getelementptr inbounds nuw i8, ptr %99, i64 1
  store i1 %93, ptr %101, align 1
  %102 = load i64, ptr %100, align 4
  %103 = and i64 %102, 3
  store i64 %103, ptr %100, align 4
  %104 = icmp eq i64 %103, 0
  br i1 %104, label %__barray_check_none_borrowed.exit733, label %mask_block_err.i731

mask_block_err.i731:                              ; preds = %__barray_check_none_borrowed.exit730
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit733:             ; preds = %__barray_check_none_borrowed.exit730
  %105 = tail call ptr @heap_alloc(i64 2)
  %106 = tail call ptr @heap_alloc(i64 8)
  store i64 0, ptr %106, align 1
  %107 = load i16, ptr %99, align 1
  store i16 %107, ptr %105, align 1
  tail call void @heap_free(ptr nonnull %105)
  %108 = load i64, ptr %100, align 4
  %109 = and i64 %108, 3
  store i64 %109, ptr %100, align 4
  %110 = icmp eq i64 %109, 0
  br i1 %110, label %__barray_check_none_borrowed.exit736, label %mask_block_err.i734

mask_block_err.i734:                              ; preds = %__barray_check_none_borrowed.exit733
  tail call void @panic(i32 1002, ptr nonnull @"e_Some array.A77EF32E.0")
  unreachable

__barray_check_none_borrowed.exit736:             ; preds = %__barray_check_none_borrowed.exit733
  %111 = xor i1 %89, %91
  %out_arr_alloca = alloca <{ i32, i32, ptr, ptr }>, align 8
  %arr_ptr = getelementptr inbounds nuw i8, ptr %out_arr_alloca, i64 8
  %mask_ptr = getelementptr inbounds nuw i8, ptr %out_arr_alloca, i64 16
  %112 = alloca [3 x i1], align 1
  store i1 false, ptr %112, align 1
  %.repack683 = getelementptr inbounds nuw i8, ptr %112, i64 1
  store i1 false, ptr %.repack683, align 1
  %.repack684 = getelementptr inbounds nuw i8, ptr %112, i64 2
  store i1 false, ptr %.repack684, align 1
  store <2 x i32> <i32 3, i32 1>, ptr %out_arr_alloca, align 8
  store ptr %0, ptr %arr_ptr, align 8
  store ptr %112, ptr %mask_ptr, align 8
  call void @print_bool_arr(ptr nonnull @res_physical.316B36C1.0, i64 21, ptr nonnull %out_arr_alloca)
  call void @print_bool(ptr nonnull @res_checks.C79F19B2.0, i64 16, i1 %111)
  call void @print_bool(ptr nonnull @res_logical_z.F0F1F36B.0, i64 19, i1 %89)
  %out_arr_alloca409 = alloca <{ i32, i32, ptr, ptr }>, align 8
  %arr_ptr412 = getelementptr inbounds nuw i8, ptr %out_arr_alloca409, i64 8
  %mask_ptr413 = getelementptr inbounds nuw i8, ptr %out_arr_alloca409, i64 16
  %113 = alloca [2 x i1], align 1
  store i1 false, ptr %113, align 1
  %.repack685 = getelementptr inbounds nuw i8, ptr %113, i64 1
  store i1 false, ptr %.repack685, align 1
  store <2 x i32> <i32 2, i32 1>, ptr %out_arr_alloca409, align 8
  store ptr %99, ptr %arr_ptr412, align 8
  store ptr %113, ptr %mask_ptr413, align 8
  call void @print_bool_arr(ptr nonnull @res_logical_x.FFE63383.0, i64 22, ptr nonnull %out_arr_alloca409)
  ret void
}

declare ptr @heap_alloc(i64) local_unnamed_addr

; Function Attrs: noreturn
declare void @panic(i32, ptr) local_unnamed_addr #0

declare void @heap_free(ptr) local_unnamed_addr

declare i64 @___lazy_measure(i64) local_unnamed_addr

declare void @___qfree(i64) local_unnamed_addr

declare void @___inc_future_refcount(i64) local_unnamed_addr

declare void @___dec_future_refcount(i64) local_unnamed_addr

declare i1 @___read_future_bool(i64) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

declare void @print_bool_arr(ptr, i64, ptr) local_unnamed_addr

declare void @print_bool(ptr, i64, i1) local_unnamed_addr

declare i64 @___qalloc() local_unnamed_addr

declare void @___reset(i64) local_unnamed_addr

define i64 @qmain(i64 %0) local_unnamed_addr {
entry:
  tail call void @setup(i64 %0)
  tail call void @__hugr__.__main__.direct_outputs.1()
  %1 = tail call i64 @teardown()
  ret i64 %1
}

declare void @setup(i64) local_unnamed_addr

declare i64 @teardown() local_unnamed_addr

attributes #0 = { noreturn }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!name = !{!0}

!0 = !{!"mainlib"}
