; ModuleID = 'ArithmeticOps.TargetedAlt.ll'
source_filename = "qat-link"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [6 x i8] c"0_t0i\00"
@1 = internal constant [6 x i8] c"1_t1i\00"
@2 = internal constant [6 x i8] c"2_t2i\00"
@3 = internal constant [6 x i8] c"3_t3i\00"

define void @Tests__Common__L3__ArithmeticOps() #0 {
entry:
  call void @__quantum__qis__x__body(%Qubit* null)
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 3 to %Qubit*))
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  call void @__quantum__qis__reset__body(%Qubit* null)
  %0 = call i1 @__quantum__qis__read_result__body(%Result* null)
  br i1 %0, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  call void @__quantum__qis__x__body(%Qubit* null)
  br label %continue__1

continue__1:                                      ; preds = %then0__1, %entry
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  %1 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  br i1 %1, label %then0__2, label %continue__2

then0__2:                                         ; preds = %continue__1
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  br label %continue__2

continue__2:                                      ; preds = %then0__2, %continue__1
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 2 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  %2 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 2 to %Result*))
  br i1 %2, label %then0__3, label %continue__3

then0__3:                                         ; preds = %continue__2
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*))
  br label %continue__3

continue__3:                                      ; preds = %then0__3, %continue__2
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 3 to %Qubit*), %Result* nonnull inttoptr (i64 3 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 3 to %Qubit*))
  %3 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 3 to %Result*))
  br i1 %3, label %then0__4, label %continue__4

then0__4:                                         ; preds = %continue__3
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 3 to %Qubit*))
  br label %continue__4

continue__4:                                      ; preds = %then0__4, %continue__3
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*), %Result* nonnull inttoptr (i64 4 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
  %4 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
  br i1 %4, label %then0__5, label %continue__5

then0__5:                                         ; preds = %continue__4
  call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
  br label %continue__5

continue__5:                                      ; preds = %then0__5, %continue__4
  %5 = call i1 @__quantum__qis__read_result__body(%Result* null)
  br i1 %5, label %then0__6, label %continue__6

then0__6:                                         ; preds = %continue__5
  br label %continue__6

continue__6:                                      ; preds = %then0__6, %continue__5
  %count.0 = phi i64 [ 1, %then0__6 ], [ 0, %continue__5 ]
  %countPos.0 = phi i64 [ 5, %then0__6 ], [ 0, %continue__5 ]
  %countNeg.0 = phi i64 [ 8, %then0__6 ], [ 10, %continue__5 ]
  %countMul.0 = phi i64 [ 3, %then0__6 ], [ 1, %continue__5 ]
  %6 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  br i1 %6, label %then0__7, label %continue__7

then0__7:                                         ; preds = %continue__6
  %7 = add nuw nsw i64 %count.0, 1
  %8 = add nuw nsw i64 %countPos.0, 5
  %9 = sub nsw i64 %countNeg.0, 2
  %10 = mul nuw nsw i64 %countMul.0, 3
  br label %continue__7

continue__7:                                      ; preds = %then0__7, %continue__6
  %count.1 = phi i64 [ %7, %then0__7 ], [ %count.0, %continue__6 ]
  %countPos.1 = phi i64 [ %8, %then0__7 ], [ %countPos.0, %continue__6 ]
  %countNeg.1 = phi i64 [ %9, %then0__7 ], [ %countNeg.0, %continue__6 ]
  %countMul.1 = phi i64 [ %10, %then0__7 ], [ %countMul.0, %continue__6 ]
  %11 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 2 to %Result*))
  br i1 %11, label %then0__8, label %continue__8

then0__8:                                         ; preds = %continue__7
  %12 = add nsw i64 %count.1, 1
  %13 = add nsw i64 %countPos.1, 5
  %14 = sub nsw i64 %countNeg.1, 2
  %15 = mul nsw i64 %countMul.1, 3
  br label %continue__8

continue__8:                                      ; preds = %then0__8, %continue__7
  %count.2 = phi i64 [ %12, %then0__8 ], [ %count.1, %continue__7 ]
  %countPos.2 = phi i64 [ %13, %then0__8 ], [ %countPos.1, %continue__7 ]
  %countNeg.2 = phi i64 [ %14, %then0__8 ], [ %countNeg.1, %continue__7 ]
  %countMul.2 = phi i64 [ %15, %then0__8 ], [ %countMul.1, %continue__7 ]
  %16 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 3 to %Result*))
  br i1 %16, label %then0__9, label %continue__9

then0__9:                                         ; preds = %continue__8
  %17 = add nsw i64 %count.2, 1
  %18 = add nsw i64 %countPos.2, 5
  %19 = sub nsw i64 %countNeg.2, 2
  %20 = mul nsw i64 %countMul.2, 3
  br label %continue__9

continue__9:                                      ; preds = %then0__9, %continue__8
  %count.3 = phi i64 [ %17, %then0__9 ], [ %count.2, %continue__8 ]
  %countPos.3 = phi i64 [ %18, %then0__9 ], [ %countPos.2, %continue__8 ]
  %countNeg.3 = phi i64 [ %19, %then0__9 ], [ %countNeg.2, %continue__8 ]
  %countMul.3 = phi i64 [ %20, %then0__9 ], [ %countMul.2, %continue__8 ]
  %21 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
  br i1 %21, label %then0__10, label %continue__10

then0__10:                                        ; preds = %continue__9
  %22 = add i64 %count.3, 1
  %23 = add i64 %countPos.3, 5
  %24 = sub i64 %countNeg.3, 2
  %25 = mul i64 %countMul.3, 3
  br label %continue__10

continue__10:                                     ; preds = %then0__10, %continue__9
  %count.4 = phi i64 [ %22, %then0__10 ], [ %count.3, %continue__9 ]
  %countPos.4 = phi i64 [ %23, %then0__10 ], [ %countPos.3, %continue__9 ]
  %countNeg.4 = phi i64 [ %24, %then0__10 ], [ %countNeg.3, %continue__9 ]
  %countMul.4 = phi i64 [ %25, %then0__10 ], [ %countMul.3, %continue__9 ]
  call void @__quantum__rt__tuple_start_record_output()
  call void @__quantum__rt__int_record_output(i64 %count.4, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @0, i64 0, i64 0))
  call void @__quantum__rt__int_record_output(i64 %countPos.4, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @1, i64 0, i64 0))
  call void @__quantum__rt__int_record_output(i64 %countNeg.4, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @2, i64 0, i64 0))
  call void @__quantum__rt__int_record_output(i64 %countMul.4, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @3, i64 0, i64 0))
  call void @__quantum__rt__tuple_end_record_output()
  ret void
}

declare %Qubit* @__quantum__rt__qubit_allocate()

declare void @__quantum__qis__x__body(%Qubit*)

declare %Result* @__quantum__rt__result_get_zero()

declare void @__quantum__rt__result_update_reference_count(%Result*, i32)

declare %Result* @__quantum__qis__m__body(%Qubit*)

declare void @__quantum__qis__reset__body(%Qubit*)

declare %Result* @__quantum__rt__result_get_one()

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare void @__quantum__rt__qubit_release(%Qubit*)

declare void @__quantum__rt__tuple_start_record_output()

declare void @__quantum__rt__int_record_output(i64, i8*)

declare void @__quantum__rt__tuple_end_record_output()

declare void @__quantum__qis__mz__body(%Qubit*, %Result*)

declare i1 @__quantum__qis__read_result__body(%Result*)

attributes #0 = { "EntryPoint" "maxQubitIndex"="4" "maxResultIndex"="4" "requiredQubits"="5" "requiredResults"="5" }
