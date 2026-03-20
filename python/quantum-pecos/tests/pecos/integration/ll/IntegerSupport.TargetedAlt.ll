; ModuleID = 'IntegerSupport.Targeted.ll'
source_filename = "qat-link"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [6 x i8] c"0_t0b\00"
@1 = internal constant [6 x i8] c"1_t1b\00"
@2 = internal constant [6 x i8] c"2_t2b\00"
@3 = internal constant [6 x i8] c"3_t3b\00"
@4 = internal constant [6 x i8] c"4_t4b\00"
@5 = internal constant [6 x i8] c"5_t5b\00"
@6 = internal constant [6 x i8] c"6_t6b\00"

define void @Tests__Common__IntegerSupport() #0 {
entry:
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  call void @__quantum__qis__reset__body(%Qubit* null)
  %0 = call i1 @__quantum__qis__read_result__body(%Result* null)
  br i1 %0, label %then0__1, label %continue__2

then0__1:                                         ; preds = %entry
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  br label %continue__2

continue__2:                                      ; preds = %then0__1, %entry
  %1 = call i1 @__quantum__qis__read_result__body(%Result* null)
  %spec.select2 = select i1 %1, i64 -2, i64 0
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* null)
  %2 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  br i1 %2, label %then0__3, label %continue__4

then0__3:                                         ; preds = %continue__2
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  br label %continue__4

continue__4:                                      ; preds = %then0__3, %continue__2
  %3 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
  %4 = add nsw i64 %spec.select2, -2
  %substraction.1 = select i1 %3, i64 %4, i64 %spec.select2
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull inttoptr (i64 2 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* null)
  %5 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 2 to %Result*))
  br i1 %5, label %then0__5, label %continue__6

then0__5:                                         ; preds = %continue__4
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  br label %continue__6

continue__6:                                      ; preds = %then0__5, %continue__4
  %6 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 2 to %Result*))
  %7 = add nsw i64 %substraction.1, -2
  %substraction.2 = select i1 %6, i64 %7, i64 %substraction.1
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull inttoptr (i64 3 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* null)
  %8 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 3 to %Result*))
  br i1 %8, label %then0__7, label %continue__8

then0__7:                                         ; preds = %continue__6
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  br label %continue__8

continue__8:                                      ; preds = %then0__7, %continue__6
  %9 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 3 to %Result*))
  %10 = add nsw i64 %substraction.2, -2
  %substraction.3 = select i1 %9, i64 %10, i64 %substraction.2
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull inttoptr (i64 4 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* null)
  %11 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
  br i1 %11, label %then0__9, label %continue__10

then0__9:                                         ; preds = %continue__8
  call void @__quantum__qis__rx__body(double 0x400921FB54442D18, %Qubit* null)
  br label %continue__10

continue__10:                                     ; preds = %then0__9, %continue__8
  %12 = select i1 %1, i64 2, i64 1
  %spec.select = select i1 %1, i64 1, i64 0
  %sum.1 = select i1 %3, i64 %12, i64 %spec.select
  %13 = select i1 %6, i64 1, i64 0
  %sum.2 = add nuw nsw i64 %sum.1, %13
  %14 = select i1 %9, i64 1, i64 0
  %sum.3 = add nuw nsw i64 %sum.2, %14
  %15 = call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
  %16 = add i64 %substraction.3, -2
  %17 = select i1 %15, i64 1, i64 0
  %sum.4 = add nuw nsw i64 %sum.3, %17
  %substraction.4 = select i1 %15, i64 %16, i64 %substraction.3
  call void @__quantum__qis__reset__body(%Qubit* null)
  %negativeMultiplication = mul i64 %substraction.4, 3
  %doubleNegativeMultiplication = mul i64 %substraction.4, -3
  call void @__quantum__rt__tuple_start_record_output()
  call void @__quantum__rt__int_record_output(i64 %sum.4,                        i8* getelementptr inbounds ([6 x i8], [6 x i8]* @0, i64 0, i64 0))
  call void @__quantum__rt__int_record_output(i64 %substraction.4,               i8* getelementptr inbounds ([6 x i8], [6 x i8]* @1, i64 0, i64 0))
  call void @__quantum__rt__int_record_output(i64 %negativeMultiplication,       i8* getelementptr inbounds ([6 x i8], [6 x i8]* @3, i64 0, i64 0))
  call void @__quantum__rt__int_record_output(i64 %doubleNegativeMultiplication, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @4, i64 0, i64 0))
  call void @__quantum__rt__tuple_end_record_output()
  ret void
}

declare void @__quantum__qis__rx__body(double, %Qubit*)

declare void @__quantum__qis__reset__body(%Qubit*)

declare void @__quantum__rt__tuple_start_record_output()

declare void @__quantum__rt__int_record_output(i64, i8*)

declare void @__quantum__rt__tuple_end_record_output()

declare void @__quantum__qis__mz__body(%Qubit*, %Result*)

declare i1 @__quantum__qis__read_result__body(%Result*)

attributes #0 = { "EntryPoint" "entry_point" "entrypoint_index"="0" "maxQubitIndex"="0" "maxResultIndex"="4" "requiredQubits"="1" "requiredResults"="5" }
