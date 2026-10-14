; ModuleID = 'quantum_module'
source_filename = "quantum_module"

@str_c = constant [2 x i8] c"c\00"
@str_c1 = constant [3 x i8] c"c1\00"

define { i1, i1 } @_hugr_bell_state() #0 {
alloca_block:
  %"0" = alloca i1, align 1
  %"1" = alloca i1, align 1
  %"4_0" = alloca i1, align 1
  %"4_1" = alloca i1, align 1
  %"01" = alloca i1, align 1
  %"12" = alloca i1, align 1
  %"19_0" = alloca {}, align 8
  %"18_0" = alloca i1, align 1
  %"18_1" = alloca i1, align 1
  %"14_0" = alloca {}, align 8
  %"12_0" = alloca {}, align 8
  %"9_0" = alloca i16, align 2
  %"10_0" = alloca i16, align 2
  %"11_0" = alloca i16, align 2
  %"13_0" = alloca i16, align 2
  %"13_1" = alloca i16, align 2
  %"15_0" = alloca i1, align 1
  %"16_0" = alloca i1, align 1
  %"17_0" = alloca { i1, i1 }, align 8
  br label %entry_block

entry_block:                                      ; preds = %alloca_block
  br label %0

0:                                                ; preds = %entry_block
  store {} undef, {}* %"19_0", align 1
  store {} undef, {}* %"14_0", align 1
  store {} undef, {}* %"12_0", align 1
  %qubit_usize = call i64 @__quantum__rt__qubit_allocate()
  %qubit = trunc i64 %qubit_usize to i16
  store i16 %qubit, i16* %"9_0", align 2
  %qubit_usize3 = call i64 @__quantum__rt__qubit_allocate()
  %qubit4 = trunc i64 %qubit_usize3 to i16
  store i16 %qubit4, i16* %"10_0", align 2
  %"9_05" = load i16, i16* %"9_0", align 2
  %qubit_i64 = zext i16 %"9_05" to i64
  call void @__quantum__qis__h__body(i64 %qubit_i64)
  store i16 %"9_05", i16* %"11_0", align 2
  %"11_06" = load i16, i16* %"11_0", align 2
  %"10_07" = load i16, i16* %"10_0", align 2
  %control_i64 = zext i16 %"11_06" to i64
  %target_i64 = zext i16 %"10_07" to i64
  call void @__quantum__qis__cx__body(i64 %control_i64, i64 %target_i64)
  store i16 %"11_06", i16* %"13_0", align 2
  store i16 %"10_07", i16* %"13_1", align 2
  %"13_08" = load i16, i16* %"13_0", align 2
  %qubit_i649 = zext i16 %"13_08" to i64
  %result_id = call i64 @__quantum__rt__result_allocate()
  %measurement_result = call i32 @__quantum__qis__m__body(i64 %qubit_i649, i64 %result_id)
  %result_ptr = inttoptr i64 %result_id to i8*
  call void @__quantum__rt__result_record_output(i8* %result_ptr, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str_c, i32 0, i32 0))
  %is_one = icmp ne i32 %measurement_result, 0
  store i1 %is_one, i1* %"15_0", align 1
  %"13_110" = load i16, i16* %"13_1", align 2
  %qubit_i6411 = zext i16 %"13_110" to i64
  %result_id12 = call i64 @__quantum__rt__result_allocate()
  %measurement_result13 = call i32 @__quantum__qis__m__body(i64 %qubit_i6411, i64 %result_id12)
  %result_ptr14 = inttoptr i64 %result_id12 to i8*
  call void @__quantum__rt__result_record_output(i8* %result_ptr14, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @str_c1, i32 0, i32 0))
  %is_one15 = icmp ne i32 %measurement_result13, 0
  store i1 %is_one15, i1* %"16_0", align 1
  %"15_016" = load i1, i1* %"15_0", align 1
  %"16_017" = load i1, i1* %"16_0", align 1
  %1 = insertvalue { i1, i1 } poison, i1 %"15_016", 0
  %2 = insertvalue { i1, i1 } %1, i1 %"16_017", 1
  store { i1, i1 } %2, { i1, i1 }* %"17_0", align 1
  %"17_018" = load { i1, i1 }, { i1, i1 }* %"17_0", align 1
  %3 = extractvalue { i1, i1 } %"17_018", 0
  %4 = extractvalue { i1, i1 } %"17_018", 1
  store i1 %3, i1* %"18_0", align 1
  store i1 %4, i1* %"18_1", align 1
  %"19_019" = load {}, {}* %"19_0", align 1
  %"18_020" = load i1, i1* %"18_0", align 1
  %"18_121" = load i1, i1* %"18_1", align 1
  store {} %"19_019", {}* %"19_0", align 1
  store i1 %"18_020", i1* %"18_0", align 1
  store i1 %"18_121", i1* %"18_1", align 1
  %"19_022" = load {}, {}* %"19_0", align 1
  %"18_023" = load i1, i1* %"18_0", align 1
  %"18_124" = load i1, i1* %"18_1", align 1
  switch i1 false, label %5 [
  ]

5:                                                ; preds = %0
  store i1 %"18_023", i1* %"01", align 1
  store i1 %"18_124", i1* %"12", align 1
  br label %6

6:                                                ; preds = %5
  %"025" = load i1, i1* %"01", align 1
  %"126" = load i1, i1* %"12", align 1
  store i1 %"025", i1* %"4_0", align 1
  store i1 %"126", i1* %"4_1", align 1
  %"4_027" = load i1, i1* %"4_0", align 1
  %"4_128" = load i1, i1* %"4_1", align 1
  store i1 %"4_027", i1* %"0", align 1
  store i1 %"4_128", i1* %"1", align 1
  %"029" = load i1, i1* %"0", align 1
  %"130" = load i1, i1* %"1", align 1
  %mrv = insertvalue { i1, i1 } undef, i1 %"029", 0
  %mrv31 = insertvalue { i1, i1 } %mrv, i1 %"130", 1
  ret { i1, i1 } %mrv31
}

declare i64 @__quantum__rt__qubit_allocate()

declare void @__quantum__qis__h__body(i64)

declare void @__quantum__qis__cx__body(i64, i64)

declare i64 @__quantum__rt__result_allocate()

declare i32 @__quantum__qis__m__body(i64, i64)

declare void @__quantum__rt__result_record_output(i8*, i8*)

attributes #0 = { "EntryPoint" }
