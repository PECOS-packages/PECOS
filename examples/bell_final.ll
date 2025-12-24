; ModuleID = 'quantum_module'
source_filename = "quantum_module"

@str_c = constant [2 x i8] c"c\00"

define void @bell_state() #0 {
alloca_block:
  %"23_0" = alloca {}, align 8
  %"22_0" = alloca {}, align 8
  %"19_0" = alloca {}, align 8
  %"14_0" = alloca {}, align 8
  %"12_0" = alloca {}, align 8
  %"9_0" = alloca i16, align 2
  %"10_0" = alloca i16, align 2
  %"11_0" = alloca i16, align 2
  %"13_0" = alloca i16, align 2
  %"13_1" = alloca i16, align 2
  %"15_0" = alloca i1, align 1
  %"16_0" = alloca i1, align 1
  %"20_0" = alloca i1, align 1
  %"17_0" = alloca i1, align 1
  br label %entry_block

entry_block:                                      ; preds = %alloca_block
  br label %0

0:                                                ; preds = %entry_block
  store {} undef, {}* %"23_0", align 1
  %"23_01" = load {}, {}* %"23_0", align 1
  store {} %"23_01", {}* %"23_0", align 1
  store {} undef, {}* %"22_0", align 1
  store {} undef, {}* %"19_0", align 1
  store {} undef, {}* %"14_0", align 1
  store {} undef, {}* %"12_0", align 1
  %qubit_usize = call i64 @__quantum__rt__qubit_allocate()
  %qubit = trunc i64 %qubit_usize to i16
  store i16 %qubit, i16* %"9_0", align 2
  %qubit_usize2 = call i64 @__quantum__rt__qubit_allocate()
  %qubit3 = trunc i64 %qubit_usize2 to i16
  store i16 %qubit3, i16* %"10_0", align 2
  %"9_04" = load i16, i16* %"9_0", align 2
  %qubit_usize5 = zext i16 %"9_04" to i64
  call void @__quantum__qis__h__body(i64 %qubit_usize5)
  store i16 %"9_04", i16* %"11_0", align 2
  %"11_06" = load i16, i16* %"11_0", align 2
  %"10_07" = load i16, i16* %"10_0", align 2
  %control_usize = zext i16 %"11_06" to i64
  %target_usize = zext i16 %"10_07" to i64
  call void @__quantum__qis__cx__body(i64 %control_usize, i64 %target_usize)
  store i16 %"11_06", i16* %"13_0", align 2
  store i16 %"10_07", i16* %"13_1", align 2
  %"13_08" = load i16, i16* %"13_0", align 2
  %result_id = call i64 @__quantum__rt__result_allocate()
  %qubit_usize9 = zext i16 %"13_08" to i64
  %measurement = call i32 @__quantum__qis__m__body(i64 %qubit_usize9, i64 %result_id)
  call void @__quantum__rt__result_record_output(i64 %result_id, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str_c, i32 0, i32 0))
  %bool_result = icmp ne i32 %measurement, 0
  store i1 %bool_result, i1* %"15_0", align 1
  %"13_110" = load i16, i16* %"13_1", align 2
  %result_id11 = call i64 @__quantum__rt__result_allocate()
  %qubit_usize12 = zext i16 %"13_110" to i64
  %measurement13 = call i32 @__quantum__qis__m__body(i64 %qubit_usize12, i64 %result_id11)
  call void @__quantum__rt__result_record_output(i64 %result_id11, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str_c, i32 0, i32 0))
  %bool_result14 = icmp ne i32 %measurement13, 0
  store i1 %bool_result14, i1* %"16_0", align 1
  %"16_015" = load i1, i1* %"16_0", align 1
  store i1 %"16_015", i1* %"20_0", align 1
  %"15_016" = load i1, i1* %"15_0", align 1
  store i1 %"15_016", i1* %"17_0", align 1
  %"17_017" = load i1, i1* %"17_0", align 1
  %"20_018" = load i1, i1* %"20_0", align 1
  %"23_019" = load {}, {}* %"23_0", align 1
  switch i1 false, label %1 [
  ]

1:                                                ; preds = %0
  br label %2

2:                                                ; preds = %1
  ret void
}

declare i64 @__quantum__rt__qubit_allocate()

declare void @__quantum__qis__h__body(i64)

declare void @__quantum__qis__cx__body(i64, i64)

declare i64 @__quantum__rt__result_allocate()

declare i32 @__quantum__qis__m__body(i64, i64)

declare void @__quantum__rt__result_record_output(i64, i8*)

attributes #0 = { "EntryPoint" }
