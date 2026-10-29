@observed_label = private constant [8 x i8] c"observed"
@after_label = private constant [5 x i8] c"after"
@second_flip = constant i1 false
declare void @__quantum__qis__reset__body(i64)
declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i1 @___read_future_bool(i64)
declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {
entry:
    %data = alloca [2 x i8]
    %second_slot = getelementptr [2 x i8], ptr %data, i64 0, i64 1
    br label %loop
loop:
    %instance = phi i64 [0, %entry], [%next, %tail]
    call void @__quantum__qis__reset__body(i64 0)
    call void @__quantum__qis__x__body(i64 0)
    %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %a = call i1 @___read_future_bool(i64 0)
    call void @print_bool_selene(ptr @observed_label, i64 8, i1 %a)
    call void @__quantum__qis__reset__body(i64 0)
    %perturb_second = load i1, ptr @second_flip
    br i1 %perturb_second, label %perturb, label %read_second
perturb:
    call void @__quantum__qis__x__body(i64 0)
    br label %read_second
read_second:
    %m1 = call i32 @__quantum__qis__m__body(i64 0, i64 1)
    %b = call i1 @___read_future_bool(i64 1)
    call void @print_bool_selene(ptr @observed_label, i64 8, i1 %b)
    %a_byte = zext i1 %a to i8
    %b_byte = zext i1 %b to i8
    store i8 %a_byte, ptr %data
    store i8 %b_byte, ptr %second_slot
    ; A fresh output qubit carries the observed parity L0 = D0 XOR D1.
    call void @__quantum__qis__reset__body(i64 1)
    %parity = xor i1 %a, %b
    br i1 %parity, label %prepare_output, label %decode
prepare_output:
    call void @__quantum__qis__x__body(i64 1)
    br label %decode
decode:
    %frame = call i64 @pecos_runtime_decode(i64 7, i64 %instance, ptr %data, i64 2)
    %correct = icmp ne i64 %frame, 0
    br i1 %correct, label %apply, label %tail
apply:
    call void @__quantum__qis__x__body(i64 1)
    br label %tail
tail:
    %m2 = call i32 @__quantum__qis__m__body(i64 1, i64 2)
    %value = call i1 @___read_future_bool(i64 2)
    call void @print_bool_selene(ptr @after_label, i64 5, i1 %value)
    %next = add i64 %instance, 1
    %more = icmp ult i64 %next, 3
    br i1 %more, label %loop, label %done
done:
    ret i64 0
}
attributes #0 = { "EntryPoint" }
