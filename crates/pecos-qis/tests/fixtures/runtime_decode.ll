@label = private constant [5 x i8] c"after"
declare void @__quantum__qis__reset__body(i64)
declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i1 @___read_future_bool(i64)
declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {
entry:
    %data = alloca i8
    br label %loop
loop:
    %instance = phi i64 [0, %entry], [%next, %tail]
    call void @__quantum__qis__reset__body(i64 0)
    call void @__quantum__qis__x__body(i64 0)
    %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %observed = call i1 @___read_future_bool(i64 0)
    %bit = zext i1 %observed to i8
    store i8 %bit, ptr %data
    %frame = call i64 @pecos_runtime_decode(i64 7, i64 %instance, ptr %data, i64 1)
    %correct = icmp ne i64 %frame, 0
    br i1 %correct, label %apply, label %tail
apply:
    call void @__quantum__qis__x__body(i64 0)
    br label %tail
tail:
    %after = call i32 @__quantum__qis__m__body(i64 0, i64 1)
    %value = call i1 @___read_future_bool(i64 1)
    call void @print_bool_selene(ptr @label, i64 5, i1 %value)
    %next = add i64 %instance, 1
    %more = icmp ult i64 %next, 3
    br i1 %more, label %loop, label %done
done:
    ret i64 0
}
attributes #0 = { "EntryPoint" }
