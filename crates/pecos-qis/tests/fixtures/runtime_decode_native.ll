; Synthetic live feedback using native qubit and future handles throughout.
@label = private constant [5 x i8] c"after"
@before = private constant [6 x i8] c"before"
declare i64 @___qalloc()
declare void @___qfree(i64)
declare void @___reset(i64)
declare void @___rxy(i64, double, double)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___dec_future_refcount(i64)
declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {
entry:
    %q = call i64 @___qalloc()
    %data = alloca i8
    br label %loop
loop:
    %instance = phi i64 [0, %entry], [%next, %tail]
    call void @___reset(i64 %q)
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    %m = call i64 @___lazy_measure(i64 %q)
    %observed = call i1 @___read_future_bool(i64 %m)
    call void @print_bool_selene(ptr @before, i64 6, i1 %observed)
    call void @___dec_future_refcount(i64 %m)
    %bit = zext i1 %observed to i8
    store i8 %bit, ptr %data
    %frame = call i64 @pecos_runtime_decode(i64 7, i64 %instance, ptr %data, i64 1)
    %correct = icmp ne i64 %frame, 0
    br i1 %correct, label %apply, label %tail
apply:
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    br label %tail
tail:
    %after = call i64 @___lazy_measure(i64 %q)
    %value = call i1 @___read_future_bool(i64 %after)
    call void @print_bool_selene(ptr @label, i64 5, i1 %value)
    call void @___dec_future_refcount(i64 %after)
    %next = add i64 %instance, 1
    %more = icmp ult i64 %next, 3
    br i1 %more, label %loop, label %done
done:
    call void @___qfree(i64 %q)
    ret i64 0
}
attributes #0 = { "EntryPoint" }
