; Public synthetic parity feedback. Context and observed-read name are supplied by the test.
@label = private constant [5 x i8] c"after"
declare i64 @___qalloc()
declare void @___qfree(i64)
declare void @___reset(i64)
declare void @___rxy(i64, double, double)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___dec_future_refcount(i64)
declare i1 @pecos_runtime_read_bool_v1(ptr, i64, i64)
declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {
entry:
    %q = call i64 @___qalloc()
    %buffer = alloca [3 x i8]
    %data = getelementptr inbounds [3 x i8], ptr %buffer, i64 0, i64 1
    %second = getelementptr inbounds [3 x i8], ptr %buffer, i64 0, i64 2
    br label %loop
loop:
    %instance = phi i64 [0, %entry], [%next, %tail]
    call void @___reset(i64 %q)
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    %m0 = call i64 @___lazy_measure(i64 %q)
    %a = call i1 @pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %m0)
    call void @___dec_future_refcount(i64 %m0)
    call void @___reset(i64 %q)
    %second_flip = icmp eq i64 %instance, 1
    br i1 %second_flip, label %perturb, label %read_second
perturb:
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    br label %read_second
read_second:
    %m1 = call i64 @___lazy_measure(i64 %q)
    %b = call i1 @pecos_runtime_read_bool_v1(ptr @read_name, i64 7, i64 %m1)
    call void @___dec_future_refcount(i64 %m1)
    %a_byte = zext i1 %a to i8
    %b_byte = zext i1 %b to i8
    call void @___reset(i64 %q)
    %parity = xor i1 %a, %b
    br i1 %parity, label %prepare_output, label %decode
prepare_output:
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    br label %decode
decode:
    store i8 %a_byte, ptr %data
    store i8 %b_byte, ptr %second
    %frame = call i64 @pecos_runtime_decode_observed_v1(i64 7, i64 %instance, ptr %data, i64 2, ptr @ctx, i64 CONTEXT_LENGTH)
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
