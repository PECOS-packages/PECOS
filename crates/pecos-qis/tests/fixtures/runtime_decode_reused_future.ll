; Synthetic transport fixture: the host deliberately recycles the same future.
@name = private constant [4 x i8] c"read"
@context = private constant [2 x i8] c"{}"
declare i64 @___lazy_measure(i64)
declare void @___dec_future_refcount(i64)
declare i1 @pecos_runtime_read_bool_v1(ptr, i64, i64)
declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)
define i64 @qmain(i64 %shot) {
    %first = call i64 @___lazy_measure(i64 0)
    %a = call i1 @pecos_runtime_read_bool_v1(ptr @name, i64 4, i64 %first)
    call void @___dec_future_refcount(i64 %first)
    %second = call i64 @___lazy_measure(i64 0)
    %b = call i1 @pecos_runtime_read_bool_v1(ptr @name, i64 4, i64 %second)
    %parity = xor i1 %a, %b
    %bit = zext i1 %parity to i8
    %data = alloca i8
    store i8 %bit, ptr %data
    %frame = call i64 @pecos_runtime_decode_observed_v1(i64 7, i64 0, ptr %data, i64 1, ptr @context, i64 2)
    ret i64 %frame
}
