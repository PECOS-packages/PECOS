; Synthetic two-detector live feedback. Test substitutes CONTEXT_SIZE and @context.
@a = private constant [1 x i8] c"a"
@b = private constant [1 x i8] c"b"
@after = private constant [5 x i8] c"after"
declare i64 @___qalloc()
declare void @___reset(i64)
declare void @___rxy(i64, double, double)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare i1 @pecos_runtime_read_bool_v1(ptr, i64, i64)
declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) {
 %q0 = call i64 @___qalloc()
 %q1 = call i64 @___qalloc()
 call void @___reset(i64 %q0)
 call void @___reset(i64 %q1)
 call void @___rxy(i64 %q0, double 3.141592653589793, double 0.0)
 %f0 = call i64 @___lazy_measure(i64 %q0)
 %f1 = call i64 @___lazy_measure(i64 %q1)
 %a = call i1 @pecos_runtime_read_bool_v1(ptr @a, i64 1, i64 %f0)
 %b = call i1 @pecos_runtime_read_bool_v1(ptr @b, i64 1, i64 %f1)
 %ab = zext i1 %a to i8
 %bb = zext i1 %b to i8
 %data = alloca [2 x i8]
 %d1 = getelementptr [2 x i8], ptr %data, i64 0, i64 1
 store i8 %bb, ptr %data
 store i8 %ab, ptr %d1
 %frame = call i64 @pecos_runtime_decode_observed_v1(i64 7, i64 0, ptr %data, i64 2, ptr @context, i64 CONTEXT_SIZE)
 %correct = icmp ne i64 %frame, 0
 br i1 %correct, label %apply, label %tail
apply:
 call void @___rxy(i64 %q0, double 3.141592653589793, double 0.0)
 br label %tail
tail:
 %f2 = call i64 @___lazy_measure(i64 %q0)
 %v = call i1 @___read_future_bool(i64 %f2)
 call void @print_bool_selene(ptr @after, i64 5, i1 %v)
 ret i64 0
}
