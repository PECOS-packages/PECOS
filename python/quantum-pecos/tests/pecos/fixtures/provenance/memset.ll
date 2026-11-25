@label = private constant [6 x i8] c"filled"

declare i64 @___qalloc()
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___qfree(i64)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @print_bool_selene(ptr, i64, i1)

define i64 @qmain(i64 %shot) {
entry:
  %q = call i64 @___qalloc()
  %future = call i64 @___lazy_measure(i64 %q)
  %m = call i1 @___read_future_bool(i64 %future)
  %byte = zext i1 %m to i8
  %buffer = alloca i8
  call void @llvm.memset.p0.i64(ptr %buffer, i8 %byte, i64 1, i1 false)
  %filled = load i1, ptr %buffer
  call void @print_bool_selene(ptr @label, i64 6, i1 %filled)
  call void @___qfree(i64 %q)
  ret i64 0
}
