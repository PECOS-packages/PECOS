; A one-second delay after the last measurement must not idle the next shot.
declare void @setup(i64)
declare i64 @___qalloc()
declare i1 @___measure(i64)
declare void @___rxy(i64, double, double)
declare void @___sleep(ptr, i64, i64)
define i64 @qmain(i64 %shot) #0 {
entry:
    call void @setup(i64 %shot)
    %q = call i64 @___qalloc()
    %qs = alloca i64
    store i64 %q, ptr %qs
    call void @___rxy(i64 %q, double 0x400921FB54442D18, double 0.0)
    %m = call i1 @___measure(i64 %q)
    call void @___sleep(ptr %qs, i64 1, i64 1000000000)
    ret i64 0
}
attributes #0 = { "EntryPoint" }
