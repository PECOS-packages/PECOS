; Same portable Helios QIS program exercised by Selene and PECOS.
; Ordinary + leakage futures, feedback, cached reads, and an unread shutdown tail.
declare void @setup(i64)
declare i64 @___qalloc()
declare void @___reset(i64)
declare void @___rxy(i64, double, double)
declare i64 @___lazy_measure(i64)
declare i64 @___lazy_measure_leaked(i64)
declare i1 @___read_future_bool(i64)
declare i64 @___read_future_uint(i64)
declare void @panic(i32, ptr)
define i64 @qmain(i64 %shot) #0 {
    call void @setup(i64 %shot)
    %a = call i64 @___qalloc()
    %b = call i64 @___qalloc()
    call void @___reset(i64 %a)
    call void @___reset(i64 %b)
    call void @___rxy(i64 %a, double 0x400921FB54442D18, double 0.0)
    %m = call i64 @___lazy_measure(i64 %a)
    %l = call i64 @___lazy_measure_leaked(i64 %b)
    %r = call i1 @___read_future_bool(i64 %m)
    %again = call i1 @___read_future_bool(i64 %m)
    %both = and i1 %r, %again
    br i1 %both, label %feedback, label %failure
feedback:
    call void @___rxy(i64 %b, double 0x400921FB54442D18, double 0.0)
    %lv = call i64 @___read_future_uint(i64 %l)
    %zero = icmp eq i64 %lv, 0
    br i1 %zero, label %verify, label %failure
verify:
    %final = call i64 @___lazy_measure(i64 %b)
    %fv = call i1 @___read_future_bool(i64 %final)
    br i1 %fv, label %tail, label %failure
tail:
    call void @___rxy(i64 %a, double 0x400921FB54442D18, double 0.0)
    %unread = call i64 @___lazy_measure(i64 %a)
    ret i64 0
failure:
    call void @panic(i32 1058, ptr null)
    ret i64 0
}
attributes #0 = { "EntryPoint" }
