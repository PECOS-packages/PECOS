; A is prepared in |1>. A timed RZZ on B/C leaves A idle.
; An exact leakage outcome of 2 triggers reset; an ordinary outcome of 1 does not.
declare void @setup(i64)
declare i64 @___qalloc()
declare void @___reset(i64)
declare void @___rxy(i64, double, double)
declare void @___rzz(i64, i64, double)
declare i64 @___lazy_measure_leaked(i64)
declare i64 @___read_future_uint(i64)
declare void @panic(i32, ptr)

define i64 @qmain(i64 %shot) #0 {
    call void @setup(i64 %shot)
    %a = call i64 @___qalloc()
    %b = call i64 @___qalloc()
    %c = call i64 @___qalloc()
    call void @___reset(i64 %a)
    call void @___reset(i64 %b)
    call void @___reset(i64 %c)
    call void @___rxy(i64 %a, double 0x400921FB54442D18, double 0.0)
    call void @___rzz(i64 %b, i64 %c, double 0.3)
    %first = call i64 @___lazy_measure_leaked(i64 %a)
    %value = call i64 @___read_future_uint(i64 %first)
    %cached = call i64 @___read_future_uint(i64 %first)
    %same = icmp eq i64 %value, %cached
    br i1 %same, label %check_leakage, label %failure
check_leakage:
    %leaked = icmp eq i64 %value, 2
    br i1 %leaked, label %recover, label %readout
recover:
    call void @___reset(i64 %a)
    br label %readout
readout:
    %second = call i64 @___lazy_measure_leaked(i64 %a)
    %unread = call i64 @___lazy_measure_leaked(i64 %a)
    ret i64 0
failure:
    call void @panic(i32 1058, ptr null)
    ret i64 0
}
attributes #0 = { "EntryPoint" }
