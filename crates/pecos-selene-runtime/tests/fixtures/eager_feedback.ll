; Hold one lazy future across repeated eager reads. With a result cap of two,
; only the temporary eager future may be retired; the lazy handle stays readable.
declare void @setup(i64)
declare i64 @___qalloc()
declare void @___reset(i64)
declare i64 @___lazy_measure(i64)
declare i1 @___measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___rxy(i64, double, double)
declare void @panic(i32, ptr)
define i64 @qmain(i64 %shot) #0 {
entry:
    call void @setup(i64 %shot)
    %q = call i64 @___qalloc()
    call void @___reset(i64 %q)
    %held = call i64 @___lazy_measure(i64 %q)
    %initial = call i1 @___read_future_bool(i64 %held)
    br i1 %initial, label %failure, label %loop
loop:
    %i = phi i64 [0, %entry], [%next, %check]
    call void @___rxy(i64 %q, double 0x400921FB54442D18, double 0.0)
    %m = call i1 @___measure(i64 %q)
    %even = and i64 %i, 1
    %expected = icmp eq i64 %even, 0
    %correct = icmp eq i1 %m, %expected
    br i1 %correct, label %check, label %failure
check:
    %next = add i64 %i, 1
    %done = icmp eq i64 %next, 8
    br i1 %done, label %verify, label %loop
verify:
    %again = call i1 @___read_future_bool(i64 %held)
    %cached = call i1 @___read_future_bool(i64 %held)
    %bad = or i1 %again, %cached
    br i1 %bad, label %failure, label %end
end:
    ret i64 0
failure:
    call void @panic(i32 1058, ptr null)
    ret i64 0
}
attributes #0 = { "EntryPoint" }
