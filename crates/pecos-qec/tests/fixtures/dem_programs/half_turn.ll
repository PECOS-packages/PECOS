; Two allocated physical qubits and one RZZ(pi), followed by lazy measurement.
define i64 @qmain(i64 %context) {
  call void @setup(i64 %context)
  %a = call i64 @___qalloc()
  %b = call i64 @___qalloc()
  call void @___reset(i64 %a)
  call void @___reset(i64 %b)
  call void @___rzz(i64 %a, i64 %b, double 0x400921FB54442D18)
  %ma = call i64 @___lazy_measure(i64 %a)
  %mb = call i64 @___lazy_measure(i64 %b)
  call void @___qfree(i64 %a)
  call void @___qfree(i64 %b)
  %va = call i1 @___read_future_bool(i64 %ma)
  %vb = call i1 @___read_future_bool(i64 %mb)
  call void @___dec_future_refcount(i64 %ma)
  call void @___dec_future_refcount(i64 %mb)
  %exit = call i64 @teardown()
  ret i64 %exit
}
declare void @setup(i64)
declare i64 @teardown()
declare i64 @___qalloc()
declare void @___reset(i64)
declare void @___rzz(i64, i64, double)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___dec_future_refcount(i64)
declare void @___qfree(i64)
!name = !{!0}
!0 = !{!"mainlib"}
