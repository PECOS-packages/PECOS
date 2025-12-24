; Simple quantum program compatible with Selene/Helios
; Uses qmain as entry point (required by libhelios.a)
; Uses only the low-level gates that Helios provides

; Declare Helios-style quantum operations
declare i64 @___qalloc()
declare void @___qfree(i64)
declare void @___rxy(i64, double, double)  ; rxy(qubit, theta, phi)
declare void @___rz(i64, double)           ; rz(qubit, theta)
declare void @___rzz(i64, i64, double)     ; rzz(q1, q2, theta)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___reset(i64)

; Entry point for Helios programs
define i64 @qmain(i64 %0) {
entry:
  ; Allocate two qubits
  %q0 = call i64 @___qalloc()
  %q1 = call i64 @___qalloc()

  ; Apply some gates
  ; H gate is rxy(q, pi/2, 0)
  call void @___rxy(i64 %q0, double 1.5707963267948966, double 0.0)

  ; Apply RZ rotation
  call void @___rz(i64 %q1, double 0.785398)

  ; Apply RZZ interaction
  call void @___rzz(i64 %q0, i64 %q1, double 0.5)

  ; Measure both qubits
  %m0 = call i64 @___lazy_measure(i64 %q0)
  %m1 = call i64 @___lazy_measure(i64 %q1)

  ; Read measurement results
  %r0 = call i1 @___read_future_bool(i64 %m0)
  %r1 = call i1 @___read_future_bool(i64 %m1)

  ; Free qubits
  call void @___qfree(i64 %q0)
  call void @___qfree(i64 %q1)

  ret i64 0
}
