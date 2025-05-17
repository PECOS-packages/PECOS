// PECOS Quantum Gate Library for hqslib1.inc compatibility

// === Basic Single-Qubit Gates ===

// General single-qubit unitary (decomposed into PECOS native gates)
gate U(theta, phi, lam) q {
  RZ(lam) q;
  R1XY(theta, pi/2) q;
  RZ(phi) q;
}

// Lowercase alias for compatibility
gate u(theta, phi, lam) q {
  U(theta, phi, lam) q;
}

// Pauli-X gate
gate x() a {
  X a;
}

// Pauli-Y gate
gate y() a {
  Y a;
}

// Pauli-Z gate
gate z() a {
  Z a;
}

// Hadamard gate
gate h() a {
  H a;
}

// === Phase Gates ===

// S gate (sqrt(Z))
gate s() a {
  RZ(pi/2) a;
}

// S-dagger gate (conjugate of sqrt(Z))
gate sdg() a {
  RZ(-pi/2) a;
}

// T gate (sqrt(S))
gate t() a {
  RZ(pi/4) a;
}

// T-dagger gate (conjugate of sqrt(S))
gate tdg() a {
  RZ(-pi/4) a;
}

// === Rotation Gates ===

// X-axis rotation
gate rx(theta) a {
  R1XY(theta, 0) a;
}

// Y-axis rotation
gate ry(theta) a {
  R1XY(theta, pi/2) a;
}

// Z-axis rotation (alias for native RZ)
gate rz(phi) a {
  RZ(phi) a;
}

// === Two-Qubit Gates ===

// CNOT gate (alias for native CX)
gate cx() c,t {
  CX c,t;
}

// Controlled-Y gate
gate cy() a,b {
  sdg b;
  cx a,b;
  s b;
}

// Controlled-Z gate
gate cz() a,b {
  H b;
  CX a,b;
  H b;
}

// === HQS-specific gate aliases for compatibility ===

// U1q is directly implemented using native R1XY
gate U1q(theta, phi) q {
  R1XY(theta, phi) q;
}

// Rz is directly implemented using native RZ (case difference)
gate Rz(lambda) q {
  RZ(lambda) q;
}

// ZZ is equivalent to SZZ in PECOS (no angle parameter)
gate ZZ q0,q1 {
  SZZ q0,q1;
}

// === Three-qubit gates ===

// Toffoli gate (CCNOT)
gate ccx a,b,c {
  H c;
  CX b,c;
  tdg c;
  CX a,c;
  t c;
  CX b,c;
  tdg c;
  CX a,c;
  t b;
  t c;
  H c;
  CX a,b;
  t a;
  tdg b;
  CX a,b;
}

// === Other common gates ===

// Phase gate
gate p(lambda) a {
  RZ(lambda) a;
}

// Controlled phase gate
gate cp(lambda) a,b {
  p(lambda/2) a;
  cx a,b;
  p(-lambda/2) b;
  cx a,b;
  p(lambda/2) b;
}

// swap gate
gate swap() a,b {
  cx a,b;
  cx b,a;
  cx a,b;
}

// sqrt(X) gates
gate sx() a {
  R1XY(pi/2, 0) a;
}

gate sxdg a {
  R1XY(-pi/2, 0) a;
}

// Identity gate (no-op)
gate id a {
  RZ(0) a;
}

// === Utility gates for backwards compatibility ===

// Only define aliases that don't conflict with native gates
gate CNOT a,b { CX a,b; }
gate S a { s a; }
gate Sdg a { sdg a; }
gate T a { t a; }
gate Tdg a { tdg a; }
gate RX(theta) a { rx(theta) a; }
gate RY(theta) a { ry(theta) a; }
