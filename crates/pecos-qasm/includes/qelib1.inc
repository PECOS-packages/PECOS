// PECOS Extended Standard Gate Library
// Version 0.2.0
// Based on the standard qelib1.inc but with implementations using PECOS native gates

// --- Basic single-qubit gates ---

// Hadamard gate (native)
gate h a { H a; }

// Pauli gates (native)
gate x a { X a; }
gate y a { Y a; }
gate z a { Z a; }

// Identity gate - implemented as no-op
gate id a { rz(0) a; }

// Symmetric Z rotation (native), not the phase-fixed p gate below
gate rz(lambda) a { RZ(lambda) a; }

// S and S-dagger gates (sqrt(Z) and its conjugate)
gate s a { S a; }
gate sdg a { SDG a; }

// T and T-dagger gates (sqrt(S) and its conjugate)
gate t a { T a; }
gate tdg a { TDG a; }

// RXY1Q(theta, 0) = RX(theta)
gate rx(theta) a { RXY1Q(theta, 0) a; }

// RXY1Q(theta, pi/2) = RY(theta)
gate ry(theta) a { RXY1Q(theta, pi/2) a; }

// sqrt(X) gates (native)
gate sx a { SX a; }
gate sxdg a { SXDG a; }

// --- Two-qubit gates ---

// Controlled-NOT (CNOT) - native
gate cx c,t { CX c,t; }

// Controlled-Z
gate cz a,b {
  h b;
  cx a,b;
  h b;
}

// Controlled-Y
gate cy a,b {
  sdg b;
  cx a,b;
  s b;
}

// Controlled-SX
// H · S · H = SX on the target
gate csx a,b {
  h b;
  cu1(pi/2) a,b;
  h b;
}

// SWAP gate
gate swap a,b {
  cx a,b;
  cx b,a;
  cx a,b;
}

// ZZ rotation (native)
gate rzz(theta) a,b { RZZ(theta) a,b; }

// Strong ZZ interaction (native)
gate szz a,b { SZZ a,b; }

// Controlled RZ
gate crz(theta) a,b {
  rz(theta/2) b;
  cx a,b;
  rz(-theta/2) b;
  cx a,b;
}

// Controlled RX
// H · RZ(theta) · H = RX(theta) on the target
gate crx(theta) a,b {
  h b;
  crz(theta) a,b;
  h b;
}

// Controlled RY
gate cry(theta) a,b {
  ry(theta/2) b;
  cx a,b;
  ry(-theta/2) b;
  cx a,b;
}

// Controlled phase
gate cphase(theta) a,b {
  p(theta/2) a;
  cx a,b;
  p(-theta/2) b;
  cx a,b;
  p(theta/2) b;
}

// Toffoli gate (controlled-controlled-X)
gate ccx a,b,c {
  h c;
  cx b,c;
  tdg c;
  cx a,c;
  t c;
  cx b,c;
  tdg c;
  cx a,c;
  t b;
  t c;
  h c;
  cx a,b;
  t a;
  tdg b;
  cx a,b;
}

// Note: More complex gates like controlled-H (ch)
// and other gates would require additional implementations

// --- Utility gates ---

// Phase-gate family: diag(1, exp(i*lambda)). These differ from the symmetric
// rotation by RZ(lambda) = exp(-i*lambda/2) * p(lambda).
gate phase(lambda) q { U(0, 0, lambda) q; }

// Standard phase gate (equivalent to u1, but not phase-exactly to rz)
gate p(lambda) q { U(0, 0, lambda) q; }

// PECOS native U gate (arbitrary single-qubit rotation)
gate u(theta, phi, lambda) q { U(theta, phi, lambda) q; }

// Single-parameter phase gate (alias)
gate u1(lambda) a { U(0, 0, lambda) a; }

// Two-parameter rotation
gate u2(phi, lambda) a { U(pi/2, phi, lambda) a; }

// Three-parameter general rotation
gate u3(theta, phi, lambda) a { U(theta, phi, lambda) a; }

// Controlled gates
gate cu1(lambda) a,b {
  u1(lambda/2) a;
  cx a,b;
  u1(-lambda/2) b;
  cx a,b;
  u1(lambda/2) b;
}

// Two-qubit XX rotation
gate rxx(theta) a,b {
  h a;
  h b;
  cx a,b;
  rz(theta) b;
  cx a,b;
  h a;
  h b;
}


// Synonyms for common gates
gate cnot a,b { cx a,b; }
gate cphase90 a,b { cphase(pi/2) a,b; }
gate cphase180 a,b { cz a,b; }
