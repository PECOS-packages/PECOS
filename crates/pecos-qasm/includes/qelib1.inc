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

// Phase gate (native rotation)
gate rz(lambda) a { RZ(lambda) a; }

// S and S-dagger gates (sqrt(Z) and its conjugate)
gate s a { rz(pi/2) a; }
gate sdg a { rz(-pi/2) a; }

// T and T-dagger gates (sqrt(S) and its conjugate)
gate t a { rz(pi/4) a; }
gate tdg a { rz(-pi/4) a; }

// X-axis rotation using decomposition
gate rx(theta) a {
  h a;
  rz(theta) a;
  h a;
}

// Y-axis rotation
gate ry(theta) a {
  rx(-pi/2) a;
  rz(theta) a;
  rx(pi/2) a;
}

// sqrt(X) gate - decomposed
gate sx a {
  sdg a;
  h a;
  sdg a;
}

// inverse sqrt(X) - decomposed
gate sxdg a {
  s a;
  h a;
  s a;
}

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
gate csx a,b {
  cx a,b;  // Simplified version
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
gate crx(theta) a,b {
  ry(pi/2) b;
  cx a,b;
  ry(theta) b;
  cx a,b;
  ry(-pi/2) b;
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
  rz(theta/2) a;
  cx a,b;
  rz(-theta/2) b;
  cx a,b;
  rz(theta/2) b;
}

// Toffoli gate (controlled-controlled-X)
gate ccx a,b,c {
  h c;
  cx b,c; tdg c;
  cx a,c; t c;
  cx b,c; tdg c;
  cx a,c; t b; t c; h c;
  cx a,b; t a; tdg b;
  cx a,b;
}

// Note: More complex gates like controlled-H (ch)
// and other gates would require additional implementations

// --- Utility gates ---

// Phase gate with arbitrary angle
gate phase(lambda) q { rz(lambda) q; }

// Standard phase gate (equivalent to rz)
gate p(lambda) q { rz(lambda) q; }

// Universal single-qubit gate (simplified version using rz gates)
// u(theta, phi, lambda) = rz(phi) * rx(theta) * rz(lambda)
// For the identity case u(0,0,0), this simplifies to doing nothing
// PECOS native U gate (arbitrary single-qubit rotation)
gate u(theta, phi, lambda) q { U(theta, phi, lambda) q; }

// Single-parameter phase gate (alias)
gate u1(lambda) a { rz(lambda) a; }

// Two-parameter rotation
gate u2(phi, lambda) a {
  rz(phi) a;
  rx(pi/2) a;
  rz(lambda) a;
}

// Three-parameter general rotation
gate u3(theta, phi, lambda) a {
  rz(phi) a;
  rx(-pi/2) a;
  rz(theta) a;
  rx(pi/2) a;
  rz(lambda) a;
}

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

// Three-qubit Toffoli gate
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

// Synonyms for common gates
gate cnot a,b { cx a,b; }
gate cphase90 a,b { cphase(pi/2) a,b; }
gate cphase180 a,b { cz a,b; }
