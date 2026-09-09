// PECOS Minimal Native Gate Library
// Version 1.0.0
// Contains only gates with direct PECOS native implementations

// --- Basic single-qubit gates ---

// Hadamard gate
gate h a { H a; }

// Pauli gates
gate x a { X a; }
gate y a { Y a; }
gate z a { Z a; }

// Rotation gates
gate rz(lambda) a { RZ(lambda) a; }
gate rxy1q(theta,phi) a { RXY1Q(theta,phi) a; }
gate r1xy(theta,phi) a { RXY1Q(theta,phi) a; }

// --- Two-qubit gates ---

// Controlled-NOT (CNOT)
gate cx c,t { CX c,t; }

// Pauli-root interactions
gate sxx a,b { SXX a,b; }
gate sxxdg a,b { SXXDG a,b; }
gate syy a,b { SYY a,b; }
gate syydg a,b { SYYDG a,b; }
gate szz a,b { SZZ a,b; }
gate szzdg a,b { SZZDG a,b; }

// --- Measurement ---
// Note: Measurement is a built-in operation in QASM
// measure q -> c;
