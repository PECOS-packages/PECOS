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
gate r1xy(theta,phi) a { R1XY(theta,phi) a; }

// --- Two-qubit gates ---

// Controlled-NOT (CNOT)
gate cx c,t { CX c,t; }

// ZZ interaction
gate szz a,b { SZZ a,b; }

// --- Measurement ---
// Note: Measurement is a built-in operation in QASM
// measure q -> c;
