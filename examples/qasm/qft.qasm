OPENQASM 2.0;
include "qelib1.inc";

// Define registers
qreg q[3];
creg c[3];

// Initialize to a simple state
x q[0];

// Apply 3-qubit QFT
// First qubit
h q[0];
cu1(pi/2) q[0],q[1];
cu1(pi/4) q[0],q[2];

// Second qubit
h q[1];
cu1(pi/2) q[1],q[2];

// Third qubit
h q[2];

// Swap qubits to match standard QFT output ordering
swap q[0],q[2];

// Measure all qubits
measure q -> c;
