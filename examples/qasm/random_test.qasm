OPENQASM 2.0;
include "qelib1.inc";

// Define registers
qreg q[2];
creg c[2];

// Apply Hadamard gate to first qubit to create a superposition
h q[0];

// Apply CNOT to create an entangled state
cx q[0], q[1];

// Apply another Hadamard to first qubit
h q[0];

// Measure both qubits to get random outcomes
measure q[0] -> c[0];
measure q[1] -> c[1];
