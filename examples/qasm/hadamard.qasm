OPENQASM 2.0;
include "qelib1.inc";

// Define registers
qreg q[3];
creg c[1];

// Apply Hadamard gates to create a superposition
h q[0];
h q[1];
h q[2];

// Measure qubit 0 to get a random outcome
measure q[0] -> c[0];
