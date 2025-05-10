OPENQASM 2.0;
include "qelib1.inc";

// Define registers
qreg q[2];
creg c[2];

// Initialize in superposition
h q[0];
h q[1];

// Oracle - marks the state |11⟩
x q[0];
x q[1];
h q[1];
cx q[0], q[1];
h q[1];
x q[0];
x q[1];

// Diffusion operator (Amplitude amplification)
h q[0];
h q[1];
x q[0];
x q[1];
h q[1];
cx q[0], q[1];
h q[1];
x q[0];
x q[1];
h q[0];
h q[1];

// Measure the result
measure q -> c;
