OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg a[2];
creg b[2];
creg c[2];

// Apply hadamards to all qubits
h q[0];
h q[1];
h q[2];

// Measure and store in different registers
measure q[0] -> a[0];
measure q[1] -> b[0];
measure q[2] -> c[0];
measure q[0] -> a[1];
measure q[1] -> b[1];
measure q[2] -> c[1];
