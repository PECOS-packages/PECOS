OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[2];

// tick syndrome_round
h q[0];
cx q[0], q[0];
cx q[0], q[1];
h q[0];
h q[1];
cx q[1], q[1];
cx q[1], q[2];
h q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
