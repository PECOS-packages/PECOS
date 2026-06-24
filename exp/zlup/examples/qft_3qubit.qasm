OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

x q[0];
x q[2];
h q[0];
rzz(pi_2) q[1], q[0];
rzz(pi_4) q[2], q[0];
h q[1];
rzz(pi_2) q[2], q[1];
h q[2];
swap q[0], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
