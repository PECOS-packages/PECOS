OPENQASM 2.0;
include "hqslib1.inc";
// Generated using: PECOS version 0.6.0.dev5
qreg q[2];
creg m[2];
h q[0];
cx q[0], q[1];
measure q -> m;
