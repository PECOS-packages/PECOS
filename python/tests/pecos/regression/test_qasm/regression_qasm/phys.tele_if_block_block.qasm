OPENQASM 2.0;
include "hqslib1.inc";
qreg q[2];
creg m[2];
h q[0];
cx q[0], q[1];
measure q -> m;
if(m == 0) h q[0];
if(m == 0) h q[1];