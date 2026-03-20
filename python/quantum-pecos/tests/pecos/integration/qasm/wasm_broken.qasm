OPENQASM 2.0;
include "hqslib1.inc";
qreg q[1];
creg m[1];
creg e[1];
creg c[1];
creg d[1];
creg a[4];
creg b[32];

x q;
measure q -> m;

b = 1;
d = 0;
c = 1;
e = meas_decoder(m, a, b, c, d);

global_reset();
