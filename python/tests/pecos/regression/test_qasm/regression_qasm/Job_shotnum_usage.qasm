OPENQASM 2.0;
include "hqslib1.inc";
qreg q[1];
creg c[1];
creg shotnum[32];
U1q(0.5*pi,0.5*pi) q[0];
shotnum = JOB_shotnum;
measure q[0] -> c[0];
