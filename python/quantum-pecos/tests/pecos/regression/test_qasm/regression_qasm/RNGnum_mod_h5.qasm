OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];

creg cond1[32];
creg hgates1[4];

/////////////////////////////////////////////
// Test RNG functions with jobvar index
/////////////////////////////////////////////

RNGseed(1000);
RNGbound(16);
RNGindex(JOB_shotnum);

h q;
hgates1[0] = 1;

cond1 = RNGnum();
cond1 = cond1 % 7;
if (cond1[0] != 0) h q;
if (cond1[0] != 0) hgates1[1] = 1;

cond1 = RNGnum();
cond1 = cond1 % 7;
if (cond1[1] != 0) h q;
if (cond1[1] != 0) hgates1[2] = 1;

cond1 = RNGnum();
cond1 = cond1 % 7;
if (cond1[3] != 0) h q;
if (cond1[3] != 0) hgates1[3] = 1;

measure q->c;
