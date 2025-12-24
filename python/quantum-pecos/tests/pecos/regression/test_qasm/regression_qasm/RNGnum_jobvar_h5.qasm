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
if (JOB_shotnum == 2) h q;
if (JOB_shotnum == 4) h q;
cond1 = JOB_shotnum + JOB_shotnum + 10;
if (cond1 == 4) h q;

cond1 = RNGnum();
if (cond1[0] != 0) h q;
if (cond1[0] != 0) hgates1[1] = 1;

cond1 = RNGnum();
if (cond1[1] != 0) h q;
if (cond1[1] != 0) hgates1[2] = 1;

cond1 = RNGnum();
if (cond1[3] != 0) h q;
if (cond1[3] != 0) hgates1[3] = 1;

measure q->c;
