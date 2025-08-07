OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];

creg cond0[32];
creg hgates0[4];

creg cond1[32];
creg hgates1[4];

creg rand_seed[64];
creg rand_bound[32];
creg rand_index[32];
creg cond2[32];
creg hgates2[4];


/////////////////////////////////////////////
// Test RNG functions using cbit arguments
/////////////////////////////////////////////

RNGseed(rand_seed[0]);
RNGbound(rand_bound[0]);
RNGindex(rand_index[0]);

cond0 = RNGnum();

h q;
hgates0[0] = 1;

if (cond0[0] != 0) h q;
if (cond0[0] != 0) hgates0[1] = 1;

if (cond0[1] != 0) h q;
if (cond0[1] != 0) hgates0[2] = 1;

if (cond0[3] != 0) h q;
if (cond0[3] != 0) hgates0[3] = 1;

///////////////////////
//////////////////////
// Test RNG functions using integer arguments
/////////////////////////////////////////////

RNGseed(1000);
RNGbound(16);
RNGindex(4);

cond1 = RNGnum();

h q;
hgates1[0] = 1;

if (cond1[0] != 0) h q;
if (cond1[0] != 0) hgates1[1] = 1;

if (cond1[1] != 0) h q;
if (cond1[1] != 0) hgates1[2] = 1;

if (cond1[3] != 0) h q;
if (cond1[3] != 0) hgates1[3] = 1;

/////////////////////////////////////////////
// Test RNG functions using creg arguments
/////////////////////////////////////////////

rand_seed = 2000;
rand_bound = 32;
rand_index = 8;

RNGseed(rand_seed);
RNGbound(rand_bound);
RNGindex(rand_index);

cond2 = RNGnum();

h q;
hgates2[0] = 1;

if (cond2[0] != 0) h q;
if (cond2[0] != 0) hgates2[1] = 1;

if (cond2[1] != 0) h q;
if (cond2[1] != 0) hgates2[2] = 1;

if (cond2[3] != 0) h q;
if (cond2[3] != 0) hgates2[3] = 1;


measure q->c;
