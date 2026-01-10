
// =========================
// BEGIN Run Z decoder
// =========================

if(flags_test != 0) syndromes_test = syn_test ^ raw_syn_test;
if(flags_test == 0) syndromes_test = 0;

// apply corrections
if(syndromes_test == 2) pf_test[0] = pf_test[0] ^ 1;
if(syndromes_test == 4) pf_test[0] = pf_test[0] ^ 1;
if(syndromes_test == 6) pf_test[0] = pf_test[0] ^ 1;

// alter correction based on flags
// ===============================

// 1&2 (1 -> 2)
// ------------
scratch_test = 0;
if(flag_test == 1) scratch_test[0] = 1;
if(syndromes_test == 2) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[0] = pf_test[0] ^ 1;

// 1&4 (1 -> 3)
// ------------
scratch_test = 0;
if(flag_test == 1) scratch_test[0] = 1;
if(syndromes_test == 4) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[0] = pf_test[0] ^ 1;

// 6&4 (2,3 -> 3)
// ------------
scratch_test = 0;
if(flag_test == 6) scratch_test[0] = 1;
if(syndromes_test == 4) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[0] = pf_test[0] ^ 1;

if(flags_test != 0) raw_syn_test = syn_test;

// =========================
// END Run Z decoder
// =========================



// ACTIVE ERROR CORRECTION FOR Z SYNDROMES

scratch_test = 0;

// only part that differs for X vs Z syns V
if(syndromes_test[0] == 1) scratch_test = scratch_test ^ 14;
if(syndromes_test[1] == 1) scratch_test = scratch_test ^ 12;
if(syndromes_test[2] == 1) scratch_test = scratch_test ^ 48;

// logical operator
if(pf_test[0] == 1) scratch_test = scratch_test ^ 112;

// not possible for X stabilizers V
// if(scratch_test[0] == 1) z q_test[0];
if(scratch_test[1] == 1) x q_test[1];
if(scratch_test[2] == 1) x q_test[2];
if(scratch_test[3] == 1) x q_test[3];
if(scratch_test[4] == 1) x q_test[4];
if(scratch_test[5] == 1) x q_test[5];
if(scratch_test[6] == 1) x q_test[6];

pf_test[0] = 0;
// syndromes_test = 0;
raw_syn_test = 0;
// syn_test = 0;
// flag_test = 0;
// flags_test = 0;
