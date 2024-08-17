
// =========================
// BEGIN Run X decoder
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
// END Run X decoder
// =========================
