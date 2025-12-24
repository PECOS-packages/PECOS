
flag_x_test = 0;
flag_z_test = 0;

// X check 1, Z check 2, Z check 3
// ===============================

reset a_test[0];
reset a_test[1];
reset a_test[2];

h a_test[0];
h a_test[1];
h a_test[2];

cx a_test[0], q_test[3];  // 5 -> 4
cz a_test[1], q_test[5];  // 6 -> 6
cz a_test[2], q_test[2];  // 7 -> 3

barrier a_test[0], a_test[1];
cz a_test[0], a_test[1];
barrier a_test[0], a_test[1];

cx a_test[0], q_test[0];  // 1 -> 1
cz a_test[1], q_test[4];  // 2 -> 5
cz a_test[2], q_test[3];  // 5 -> 4

cx a_test[0], q_test[1];  // 3 -> 2
cz a_test[1], q_test[2];  // 7 -> 3
cz a_test[2], q_test[6];  // 4 -> 7

barrier a_test[0], a_test[2];
cz a_test[0], a_test[2];
barrier a_test[0], a_test[2];

cx a_test[0], q_test[2];  // 7 -> 3
cz a_test[1], q_test[1];  // 3 -> 2
cz a_test[2], q_test[5];  // 6 -> 6

h a_test[0];
h a_test[1];
h a_test[2];

measure a_test[0] -> flag_x_test[0];
measure a_test[1] -> flag_z_test[1];
measure a_test[2] -> flag_z_test[2];

flag_x_test[0] = flag_x_test[0] ^ last_raw_syn_x_test[0];
flag_z_test[1] = flag_z_test[1] ^ last_raw_syn_z_test[1];
flag_z_test[2] = flag_z_test[2] ^ last_raw_syn_z_test[2];

flags_test = flag_x_test | flag_z_test;


// Z check 1, X check 2, X check 3
// ===============================

if(flags_test == 0) reset a_test[0];
if(flags_test == 0) reset a_test[1];
if(flags_test == 0) reset a_test[2];

if(flags_test == 0) h a_test[0];
if(flags_test == 0) h a_test[1];
if(flags_test == 0) h a_test[2];

if(flags_test == 0) barrier a_test[0], q_test[3];
if(flags_test == 0) cz a_test[0], q_test[3];  // 5 -> 4
if(flags_test == 0) barrier a_test[0], q_test[3];

if(flags_test == 0) barrier a_test[1], q_test[5];
if(flags_test == 0) cx a_test[1], q_test[5];  // 6 -> 6
if(flags_test == 0) barrier a_test[1], q_test[5];

if(flags_test == 0) barrier a_test[2], q_test[2];
if(flags_test == 0) cx a_test[2], q_test[2];  // 7 -> 3
if(flags_test == 0) barrier a_test[2], q_test[2];

if(flags_test == 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];
if(flags_test == 0) cz a_test[1], a_test[0];
if(flags_test == 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];

if(flags_test == 0) barrier a_test[0], q_test[0];
if(flags_test == 0) cz a_test[0], q_test[0];  // 1 -> 1
if(flags_test == 0) barrier a_test[0], q_test[0];

if(flags_test == 0) barrier a_test[1], q_test[4];
if(flags_test == 0) cx a_test[1], q_test[4];  // 2 -> 5
if(flags_test == 0) barrier a_test[1], q_test[4];

if(flags_test == 0) barrier a_test[2], q_test[3];
if(flags_test == 0) cx a_test[2], q_test[3];  // 5 -> 4
if(flags_test == 0) barrier a_test[2], q_test[3];

if(flags_test == 0) barrier a_test[0], q_test[1];
if(flags_test == 0) cz a_test[0], q_test[1];  // 3 -> 2
if(flags_test == 0) barrier a_test[0], q_test[1];

if(flags_test == 0) barrier a_test[1], q_test[2];
if(flags_test == 0) cx a_test[1], q_test[2];  // 7 -> 3
if(flags_test == 0) barrier a_test[1], q_test[2];

if(flags_test == 0) barrier a_test[2], q_test[6];
if(flags_test == 0) cx a_test[2], q_test[6];  // 4 -> 7
if(flags_test == 0) barrier a_test[2], q_test[6];

if(flags_test == 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];
if(flags_test == 0) cz a_test[2], a_test[0];
if(flags_test == 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];

if(flags_test == 0) barrier a_test[0], q_test[2];
if(flags_test == 0) cz a_test[0], q_test[2];  // 7 -> 3
if(flags_test == 0) barrier a_test[0], q_test[2];

if(flags_test == 0) barrier a_test[1], q_test[1];
if(flags_test == 0) cx a_test[1], q_test[1];  // 3 -> 2
if(flags_test == 0) barrier a_test[1], q_test[1];

if(flags_test == 0) barrier a_test[2], q_test[5];
if(flags_test == 0) cx a_test[2], q_test[5];  // 6 -> 6
if(flags_test == 0) barrier a_test[2], q_test[5];

if(flags_test == 0) h a_test[0];
if(flags_test == 0) h a_test[1];
if(flags_test == 0) h a_test[2];

if(flags_test == 0) measure a_test[0] -> flag_z_test[0];
if(flags_test == 0) measure a_test[1] -> flag_x_test[1];
if(flags_test == 0) measure a_test[2] -> flag_x_test[2];

// XOR flags/syndromes
if(flags_test == 0) flag_z_test[0] = flag_z_test[0] ^ last_raw_syn_z_test[0];
if(flags_test == 0) flag_x_test[1] = flag_x_test[1] ^ last_raw_syn_x_test[1];
if(flags_test == 0) flag_x_test[2] = flag_x_test[2] ^ last_raw_syn_x_test[2];

if(flags_test == 0) flags_test = flag_x_test | flag_z_test;


// Run the 6 non-flagged checks (if non-trivial flags)
// ===================================================
// // X check 1, Z check 2, Z check 3

if(flags_test != 0) syn_x_test = 0;
if(flags_test != 0) syn_z_test = 0;

if(flags_test != 0) reset a_test[0];
if(flags_test != 0) reset a_test[1];
if(flags_test != 0) reset a_test[2];

if(flags_test != 0) h a_test[0];
if(flags_test != 0) h a_test[1];
if(flags_test != 0) h a_test[2];

if(flags_test != 0) cx a_test[0], q_test[2];  // X1
if(flags_test != 0) cz a_test[1], q_test[5];  // Z2
if(flags_test != 0) cz a_test[2], q_test[6];  // Z3

if(flags_test != 0) cx a_test[0], q_test[1];  // X1
if(flags_test != 0) cz a_test[1], q_test[2];  // Z2
if(flags_test != 0) cz a_test[2], q_test[5];  // Z3

if(flags_test != 0) cx a_test[0], q_test[3];  // X1
if(flags_test != 0) cz a_test[1], q_test[1];  // Z2
if(flags_test != 0) cz a_test[2], q_test[2];  // Z3

if(flags_test != 0) cx a_test[0], q_test[0];  // X1
if(flags_test != 0) cz a_test[1], q_test[4];  // Z2
if(flags_test != 0) cz a_test[2], q_test[3];  // Z3

if(flags_test != 0) h a_test[0];
if(flags_test != 0) h a_test[1];
if(flags_test != 0) h a_test[2];

if(flags_test != 0) measure a_test[0] -> syn_x_test[0];
if(flags_test != 0) measure a_test[1] -> syn_z_test[1];
if(flags_test != 0) measure a_test[2] -> syn_z_test[2];

// // Z check 1, X check 2, X check 3

if(flags_test != 0) reset a_test[0];
if(flags_test != 0) reset a_test[1];
if(flags_test != 0) reset a_test[2];

if(flags_test != 0) h a_test[0];
if(flags_test != 0) h a_test[1];
if(flags_test != 0) h a_test[2];

if(flags_test != 0) cz a_test[0], q_test[2];  // Z1 0,3
if(flags_test != 0) cx a_test[1], q_test[5];  // X2 6,8
if(flags_test != 0) cx a_test[2], q_test[6];  // X3 7,9

if(flags_test != 0) cz a_test[0], q_test[1];  // Z1 0,2
if(flags_test != 0) cx a_test[1], q_test[2];  // X2 3,8
if(flags_test != 0) cx a_test[2], q_test[5];  // X3 6,9

if(flags_test != 0) cz a_test[0], q_test[3];  // Z1 0,4
if(flags_test != 0) cx a_test[1], q_test[1];  // X2 2,8
if(flags_test != 0) cx a_test[2], q_test[2];  // X3 3,9

if(flags_test != 0) cz a_test[0], q_test[0];  // Z1 0,1
if(flags_test != 0) cx a_test[1], q_test[4];  // X2 5,8
if(flags_test != 0) cx a_test[2], q_test[3];  // X3 4,9

if(flags_test != 0) h a_test[0];
if(flags_test != 0) h a_test[1];
if(flags_test != 0) h a_test[2];

if(flags_test != 0) measure a_test[0] -> syn_z_test[0];
if(flags_test != 0) measure a_test[1] -> syn_x_test[1];
if(flags_test != 0) measure a_test[2] -> syn_x_test[2];


// =========================
// BEGIN Run X decoder
// =========================

if(flags_test != 0) syndromes_test = syn_x_test ^ last_raw_syn_x_test;
if(flags_test == 0) syndromes_test = 0;

// apply corrections
if(syndromes_test == 2) pf_test[1] = pf_test[1] ^ 1;
if(syndromes_test == 4) pf_test[1] = pf_test[1] ^ 1;
if(syndromes_test == 6) pf_test[1] = pf_test[1] ^ 1;

// alter correction based on flags
// ===============================

// 1&2 (1 -> 2)
// ------------
scratch_test = 0;
if(flag_x_test == 1) scratch_test[0] = 1;
if(syndromes_test == 2) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[1] = pf_test[1] ^ 1;

// 1&4 (1 -> 3)
// ------------
scratch_test = 0;
if(flag_x_test == 1) scratch_test[0] = 1;
if(syndromes_test == 4) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[1] = pf_test[1] ^ 1;

// 6&4 (2,3 -> 3)
// ------------
scratch_test = 0;
if(flag_x_test == 6) scratch_test[0] = 1;
if(syndromes_test == 4) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[1] = pf_test[1] ^ 1;

if(flags_test != 0) last_raw_syn_x_test = syn_x_test;

// =========================
// END Run X decoder
// =========================



// ACTIVE ERROR CORRECTION FOR X SYNDROMES

scratch_test = 0;

// only part that differs for X vs Z syns V
if(syndromes_test[0] == 1) scratch_test = scratch_test ^ 1;
if(syndromes_test[1] == 1) scratch_test = scratch_test ^ 12;
if(syndromes_test[2] == 1) scratch_test = scratch_test ^ 48;

// logical operator
if(pf_test[1] == 1) scratch_test = scratch_test ^ 112;

if(scratch_test[0] == 1) z q_test[0];
// not possible for X stabilizers V
// if(scratch_test[1] == 1) z q_test[1];
if(scratch_test[2] == 1) z q_test[2];
if(scratch_test[3] == 1) z q_test[3];
if(scratch_test[4] == 1) z q_test[4];
if(scratch_test[5] == 1) z q_test[5];
if(scratch_test[6] == 1) z q_test[6];

pf_test[1] = 0;
// syndromes_test = 0;
last_raw_syn_x_test = 0;
// syn_x_test = 0;
// flag_x_test = 0;
// flags_test = 0;



// =========================
// BEGIN Run Z decoder
// =========================

if(flags_test != 0) syndromes_test = syn_z_test ^ last_raw_syn_z_test;
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
if(flag_z_test == 1) scratch_test[0] = 1;
if(syndromes_test == 2) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[0] = pf_test[0] ^ 1;

// 1&4 (1 -> 3)
// ------------
scratch_test = 0;
if(flag_z_test == 1) scratch_test[0] = 1;
if(syndromes_test == 4) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[0] = pf_test[0] ^ 1;

// 6&4 (2,3 -> 3)
// ------------
scratch_test = 0;
if(flag_z_test == 6) scratch_test[0] = 1;
if(syndromes_test == 4) scratch_test[1] = 1;

scratch_test[2] = scratch_test[0] & scratch_test[1];
if(scratch_test[2] == 1) pf_test[0] = pf_test[0] ^ 1;

if(flags_test != 0) last_raw_syn_z_test = syn_z_test;

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
last_raw_syn_z_test = 0;
// syn_z_test = 0;
// flag_z_test = 0;
// flags_test = 0;
