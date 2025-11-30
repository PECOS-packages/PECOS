OPENQASM 2.0;
include "hqslib1.inc";
creg m_reject[2];
creg m_t[1];
creg m_out[2];
qreg sin_d[7];
qreg sin_a[3];
creg sin_c[32];
creg sin_syn_meas[32];
creg sin_last_raw_syn_x[32];
creg sin_last_raw_syn_z[32];
creg sin_scratch[32];
creg sin_flag_x[3];
creg sin_flags_z[3];
creg sin_flags[3];
creg sin_raw_meas[7];
creg sin_syn_x[3];
creg sin_syn_z[3];
creg sin_syndromes[3];
creg sin_verify_prep[32];
qreg saux_d[7];
qreg saux_a[3];
creg saux_c[32];
creg saux_syn_meas[32];
creg saux_last_raw_syn_x[32];
creg saux_last_raw_syn_z[32];
creg saux_scratch[32];
creg saux_flag_x[3];
creg saux_flags_z[3];
creg saux_flags[3];
creg saux_raw_meas[7];
creg saux_syn_x[3];
creg saux_syn_z[3];
creg saux_syndromes[3];
creg saux_verify_prep[32];

barrier sin_d[0], sin_d[1], sin_d[2], sin_d[3], sin_d[4], sin_d[5], sin_d[6], sin_a[0];

reset sin_d[0];
reset sin_d[1];
reset sin_d[2];
reset sin_d[3];
reset sin_d[4];
reset sin_d[5];
reset sin_d[6];
reset sin_a[0];
barrier sin_d, sin_a[0];
h sin_d[0];
h sin_d[4];
h sin_d[6];

cx sin_d[4], sin_d[5];
cx sin_d[0], sin_d[1];
cx sin_d[6], sin_d[3];
cx sin_d[4], sin_d[2];
cx sin_d[6], sin_d[5];
cx sin_d[0], sin_d[3];
cx sin_d[4], sin_d[1];
cx sin_d[3], sin_d[2];

barrier sin_a[0], sin_d[1], sin_d[3], sin_d[5];
// verification step
cx sin_d[5], sin_a[0];
cx sin_d[1], sin_a[0];
cx sin_d[3], sin_a[0];
measure sin_a[0] -> sin_verify_prep[0];


if(sin_verify_prep[0] == 1) barrier sin_d[0], sin_d[1], sin_d[2], sin_d[3], sin_d[4], sin_d[5], sin_d[6], sin_a[0];

if(sin_verify_prep[0] == 1) reset sin_d[0];
if(sin_verify_prep[0] == 1) reset sin_d[1];
if(sin_verify_prep[0] == 1) reset sin_d[2];
if(sin_verify_prep[0] == 1) reset sin_d[3];
if(sin_verify_prep[0] == 1) reset sin_d[4];
if(sin_verify_prep[0] == 1) reset sin_d[5];
if(sin_verify_prep[0] == 1) reset sin_d[6];
if(sin_verify_prep[0] == 1) reset sin_a[0];
if(sin_verify_prep[0] == 1) barrier sin_d, sin_a[0];
if(sin_verify_prep[0] == 1) h sin_d[0];
if(sin_verify_prep[0] == 1) h sin_d[4];
if(sin_verify_prep[0] == 1) h sin_d[6];

if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[5];
if(sin_verify_prep[0] == 1) cx sin_d[0], sin_d[1];
if(sin_verify_prep[0] == 1) cx sin_d[6], sin_d[3];
if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[2];
if(sin_verify_prep[0] == 1) cx sin_d[6], sin_d[5];
if(sin_verify_prep[0] == 1) cx sin_d[0], sin_d[3];
if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[1];
if(sin_verify_prep[0] == 1) cx sin_d[3], sin_d[2];

if(sin_verify_prep[0] == 1) barrier sin_a[0], sin_d[1], sin_d[3], sin_d[5];
// verification step
if(sin_verify_prep[0] == 1) cx sin_d[5], sin_a[0];
if(sin_verify_prep[0] == 1) cx sin_d[1], sin_a[0];
if(sin_verify_prep[0] == 1) cx sin_d[3], sin_a[0];
if(sin_verify_prep[0] == 1) measure sin_a[0] -> sin_verify_prep[0];


if(sin_verify_prep[0] == 1) barrier sin_d[0], sin_d[1], sin_d[2], sin_d[3], sin_d[4], sin_d[5], sin_d[6], sin_a[0];

if(sin_verify_prep[0] == 1) reset sin_d[0];
if(sin_verify_prep[0] == 1) reset sin_d[1];
if(sin_verify_prep[0] == 1) reset sin_d[2];
if(sin_verify_prep[0] == 1) reset sin_d[3];
if(sin_verify_prep[0] == 1) reset sin_d[4];
if(sin_verify_prep[0] == 1) reset sin_d[5];
if(sin_verify_prep[0] == 1) reset sin_d[6];
if(sin_verify_prep[0] == 1) reset sin_a[0];
if(sin_verify_prep[0] == 1) barrier sin_d, sin_a[0];
if(sin_verify_prep[0] == 1) h sin_d[0];
if(sin_verify_prep[0] == 1) h sin_d[4];
if(sin_verify_prep[0] == 1) h sin_d[6];

if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[5];
if(sin_verify_prep[0] == 1) cx sin_d[0], sin_d[1];
if(sin_verify_prep[0] == 1) cx sin_d[6], sin_d[3];
if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[2];
if(sin_verify_prep[0] == 1) cx sin_d[6], sin_d[5];
if(sin_verify_prep[0] == 1) cx sin_d[0], sin_d[3];
if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[1];
if(sin_verify_prep[0] == 1) cx sin_d[3], sin_d[2];

if(sin_verify_prep[0] == 1) barrier sin_a[0], sin_d[1], sin_d[3], sin_d[5];
// verification step
if(sin_verify_prep[0] == 1) cx sin_d[5], sin_a[0];
if(sin_verify_prep[0] == 1) cx sin_d[1], sin_a[0];
if(sin_verify_prep[0] == 1) cx sin_d[3], sin_a[0];
if(sin_verify_prep[0] == 1) measure sin_a[0] -> sin_verify_prep[0];

saux_scratch = 0;
reset saux_d[6];
ry(0.7853981633974483) saux_d[6];

// Encoding circuit
// ---------------
reset saux_d[0];
reset saux_d[1];
reset saux_d[2];
reset saux_d[3];
reset saux_d[4];
reset saux_d[5];

// q[6] is the input qubit

cx saux_d[6], saux_d[5];

h saux_d[1];
cx saux_d[1], saux_d[0];

h saux_d[2];
cx saux_d[2], saux_d[4];

// ---------------
h saux_d[3];
cx saux_d[3], saux_d[5];
cx saux_d[2], saux_d[0];
cx saux_d[6], saux_d[4];

// ---------------
cx saux_d[2], saux_d[6];
cx saux_d[3], saux_d[4];
cx saux_d[1], saux_d[5];

// ---------------
cx saux_d[1], saux_d[6];
cx saux_d[3], saux_d[0];

// Measure check HHHHHHH
reset saux_a[0];
reset saux_a[1];
h saux_a[0];
barrier saux_a[0], saux_d[0];
ch saux_a[0], saux_d[0];
barrier saux_a[0], saux_a[1];
cx saux_a[0], saux_a[1];
barrier saux_a[0], saux_a[1];
ch saux_a[0], saux_d[1];
barrier saux_a[0], saux_d[1];
ch saux_a[0], saux_d[2];
barrier saux_a[0], saux_d[2];
ch saux_a[0], saux_d[3];
barrier saux_a[0], saux_d[3];
ch saux_a[0], saux_d[4];
barrier saux_a[0], saux_d[4];
ch saux_a[0], saux_d[5];
barrier saux_a[0], saux_d[5];
barrier saux_a[0], saux_a[1];
cx saux_a[0], saux_a[1];
barrier saux_a[0], saux_a[1];
ch saux_a[0], saux_d[6];
barrier saux_a[0], saux_d[6];
h saux_a[0];
measure saux_a[0] -> saux_scratch[0];
measure saux_a[1] -> saux_scratch[1];

saux_flag_x = 0;
saux_flags_z = 0;

// X check 1, Z check 2, Z check 3
// ===============================

reset saux_a[0];
reset saux_a[1];
reset saux_a[2];

h saux_a[0];
h saux_a[1];
h saux_a[2];

cx saux_a[0], saux_d[3];  // 5 -> 4
cz saux_a[1], saux_d[5];  // 6 -> 6
cz saux_a[2], saux_d[2];  // 7 -> 3

barrier saux_a[0], saux_a[1];
cz saux_a[0], saux_a[1];
barrier saux_a[0], saux_a[1];

cx saux_a[0], saux_d[0];  // 1 -> 1
cz saux_a[1], saux_d[4];  // 2 -> 5
cz saux_a[2], saux_d[3];  // 5 -> 4

cx saux_a[0], saux_d[1];  // 3 -> 2
cz saux_a[1], saux_d[2];  // 7 -> 3
cz saux_a[2], saux_d[6];  // 4 -> 7

barrier saux_a[0], saux_a[2];
cz saux_a[0], saux_a[2];
barrier saux_a[0], saux_a[2];

cx saux_a[0], saux_d[2];  // 7 -> 3
cz saux_a[1], saux_d[1];  // 3 -> 2
cz saux_a[2], saux_d[5];  // 6 -> 6

h saux_a[0];
h saux_a[1];
h saux_a[2];

measure saux_a[0] -> saux_flag_x[0];
measure saux_a[1] -> saux_flags_z[1];
measure saux_a[2] -> saux_flags_z[2];

saux_flag_x[0] = saux_flag_x[0] ^ saux_last_raw_syn_x[0];
saux_flags_z[1] = saux_flags_z[1] ^ saux_last_raw_syn_z[1];
saux_flags_z[2] = saux_flags_z[2] ^ saux_last_raw_syn_z[2];

saux_flags = saux_flag_x | saux_flags_z;


// Z check 1, X check 2, X check 3
// ===============================

if(saux_flags == 0) reset saux_a[0];
if(saux_flags == 0) reset saux_a[1];
if(saux_flags == 0) reset saux_a[2];

if(saux_flags == 0) h saux_a[0];
if(saux_flags == 0) h saux_a[1];
if(saux_flags == 0) h saux_a[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[3];
if(saux_flags == 0) cz saux_a[0], saux_d[3];  // 5 -> 4
if(saux_flags == 0) barrier saux_a[0], saux_d[3];

if(saux_flags == 0) barrier saux_a[1], saux_d[5];
if(saux_flags == 0) cx saux_a[1], saux_d[5];  // 6 -> 6
if(saux_flags == 0) barrier saux_a[1], saux_d[5];

if(saux_flags == 0) barrier saux_a[2], saux_d[2];
if(saux_flags == 0) cx saux_a[2], saux_d[2];  // 7 -> 3
if(saux_flags == 0) barrier saux_a[2], saux_d[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];
if(saux_flags == 0) cz saux_a[1], saux_a[0];
if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[0];
if(saux_flags == 0) cz saux_a[0], saux_d[0];  // 1 -> 1
if(saux_flags == 0) barrier saux_a[0], saux_d[0];

if(saux_flags == 0) barrier saux_a[1], saux_d[4];
if(saux_flags == 0) cx saux_a[1], saux_d[4];  // 2 -> 5
if(saux_flags == 0) barrier saux_a[1], saux_d[4];

if(saux_flags == 0) barrier saux_a[2], saux_d[3];
if(saux_flags == 0) cx saux_a[2], saux_d[3];  // 5 -> 4
if(saux_flags == 0) barrier saux_a[2], saux_d[3];

if(saux_flags == 0) barrier saux_a[0], saux_d[1];
if(saux_flags == 0) cz saux_a[0], saux_d[1];  // 3 -> 2
if(saux_flags == 0) barrier saux_a[0], saux_d[1];

if(saux_flags == 0) barrier saux_a[1], saux_d[2];
if(saux_flags == 0) cx saux_a[1], saux_d[2];  // 7 -> 3
if(saux_flags == 0) barrier saux_a[1], saux_d[2];

if(saux_flags == 0) barrier saux_a[2], saux_d[6];
if(saux_flags == 0) cx saux_a[2], saux_d[6];  // 4 -> 7
if(saux_flags == 0) barrier saux_a[2], saux_d[6];

if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];
if(saux_flags == 0) cz saux_a[2], saux_a[0];
if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[2];
if(saux_flags == 0) cz saux_a[0], saux_d[2];  // 7 -> 3
if(saux_flags == 0) barrier saux_a[0], saux_d[2];

if(saux_flags == 0) barrier saux_a[1], saux_d[1];
if(saux_flags == 0) cx saux_a[1], saux_d[1];  // 3 -> 2
if(saux_flags == 0) barrier saux_a[1], saux_d[1];

if(saux_flags == 0) barrier saux_a[2], saux_d[5];
if(saux_flags == 0) cx saux_a[2], saux_d[5];  // 6 -> 6
if(saux_flags == 0) barrier saux_a[2], saux_d[5];

if(saux_flags == 0) h saux_a[0];
if(saux_flags == 0) h saux_a[1];
if(saux_flags == 0) h saux_a[2];

if(saux_flags == 0) measure saux_a[0] -> saux_flags_z[0];
if(saux_flags == 0) measure saux_a[1] -> saux_flag_x[1];
if(saux_flags == 0) measure saux_a[2] -> saux_flag_x[2];

// XOR flags/syndromes
if(saux_flags == 0) saux_flags_z[0] = saux_flags_z[0] ^ saux_last_raw_syn_z[0];
if(saux_flags == 0) saux_flag_x[1] = saux_flag_x[1] ^ saux_last_raw_syn_x[1];
if(saux_flags == 0) saux_flag_x[2] = saux_flag_x[2] ^ saux_last_raw_syn_x[2];

if(saux_flags == 0) saux_flags = saux_flag_x | saux_flags_z;

saux_scratch[2] = (((saux_scratch[0] | saux_scratch[1]) | saux_flags[0]) | saux_flags[1]) | saux_flags[2];

rx(-pi/2) saux_d[0];
rx(-pi/2) saux_d[1];
rx(-pi/2) saux_d[2];
rx(-pi/2) saux_d[3];
rx(-pi/2) saux_d[4];
rx(-pi/2) saux_d[5];
rx(-pi/2) saux_d[6];
rz(-pi/2) saux_d[0];
rz(-pi/2) saux_d[1];
rz(-pi/2) saux_d[2];
rz(-pi/2) saux_d[3];
rz(-pi/2) saux_d[4];
rz(-pi/2) saux_d[5];
rz(-pi/2) saux_d[6];
m_reject[0] = saux_scratch[2];
// Transversal Logical CX
barrier sin_d, saux_d;
cx sin_d[0], saux_d[0];
cx sin_d[1], saux_d[1];
cx sin_d[2], saux_d[2];
cx sin_d[3], saux_d[3];
cx sin_d[4], saux_d[4];
cx sin_d[5], saux_d[5];
cx sin_d[6], saux_d[6];
barrier sin_d, saux_d;
// Destructive logical Z measurement

barrier saux_d;

measure saux_d[0] -> saux_raw_meas[0];
measure saux_d[1] -> saux_raw_meas[1];
measure saux_d[2] -> saux_raw_meas[2];
measure saux_d[3] -> saux_raw_meas[3];
measure saux_d[4] -> saux_raw_meas[4];
measure saux_d[5] -> saux_raw_meas[5];
measure saux_d[6] -> saux_raw_meas[6];

// determine raw logical output
// ============================
saux_c[1] = (saux_raw_meas[4] ^ saux_raw_meas[5]) ^ saux_raw_meas[6];



// =================== //
// PROCESS MEASUREMENT //
// =================== //

// Determine correction to get logical output
// ==========================================
saux_syn_meas[0] = ((saux_raw_meas[0] ^ saux_raw_meas[1]) ^ saux_raw_meas[2]) ^ saux_raw_meas[3];
saux_syn_meas[1] = ((saux_raw_meas[1] ^ saux_raw_meas[2]) ^ saux_raw_meas[4]) ^ saux_raw_meas[5];
saux_syn_meas[2] = ((saux_raw_meas[2] ^ saux_raw_meas[3]) ^ saux_raw_meas[5]) ^ saux_raw_meas[6];

// XOR syndromes
saux_syn_meas = saux_syn_meas ^ saux_last_raw_syn_z;

// Correct logical output based on measured out syndromes
saux_c[2] = saux_c[1];
if(saux_syn_meas == 2) saux_c[2] = saux_c[2] ^ 1;
if(saux_syn_meas == 4) saux_c[2] = saux_c[2] ^ 1;
if(saux_syn_meas == 6) saux_c[2] = saux_c[2] ^ 1;

// Apply Pauli frame update (flip the logical output)
// Update for logical Z out
saux_c[2] = saux_c[2] ^ saux_c[3];
sin_c[5] = saux_c[2];
// Logical SZ
if(sin_c[5] == 1) rz(-pi/2) sin_d[0];
if(sin_c[5] == 1) rz(-pi/2) sin_d[1];
if(sin_c[5] == 1) rz(-pi/2) sin_d[2];
if(sin_c[5] == 1) rz(-pi/2) sin_d[3];
if(sin_c[5] == 1) rz(-pi/2) sin_d[4];
if(sin_c[5] == 1) rz(-pi/2) sin_d[5];
if(sin_c[5] == 1) rz(-pi/2) sin_d[6];
// Destructive logical X measurement
// Logical SYdg
ry(-pi/2) sin_d[0];
ry(-pi/2) sin_d[1];
ry(-pi/2) sin_d[2];
ry(-pi/2) sin_d[3];
ry(-pi/2) sin_d[4];
ry(-pi/2) sin_d[5];
ry(-pi/2) sin_d[6];

barrier sin_d;

measure sin_d[0] -> sin_raw_meas[0];
measure sin_d[1] -> sin_raw_meas[1];
measure sin_d[2] -> sin_raw_meas[2];
measure sin_d[3] -> sin_raw_meas[3];
measure sin_d[4] -> sin_raw_meas[4];
measure sin_d[5] -> sin_raw_meas[5];
measure sin_d[6] -> sin_raw_meas[6];

// determine raw logical output
// ============================
sin_c[1] = (sin_raw_meas[4] ^ sin_raw_meas[5]) ^ sin_raw_meas[6];



// =================== //
// PROCESS MEASUREMENT //
// =================== //

// Determine correction to get logical output
// ==========================================
sin_syn_meas[0] = ((sin_raw_meas[0] ^ sin_raw_meas[1]) ^ sin_raw_meas[2]) ^ sin_raw_meas[3];
sin_syn_meas[1] = ((sin_raw_meas[1] ^ sin_raw_meas[2]) ^ sin_raw_meas[4]) ^ sin_raw_meas[5];
sin_syn_meas[2] = ((sin_raw_meas[2] ^ sin_raw_meas[3]) ^ sin_raw_meas[5]) ^ sin_raw_meas[6];

// XOR syndromes
sin_syn_meas = sin_syn_meas ^ sin_last_raw_syn_x;

// Correct logical output based on measured out syndromes
sin_c[2] = sin_c[1];
if(sin_syn_meas == 2) sin_c[2] = sin_c[2] ^ 1;
if(sin_syn_meas == 4) sin_c[2] = sin_c[2] ^ 1;
if(sin_syn_meas == 6) sin_c[2] = sin_c[2] ^ 1;

// Apply Pauli frame update (flip the logical output)
// Update for logical X out
sin_c[2] = sin_c[2] ^ sin_c[4];
m_out[0] = sin_c[2];
saux_scratch = 0;
reset saux_d[6];
ry(0.7853981633974483) saux_d[6];

// Encoding circuit
// ---------------
reset saux_d[0];
reset saux_d[1];
reset saux_d[2];
reset saux_d[3];
reset saux_d[4];
reset saux_d[5];

// q[6] is the input qubit

cx saux_d[6], saux_d[5];

h saux_d[1];
cx saux_d[1], saux_d[0];

h saux_d[2];
cx saux_d[2], saux_d[4];

// ---------------
h saux_d[3];
cx saux_d[3], saux_d[5];
cx saux_d[2], saux_d[0];
cx saux_d[6], saux_d[4];

// ---------------
cx saux_d[2], saux_d[6];
cx saux_d[3], saux_d[4];
cx saux_d[1], saux_d[5];

// ---------------
cx saux_d[1], saux_d[6];
cx saux_d[3], saux_d[0];

// Measure check HHHHHHH
reset saux_a[0];
reset saux_a[1];
h saux_a[0];
barrier saux_a[0], saux_d[0];
ch saux_a[0], saux_d[0];
barrier saux_a[0], saux_a[1];
cx saux_a[0], saux_a[1];
barrier saux_a[0], saux_a[1];
ch saux_a[0], saux_d[1];
barrier saux_a[0], saux_d[1];
ch saux_a[0], saux_d[2];
barrier saux_a[0], saux_d[2];
ch saux_a[0], saux_d[3];
barrier saux_a[0], saux_d[3];
ch saux_a[0], saux_d[4];
barrier saux_a[0], saux_d[4];
ch saux_a[0], saux_d[5];
barrier saux_a[0], saux_d[5];
barrier saux_a[0], saux_a[1];
cx saux_a[0], saux_a[1];
barrier saux_a[0], saux_a[1];
ch saux_a[0], saux_d[6];
barrier saux_a[0], saux_d[6];
h saux_a[0];
measure saux_a[0] -> saux_scratch[0];
measure saux_a[1] -> saux_scratch[1];

saux_flag_x = 0;
saux_flags_z = 0;

// X check 1, Z check 2, Z check 3
// ===============================

reset saux_a[0];
reset saux_a[1];
reset saux_a[2];

h saux_a[0];
h saux_a[1];
h saux_a[2];

cx saux_a[0], saux_d[3];  // 5 -> 4
cz saux_a[1], saux_d[5];  // 6 -> 6
cz saux_a[2], saux_d[2];  // 7 -> 3

barrier saux_a[0], saux_a[1];
cz saux_a[0], saux_a[1];
barrier saux_a[0], saux_a[1];

cx saux_a[0], saux_d[0];  // 1 -> 1
cz saux_a[1], saux_d[4];  // 2 -> 5
cz saux_a[2], saux_d[3];  // 5 -> 4

cx saux_a[0], saux_d[1];  // 3 -> 2
cz saux_a[1], saux_d[2];  // 7 -> 3
cz saux_a[2], saux_d[6];  // 4 -> 7

barrier saux_a[0], saux_a[2];
cz saux_a[0], saux_a[2];
barrier saux_a[0], saux_a[2];

cx saux_a[0], saux_d[2];  // 7 -> 3
cz saux_a[1], saux_d[1];  // 3 -> 2
cz saux_a[2], saux_d[5];  // 6 -> 6

h saux_a[0];
h saux_a[1];
h saux_a[2];

measure saux_a[0] -> saux_flag_x[0];
measure saux_a[1] -> saux_flags_z[1];
measure saux_a[2] -> saux_flags_z[2];

saux_flag_x[0] = saux_flag_x[0] ^ saux_last_raw_syn_x[0];
saux_flags_z[1] = saux_flags_z[1] ^ saux_last_raw_syn_z[1];
saux_flags_z[2] = saux_flags_z[2] ^ saux_last_raw_syn_z[2];

saux_flags = saux_flag_x | saux_flags_z;


// Z check 1, X check 2, X check 3
// ===============================

if(saux_flags == 0) reset saux_a[0];
if(saux_flags == 0) reset saux_a[1];
if(saux_flags == 0) reset saux_a[2];

if(saux_flags == 0) h saux_a[0];
if(saux_flags == 0) h saux_a[1];
if(saux_flags == 0) h saux_a[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[3];
if(saux_flags == 0) cz saux_a[0], saux_d[3];  // 5 -> 4
if(saux_flags == 0) barrier saux_a[0], saux_d[3];

if(saux_flags == 0) barrier saux_a[1], saux_d[5];
if(saux_flags == 0) cx saux_a[1], saux_d[5];  // 6 -> 6
if(saux_flags == 0) barrier saux_a[1], saux_d[5];

if(saux_flags == 0) barrier saux_a[2], saux_d[2];
if(saux_flags == 0) cx saux_a[2], saux_d[2];  // 7 -> 3
if(saux_flags == 0) barrier saux_a[2], saux_d[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];
if(saux_flags == 0) cz saux_a[1], saux_a[0];
if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[0];
if(saux_flags == 0) cz saux_a[0], saux_d[0];  // 1 -> 1
if(saux_flags == 0) barrier saux_a[0], saux_d[0];

if(saux_flags == 0) barrier saux_a[1], saux_d[4];
if(saux_flags == 0) cx saux_a[1], saux_d[4];  // 2 -> 5
if(saux_flags == 0) barrier saux_a[1], saux_d[4];

if(saux_flags == 0) barrier saux_a[2], saux_d[3];
if(saux_flags == 0) cx saux_a[2], saux_d[3];  // 5 -> 4
if(saux_flags == 0) barrier saux_a[2], saux_d[3];

if(saux_flags == 0) barrier saux_a[0], saux_d[1];
if(saux_flags == 0) cz saux_a[0], saux_d[1];  // 3 -> 2
if(saux_flags == 0) barrier saux_a[0], saux_d[1];

if(saux_flags == 0) barrier saux_a[1], saux_d[2];
if(saux_flags == 0) cx saux_a[1], saux_d[2];  // 7 -> 3
if(saux_flags == 0) barrier saux_a[1], saux_d[2];

if(saux_flags == 0) barrier saux_a[2], saux_d[6];
if(saux_flags == 0) cx saux_a[2], saux_d[6];  // 4 -> 7
if(saux_flags == 0) barrier saux_a[2], saux_d[6];

if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];
if(saux_flags == 0) cz saux_a[2], saux_a[0];
if(saux_flags == 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];

if(saux_flags == 0) barrier saux_a[0], saux_d[2];
if(saux_flags == 0) cz saux_a[0], saux_d[2];  // 7 -> 3
if(saux_flags == 0) barrier saux_a[0], saux_d[2];

if(saux_flags == 0) barrier saux_a[1], saux_d[1];
if(saux_flags == 0) cx saux_a[1], saux_d[1];  // 3 -> 2
if(saux_flags == 0) barrier saux_a[1], saux_d[1];

if(saux_flags == 0) barrier saux_a[2], saux_d[5];
if(saux_flags == 0) cx saux_a[2], saux_d[5];  // 6 -> 6
if(saux_flags == 0) barrier saux_a[2], saux_d[5];

if(saux_flags == 0) h saux_a[0];
if(saux_flags == 0) h saux_a[1];
if(saux_flags == 0) h saux_a[2];

if(saux_flags == 0) measure saux_a[0] -> saux_flags_z[0];
if(saux_flags == 0) measure saux_a[1] -> saux_flag_x[1];
if(saux_flags == 0) measure saux_a[2] -> saux_flag_x[2];

// XOR flags/syndromes
if(saux_flags == 0) saux_flags_z[0] = saux_flags_z[0] ^ saux_last_raw_syn_z[0];
if(saux_flags == 0) saux_flag_x[1] = saux_flag_x[1] ^ saux_last_raw_syn_x[1];
if(saux_flags == 0) saux_flag_x[2] = saux_flag_x[2] ^ saux_last_raw_syn_x[2];

if(saux_flags == 0) saux_flags = saux_flag_x | saux_flags_z;

saux_scratch[2] = (((saux_scratch[0] | saux_scratch[1]) | saux_flags[0]) | saux_flags[1]) | saux_flags[2];
if(saux_scratch[2] != 0) reset saux_d[6];
if(saux_scratch[2] != 0) ry(0.7853981633974483) saux_d[6];

// Encoding circuit
// ---------------
if(saux_scratch[2] != 0) reset saux_d[0];
if(saux_scratch[2] != 0) reset saux_d[1];
if(saux_scratch[2] != 0) reset saux_d[2];
if(saux_scratch[2] != 0) reset saux_d[3];
if(saux_scratch[2] != 0) reset saux_d[4];
if(saux_scratch[2] != 0) reset saux_d[5];

// q[6] is the input qubit

if(saux_scratch[2] != 0) cx saux_d[6], saux_d[5];

if(saux_scratch[2] != 0) h saux_d[1];
if(saux_scratch[2] != 0) cx saux_d[1], saux_d[0];

if(saux_scratch[2] != 0) h saux_d[2];
if(saux_scratch[2] != 0) cx saux_d[2], saux_d[4];

// ---------------
if(saux_scratch[2] != 0) h saux_d[3];
if(saux_scratch[2] != 0) cx saux_d[3], saux_d[5];
if(saux_scratch[2] != 0) cx saux_d[2], saux_d[0];
if(saux_scratch[2] != 0) cx saux_d[6], saux_d[4];

// ---------------
if(saux_scratch[2] != 0) cx saux_d[2], saux_d[6];
if(saux_scratch[2] != 0) cx saux_d[3], saux_d[4];
if(saux_scratch[2] != 0) cx saux_d[1], saux_d[5];

// ---------------
if(saux_scratch[2] != 0) cx saux_d[1], saux_d[6];
if(saux_scratch[2] != 0) cx saux_d[3], saux_d[0];

// Measure check HHHHHHH
if(saux_scratch[2] != 0) reset saux_a[0];
if(saux_scratch[2] != 0) reset saux_a[1];
if(saux_scratch[2] != 0) h saux_a[0];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[0];
if(saux_scratch[2] != 0) ch saux_a[0], saux_d[0];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) cx saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) ch saux_a[0], saux_d[1];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[1];
if(saux_scratch[2] != 0) ch saux_a[0], saux_d[2];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[2];
if(saux_scratch[2] != 0) ch saux_a[0], saux_d[3];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[3];
if(saux_scratch[2] != 0) ch saux_a[0], saux_d[4];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[4];
if(saux_scratch[2] != 0) ch saux_a[0], saux_d[5];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[5];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) cx saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) ch saux_a[0], saux_d[6];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[6];
if(saux_scratch[2] != 0) h saux_a[0];
if(saux_scratch[2] != 0) measure saux_a[0] -> saux_scratch[0];
if(saux_scratch[2] != 0) measure saux_a[1] -> saux_scratch[1];

if(saux_scratch[2] != 0) saux_flag_x = 0;
if(saux_scratch[2] != 0) saux_flags_z = 0;

// X check 1, Z check 2, Z check 3
// ===============================

if(saux_scratch[2] != 0) reset saux_a[0];
if(saux_scratch[2] != 0) reset saux_a[1];
if(saux_scratch[2] != 0) reset saux_a[2];

if(saux_scratch[2] != 0) h saux_a[0];
if(saux_scratch[2] != 0) h saux_a[1];
if(saux_scratch[2] != 0) h saux_a[2];

if(saux_scratch[2] != 0) cx saux_a[0], saux_d[3];  // 5 -> 4
if(saux_scratch[2] != 0) cz saux_a[1], saux_d[5];  // 6 -> 6
if(saux_scratch[2] != 0) cz saux_a[2], saux_d[2];  // 7 -> 3

if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) cz saux_a[0], saux_a[1];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[1];

if(saux_scratch[2] != 0) cx saux_a[0], saux_d[0];  // 1 -> 1
if(saux_scratch[2] != 0) cz saux_a[1], saux_d[4];  // 2 -> 5
if(saux_scratch[2] != 0) cz saux_a[2], saux_d[3];  // 5 -> 4

if(saux_scratch[2] != 0) cx saux_a[0], saux_d[1];  // 3 -> 2
if(saux_scratch[2] != 0) cz saux_a[1], saux_d[2];  // 7 -> 3
if(saux_scratch[2] != 0) cz saux_a[2], saux_d[6];  // 4 -> 7

if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[2];
if(saux_scratch[2] != 0) cz saux_a[0], saux_a[2];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_a[2];

if(saux_scratch[2] != 0) cx saux_a[0], saux_d[2];  // 7 -> 3
if(saux_scratch[2] != 0) cz saux_a[1], saux_d[1];  // 3 -> 2
if(saux_scratch[2] != 0) cz saux_a[2], saux_d[5];  // 6 -> 6

if(saux_scratch[2] != 0) h saux_a[0];
if(saux_scratch[2] != 0) h saux_a[1];
if(saux_scratch[2] != 0) h saux_a[2];

if(saux_scratch[2] != 0) measure saux_a[0] -> saux_flag_x[0];
if(saux_scratch[2] != 0) measure saux_a[1] -> saux_flags_z[1];
if(saux_scratch[2] != 0) measure saux_a[2] -> saux_flags_z[2];

if(saux_scratch[2] != 0) saux_flag_x[0] = saux_flag_x[0] ^ saux_last_raw_syn_x[0];
if(saux_scratch[2] != 0) saux_flags_z[1] = saux_flags_z[1] ^ saux_last_raw_syn_z[1];
if(saux_scratch[2] != 0) saux_flags_z[2] = saux_flags_z[2] ^ saux_last_raw_syn_z[2];

if(saux_scratch[2] != 0) saux_flags = saux_flag_x | saux_flags_z;


// Z check 1, X check 2, X check 3
// ===============================

if(saux_scratch[2] != 0) reset saux_a[0];
if(saux_scratch[2] != 0) reset saux_a[1];
if(saux_scratch[2] != 0) reset saux_a[2];

if(saux_scratch[2] != 0) h saux_a[0];
if(saux_scratch[2] != 0) h saux_a[1];
if(saux_scratch[2] != 0) h saux_a[2];

if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[3];
if(saux_scratch[2] != 0) cz saux_a[0], saux_d[3];  // 5 -> 4
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[3];

if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[5];
if(saux_scratch[2] != 0) cx saux_a[1], saux_d[5];  // 6 -> 6
if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[5];

if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[2];
if(saux_scratch[2] != 0) cx saux_a[2], saux_d[2];  // 7 -> 3
if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[2];

if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];
if(saux_scratch[2] != 0) cz saux_a[1], saux_a[0];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];

if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[0];
if(saux_scratch[2] != 0) cz saux_a[0], saux_d[0];  // 1 -> 1
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[0];

if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[4];
if(saux_scratch[2] != 0) cx saux_a[1], saux_d[4];  // 2 -> 5
if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[4];

if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[3];
if(saux_scratch[2] != 0) cx saux_a[2], saux_d[3];  // 5 -> 4
if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[3];

if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[1];
if(saux_scratch[2] != 0) cz saux_a[0], saux_d[1];  // 3 -> 2
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[1];

if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[2];
if(saux_scratch[2] != 0) cx saux_a[1], saux_d[2];  // 7 -> 3
if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[2];

if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[6];
if(saux_scratch[2] != 0) cx saux_a[2], saux_d[6];  // 4 -> 7
if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[6];

if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];
if(saux_scratch[2] != 0) cz saux_a[2], saux_a[0];
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[0], saux_d[1], saux_d[2], saux_d[3], saux_d[4], saux_d[5], saux_d[6], saux_a[1], saux_a[2];

if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[2];
if(saux_scratch[2] != 0) cz saux_a[0], saux_d[2];  // 7 -> 3
if(saux_scratch[2] != 0) barrier saux_a[0], saux_d[2];

if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[1];
if(saux_scratch[2] != 0) cx saux_a[1], saux_d[1];  // 3 -> 2
if(saux_scratch[2] != 0) barrier saux_a[1], saux_d[1];

if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[5];
if(saux_scratch[2] != 0) cx saux_a[2], saux_d[5];  // 6 -> 6
if(saux_scratch[2] != 0) barrier saux_a[2], saux_d[5];

if(saux_scratch[2] != 0) h saux_a[0];
if(saux_scratch[2] != 0) h saux_a[1];
if(saux_scratch[2] != 0) h saux_a[2];

if(saux_scratch[2] != 0) measure saux_a[0] -> saux_flags_z[0];
if(saux_scratch[2] != 0) measure saux_a[1] -> saux_flag_x[1];
if(saux_scratch[2] != 0) measure saux_a[2] -> saux_flag_x[2];

// XOR flags/syndromes
if(saux_scratch[2] != 0) saux_flags_z[0] = saux_flags_z[0] ^ saux_last_raw_syn_z[0];
if(saux_scratch[2] != 0) saux_flag_x[1] = saux_flag_x[1] ^ saux_last_raw_syn_x[1];
if(saux_scratch[2] != 0) saux_flag_x[2] = saux_flag_x[2] ^ saux_last_raw_syn_x[2];

if(saux_scratch[2] != 0) saux_flags = saux_flag_x | saux_flags_z;

if(saux_scratch[2] != 0) saux_scratch[2] = (((saux_scratch[0] | saux_scratch[1]) | saux_flags[0]) | saux_flags[1]) | saux_flags[2];
rx(-pi/2) saux_d[0];
rx(-pi/2) saux_d[1];
rx(-pi/2) saux_d[2];
rx(-pi/2) saux_d[3];
rx(-pi/2) saux_d[4];
rx(-pi/2) saux_d[5];
rx(-pi/2) saux_d[6];
rz(-pi/2) saux_d[0];
rz(-pi/2) saux_d[1];
rz(-pi/2) saux_d[2];
rz(-pi/2) saux_d[3];
rz(-pi/2) saux_d[4];
rz(-pi/2) saux_d[5];
rz(-pi/2) saux_d[6];
m_reject[1] = saux_scratch[2];

barrier sin_d[0], sin_d[1], sin_d[2], sin_d[3], sin_d[4], sin_d[5], sin_d[6], sin_a[0];

reset sin_d[0];
reset sin_d[1];
reset sin_d[2];
reset sin_d[3];
reset sin_d[4];
reset sin_d[5];
reset sin_d[6];
reset sin_a[0];
barrier sin_d, sin_a[0];
h sin_d[0];
h sin_d[4];
h sin_d[6];

cx sin_d[4], sin_d[5];
cx sin_d[0], sin_d[1];
cx sin_d[6], sin_d[3];
cx sin_d[4], sin_d[2];
cx sin_d[6], sin_d[5];
cx sin_d[0], sin_d[3];
cx sin_d[4], sin_d[1];
cx sin_d[3], sin_d[2];

barrier sin_a[0], sin_d[1], sin_d[3], sin_d[5];
// verification step
cx sin_d[5], sin_a[0];
cx sin_d[1], sin_a[0];
cx sin_d[3], sin_a[0];
measure sin_a[0] -> sin_verify_prep[0];


if(sin_verify_prep[0] == 1) barrier sin_d[0], sin_d[1], sin_d[2], sin_d[3], sin_d[4], sin_d[5], sin_d[6], sin_a[0];

if(sin_verify_prep[0] == 1) reset sin_d[0];
if(sin_verify_prep[0] == 1) reset sin_d[1];
if(sin_verify_prep[0] == 1) reset sin_d[2];
if(sin_verify_prep[0] == 1) reset sin_d[3];
if(sin_verify_prep[0] == 1) reset sin_d[4];
if(sin_verify_prep[0] == 1) reset sin_d[5];
if(sin_verify_prep[0] == 1) reset sin_d[6];
if(sin_verify_prep[0] == 1) reset sin_a[0];
if(sin_verify_prep[0] == 1) barrier sin_d, sin_a[0];
if(sin_verify_prep[0] == 1) h sin_d[0];
if(sin_verify_prep[0] == 1) h sin_d[4];
if(sin_verify_prep[0] == 1) h sin_d[6];

if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[5];
if(sin_verify_prep[0] == 1) cx sin_d[0], sin_d[1];
if(sin_verify_prep[0] == 1) cx sin_d[6], sin_d[3];
if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[2];
if(sin_verify_prep[0] == 1) cx sin_d[6], sin_d[5];
if(sin_verify_prep[0] == 1) cx sin_d[0], sin_d[3];
if(sin_verify_prep[0] == 1) cx sin_d[4], sin_d[1];
if(sin_verify_prep[0] == 1) cx sin_d[3], sin_d[2];

if(sin_verify_prep[0] == 1) barrier sin_a[0], sin_d[1], sin_d[3], sin_d[5];
// verification step
if(sin_verify_prep[0] == 1) cx sin_d[5], sin_a[0];
if(sin_verify_prep[0] == 1) cx sin_d[1], sin_a[0];
if(sin_verify_prep[0] == 1) cx sin_d[3], sin_a[0];
if(sin_verify_prep[0] == 1) measure sin_a[0] -> sin_verify_prep[0];

// Transversal Logical CX
barrier sin_d, saux_d;
cx sin_d[0], saux_d[0];
cx sin_d[1], saux_d[1];
cx sin_d[2], saux_d[2];
cx sin_d[3], saux_d[3];
cx sin_d[4], saux_d[4];
cx sin_d[5], saux_d[5];
cx sin_d[6], saux_d[6];
barrier sin_d, saux_d;
// Destructive logical Z measurement

barrier saux_d;

measure saux_d[0] -> saux_raw_meas[0];
measure saux_d[1] -> saux_raw_meas[1];
measure saux_d[2] -> saux_raw_meas[2];
measure saux_d[3] -> saux_raw_meas[3];
measure saux_d[4] -> saux_raw_meas[4];
measure saux_d[5] -> saux_raw_meas[5];
measure saux_d[6] -> saux_raw_meas[6];

// determine raw logical output
// ============================
saux_c[1] = (saux_raw_meas[4] ^ saux_raw_meas[5]) ^ saux_raw_meas[6];



// =================== //
// PROCESS MEASUREMENT //
// =================== //

// Determine correction to get logical output
// ==========================================
saux_syn_meas[0] = ((saux_raw_meas[0] ^ saux_raw_meas[1]) ^ saux_raw_meas[2]) ^ saux_raw_meas[3];
saux_syn_meas[1] = ((saux_raw_meas[1] ^ saux_raw_meas[2]) ^ saux_raw_meas[4]) ^ saux_raw_meas[5];
saux_syn_meas[2] = ((saux_raw_meas[2] ^ saux_raw_meas[3]) ^ saux_raw_meas[5]) ^ saux_raw_meas[6];

// XOR syndromes
saux_syn_meas = saux_syn_meas ^ saux_last_raw_syn_z;

// Correct logical output based on measured out syndromes
saux_c[2] = saux_c[1];
if(saux_syn_meas == 2) saux_c[2] = saux_c[2] ^ 1;
if(saux_syn_meas == 4) saux_c[2] = saux_c[2] ^ 1;
if(saux_syn_meas == 6) saux_c[2] = saux_c[2] ^ 1;

// Apply Pauli frame update (flip the logical output)
// Update for logical Z out
saux_c[2] = saux_c[2] ^ saux_c[3];
m_t[0] = saux_c[2];
// Logical SZ
if(sin_c[5] == 1) rz(-pi/2) sin_d[0];
if(sin_c[5] == 1) rz(-pi/2) sin_d[1];
if(sin_c[5] == 1) rz(-pi/2) sin_d[2];
if(sin_c[5] == 1) rz(-pi/2) sin_d[3];
if(sin_c[5] == 1) rz(-pi/2) sin_d[4];
if(sin_c[5] == 1) rz(-pi/2) sin_d[5];
if(sin_c[5] == 1) rz(-pi/2) sin_d[6];
// Destructive logical X measurement
// Logical SYdg
ry(-pi/2) sin_d[0];
ry(-pi/2) sin_d[1];
ry(-pi/2) sin_d[2];
ry(-pi/2) sin_d[3];
ry(-pi/2) sin_d[4];
ry(-pi/2) sin_d[5];
ry(-pi/2) sin_d[6];

barrier sin_d;

measure sin_d[0] -> sin_raw_meas[0];
measure sin_d[1] -> sin_raw_meas[1];
measure sin_d[2] -> sin_raw_meas[2];
measure sin_d[3] -> sin_raw_meas[3];
measure sin_d[4] -> sin_raw_meas[4];
measure sin_d[5] -> sin_raw_meas[5];
measure sin_d[6] -> sin_raw_meas[6];

// determine raw logical output
// ============================
sin_c[1] = (sin_raw_meas[4] ^ sin_raw_meas[5]) ^ sin_raw_meas[6];



// =================== //
// PROCESS MEASUREMENT //
// =================== //

// Determine correction to get logical output
// ==========================================
sin_syn_meas[0] = ((sin_raw_meas[0] ^ sin_raw_meas[1]) ^ sin_raw_meas[2]) ^ sin_raw_meas[3];
sin_syn_meas[1] = ((sin_raw_meas[1] ^ sin_raw_meas[2]) ^ sin_raw_meas[4]) ^ sin_raw_meas[5];
sin_syn_meas[2] = ((sin_raw_meas[2] ^ sin_raw_meas[3]) ^ sin_raw_meas[5]) ^ sin_raw_meas[6];

// XOR syndromes
sin_syn_meas = sin_syn_meas ^ sin_last_raw_syn_x;

// Correct logical output based on measured out syndromes
sin_c[2] = sin_c[1];
if(sin_syn_meas == 2) sin_c[2] = sin_c[2] ^ 1;
if(sin_syn_meas == 4) sin_c[2] = sin_c[2] ^ 1;
if(sin_syn_meas == 6) sin_c[2] = sin_c[2] ^ 1;

// Apply Pauli frame update (flip the logical output)
// Update for logical X out
sin_c[2] = sin_c[2] ^ sin_c[4];
m_out[1] = sin_c[2];
