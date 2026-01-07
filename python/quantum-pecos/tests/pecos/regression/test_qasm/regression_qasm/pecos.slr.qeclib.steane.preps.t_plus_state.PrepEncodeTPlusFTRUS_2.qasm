reset q_test[6];
ry(0.7853981633974483) q_test[6];

// Encoding circuit
// ---------------
reset q_test[0];
reset q_test[1];
reset q_test[2];
reset q_test[3];
reset q_test[4];
reset q_test[5];

// q[6] is the input qubit

cx q_test[6], q_test[5];

h q_test[1];
cx q_test[1], q_test[0];

h q_test[2];
cx q_test[2], q_test[4];

// ---------------
h q_test[3];
cx q_test[3], q_test[5];
cx q_test[2], q_test[0];
cx q_test[6], q_test[4];

// ---------------
cx q_test[2], q_test[6];
cx q_test[3], q_test[4];
cx q_test[1], q_test[5];

// ---------------
cx q_test[1], q_test[6];
cx q_test[3], q_test[0];

// Measure check HHHHHHH
reset a_test[0];
reset a_test[1];
h a_test[0];
barrier a_test[0], q_test[0];
ch a_test[0], q_test[0];
barrier a_test[0], a_test[1];
cx a_test[0], a_test[1];
barrier a_test[0], a_test[1];
ch a_test[0], q_test[1];
barrier a_test[0], q_test[1];
ch a_test[0], q_test[2];
barrier a_test[0], q_test[2];
ch a_test[0], q_test[3];
barrier a_test[0], q_test[3];
ch a_test[0], q_test[4];
barrier a_test[0], q_test[4];
ch a_test[0], q_test[5];
barrier a_test[0], q_test[5];
barrier a_test[0], a_test[1];
cx a_test[0], a_test[1];
barrier a_test[0], a_test[1];
ch a_test[0], q_test[6];
barrier a_test[0], q_test[6];
h a_test[0];
measure a_test[0] -> out_test[0];
measure a_test[1] -> out_test[1];

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

reject_test[0] = (((out_test[0] | out_test[1]) | flags_test[0]) | flags_test[1]) | flags_test[2];
if(reject_test[0] != 0) reset q_test[6];
if(reject_test[0] != 0) ry(0.7853981633974483) q_test[6];

// Encoding circuit
// ---------------
if(reject_test[0] != 0) reset q_test[0];
if(reject_test[0] != 0) reset q_test[1];
if(reject_test[0] != 0) reset q_test[2];
if(reject_test[0] != 0) reset q_test[3];
if(reject_test[0] != 0) reset q_test[4];
if(reject_test[0] != 0) reset q_test[5];

// q[6] is the input qubit

if(reject_test[0] != 0) cx q_test[6], q_test[5];

if(reject_test[0] != 0) h q_test[1];
if(reject_test[0] != 0) cx q_test[1], q_test[0];

if(reject_test[0] != 0) h q_test[2];
if(reject_test[0] != 0) cx q_test[2], q_test[4];

// ---------------
if(reject_test[0] != 0) h q_test[3];
if(reject_test[0] != 0) cx q_test[3], q_test[5];
if(reject_test[0] != 0) cx q_test[2], q_test[0];
if(reject_test[0] != 0) cx q_test[6], q_test[4];

// ---------------
if(reject_test[0] != 0) cx q_test[2], q_test[6];
if(reject_test[0] != 0) cx q_test[3], q_test[4];
if(reject_test[0] != 0) cx q_test[1], q_test[5];

// ---------------
if(reject_test[0] != 0) cx q_test[1], q_test[6];
if(reject_test[0] != 0) cx q_test[3], q_test[0];

// Measure check HHHHHHH
if(reject_test[0] != 0) reset a_test[0];
if(reject_test[0] != 0) reset a_test[1];
if(reject_test[0] != 0) h a_test[0];
if(reject_test[0] != 0) barrier a_test[0], q_test[0];
if(reject_test[0] != 0) ch a_test[0], q_test[0];
if(reject_test[0] != 0) barrier a_test[0], a_test[1];
if(reject_test[0] != 0) cx a_test[0], a_test[1];
if(reject_test[0] != 0) barrier a_test[0], a_test[1];
if(reject_test[0] != 0) ch a_test[0], q_test[1];
if(reject_test[0] != 0) barrier a_test[0], q_test[1];
if(reject_test[0] != 0) ch a_test[0], q_test[2];
if(reject_test[0] != 0) barrier a_test[0], q_test[2];
if(reject_test[0] != 0) ch a_test[0], q_test[3];
if(reject_test[0] != 0) barrier a_test[0], q_test[3];
if(reject_test[0] != 0) ch a_test[0], q_test[4];
if(reject_test[0] != 0) barrier a_test[0], q_test[4];
if(reject_test[0] != 0) ch a_test[0], q_test[5];
if(reject_test[0] != 0) barrier a_test[0], q_test[5];
if(reject_test[0] != 0) barrier a_test[0], a_test[1];
if(reject_test[0] != 0) cx a_test[0], a_test[1];
if(reject_test[0] != 0) barrier a_test[0], a_test[1];
if(reject_test[0] != 0) ch a_test[0], q_test[6];
if(reject_test[0] != 0) barrier a_test[0], q_test[6];
if(reject_test[0] != 0) h a_test[0];
if(reject_test[0] != 0) measure a_test[0] -> out_test[0];
if(reject_test[0] != 0) measure a_test[1] -> out_test[1];

if(reject_test[0] != 0) flag_x_test = 0;
if(reject_test[0] != 0) flag_z_test = 0;

// X check 1, Z check 2, Z check 3
// ===============================

if(reject_test[0] != 0) reset a_test[0];
if(reject_test[0] != 0) reset a_test[1];
if(reject_test[0] != 0) reset a_test[2];

if(reject_test[0] != 0) h a_test[0];
if(reject_test[0] != 0) h a_test[1];
if(reject_test[0] != 0) h a_test[2];

if(reject_test[0] != 0) cx a_test[0], q_test[3];  // 5 -> 4
if(reject_test[0] != 0) cz a_test[1], q_test[5];  // 6 -> 6
if(reject_test[0] != 0) cz a_test[2], q_test[2];  // 7 -> 3

if(reject_test[0] != 0) barrier a_test[0], a_test[1];
if(reject_test[0] != 0) cz a_test[0], a_test[1];
if(reject_test[0] != 0) barrier a_test[0], a_test[1];

if(reject_test[0] != 0) cx a_test[0], q_test[0];  // 1 -> 1
if(reject_test[0] != 0) cz a_test[1], q_test[4];  // 2 -> 5
if(reject_test[0] != 0) cz a_test[2], q_test[3];  // 5 -> 4

if(reject_test[0] != 0) cx a_test[0], q_test[1];  // 3 -> 2
if(reject_test[0] != 0) cz a_test[1], q_test[2];  // 7 -> 3
if(reject_test[0] != 0) cz a_test[2], q_test[6];  // 4 -> 7

if(reject_test[0] != 0) barrier a_test[0], a_test[2];
if(reject_test[0] != 0) cz a_test[0], a_test[2];
if(reject_test[0] != 0) barrier a_test[0], a_test[2];

if(reject_test[0] != 0) cx a_test[0], q_test[2];  // 7 -> 3
if(reject_test[0] != 0) cz a_test[1], q_test[1];  // 3 -> 2
if(reject_test[0] != 0) cz a_test[2], q_test[5];  // 6 -> 6

if(reject_test[0] != 0) h a_test[0];
if(reject_test[0] != 0) h a_test[1];
if(reject_test[0] != 0) h a_test[2];

if(reject_test[0] != 0) measure a_test[0] -> flag_x_test[0];
if(reject_test[0] != 0) measure a_test[1] -> flag_z_test[1];
if(reject_test[0] != 0) measure a_test[2] -> flag_z_test[2];

if(reject_test[0] != 0) flag_x_test[0] = flag_x_test[0] ^ last_raw_syn_x_test[0];
if(reject_test[0] != 0) flag_z_test[1] = flag_z_test[1] ^ last_raw_syn_z_test[1];
if(reject_test[0] != 0) flag_z_test[2] = flag_z_test[2] ^ last_raw_syn_z_test[2];

if(reject_test[0] != 0) flags_test = flag_x_test | flag_z_test;


// Z check 1, X check 2, X check 3
// ===============================

if(reject_test[0] != 0) reset a_test[0];
if(reject_test[0] != 0) reset a_test[1];
if(reject_test[0] != 0) reset a_test[2];

if(reject_test[0] != 0) h a_test[0];
if(reject_test[0] != 0) h a_test[1];
if(reject_test[0] != 0) h a_test[2];

if(reject_test[0] != 0) barrier a_test[0], q_test[3];
if(reject_test[0] != 0) cz a_test[0], q_test[3];  // 5 -> 4
if(reject_test[0] != 0) barrier a_test[0], q_test[3];

if(reject_test[0] != 0) barrier a_test[1], q_test[5];
if(reject_test[0] != 0) cx a_test[1], q_test[5];  // 6 -> 6
if(reject_test[0] != 0) barrier a_test[1], q_test[5];

if(reject_test[0] != 0) barrier a_test[2], q_test[2];
if(reject_test[0] != 0) cx a_test[2], q_test[2];  // 7 -> 3
if(reject_test[0] != 0) barrier a_test[2], q_test[2];

if(reject_test[0] != 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];
if(reject_test[0] != 0) cz a_test[1], a_test[0];
if(reject_test[0] != 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];

if(reject_test[0] != 0) barrier a_test[0], q_test[0];
if(reject_test[0] != 0) cz a_test[0], q_test[0];  // 1 -> 1
if(reject_test[0] != 0) barrier a_test[0], q_test[0];

if(reject_test[0] != 0) barrier a_test[1], q_test[4];
if(reject_test[0] != 0) cx a_test[1], q_test[4];  // 2 -> 5
if(reject_test[0] != 0) barrier a_test[1], q_test[4];

if(reject_test[0] != 0) barrier a_test[2], q_test[3];
if(reject_test[0] != 0) cx a_test[2], q_test[3];  // 5 -> 4
if(reject_test[0] != 0) barrier a_test[2], q_test[3];

if(reject_test[0] != 0) barrier a_test[0], q_test[1];
if(reject_test[0] != 0) cz a_test[0], q_test[1];  // 3 -> 2
if(reject_test[0] != 0) barrier a_test[0], q_test[1];

if(reject_test[0] != 0) barrier a_test[1], q_test[2];
if(reject_test[0] != 0) cx a_test[1], q_test[2];  // 7 -> 3
if(reject_test[0] != 0) barrier a_test[1], q_test[2];

if(reject_test[0] != 0) barrier a_test[2], q_test[6];
if(reject_test[0] != 0) cx a_test[2], q_test[6];  // 4 -> 7
if(reject_test[0] != 0) barrier a_test[2], q_test[6];

if(reject_test[0] != 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];
if(reject_test[0] != 0) cz a_test[2], a_test[0];
if(reject_test[0] != 0) barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];

if(reject_test[0] != 0) barrier a_test[0], q_test[2];
if(reject_test[0] != 0) cz a_test[0], q_test[2];  // 7 -> 3
if(reject_test[0] != 0) barrier a_test[0], q_test[2];

if(reject_test[0] != 0) barrier a_test[1], q_test[1];
if(reject_test[0] != 0) cx a_test[1], q_test[1];  // 3 -> 2
if(reject_test[0] != 0) barrier a_test[1], q_test[1];

if(reject_test[0] != 0) barrier a_test[2], q_test[5];
if(reject_test[0] != 0) cx a_test[2], q_test[5];  // 6 -> 6
if(reject_test[0] != 0) barrier a_test[2], q_test[5];

if(reject_test[0] != 0) h a_test[0];
if(reject_test[0] != 0) h a_test[1];
if(reject_test[0] != 0) h a_test[2];

if(reject_test[0] != 0) measure a_test[0] -> flag_z_test[0];
if(reject_test[0] != 0) measure a_test[1] -> flag_x_test[1];
if(reject_test[0] != 0) measure a_test[2] -> flag_x_test[2];

// XOR flags/syndromes
if(reject_test[0] != 0) flag_z_test[0] = flag_z_test[0] ^ last_raw_syn_z_test[0];
if(reject_test[0] != 0) flag_x_test[1] = flag_x_test[1] ^ last_raw_syn_x_test[1];
if(reject_test[0] != 0) flag_x_test[2] = flag_x_test[2] ^ last_raw_syn_x_test[2];

if(reject_test[0] != 0) flags_test = flag_x_test | flag_z_test;

if(reject_test[0] != 0) reject_test[0] = (((out_test[0] | out_test[1]) | flags_test[0]) | flags_test[1]) | flags_test[2];
rx(-pi/2) q_test[0];
rx(-pi/2) q_test[1];
rx(-pi/2) q_test[2];
rx(-pi/2) q_test[3];
rx(-pi/2) q_test[4];
rx(-pi/2) q_test[5];
rx(-pi/2) q_test[6];
rz(-pi/2) q_test[0];
rz(-pi/2) q_test[1];
rz(-pi/2) q_test[2];
rz(-pi/2) q_test[3];
rz(-pi/2) q_test[4];
rz(-pi/2) q_test[5];
rz(-pi/2) q_test[6];
