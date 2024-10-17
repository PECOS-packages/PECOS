
// Z check 1, X check 2, X check 3
// ===============================

reset a_test[0];
reset a_test[1];
reset a_test[2];

h a_test[0];
h a_test[1];
h a_test[2];

barrier a_test[0], q_test[3];
cz a_test[0], q_test[3];  // 5 -> 4
barrier a_test[0], q_test[3];

barrier a_test[1], q_test[5];
cx a_test[1], q_test[5];  // 6 -> 6
barrier a_test[1], q_test[5];

barrier a_test[2], q_test[2];
cx a_test[2], q_test[2];  // 7 -> 3
barrier a_test[2], q_test[2];

barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];
cz a_test[1], a_test[0];
barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];

barrier a_test[0], q_test[0];
cz a_test[0], q_test[0];  // 1 -> 1
barrier a_test[0], q_test[0];

barrier a_test[1], q_test[4];
cx a_test[1], q_test[4];  // 2 -> 5
barrier a_test[1], q_test[4];

barrier a_test[2], q_test[3];
cx a_test[2], q_test[3];  // 5 -> 4
barrier a_test[2], q_test[3];

barrier a_test[0], q_test[1];
cz a_test[0], q_test[1];  // 3 -> 2
barrier a_test[0], q_test[1];

barrier a_test[1], q_test[2];
cx a_test[1], q_test[2];  // 7 -> 3
barrier a_test[1], q_test[2];

barrier a_test[2], q_test[6];
cx a_test[2], q_test[6];  // 4 -> 7
barrier a_test[2], q_test[6];

barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];
cz a_test[2], a_test[0];
barrier a_test[0], q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[1], a_test[2];

barrier a_test[0], q_test[2];
cz a_test[0], q_test[2];  // 7 -> 3
barrier a_test[0], q_test[2];

barrier a_test[1], q_test[1];
cx a_test[1], q_test[1];  // 3 -> 2
barrier a_test[1], q_test[1];

barrier a_test[2], q_test[5];
cx a_test[2], q_test[5];  // 6 -> 6
barrier a_test[2], q_test[5];

h a_test[0];
h a_test[1];
h a_test[2];

measure a_test[0] -> flag_z_test[0];
measure a_test[1] -> flag_x_test[1];
measure a_test[2] -> flag_x_test[2];

// XOR flags/syndromes
flag_z_test[0] = flag_z_test[0] ^ last_raw_syn_z_test[0];
flag_x_test[1] = flag_x_test[1] ^ last_raw_syn_x_test[1];
flag_x_test[2] = flag_x_test[2] ^ last_raw_syn_x_test[2];

flags_test = flag_x_test | flag_z_test;
