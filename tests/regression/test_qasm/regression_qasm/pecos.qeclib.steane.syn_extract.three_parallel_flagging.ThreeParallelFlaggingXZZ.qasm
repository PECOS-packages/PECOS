
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
