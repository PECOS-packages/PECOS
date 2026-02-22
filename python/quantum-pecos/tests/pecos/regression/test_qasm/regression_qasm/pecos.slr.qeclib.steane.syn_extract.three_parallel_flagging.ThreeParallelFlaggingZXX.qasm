OPENQASM 2.0;
include "hqslib1.inc";
// Z check 1, X check 2, X check 3
// ===============================
reset a_test[0];
reset a_test[1];
reset a_test[2];
h a_test[0];
h a_test[1];
h a_test[2];
barrier ;
cz a_test[0], q_test[3];
// 5 -> 4
barrier ;
barrier ;
cx a_test[1], q_test[5];
// 6 -> 6
barrier ;
barrier ;
cx a_test[2], q_test[2];
// 7 -> 3
barrier ;
barrier ;
cz a_test[1], a_test[0];
barrier ;
barrier ;
cz a_test[0], q_test[0];
// 1 -> 1
barrier ;
barrier ;
cx a_test[1], q_test[4];
// 2 -> 5
barrier ;
barrier ;
cx a_test[2], q_test[3];
// 5 -> 4
barrier ;
barrier ;
cz a_test[0], q_test[1];
// 3 -> 2
barrier ;
barrier ;
cx a_test[1], q_test[2];
// 7 -> 3
barrier ;
barrier ;
cx a_test[2], q_test[6];
// 4 -> 7
barrier ;
barrier ;
cz a_test[2], a_test[0];
barrier ;
barrier ;
cz a_test[0], q_test[2];
// 7 -> 3
barrier ;
barrier ;
cx a_test[1], q_test[1];
// 3 -> 2
barrier ;
barrier ;
cx a_test[2], q_test[5];
// 6 -> 6
barrier ;
h a_test[0];
h a_test[1];
h a_test[2];
measure a_test[0] -> flag_z_test[0];
measure a_test[1] -> flag_x_test[1];
measure a_test[2] -> flag_x_test[2];
// XOR flags/syndromes
flag_z_test[0] = (flag_z_test[0] ^ last_raw_syn_z_test[0]);
flag_x_test[1] = (flag_x_test[1] ^ last_raw_syn_x_test[1]);
flag_x_test[2] = (flag_x_test[2] ^ last_raw_syn_x_test[2]);
flags_test = (flag_x_test | flag_z_test);
