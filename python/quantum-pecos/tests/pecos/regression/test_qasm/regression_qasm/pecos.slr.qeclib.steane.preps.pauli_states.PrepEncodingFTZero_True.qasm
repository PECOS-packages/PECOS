
barrier q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[0];

reset q_test[0];
reset q_test[1];
reset q_test[2];
reset q_test[3];
reset q_test[4];
reset q_test[5];
reset q_test[6];
reset a_test[0];
barrier q_test, a_test[0];
h q_test[0];
h q_test[4];
h q_test[6];

cx q_test[4], q_test[5];
cx q_test[0], q_test[1];
cx q_test[6], q_test[3];
cx q_test[4], q_test[2];
cx q_test[6], q_test[5];
cx q_test[0], q_test[3];
cx q_test[4], q_test[1];
cx q_test[3], q_test[2];

barrier a_test[0], q_test[1], q_test[3], q_test[5];
// verification step
cx q_test[5], a_test[0];
cx q_test[1], a_test[0];
cx q_test[3], a_test[0];
measure a_test[0] -> init_bit_test[0];
