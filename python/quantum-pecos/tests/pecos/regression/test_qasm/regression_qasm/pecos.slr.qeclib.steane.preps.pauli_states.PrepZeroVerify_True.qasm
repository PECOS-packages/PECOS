
barrier a_test[0], q_test[1], q_test[3], q_test[5];
// verification step

reset a_test[0];
cx q_test[5], a_test[0];
cx q_test[1], a_test[0];
cx q_test[3], a_test[0];
measure a_test[0] -> init_bit[0];
