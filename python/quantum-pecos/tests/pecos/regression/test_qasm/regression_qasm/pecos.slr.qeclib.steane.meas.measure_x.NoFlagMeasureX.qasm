// Measure logical X with no flagging
reset a_test[0];
h a_test[0];
cx q_test[0], a_test[0];
cx q_test[1], a_test[0];
cx q_test[2], a_test[0];
h a_test[0];
measure a_test[0] -> out_test[0];
