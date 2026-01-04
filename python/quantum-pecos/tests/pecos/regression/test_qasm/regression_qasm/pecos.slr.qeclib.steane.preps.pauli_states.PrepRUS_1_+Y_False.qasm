
barrier q_test[0], q_test[1], q_test[2], q_test[3], q_test[4], q_test[5], q_test[6], a_test[0];

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
measure a_test[0] -> init_test[0];


// Logical SXdg
rx(pi/2) q_test[0];
rx(pi/2) q_test[1];
rx(pi/2) q_test[2];
rx(pi/2) q_test[3];
rx(pi/2) q_test[4];
rx(pi/2) q_test[5];
rx(pi/2) q_test[6];
