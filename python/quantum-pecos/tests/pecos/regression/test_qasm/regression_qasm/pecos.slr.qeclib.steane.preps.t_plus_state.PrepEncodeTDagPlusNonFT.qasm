// Initialize logical |T> = T|+>
// =============================
reset q_test[6];
h q_test[6];
rz(-pi/4) q_test[6];

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
