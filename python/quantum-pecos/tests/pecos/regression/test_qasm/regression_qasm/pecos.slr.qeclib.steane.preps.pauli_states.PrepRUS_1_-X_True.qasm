OPENQASM 2.0;
include "hqslib1.inc";
barrier ;
reset q_test[0];
reset q_test[1];
reset q_test[2];
reset q_test[3];
reset q_test[4];
reset q_test[5];
reset q_test[6];
reset a_test[0];
barrier q_test;
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
barrier ;
// verification step
cx q_test[5], a_test[0];
cx q_test[1], a_test[0];
cx q_test[3], a_test[0];
measure a_test[0] -> init_test[0];
// Repeat 0 times (unrolled)
// Logical H
h q_test[0];
h q_test[1];
h q_test[2];
h q_test[3];
h q_test[4];
h q_test[5];
h q_test[6];
// Logical Z
z q_test[4];
z q_test[5];
z q_test[6];
