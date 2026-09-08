OPENQASM 2.0;
include "qelib1.inc";
// Transversal Logical SZZ
barrier q1_test, q2_test;
SZZ q1_test[0], q2_test[0];
SZZ q1_test[1], q2_test[1];
SZZ q1_test[2], q2_test[2];
SZZ q1_test[3], q2_test[3];
SZZ q1_test[4], q2_test[4];
SZZ q1_test[5], q2_test[5];
SZZ q1_test[6], q2_test[6];
barrier q1_test, q2_test;
