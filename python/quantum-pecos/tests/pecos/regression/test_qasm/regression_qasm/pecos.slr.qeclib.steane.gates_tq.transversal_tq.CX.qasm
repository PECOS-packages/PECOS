OPENQASM 2.0;
include "hqslib1.inc";
// Transversal Logical CX
barrier q_test1, q_test2;
cx q_test1[0], q_test2[0];
cx q_test1[1], q_test2[1];
cx q_test1[2], q_test2[2];
cx q_test1[3], q_test2[3];
cx q_test1[4], q_test2[4];
cx q_test1[5], q_test2[5];
cx q_test1[6], q_test2[6];
barrier q_test1, q_test2;
