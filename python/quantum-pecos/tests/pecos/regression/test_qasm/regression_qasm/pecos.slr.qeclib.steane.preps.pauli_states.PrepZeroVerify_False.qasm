OPENQASM 2.0;
include "hqslib1.inc";
barrier ;
// verification step
cx q_test[5], a_test[0];
cx q_test[1], a_test[0];
cx q_test[3], a_test[0];
measure a_test[0] -> init_bit[0];
