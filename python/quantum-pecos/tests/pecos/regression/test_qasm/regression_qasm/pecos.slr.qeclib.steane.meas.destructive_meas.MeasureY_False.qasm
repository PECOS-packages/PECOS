// Logical SX
rx(-pi/2) q_test[0];
rx(-pi/2) q_test[1];
rx(-pi/2) q_test[2];
rx(-pi/2) q_test[3];
rx(-pi/2) q_test[4];
rx(-pi/2) q_test[5];
rx(-pi/2) q_test[6];

measure q_test[0] -> meas_creg_test[0];
measure q_test[1] -> meas_creg_test[1];
measure q_test[2] -> meas_creg_test[2];
measure q_test[3] -> meas_creg_test[3];
measure q_test[4] -> meas_creg_test[4];
measure q_test[5] -> meas_creg_test[5];
measure q_test[6] -> meas_creg_test[6];

// determine raw logical output
// ============================
log_raw_test = (meas_creg_test[4] ^ meas_creg_test[5]) ^ meas_creg_test[6];
