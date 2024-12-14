// Logical SYdg
ry(-pi/2) q_test[0];
ry(-pi/2) q_test[1];
ry(-pi/2) q_test[2];
ry(-pi/2) q_test[3];
ry(-pi/2) q_test[4];
ry(-pi/2) q_test[5];
ry(-pi/2) q_test[6];

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