// Destructive logical X measurement
// Logical SYdg
ry(-pi/2) q_test[0];
ry(-pi/2) q_test[1];
ry(-pi/2) q_test[2];
ry(-pi/2) q_test[3];
ry(-pi/2) q_test[4];
ry(-pi/2) q_test[5];
ry(-pi/2) q_test[6];

barrier q_test;

measure q_test[0] -> meas_test[0];
measure q_test[1] -> meas_test[1];
measure q_test[2] -> meas_test[2];
measure q_test[3] -> meas_test[3];
measure q_test[4] -> meas_test[4];
measure q_test[5] -> meas_test[5];
measure q_test[6] -> meas_test[6];

// determine raw logical output
// ============================
log_test[0] = (meas_test[4] ^ meas_test[5]) ^ meas_test[6];



// =================== //
// PROCESS MEASUREMENT //
// =================== //

// Determine correction to get logical output
// ==========================================
syn_meas_test[0] = ((meas_test[0] ^ meas_test[1]) ^ meas_test[2]) ^ meas_test[3];
syn_meas_test[1] = ((meas_test[1] ^ meas_test[2]) ^ meas_test[4]) ^ meas_test[5];
syn_meas_test[2] = ((meas_test[2] ^ meas_test[3]) ^ meas_test[5]) ^ meas_test[6];

// XOR syndromes
syn_meas_test = syn_meas_test ^ last_raw_syn_x_test;

// Correct logical output based on measured out syndromes
log_test[1] = log_test[0];
if(syn_meas_test == 2) log_test[1] = log_test[1] ^ 1;
if(syn_meas_test == 4) log_test[1] = log_test[1] ^ 1;
if(syn_meas_test == 6) log_test[1] = log_test[1] ^ 1;

// Apply Pauli frame update (flip the logical output)
// Update for logical X out
log_test[1] = log_test[1] ^ pf_test[1];
