
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
syn_meas_test = syn_meas_test ^ last_raw_syn_z_test;

// Correct logical output based on measured out syndromes
log_test[1] = log_test[0];
if(syn_meas_test == 2) log_test[1] = log_test[1] ^ 1;
if(syn_meas_test == 4) log_test[1] = log_test[1] ^ 1;
if(syn_meas_test == 6) log_test[1] = log_test[1] ^ 1;

// Apply Pauli frame update (flip the logical output)
// Update for logical Y out
log_test[1] = log_test[1] ^ pf_test[0];
log_test[1] = log_test[1] ^ pf_test[1];
