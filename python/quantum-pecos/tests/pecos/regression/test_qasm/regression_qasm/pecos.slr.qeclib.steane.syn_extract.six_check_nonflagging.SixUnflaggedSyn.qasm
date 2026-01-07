
// Run the 6 non-flagged checks (if non-trivial flags)
// ===================================================
// // X check 1, Z check 2, Z check 3

syn_x_test = 0;
syn_z_test = 0;

reset a_test[0];
reset a_test[1];
reset a_test[2];

h a_test[0];
h a_test[1];
h a_test[2];

cx a_test[0], q_test[2];  // X1
cz a_test[1], q_test[5];  // Z2
cz a_test[2], q_test[6];  // Z3

cx a_test[0], q_test[1];  // X1
cz a_test[1], q_test[2];  // Z2
cz a_test[2], q_test[5];  // Z3

cx a_test[0], q_test[3];  // X1
cz a_test[1], q_test[1];  // Z2
cz a_test[2], q_test[2];  // Z3

cx a_test[0], q_test[0];  // X1
cz a_test[1], q_test[4];  // Z2
cz a_test[2], q_test[3];  // Z3

h a_test[0];
h a_test[1];
h a_test[2];

measure a_test[0] -> syn_x_test[0];
measure a_test[1] -> syn_z_test[1];
measure a_test[2] -> syn_z_test[2];

// // Z check 1, X check 2, X check 3

reset a_test[0];
reset a_test[1];
reset a_test[2];

h a_test[0];
h a_test[1];
h a_test[2];

cz a_test[0], q_test[2];  // Z1 0,3
cx a_test[1], q_test[5];  // X2 6,8
cx a_test[2], q_test[6];  // X3 7,9

cz a_test[0], q_test[1];  // Z1 0,2
cx a_test[1], q_test[2];  // X2 3,8
cx a_test[2], q_test[5];  // X3 6,9

cz a_test[0], q_test[3];  // Z1 0,4
cx a_test[1], q_test[1];  // X2 2,8
cx a_test[2], q_test[2];  // X3 3,9

cz a_test[0], q_test[0];  // Z1 0,1
cx a_test[1], q_test[4];  // X2 5,8
cx a_test[2], q_test[3];  // X3 4,9

h a_test[0];
h a_test[1];
h a_test[2];

measure a_test[0] -> syn_z_test[0];
measure a_test[1] -> syn_x_test[1];
measure a_test[2] -> syn_x_test[2];
