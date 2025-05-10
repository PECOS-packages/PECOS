OPENQASM 2.0;
include "qelib1.inc";

// Define quantum and classical registers
qreg q[1];
creg c[3];

// Set some values in classical registers
// We'll do this by conditionally flipping bits based on measurements

// First set qubit to |1⟩ state
x q[0];

// Measure to c[0] - should always be 1
measure q[0] -> c[0];

// Assert that c[0] is 1 by checking if it's 1, and if so, set c[1] to 1
if(c[0]==1) x q[0];
if(c[0]==1) measure q[0] -> c[1];

// Assert that c[0] and c[1] are both 1, and if so, set c[2] to 1
if(c[1]==1) x q[0];
if(c[1]==1) measure q[0] -> c[2];
