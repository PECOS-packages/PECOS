OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Define a custom gate
gate bell a, b {
    h a;
    cx a, b;
}

// Define another gate that uses our custom gate
gate triple_bell a, b, c {
    bell a, b;
    bell b, c;
}

// Use the gates
h q[0];
bell q[0], q[1];
triple_bell q[0], q[1], q[2];
measure q -> c;