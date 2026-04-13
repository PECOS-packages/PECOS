// Hardware validation: multiple classical registers of different sizes
// Expected: a = 1, b = 0, c = 3 (all bits set in 2-bit register), bell = 0 or 3
// Tests: independent registers, different sizes, measurement routing
OPENQASM 2.0;
include "qelib1.inc";

qreg bp[2];
qreg q[4];
creg bell[2];
creg a[1];
creg b[1];
creg c[2];

// Bell state (real quantum content)
h bp[0];
cx bp[0], bp[1];
measure bp[0] -> bell[0];
measure bp[1] -> bell[1];

// Deterministic multi-register test
x q[0];
x q[2];
x q[3];
measure q[0] -> a[0];
measure q[1] -> b[0];
measure q[2] -> c[0];
measure q[3] -> c[1];
