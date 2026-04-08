// Hardware validation: conditional on multi-bit register value
// Expected: m = 3, r = 0 (conditional triggers, flips q[0] back), bell = 0 or 3
// Tests: if(creg == val) with val > 1, register comparison semantics
OPENQASM 2.0;
include "qelib1.inc";

qreg b[2];
qreg q[2];
creg bell[2];
creg m[2];
creg r[1];

// Bell state (real quantum content)
h b[0];
cx b[0], b[1];
measure b[0] -> bell[0];
measure b[1] -> bell[1];

// Deterministic conditional test
x q[0];
x q[1];
measure q[0] -> m[0];
measure q[1] -> m[1];
if(m==3) x q[0];
measure q[0] -> r[0];
