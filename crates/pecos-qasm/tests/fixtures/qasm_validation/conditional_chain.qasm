// Hardware validation: chain of measure-conditional-correct steps
// Expected: m0 = 1, m1 = 1, m2 = 0 (each correction flips state), bell = 0 or 3
// Tests: sequential mid-circuit measurement and feedback
OPENQASM 2.0;
include "qelib1.inc";

qreg b[2];
qreg q[1];
creg bell[2];
creg m0[1];
creg m1[1];
creg m2[1];

// Bell state (real quantum content)
h b[0];
cx b[0], b[1];
measure b[0] -> bell[0];
measure b[1] -> bell[1];

// Deterministic feedback chain
x q[0];
measure q[0] -> m0[0];
if(m0==1) x q[0];
x q[0];
measure q[0] -> m1[0];
if(m1==1) x q[0];
measure q[0] -> m2[0];
