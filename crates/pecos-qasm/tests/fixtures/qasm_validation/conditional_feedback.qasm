// Hardware validation: measure-conditional-correct feedback loop
// Expected: m = 1 (first measure), r = 0 (after correction), bell = 0 or 3
// Tests: mid-circuit measurement, conditional gate, re-measurement
OPENQASM 2.0;
include "qelib1.inc";

qreg b[2];
qreg q[1];
creg bell[2];
creg m[1];
creg r[1];

// Bell state (real quantum content)
h b[0];
cx b[0], b[1];
measure b[0] -> bell[0];
measure b[1] -> bell[1];

// Deterministic feedback test
x q[0];
measure q[0] -> m[0];
if(m==1) x q[0];
measure q[0] -> r[0];
