// Hardware validation: conditional that should NOT trigger
// Expected: m = 1 (bit 0 set), r = 1 (q[0] unchanged because m != 0), bell = 0 or 3
// Tests: if(creg == 0) when register is nonzero
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

// Deterministic non-triggering conditional
x q[0];
measure q[0] -> m[0];
if(m==0) x q[0];
measure q[0] -> r[0];
