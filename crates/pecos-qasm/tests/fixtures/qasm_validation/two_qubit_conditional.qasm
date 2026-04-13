// Hardware validation: conditional two-qubit gate
// Expected: m = 1, r = 3 (CX triggers, both measure 1), bell = 0 or 3
// Tests: conditional multi-qubit gate execution
OPENQASM 2.0;
include "qelib1.inc";

qreg b[2];
qreg q[2];
creg bell[2];
creg m[1];
creg r[2];

// Bell state (real quantum content)
h b[0];
cx b[0], b[1];
measure b[0] -> bell[0];
measure b[1] -> bell[1];

// Deterministic conditional CX test
x q[0];
measure q[0] -> m[0];
if(m==1) cx q[0], q[1];
measure q[0] -> r[0];
measure q[1] -> r[1];
