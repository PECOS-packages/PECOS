// Hardware validation: measurement into 4-bit register with pattern
// Expected: m = 5 (bits 0 and 2 set = binary 0101), bell = 0 or 3
// Tests: measurement bit packing in wider register
OPENQASM 2.0;
include "qelib1.inc";

qreg b[2];
qreg q[4];
creg bell[2];
creg m[4];

// Bell state (real quantum content)
h b[0];
cx b[0], b[1];
measure b[0] -> bell[0];
measure b[1] -> bell[1];

// Deterministic test
x q[0];
x q[2];
measure q[0] -> m[0];
measure q[1] -> m[1];
measure q[2] -> m[2];
measure q[3] -> m[3];
