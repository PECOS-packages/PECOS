// Hardware validation: measurement into 8-bit register
// Expected: m = 170 (binary 10101010, bits 1,3,5,7 set), bell = 0 or 3
// Tests: wider register measurement packing, bit ordering
OPENQASM 2.0;
include "qelib1.inc";

qreg b[2];
qreg q[8];
creg bell[2];
creg m[8];

// Bell state (real quantum content)
h b[0];
cx b[0], b[1];
measure b[0] -> bell[0];
measure b[1] -> bell[1];

// Deterministic pattern
x q[1];
x q[3];
x q[5];
x q[7];
measure q[0] -> m[0];
measure q[1] -> m[1];
measure q[2] -> m[2];
measure q[3] -> m[3];
measure q[4] -> m[4];
measure q[5] -> m[5];
measure q[6] -> m[6];
measure q[7] -> m[7];
