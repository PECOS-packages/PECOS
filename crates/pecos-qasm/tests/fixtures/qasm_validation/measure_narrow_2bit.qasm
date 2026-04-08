// Hardware validation: measurement into 2-bit register
// Expected: m = 3 (both bits set), bell = 0 or 3 (entangled pair)
// Tests: narrow register measurement packing
OPENQASM 2.0;
include "qelib1.inc";

qreg b[2];
qreg q[2];
creg bell[2];
creg m[2];

// Bell state on spare qubits (real quantum content)
h b[0];
cx b[0], b[1];
measure b[0] -> bell[0];
measure b[1] -> bell[1];

// Deterministic test: set both qubits to |1>
x q[0];
x q[1];
measure q[0] -> m[0];
measure q[1] -> m[1];
