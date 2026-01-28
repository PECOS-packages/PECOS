//! Single-qubit Clifford gate fusion
//!
//! Fuses consecutive single-qubit gates on the same qubit to reduce GPU dispatches.
//! This handles common patterns like H*H=I, S*S=Z, and accumulates more complex
//! sequences for batch processing.

use std::collections::HashMap;

/// Gate type constants (must match `gpu_stab` shader)
pub const GATE_H: u32 = 0;
pub const GATE_S: u32 = 1;
pub const GATE_SDG: u32 = 2;
pub const GATE_X: u32 = 3;
pub const GATE_Y: u32 = 4;
pub const GATE_Z: u32 = 5;
// Two-qubit gate constants - defined for completeness but fusion not yet implemented
#[allow(dead_code)]
pub const GATE_CX: u32 = 6;
#[allow(dead_code)]
pub const GATE_CZ: u32 = 7;
#[allow(dead_code)]
pub const GATE_SWAP: u32 = 8;

/// Single-qubit gate sequence for one qubit
/// Accumulates gates and simplifies common patterns
#[derive(Clone, Debug, Default)]
struct GateSequence {
    /// Accumulated gates in order applied
    gates: Vec<u32>,
}

impl GateSequence {
    #[allow(dead_code)]
    fn new() -> Self {
        Self { gates: Vec::new() }
    }

    /// Add a gate and simplify if possible
    fn add(&mut self, gate: u32) {
        // Try to simplify with the last gate
        if let Some(&last) = self.gates.last()
            && let Some(simplified) = simplify_pair(last, gate)
        {
            self.gates.pop();
            if let Some(g) = simplified {
                self.add(g); // Recursively simplify
            }
            return;
        }
        self.gates.push(gate);
    }

    /// Check if sequence is empty (identity)
    fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Get the gates to emit
    fn into_gates(self) -> Vec<u32> {
        self.gates
    }
}

/// Try to simplify a pair of gates. Returns:
/// - Some(None) if they cancel to identity
/// - Some(Some(gate)) if they simplify to a single gate
/// - None if they can't be simplified
#[allow(clippy::option_option)]
fn simplify_pair(first: u32, second: u32) -> Option<Option<u32>> {
    match (first, second) {
        // Self-inverse gates: H*H = I, X*X = I, Y*Y = I, Z*Z = I
        (GATE_H, GATE_H) | (GATE_X, GATE_X) | (GATE_Y, GATE_Y) | (GATE_Z, GATE_Z) => {
            Some(None) // Cancel to identity
        }

        // S gates: S*S = Z, Sdg*Sdg = Z, S*Sdg = I, Sdg*S = I
        // X and Y: X*Y = Z, Y*X = Z (up to global phase)
        (GATE_S, GATE_S) | (GATE_SDG, GATE_SDG) | (GATE_X, GATE_Y) | (GATE_Y, GATE_X) => {
            Some(Some(GATE_Z))
        }
        (GATE_S, GATE_SDG) | (GATE_SDG, GATE_S) => Some(None),

        // S and Z: S*Z = Sdg, Sdg*Z = S, Z*S = Sdg, Z*Sdg = S
        (GATE_S, GATE_Z) | (GATE_Z, GATE_S) => Some(Some(GATE_SDG)),
        (GATE_SDG, GATE_Z) | (GATE_Z, GATE_SDG) => Some(Some(GATE_S)),

        // X and Z: X*Z = Y (up to phase), Z*X = Y
        // Actually XZ = -iY, ZX = iY - for Clifford simulation phases don't matter
        (GATE_X, GATE_Z) | (GATE_Z, GATE_X) => Some(Some(GATE_Y)),

        // Y and Z: Y*Z = X, Z*Y = X
        (GATE_Y, GATE_Z) | (GATE_Z, GATE_Y) => Some(Some(GATE_X)),

        _ => None, // Can't simplify
    }
}

/// Gate fusion optimizer
///
/// Accumulates gates and fuses consecutive single-qubit gates on the same qubit.
pub struct CliffordFuser {
    /// Pending gate sequence for each qubit
    pending: HashMap<u32, GateSequence>,
    /// Output gate queue (packed gates)
    output: Vec<u32>,
}

impl CliffordFuser {
    /// Create a new gate fuser
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            output: Vec::new(),
        }
    }

    /// Check if a gate is a single-qubit gate
    fn is_single_qubit(gate_type: u32) -> bool {
        matches!(
            gate_type,
            GATE_H | GATE_S | GATE_SDG | GATE_X | GATE_Y | GATE_Z
        )
    }

    /// Add a gate to the fuser
    ///
    /// Returns true if the gate was absorbed (fused with pending), false if it was emitted.
    pub fn add_gate(&mut self, gate_type: u32, target: u32, control: u32) -> bool {
        if Self::is_single_qubit(gate_type) {
            // Single-qubit gate - try to fuse
            let seq = self.pending.entry(target).or_default();
            seq.add(gate_type);
            true
        } else {
            // Two-qubit gate - flush any pending gates on these qubits first
            self.flush_qubit(target);
            self.flush_qubit(control);

            // Emit the two-qubit gate
            let packed = pack_gate(gate_type, target, control);
            self.output.push(packed);
            false
        }
    }

    /// Flush pending gates for a specific qubit
    fn flush_qubit(&mut self, qubit: u32) {
        if let Some(seq) = self.pending.remove(&qubit)
            && !seq.is_empty()
        {
            for gate_type in seq.into_gates() {
                let packed = pack_gate(gate_type, qubit, 0);
                self.output.push(packed);
            }
        }
    }

    /// Flush all pending gates and return the optimized gate queue
    pub fn flush_all(&mut self) -> Vec<u32> {
        let qubits: Vec<u32> = self.pending.keys().copied().collect();
        for qubit in qubits {
            self.flush_qubit(qubit);
        }
        std::mem::take(&mut self.output)
    }
}

impl Default for CliffordFuser {
    fn default() -> Self {
        Self::new()
    }
}

/// Pack a gate into the queue format
fn pack_gate(gate_type: u32, target: u32, control: u32) -> u32 {
    (gate_type & 0xF) | ((target & 0x3FFF) << 4) | ((control & 0x3FFF) << 18)
}

/// Unpack a gate from the queue format
#[allow(dead_code)]
fn unpack_gate(packed: u32) -> (u32, u32, u32) {
    let gate_type = packed & 0xF;
    let target = (packed >> 4) & 0x3FFF;
    let control = (packed >> 18) & 0x3FFF;
    (gate_type, target, control)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_fusion() {
        let mut fuser = CliffordFuser::new();

        // H * H = I
        fuser.add_gate(GATE_H, 0, 0);
        fuser.add_gate(GATE_H, 0, 0);

        let gates = fuser.flush_all();
        assert!(gates.is_empty(), "H*H should cancel to identity");
    }

    #[test]
    fn test_s_s_fusion() {
        let mut fuser = CliffordFuser::new();

        // S * S = Z
        fuser.add_gate(GATE_S, 0, 0);
        fuser.add_gate(GATE_S, 0, 0);

        let gates = fuser.flush_all();
        assert_eq!(gates.len(), 1, "S*S should fuse to Z");
        let (gate_type, _, _) = unpack_gate(gates[0]);
        assert_eq!(gate_type, GATE_Z, "S*S should be Z");
    }

    #[test]
    fn test_s_sdg_cancel() {
        let mut fuser = CliffordFuser::new();

        // S * Sdg = I
        fuser.add_gate(GATE_S, 0, 0);
        fuser.add_gate(GATE_SDG, 0, 0);

        let gates = fuser.flush_all();
        assert!(gates.is_empty(), "S*Sdg should cancel to identity");
    }

    #[test]
    fn test_two_qubit_flushes() {
        let mut fuser = CliffordFuser::new();

        // H on qubit 0, then CX(1, 0) should flush H first
        fuser.add_gate(GATE_H, 0, 0);
        fuser.add_gate(GATE_CX, 1, 0); // target=1, control=0

        let gates = fuser.flush_all();
        assert_eq!(gates.len(), 2, "Should have H then CX");
    }

    #[test]
    fn test_different_qubits() {
        let mut fuser = CliffordFuser::new();

        // Gates on different qubits don't fuse
        fuser.add_gate(GATE_H, 0, 0);
        fuser.add_gate(GATE_H, 1, 0);

        let gates = fuser.flush_all();
        assert_eq!(gates.len(), 2, "Different qubits should not fuse");
    }

    #[test]
    fn test_pauli_fusion() {
        let mut fuser = CliffordFuser::new();

        // X * Z = Y
        fuser.add_gate(GATE_X, 0, 0);
        fuser.add_gate(GATE_Z, 0, 0);

        let gates = fuser.flush_all();
        assert_eq!(gates.len(), 1, "X*Z should fuse to Y");
        let (gate_type, _, _) = unpack_gate(gates[0]);
        assert_eq!(gate_type, GATE_Y, "X*Z should be Y");
    }

    #[test]
    fn test_triple_cancel() {
        let mut fuser = CliffordFuser::new();

        // X * X * H = H (X*X cancels)
        fuser.add_gate(GATE_X, 0, 0);
        fuser.add_gate(GATE_X, 0, 0);
        fuser.add_gate(GATE_H, 0, 0);

        let gates = fuser.flush_all();
        assert_eq!(gates.len(), 1, "X*X*H should simplify to H");
        let (gate_type, _, _) = unpack_gate(gates[0]);
        assert_eq!(gate_type, GATE_H, "Should be H");
    }

    #[test]
    fn test_s_z_fusion() {
        let mut fuser = CliffordFuser::new();

        // S * Z = Sdg
        fuser.add_gate(GATE_S, 0, 0);
        fuser.add_gate(GATE_Z, 0, 0);

        let gates = fuser.flush_all();
        assert_eq!(gates.len(), 1, "S*Z should fuse to Sdg");
        let (gate_type, _, _) = unpack_gate(gates[0]);
        assert_eq!(gate_type, GATE_SDG, "S*Z should be Sdg");
    }
}
