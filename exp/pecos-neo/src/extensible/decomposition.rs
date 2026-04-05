//! Gate decomposition system.
//!
//! Provides a trait-like mechanism for defining how gates decompose into simpler gates:
//! - Simulators that natively support a gate use the native implementation
//! - Simulators that support base gates use the decomposition
//! - Resolution happens at build time, not execution time
//!
//! # Example
//!
//! ```
//! use pecos_neo::extensible::*;
//!
//! // Define a custom gate with decomposition
//! let mut registry = DecompositionRegistry::new();
//!
//! // SWAP decomposes into 3 CX gates
//! registry.register(
//!     gates::SWAP,
//!     GateSupportSet::from_iter([gates::CX]),
//!     Decomposition::SwapViaCx,
//! );
//!
//! // Check if a simulator can handle SWAP
//! let sim_support = GateSupportSet::from_iter([gates::CX, gates::H, gates::T]);
//! assert!(registry.can_execute(gates::SWAP, &sim_support));
//! ```

use super::{GateId, GateSupportSet, gates};
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;
use std::sync::Arc;

/// A single operation in a decomposition sequence.
///
/// Uses indices (0, 1, 2, ...) to refer to input qubits,
/// which get mapped to actual `QubitId`s at resolution time.
#[derive(Clone, Debug)]
pub struct DecompOp {
    /// The gate to apply.
    pub gate: GateId,
    /// Qubit indices (0 = first input qubit, 1 = second, etc.)
    pub qubit_indices: SmallVec<[u8; 4]>,
    /// Angle sources for parameterized gates.
    pub angles: SmallVec<[AngleSource; 2]>,
}

impl DecompOp {
    /// Create a single-qubit gate operation.
    #[must_use]
    pub fn gate1(gate: GateId, q: u8) -> Self {
        Self {
            gate,
            qubit_indices: SmallVec::from_buf([q, 0, 0, 0]),
            angles: SmallVec::new(),
        }
    }

    /// Create a two-qubit gate operation.
    #[must_use]
    pub fn gate2(gate: GateId, q0: u8, q1: u8) -> Self {
        Self {
            gate,
            qubit_indices: SmallVec::from_buf([q0, q1, 0, 0]),
            angles: SmallVec::new(),
        }
    }

    /// Create a rotation gate operation with an angle from input.
    #[must_use]
    pub fn rotation(gate: GateId, q: u8, angle_idx: u8) -> Self {
        let mut angles = SmallVec::new();
        angles.push(AngleSource::Input(angle_idx));
        Self {
            gate,
            qubit_indices: SmallVec::from_buf([q, 0, 0, 0]),
            angles,
        }
    }

    /// Create a rotation gate operation with a fixed angle.
    #[must_use]
    pub fn rotation_fixed(gate: GateId, q: u8, angle: Angle64) -> Self {
        let mut angles = SmallVec::new();
        angles.push(AngleSource::Fixed(angle));
        Self {
            gate,
            qubit_indices: SmallVec::from_buf([q, 0, 0, 0]),
            angles,
        }
    }

    /// Instantiate this operation with actual qubit IDs and angles.
    #[must_use]
    pub fn instantiate(&self, qubits: &[QubitId], input_angles: &[Angle64]) -> InstantiatedOp {
        let mapped_qubits: SmallVec<[QubitId; 4]> = self
            .qubit_indices
            .iter()
            .take(self.qubit_count())
            .map(|&idx| qubits[idx as usize])
            .collect();

        let resolved_angles: SmallVec<[Angle64; 2]> = self
            .angles
            .iter()
            .map(|src| src.resolve(input_angles))
            .collect();

        InstantiatedOp {
            gate: self.gate,
            qubits: mapped_qubits,
            angles: resolved_angles,
        }
    }

    /// Get the number of qubits this operation uses.
    fn qubit_count(&self) -> usize {
        // Count non-zero indices (hacky but works for small counts)
        self.qubit_indices
            .iter()
            .enumerate()
            .take_while(|(i, idx)| *i == 0 || **idx != 0)
            .count()
            .max(1)
    }
}

/// Where an angle value comes from in a decomposition.
#[derive(Clone, Copy, Debug)]
pub enum AngleSource {
    /// Use input angle at the given index.
    Input(u8),
    /// Use a fixed constant angle.
    Fixed(Angle64),
    /// Negate the input angle.
    NegInput(u8),
    /// Half of the input angle.
    HalfInput(u8),
}

impl AngleSource {
    /// Resolve the angle source to an actual angle value.
    #[must_use]
    pub fn resolve(&self, input_angles: &[Angle64]) -> Angle64 {
        match *self {
            Self::Input(idx) => input_angles[idx as usize],
            Self::Fixed(a) => a,
            Self::NegInput(idx) => -input_angles[idx as usize],
            Self::HalfInput(idx) => input_angles[idx as usize] / 2_u64,
        }
    }
}

/// An instantiated operation with actual qubit IDs and angles.
#[derive(Clone, Debug)]
pub struct InstantiatedOp {
    /// The gate to apply.
    pub gate: GateId,
    /// Actual qubit IDs.
    pub qubits: SmallVec<[QubitId; 4]>,
    /// Resolved angle values.
    pub angles: SmallVec<[Angle64; 2]>,
}

/// How to decompose a gate into simpler gates.
///
/// Uses enum variants for common patterns (no allocation),
/// with a fallback to dynamic sequences.
#[derive(Clone)]
pub enum Decomposition {
    /// Gate is primitive - no decomposition, must be natively supported.
    Native,

    /// SWAP = CX(0,1); CX(1,0); CX(0,1)
    SwapViaCx,

    /// iSWAP decomposition
    ISwapDecomp,

    /// Static decomposition stored as a reference (no allocation).
    Static(&'static [DecompOp]),

    /// Dynamic decomposition (for user-defined gates).
    Dynamic(Arc<Vec<DecompOp>>),
}

impl std::fmt::Debug for Decomposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => write!(f, "Native"),
            Self::SwapViaCx => write!(f, "SwapViaCx"),
            Self::ISwapDecomp => write!(f, "ISwapDecomp"),
            Self::Static(ops) => write!(f, "Static({} ops)", ops.len()),
            Self::Dynamic(ops) => write!(f, "Dynamic({} ops)", ops.len()),
        }
    }
}

impl Decomposition {
    /// Expand this decomposition into a sequence of operations.
    ///
    /// Returns an iterator over `DecompOp` that can be instantiated.
    #[must_use]
    pub fn expand(&self) -> DecompIter<'_> {
        match self {
            Self::Native => DecompIter::Empty,
            Self::SwapViaCx => DecompIter::Swap { idx: 0 },
            Self::ISwapDecomp => DecompIter::ISwap { idx: 0 },
            Self::Static(ops) => DecompIter::Slice(ops.iter()),
            Self::Dynamic(ops) => DecompIter::Slice(ops.iter()),
        }
    }

    /// Get the number of operations in this decomposition.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Native => 0,
            Self::SwapViaCx => 3,
            Self::ISwapDecomp => 6,
            Self::Static(ops) => ops.len(),
            Self::Dynamic(ops) => ops.len(),
        }
    }

    /// Check if this is a native gate (no decomposition).
    #[must_use]
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native)
    }

    /// Check if this decomposition is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Iterator over decomposition operations.
pub enum DecompIter<'a> {
    Empty,
    Swap { idx: u8 },
    ISwap { idx: u8 },
    Slice(std::slice::Iter<'a, DecompOp>),
}

impl Iterator for DecompIter<'_> {
    type Item = DecompOp;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Swap { idx } => {
                // SWAP = CX(0,1); CX(1,0); CX(0,1)
                let op = match *idx {
                    0 | 2 => Some(DecompOp::gate2(gates::CX, 0, 1)),
                    1 => Some(DecompOp::gate2(gates::CX, 1, 0)),
                    _ => None,
                };
                if op.is_some() {
                    *idx += 1;
                }
                op
            }
            Self::ISwap { idx } => {
                // iSWAP decomposition: S(0); S(1); H(0); CX(0,1); CX(1,0); H(1)
                let op = match *idx {
                    0 => Some(DecompOp::gate1(gates::SZ, 0)),
                    1 => Some(DecompOp::gate1(gates::SZ, 1)),
                    2 => Some(DecompOp::gate1(gates::H, 0)),
                    3 => Some(DecompOp::gate2(gates::CX, 0, 1)),
                    4 => Some(DecompOp::gate2(gates::CX, 1, 0)),
                    5 => Some(DecompOp::gate1(gates::H, 1)),
                    _ => None,
                };
                if op.is_some() {
                    *idx += 1;
                }
                op
            }
            Self::Slice(iter) => iter.next().cloned(),
        }
    }
}

/// Entry in the decomposition registry.
#[derive(Clone, Debug)]
pub struct DecompEntry {
    /// What base gates are required for this decomposition.
    pub requires: GateSupportSet,
    /// How to decompose the gate.
    pub decomposition: Decomposition,
}

/// Registry of gate decompositions.
///
/// Stores how each gate decomposes, indexed by `GateId` for O(1) lookup.
#[derive(Clone)]
pub struct DecompositionRegistry {
    /// Decomposition entries indexed by `GateId`.
    /// Core gates (0-255) use the first 256 slots.
    entries: Vec<Option<DecompEntry>>,
}

impl Default for DecompositionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DecompositionRegistry {
    /// Create a new registry with core gate decompositions.
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self {
            entries: vec![None; 256],
        };
        registry.init_core_decompositions();
        registry
    }

    /// Initialize decompositions for core gates.
    fn init_core_decompositions(&mut self) {
        // Native gates (no decomposition needed)
        for &gate in &[
            gates::I,
            gates::X,
            gates::Y,
            gates::Z,
            gates::H,
            gates::SX,
            gates::SXdg,
            gates::SY,
            gates::SYdg,
            gates::SZ,
            gates::SZdg,
            gates::T,
            gates::Tdg,
            gates::RX,
            gates::RY,
            gates::RZ,
            gates::CX,
            gates::CY,
            gates::CZ,
        ] {
            self.register_native(gate);
        }

        // SWAP = 3 CX gates
        self.register(
            gates::SWAP,
            GateSupportSet::from_iter([gates::CX]),
            Decomposition::SwapViaCx,
        );

        // iSWAP decomposition
        self.register(
            gates::ISWAP,
            GateSupportSet::from_iter([gates::SZ, gates::H, gates::CX]),
            Decomposition::ISwapDecomp,
        );
    }

    /// Ensure capacity for a gate ID.
    fn ensure_capacity(&mut self, gate: GateId) {
        let idx = gate.0 as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, None);
        }
    }

    /// Register a gate as native (no decomposition).
    pub fn register_native(&mut self, gate: GateId) {
        self.ensure_capacity(gate);
        self.entries[gate.0 as usize] = Some(DecompEntry {
            requires: GateSupportSet::new(),
            decomposition: Decomposition::Native,
        });
    }

    /// Register a gate with a decomposition.
    pub fn register(
        &mut self,
        gate: GateId,
        requires: GateSupportSet,
        decomposition: Decomposition,
    ) {
        self.ensure_capacity(gate);
        self.entries[gate.0 as usize] = Some(DecompEntry {
            requires,
            decomposition,
        });
    }

    /// Register a gate with a dynamic decomposition.
    pub fn register_dynamic(&mut self, gate: GateId, requires: GateSupportSet, ops: Vec<DecompOp>) {
        self.register(gate, requires, Decomposition::Dynamic(Arc::new(ops)));
    }

    /// Get the decomposition entry for a gate.
    #[inline]
    #[must_use]
    pub fn get(&self, gate: GateId) -> Option<&DecompEntry> {
        self.entries.get(gate.0 as usize).and_then(|e| e.as_ref())
    }

    /// Check if a gate is registered.
    #[inline]
    #[must_use]
    pub fn contains(&self, gate: GateId) -> bool {
        self.get(gate).is_some()
    }

    /// Check if a gate is native (no decomposition needed).
    #[inline]
    #[must_use]
    pub fn is_native(&self, gate: GateId) -> bool {
        self.get(gate).is_some_and(|e| e.decomposition.is_native())
    }

    /// Check if a simulator can execute a gate.
    ///
    /// Returns true if either:
    /// 1. The simulator natively supports the gate, OR
    /// 2. All gates in the decomposition chain can ultimately be resolved
    ///    to gates the simulator supports
    ///
    /// This method performs recursive resolution, so gates can list their
    /// immediate dependencies in `requires`, not just ultimate native gates.
    #[must_use]
    pub fn can_execute(&self, gate: GateId, simulator_support: &GateSupportSet) -> bool {
        let mut visited = GateSupportSet::new();
        self.can_execute_recursive(gate, simulator_support, &mut visited)
    }

    /// Recursive helper for `can_execute` with cycle detection.
    fn can_execute_recursive(
        &self,
        gate: GateId,
        simulator_support: &GateSupportSet,
        visited: &mut GateSupportSet,
    ) -> bool {
        // If simulator natively supports it, great
        if simulator_support.contains(gate) {
            return true;
        }

        // Cycle detection: if we've already visited this gate, we have a cycle
        if visited.contains(gate) {
            return false;
        }
        visited.insert(gate);

        // Check if we can decompose
        if let Some(entry) = self.get(gate) {
            if entry.decomposition.is_native() {
                // Native gate not supported by simulator
                false
            } else {
                // Recursively check each required gate
                entry
                    .requires
                    .iter()
                    .all(|req| self.can_execute_recursive(req, simulator_support, visited))
            }
        } else {
            // Unknown gate
            false
        }
    }

    /// Resolve how to execute a gate on a simulator.
    ///
    /// Returns `Native` if simulator supports it directly,
    /// `Decompose` if we need to decompose, or an error.
    ///
    /// This method performs recursive resolution, so gates can list their
    /// immediate dependencies in `requires`, not just ultimate native gates.
    ///
    /// # Errors
    /// Returns `ResolutionError` if the gate has no decomposition or a cycle is detected.
    pub fn resolve(
        &self,
        gate: GateId,
        simulator_support: &GateSupportSet,
    ) -> Result<Resolution, ResolutionError> {
        let mut visited = GateSupportSet::new();
        self.resolve_recursive(gate, simulator_support, &mut visited)
    }

    /// Recursive helper for `resolve` with cycle detection.
    fn resolve_recursive(
        &self,
        gate: GateId,
        simulator_support: &GateSupportSet,
        visited: &mut GateSupportSet,
    ) -> Result<Resolution, ResolutionError> {
        // If simulator natively supports it, use native
        if simulator_support.contains(gate) {
            return Ok(Resolution::Native);
        }

        // Cycle detection
        if visited.contains(gate) {
            return Err(ResolutionError::CircularDependency(gate));
        }
        visited.insert(gate);

        // Get decomposition info
        let entry = self.get(gate).ok_or(ResolutionError::UnknownGate(gate))?;

        if entry.decomposition.is_native() {
            // This is a native gate but simulator doesn't support it
            Err(ResolutionError::UnsupportedNativeGate(gate))
        } else {
            // Recursively check each required gate can be resolved
            for req in entry.requires.iter() {
                self.resolve_recursive(req, simulator_support, visited)?;
            }
            // All requirements can be resolved, so we can decompose this gate
            Ok(Resolution::Decompose(gate))
        }
    }
}

/// Result of resolving how to execute a gate.
#[derive(Clone, Debug)]
pub enum Resolution {
    /// Execute the gate natively.
    Native,
    /// Decompose the gate (look up decomposition in registry).
    Decompose(GateId),
}

/// Error during gate resolution.
#[derive(Clone, Debug)]
pub enum ResolutionError {
    /// Gate is not registered in the decomposition registry.
    UnknownGate(GateId),
    /// Gate is marked as native but simulator doesn't support it.
    UnsupportedNativeGate(GateId),
    /// Simulator is missing some gates required for decomposition.
    MissingRequirements {
        gate: GateId,
        missing: GateSupportSet,
    },
    /// Circular dependency detected in decomposition chain.
    CircularDependency(GateId),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownGate(gate) => write!(f, "Unknown gate: {gate:?}"),
            Self::UnsupportedNativeGate(gate) => {
                write!(f, "Native gate {gate:?} not supported by simulator")
            }
            Self::MissingRequirements { gate, missing } => {
                write!(
                    f,
                    "Cannot decompose {gate:?}: missing {} required gates",
                    missing.len()
                )
            }
            Self::CircularDependency(gate) => {
                write!(f, "Circular dependency detected at gate {gate:?}")
            }
        }
    }
}

impl std::error::Error for ResolutionError {}

/// A resolved circuit where all gates are native to the target simulator.
///
/// This is the result of running `CircuitResolver::resolve` on an `AdaptedSequence`.
#[derive(Clone, Debug)]
pub struct ResolvedCircuit {
    /// The sequence of resolved operations.
    pub ops: Vec<ResolvedOp>,
    /// Number of result slots used.
    pub result_count: usize,
}

impl ResolvedCircuit {
    /// Create a new resolved circuit.
    #[must_use]
    pub fn new(ops: Vec<ResolvedOp>) -> Self {
        let result_count = Self::count_results(&ops);
        Self { ops, result_count }
    }

    /// Get the number of operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if the circuit is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    fn count_results(ops: &[ResolvedOp]) -> usize {
        let mut max_id = 0usize;
        for op in ops {
            match op {
                ResolvedOp::Measure { result, .. } | ResolvedOp::OutputResult { result } => {
                    max_id = max_id.max(result.0 as usize + 1);
                }
                ResolvedOp::Conditional {
                    if_one, if_zero, ..
                } => {
                    max_id = max_id.max(Self::count_results(if_one));
                    max_id = max_id.max(Self::count_results(if_zero));
                }
                ResolvedOp::XorResult { target, source } => {
                    max_id = max_id.max(target.0 as usize + 1);
                    max_id = max_id.max(source.0 as usize + 1);
                }
                _ => {}
            }
        }
        max_id
    }
}

/// A resolved operation where all gates are native.
#[derive(Clone, Debug)]
pub enum ResolvedOp {
    /// A native quantum gate.
    Gate {
        gate_id: GateId,
        qubits: SmallVec<[QubitId; 4]>,
        angles: SmallVec<[Angle64; 3]>,
    },

    /// Prepare a qubit in a specific basis.
    Prep {
        qubit: QubitId,
        basis: super::PrepBasis,
    },

    /// Measure a qubit.
    Measure {
        qubit: QubitId,
        basis: super::MeasBasis,
        result: super::ResultId,
    },

    /// Conditional operations.
    Conditional {
        condition: super::ResultId,
        if_one: Vec<ResolvedOp>,
        if_zero: Vec<ResolvedOp>,
    },

    /// XOR result operation.
    XorResult {
        target: super::ResultId,
        source: super::ResultId,
    },

    /// Output a result.
    OutputResult { result: super::ResultId },
}

impl ResolvedOp {
    /// Create a single-qubit gate.
    #[must_use]
    pub fn gate1(gate_id: GateId, qubit: QubitId) -> Self {
        Self::Gate {
            gate_id,
            qubits: smallvec::smallvec![qubit],
            angles: SmallVec::new(),
        }
    }

    /// Create a two-qubit gate.
    #[must_use]
    pub fn gate2(gate_id: GateId, q0: QubitId, q1: QubitId) -> Self {
        Self::Gate {
            gate_id,
            qubits: smallvec::smallvec![q0, q1],
            angles: SmallVec::new(),
        }
    }

    /// Create a rotation gate.
    #[must_use]
    pub fn rotation(gate_id: GateId, qubit: QubitId, angle: Angle64) -> Self {
        Self::Gate {
            gate_id,
            qubits: smallvec::smallvec![qubit],
            angles: smallvec::smallvec![angle],
        }
    }
}

/// Resolves circuits by expanding decompositions for a target simulator.
///
/// This performs build-time resolution, ensuring all gates in the output
/// are natively supported by the target simulator.
pub struct CircuitResolver<'a> {
    registry: &'a DecompositionRegistry,
    simulator_support: &'a GateSupportSet,
}

impl<'a> CircuitResolver<'a> {
    /// Create a new resolver for a target simulator.
    #[must_use]
    pub fn new(registry: &'a DecompositionRegistry, simulator_support: &'a GateSupportSet) -> Self {
        Self {
            registry,
            simulator_support,
        }
    }

    /// Resolve an adapted sequence to native gates.
    ///
    /// # Errors
    ///
    /// Returns an error if a gate cannot be resolved (unknown gate,
    /// unsupported native gate, or missing decomposition requirements).
    pub fn resolve(
        &self,
        sequence: &super::AdaptedSequence,
    ) -> Result<ResolvedCircuit, ResolutionError> {
        let mut resolved = Vec::with_capacity(sequence.ops.len() * 2);
        for op in &sequence.ops {
            self.resolve_op(op, &mut resolved)?;
        }
        Ok(ResolvedCircuit::new(resolved))
    }

    fn resolve_op(
        &self,
        op: &super::AdaptedOp,
        out: &mut Vec<ResolvedOp>,
    ) -> Result<(), ResolutionError> {
        use super::AdaptedOp;

        match op {
            AdaptedOp::Gate {
                gate_id,
                qubits,
                angles,
            } => {
                if self.simulator_support.contains(*gate_id) {
                    // Native support
                    out.push(ResolvedOp::Gate {
                        gate_id: *gate_id,
                        qubits: qubits.clone(),
                        angles: angles.clone(),
                    });
                } else {
                    // Need to decompose
                    self.expand_gate(*gate_id, qubits, angles, out)?;
                }
            }

            AdaptedOp::Prep { qubit, basis } => {
                out.push(ResolvedOp::Prep {
                    qubit: *qubit,
                    basis: *basis,
                });
            }

            AdaptedOp::Measure {
                qubit,
                basis,
                result,
            } => {
                out.push(ResolvedOp::Measure {
                    qubit: *qubit,
                    basis: *basis,
                    result: *result,
                });
            }

            AdaptedOp::Conditional(cond) => {
                let mut resolved_one = Vec::with_capacity(cond.if_one.len());
                let mut resolved_zero = Vec::with_capacity(cond.if_zero.len());
                for op in &cond.if_one {
                    self.resolve_op(op, &mut resolved_one)?;
                }
                for op in &cond.if_zero {
                    self.resolve_op(op, &mut resolved_zero)?;
                }
                out.push(ResolvedOp::Conditional {
                    condition: cond.condition,
                    if_one: resolved_one,
                    if_zero: resolved_zero,
                });
            }

            AdaptedOp::XorResult { target, source } => {
                out.push(ResolvedOp::XorResult {
                    target: *target,
                    source: *source,
                });
            }

            AdaptedOp::OutputResult { result } => {
                out.push(ResolvedOp::OutputResult { result: *result });
            }
        }

        Ok(())
    }

    fn expand_gate(
        &self,
        gate_id: GateId,
        qubits: &SmallVec<[QubitId; 4]>,
        angles: &SmallVec<[Angle64; 3]>,
        out: &mut Vec<ResolvedOp>,
    ) -> Result<(), ResolutionError> {
        let entry = self
            .registry
            .get(gate_id)
            .ok_or(ResolutionError::UnknownGate(gate_id))?;

        if entry.decomposition.is_native() {
            return Err(ResolutionError::UnsupportedNativeGate(gate_id));
        }

        // Note: We don't check entry.requires here because the recursive expansion
        // below handles gates that need further decomposition. This enables chaining
        // like DOUBLE_SWAP → SWAP → CX where DOUBLE_SWAP requires SWAP (not CX directly).

        // Expand the decomposition
        for decomp_op in entry.decomposition.expand() {
            let instantiated = decomp_op.instantiate(qubits, angles);

            // Recursively resolve (in case decomposition uses non-native gates)
            if self.simulator_support.contains(instantiated.gate) {
                out.push(ResolvedOp::Gate {
                    gate_id: instantiated.gate,
                    qubits: instantiated.qubits,
                    angles: instantiated.angles.iter().copied().collect(),
                });
            } else {
                // Need further decomposition
                self.expand_gate(
                    instantiated.gate,
                    &instantiated.qubits,
                    &instantiated.angles.iter().copied().collect(),
                    out,
                )?;
            }
        }

        Ok(())
    }
}

/// Macro for creating a `GateSupportSet` from gate constants.
#[macro_export]
macro_rules! support_set {
    ($($gate:expr),* $(,)?) => {{
        let mut set = $crate::GateSupportSet::new();
        $(set.insert($gate);)*
        set
    }};
}

/// Macro for creating a `GateSupportSet` (alias for `support_set!`).
#[macro_export]
macro_rules! requires {
    ($($gate:expr),* $(,)?) => {
        $crate::support_set!($($gate),*)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_registry_new() {
        let registry = DecompositionRegistry::new();

        // Core gates should be registered
        assert!(registry.contains(gates::H));
        assert!(registry.contains(gates::CX));
        assert!(registry.contains(gates::SWAP));

        // Native gates should be marked as native
        assert!(registry.is_native(gates::H));
        assert!(registry.is_native(gates::CX));

        // SWAP should not be native (has decomposition)
        assert!(!registry.is_native(gates::SWAP));
    }

    #[test]
    fn test_can_execute_native() {
        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);

        // Simulator supports H and CX natively
        assert!(registry.can_execute(gates::H, &sim_support));
        assert!(registry.can_execute(gates::CX, &sim_support));

        // Simulator can also do SWAP via decomposition (needs CX)
        assert!(registry.can_execute(gates::SWAP, &sim_support));
    }

    #[test]
    fn test_can_execute_via_decomposition() {
        let registry = DecompositionRegistry::new();

        // Simulator only supports CX, not SWAP
        let sim_support = GateSupportSet::from_iter([gates::CX]);

        // SWAP needs CX, which is supported
        assert!(registry.can_execute(gates::SWAP, &sim_support));
    }

    #[test]
    fn test_cannot_execute_missing_requirements() {
        let registry = DecompositionRegistry::new();

        // Simulator supports nothing
        let sim_support = GateSupportSet::new();

        // Can't do SWAP without CX
        assert!(!registry.can_execute(gates::SWAP, &sim_support));
    }

    #[test]
    fn test_resolve_native() {
        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::H, gates::CX, gates::SWAP]);

        // If simulator supports SWAP natively, use native
        let resolution = registry.resolve(gates::SWAP, &sim_support).unwrap();
        assert!(matches!(resolution, Resolution::Native));
    }

    #[test]
    fn test_resolve_decompose() {
        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);

        // Simulator doesn't support SWAP natively, but can decompose
        let resolution = registry.resolve(gates::SWAP, &sim_support).unwrap();
        assert!(matches!(resolution, Resolution::Decompose(_)));
    }

    #[test]
    fn test_resolve_error_unsupported_native_in_chain() {
        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::new();

        // SWAP decomposes to CX, CX is native but not supported
        // With recursive resolution, this results in UnsupportedNativeGate
        let result = registry.resolve(gates::SWAP, &sim_support);
        assert!(
            matches!(result, Err(ResolutionError::UnsupportedNativeGate(g)) if g == gates::CX),
            "Expected UnsupportedNativeGate(CX), got {result:?}"
        );
    }

    #[test]
    fn test_decomp_op_instantiate() {
        let op = DecompOp::gate2(gates::CX, 0, 1);
        let qubits = [QubitId(5), QubitId(10)];

        let instantiated = op.instantiate(&qubits, &[]);

        assert_eq!(instantiated.gate, gates::CX);
        assert_eq!(instantiated.qubits[0], QubitId(5));
        assert_eq!(instantiated.qubits[1], QubitId(10));
    }

    #[test]
    fn test_decomp_op_with_angles() {
        let op = DecompOp::rotation(gates::RZ, 0, 0);
        let qubits = [QubitId(3)];
        let angles = [Angle64::QUARTER_TURN];

        let instantiated = op.instantiate(&qubits, &angles);

        assert_eq!(instantiated.gate, gates::RZ);
        assert_eq!(instantiated.qubits[0], QubitId(3));
        assert_eq!(instantiated.angles[0], Angle64::QUARTER_TURN);
    }

    #[test]
    fn test_swap_decomposition_length() {
        assert_eq!(Decomposition::SwapViaCx.len(), 3);
    }

    #[test]
    fn test_support_set_macro() {
        let set = support_set![gates::H, gates::CX, gates::T];

        assert!(set.contains(gates::H));
        assert!(set.contains(gates::CX));
        assert!(set.contains(gates::T));
        assert!(!set.contains(gates::SWAP));
    }

    #[test]
    fn test_register_dynamic() {
        let mut registry = DecompositionRegistry::new();

        // Register a custom gate with dynamic decomposition
        let custom_gate = GateId(256);
        let ops = vec![
            DecompOp::gate1(gates::H, 0),
            DecompOp::gate2(gates::CX, 0, 1),
            DecompOp::gate1(gates::H, 0),
        ];

        registry.register_dynamic(
            custom_gate,
            GateSupportSet::from_iter([gates::H, gates::CX]),
            ops,
        );

        assert!(registry.contains(custom_gate));
        assert!(!registry.is_native(custom_gate));

        let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);
        assert!(registry.can_execute(custom_gate, &sim_support));
    }

    #[test]
    fn test_swap_iterator_produces_correct_ops() {
        let decomp = Decomposition::SwapViaCx;
        let ops: Vec<_> = decomp.expand().collect();

        assert_eq!(ops.len(), 3);
        // SWAP = CX(0,1); CX(1,0); CX(0,1)
        assert_eq!(ops[0].gate, gates::CX);
        assert_eq!(ops[0].qubit_indices[0], 0);
        assert_eq!(ops[0].qubit_indices[1], 1);

        assert_eq!(ops[1].gate, gates::CX);
        assert_eq!(ops[1].qubit_indices[0], 1);
        assert_eq!(ops[1].qubit_indices[1], 0);

        assert_eq!(ops[2].gate, gates::CX);
        assert_eq!(ops[2].qubit_indices[0], 0);
        assert_eq!(ops[2].qubit_indices[1], 1);
    }

    #[test]
    fn test_iswap_iterator_produces_correct_ops() {
        let decomp = Decomposition::ISwapDecomp;
        let ops: Vec<_> = decomp.expand().collect();

        assert_eq!(ops.len(), 6);
        // iSWAP = S(0); S(1); H(0); CX(0,1); CX(1,0); H(1)
        assert_eq!(ops[0].gate, gates::SZ);
        assert_eq!(ops[1].gate, gates::SZ);
        assert_eq!(ops[2].gate, gates::H);
        assert_eq!(ops[3].gate, gates::CX);
        assert_eq!(ops[4].gate, gates::CX);
        assert_eq!(ops[5].gate, gates::H);
    }

    #[test]
    fn test_angle_source_neg_input() {
        let input_angles = [Angle64::QUARTER_TURN];
        let src = AngleSource::NegInput(0);
        let resolved = src.resolve(&input_angles);

        // Negating quarter turn should give three-quarter turn
        assert_eq!(resolved, -Angle64::QUARTER_TURN);
    }

    #[test]
    fn test_angle_source_half_input() {
        let input_angles = [Angle64::QUARTER_TURN];
        let src = AngleSource::HalfInput(0);
        let resolved = src.resolve(&input_angles);

        // Half of quarter turn is eighth turn
        assert_eq!(resolved, Angle64::QUARTER_TURN / 2_u64);
    }

    #[test]
    fn test_angle_source_fixed() {
        let input_angles: [Angle64; 0] = [];
        let src = AngleSource::Fixed(Angle64::HALF_TURN);
        let resolved = src.resolve(&input_angles);

        assert_eq!(resolved, Angle64::HALF_TURN);
    }

    #[test]
    fn test_iterator_is_reusable() {
        // Verify multiple iterations work (no global state issues)
        let decomp = Decomposition::SwapViaCx;

        let ops1: Vec<_> = decomp.expand().collect();
        let ops2: Vec<_> = decomp.expand().collect();

        assert_eq!(ops1.len(), ops2.len());
        for (op1, op2) in ops1.iter().zip(ops2.iter()) {
            assert_eq!(op1.gate, op2.gate);
        }
    }

    #[test]
    fn test_native_decomposition_empty() {
        let decomp = Decomposition::Native;
        let ops: Vec<_> = decomp.expand().collect();
        assert!(ops.is_empty());
        assert!(decomp.is_native());
        assert!(decomp.is_empty());
    }

    // === CircuitResolver tests ===

    #[test]
    fn test_resolver_native_gate() {
        use super::super::{AdaptedOp, AdaptedSequence};

        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);

        let resolver = CircuitResolver::new(&registry, &sim_support);

        // Circuit with only native gates
        let seq = AdaptedSequence::new(vec![
            AdaptedOp::gate1(gates::H, QubitId(0)),
            AdaptedOp::gate2(gates::CX, QubitId(0), QubitId(1)),
        ]);

        let resolved = resolver.resolve(&seq).unwrap();
        assert_eq!(resolved.len(), 2);

        // Should be unchanged
        match &resolved.ops[0] {
            ResolvedOp::Gate {
                gate_id, qubits, ..
            } => {
                assert_eq!(*gate_id, gates::H);
                assert_eq!(qubits[0], QubitId(0));
            }
            _ => panic!("Expected Gate"),
        }
    }

    #[test]
    fn test_resolver_decompose_swap() {
        use super::super::{AdaptedOp, AdaptedSequence};

        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::CX]);

        let resolver = CircuitResolver::new(&registry, &sim_support);

        // Circuit with SWAP (needs decomposition)
        let seq = AdaptedSequence::new(vec![AdaptedOp::gate2(gates::SWAP, QubitId(0), QubitId(1))]);

        let resolved = resolver.resolve(&seq).unwrap();

        // SWAP should decompose to 3 CX gates
        assert_eq!(resolved.len(), 3);

        for op in &resolved.ops {
            match op {
                ResolvedOp::Gate { gate_id, .. } => {
                    assert_eq!(*gate_id, gates::CX);
                }
                _ => panic!("Expected Gate"),
            }
        }
    }

    #[test]
    fn test_resolver_preserves_prep_meas() {
        use super::super::{AdaptedOp, AdaptedSequence, MeasBasis, PrepBasis, ResultId};

        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::H]);

        let resolver = CircuitResolver::new(&registry, &sim_support);

        let seq = AdaptedSequence::new(vec![
            AdaptedOp::Prep {
                qubit: QubitId(0),
                basis: PrepBasis::Z,
            },
            AdaptedOp::gate1(gates::H, QubitId(0)),
            AdaptedOp::Measure {
                qubit: QubitId(0),
                basis: MeasBasis::Z,
                result: ResultId(0),
            },
        ]);

        let resolved = resolver.resolve(&seq).unwrap();
        assert_eq!(resolved.len(), 3);

        assert!(matches!(resolved.ops[0], ResolvedOp::Prep { .. }));
        assert!(matches!(resolved.ops[1], ResolvedOp::Gate { .. }));
        assert!(matches!(resolved.ops[2], ResolvedOp::Measure { .. }));
    }

    #[test]
    fn test_resolver_handles_conditionals() {
        use super::super::{AdaptedOp, AdaptedSequence, ConditionalOp, ResultId};

        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::X, gates::Z]);

        let resolver = CircuitResolver::new(&registry, &sim_support);

        let seq = AdaptedSequence::new(vec![AdaptedOp::Conditional(Box::new(ConditionalOp {
            condition: ResultId(0),
            if_one: vec![AdaptedOp::gate1(gates::X, QubitId(0))],
            if_zero: vec![AdaptedOp::gate1(gates::Z, QubitId(0))],
        }))]);

        let resolved = resolver.resolve(&seq).unwrap();
        assert_eq!(resolved.len(), 1);

        match &resolved.ops[0] {
            ResolvedOp::Conditional {
                if_one, if_zero, ..
            } => {
                assert_eq!(if_one.len(), 1);
                assert_eq!(if_zero.len(), 1);
            }
            _ => panic!("Expected Conditional"),
        }
    }

    #[test]
    fn test_resolver_error_unsupported() {
        use super::super::{AdaptedOp, AdaptedSequence};

        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::new(); // supports nothing

        let resolver = CircuitResolver::new(&registry, &sim_support);

        // Try to use H which requires native support
        let seq = AdaptedSequence::new(vec![AdaptedOp::gate1(gates::H, QubitId(0))]);

        let result = resolver.resolve(&seq);
        assert!(matches!(
            result,
            Err(ResolutionError::UnsupportedNativeGate(_))
        ));
    }

    #[test]
    fn test_resolver_error_missing_decomp_requirements() {
        use super::super::{AdaptedOp, AdaptedSequence};

        let registry = DecompositionRegistry::new();
        let sim_support = GateSupportSet::from_iter([gates::H]); // has H but not CX

        let resolver = CircuitResolver::new(&registry, &sim_support);

        // Try to use SWAP which decomposes to CX
        // CX is native but not supported, so we get UnsupportedNativeGate
        let seq = AdaptedSequence::new(vec![AdaptedOp::gate2(gates::SWAP, QubitId(0), QubitId(1))]);

        let result = resolver.resolve(&seq);
        assert!(
            matches!(result, Err(ResolutionError::UnsupportedNativeGate(g)) if g == gates::CX),
            "Expected UnsupportedNativeGate(CX), got {result:?}"
        );
    }

    #[test]
    fn test_resolved_circuit_result_count() {
        use super::super::{MeasBasis, ResultId};

        let ops = vec![
            ResolvedOp::Measure {
                qubit: QubitId(0),
                basis: MeasBasis::Z,
                result: ResultId(0),
            },
            ResolvedOp::Measure {
                qubit: QubitId(1),
                basis: MeasBasis::Z,
                result: ResultId(1),
            },
        ];

        let circuit = ResolvedCircuit::new(ops);
        assert_eq!(circuit.result_count, 2);
    }

    #[test]
    fn test_resolved_op_constructors() {
        let h = ResolvedOp::gate1(gates::H, QubitId(0));
        match h {
            ResolvedOp::Gate {
                gate_id, qubits, ..
            } => {
                assert_eq!(gate_id, gates::H);
                assert_eq!(qubits[0], QubitId(0));
            }
            _ => panic!("Expected Gate"),
        }

        let cx = ResolvedOp::gate2(gates::CX, QubitId(0), QubitId(1));
        match cx {
            ResolvedOp::Gate {
                gate_id, qubits, ..
            } => {
                assert_eq!(gate_id, gates::CX);
                assert_eq!(qubits.len(), 2);
            }
            _ => panic!("Expected Gate"),
        }

        let rz = ResolvedOp::rotation(gates::RZ, QubitId(0), Angle64::QUARTER_TURN);
        match rz {
            ResolvedOp::Gate {
                gate_id, angles, ..
            } => {
                assert_eq!(gate_id, gates::RZ);
                assert_eq!(angles[0], Angle64::QUARTER_TURN);
            }
            _ => panic!("Expected Gate"),
        }
    }
}
