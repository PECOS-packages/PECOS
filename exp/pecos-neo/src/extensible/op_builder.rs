//! Builder for constructing arbitrary operation sequences.
//!
//! Provides a fluent API for building `AdaptedSequence` that can include:
//! - Arbitrary unitary gates
//! - Preparations in any basis
//! - Measurements with result tracking
//! - Conditional operations (if statements)
//! - Stabilizer measurements and preparations
//! - Named subcircuits and custom gates
//!
//! # Example
//!
//! ```
//! use pecos_neo::prelude::*;
//!
//! let (q0, q1) = (QubitId(0), QubitId(1));
//!
//! // Fluent chaining
//! let seq = OpBuilder::new()
//!     .pz(q0)
//!     .h(q0)
//!     .cx(q0, q1)
//!     .mz(q0, ResultId(0))
//!     .build();
//! ```

use super::GateId;
use super::gates;
use super::operation::{
    AdaptedOp, AdaptedSequence, AncillaRequirements, ConditionalOp, MeasBasis, PrepBasis, ResultId,
};
use super::pauli::{PauliString, StabilizerMeasurement, StabilizerPreparation};
use crate::command::{CommandQueue, GateCommand, GateType};
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;

/// A named subcircuit that can be called multiple times.
#[derive(Clone)]
pub struct Subcircuit {
    /// The operations in this subcircuit.
    pub ops: Arc<Vec<AdaptedOp>>,
    /// Number of qubits this subcircuit operates on.
    pub qubit_count: usize,
    /// Number of results this subcircuit produces.
    pub result_count: usize,
}

impl Subcircuit {
    /// Create a new subcircuit from an operation sequence.
    #[must_use]
    pub fn new(seq: AdaptedSequence, qubit_count: usize) -> Self {
        Self {
            result_count: seq.result_count,
            ops: Arc::new(seq.ops),
            qubit_count,
        }
    }

    /// Create from an `OpBuilder`.
    #[must_use]
    pub fn from_builder(builder: OpBuilder, qubit_count: usize) -> Self {
        Self::new(builder.build(), qubit_count)
    }
}

/// Registry of named subcircuits/custom gates.
#[derive(Clone, Default)]
pub struct GateLibrary {
    gates: HashMap<String, Subcircuit>,
}

impl GateLibrary {
    /// Create a new empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Define a named gate/subcircuit.
    pub fn define(&mut self, name: impl Into<String>, subcircuit: Subcircuit) {
        self.gates.insert(name.into(), subcircuit);
    }

    /// Get a named gate/subcircuit.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Subcircuit> {
        self.gates.get(name)
    }

    /// Check if a gate is defined.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.gates.contains_key(name)
    }

    /// Define a gate using a builder function.
    ///
    /// The function receives placeholder qubit IDs (0, 1, 2, ...).
    pub fn define_with<F>(&mut self, name: impl Into<String>, qubit_count: usize, f: F)
    where
        F: FnOnce(OpBuilder, &[QubitId]) -> OpBuilder,
    {
        let qubits: Vec<QubitId> = (0..qubit_count).map(QubitId).collect();
        let builder = f(OpBuilder::new(), &qubits);
        self.define(name, Subcircuit::from_builder(builder, qubit_count));
    }
}

/// Builder for constructing operation sequences.
///
/// Uses a consuming builder pattern for fluent chaining:
/// ```
/// # use pecos_neo::prelude::*;
/// # let (q0, q1) = (QubitId(0), QubitId(1));
/// let seq = OpBuilder::new()
///     .pz(q0)
///     .h(q0)
///     .cx(q0, q1)
///     .build();
/// ```
#[derive(Clone, Debug, Default)]
pub struct OpBuilder {
    ops: Vec<AdaptedOp>,
    next_result: u16,
    ancilla_count: usize,
    clean_ancillas: bool,
    ancillas: SmallVec<[QubitId; 4]>,
    next_ancilla_id: usize,
}

impl OpBuilder {
    /// Create a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            next_result: 0,
            ancilla_count: 0,
            clean_ancillas: false,
            ancillas: SmallVec::new(),
            next_ancilla_id: 1000,
        }
    }

    /// Create a builder with a specific starting ancilla ID.
    #[must_use]
    pub fn with_ancilla_start(mut self, start_id: usize) -> Self {
        self.next_ancilla_id = start_id;
        self
    }

    /// Build the operation sequence.
    #[must_use]
    pub fn build(self) -> AdaptedSequence {
        AdaptedSequence::new(self.ops)
    }

    /// Get the current number of operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if the builder is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Allocate a new result ID.
    pub fn alloc_result(&mut self) -> ResultId {
        let id = ResultId(self.next_result);
        self.next_result += 1;
        id
    }

    /// Get ancilla requirements for this sequence.
    #[must_use]
    pub fn ancilla_requirements(&self) -> AncillaRequirements {
        let count = self.ancilla_count.max(self.ancillas.len());
        if count == 0 {
            AncillaRequirements::none()
        } else if self.clean_ancillas {
            AncillaRequirements::clean(count)
        } else {
            AncillaRequirements::dirty(count)
        }
    }

    /// Get all allocated ancillas.
    #[must_use]
    pub fn ancillas(&self) -> &[QubitId] {
        &self.ancillas
    }

    // --- Ancilla management ---

    /// Allocate a clean ancilla qubit.
    pub fn alloc_ancilla(&mut self) -> QubitId {
        let id = QubitId(self.next_ancilla_id);
        self.next_ancilla_id += 1;
        self.ancillas.push(id);
        self.clean_ancillas = true;
        id
    }

    /// Allocate and prepare a fresh ancilla in |0⟩.
    pub fn fresh_ancilla(&mut self) -> QubitId {
        let anc = self.alloc_ancilla();
        self.ops.push(AdaptedOp::pz(anc));
        anc
    }

    // --- Raw operation insertion ---

    /// Add a raw operation.
    #[must_use]
    pub fn op(mut self, op: AdaptedOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Add multiple operations.
    #[must_use]
    pub fn ops(mut self, ops: impl IntoIterator<Item = AdaptedOp>) -> Self {
        self.ops.extend(ops);
        self
    }

    // --- Preparations ---

    /// Prepare a qubit in the Z basis (|0⟩).
    #[must_use]
    pub fn pz(mut self, qubit: QubitId) -> Self {
        self.ops.push(AdaptedOp::pz(qubit));
        self
    }

    /// Prepare a qubit in the X basis (|+⟩).
    #[must_use]
    pub fn px(mut self, qubit: QubitId) -> Self {
        self.ops.push(AdaptedOp::px(qubit));
        self
    }

    /// Prepare a qubit in the Y basis (|+i⟩).
    #[must_use]
    pub fn prep_y(mut self, qubit: QubitId) -> Self {
        self.ops.push(AdaptedOp::Prep {
            qubit,
            basis: PrepBasis::Y,
        });
        self
    }

    /// Prepare a qubit in a specific basis.
    #[must_use]
    pub fn prep(mut self, qubit: QubitId, basis: PrepBasis) -> Self {
        self.ops.push(AdaptedOp::Prep { qubit, basis });
        self
    }

    // --- Measurements ---

    /// Measure a qubit in the Z basis.
    #[must_use]
    pub fn mz(mut self, qubit: QubitId, result: ResultId) -> Self {
        self.ops.push(AdaptedOp::mz(qubit, result));
        self.next_result = self.next_result.max(result.0 + 1);
        self
    }

    /// Measure a qubit in the X basis.
    #[must_use]
    pub fn mx(mut self, qubit: QubitId, result: ResultId) -> Self {
        self.ops.push(AdaptedOp::mx(qubit, result));
        self.next_result = self.next_result.max(result.0 + 1);
        self
    }

    /// Measure a qubit in the Y basis.
    #[must_use]
    pub fn meas_y(mut self, qubit: QubitId, result: ResultId) -> Self {
        self.ops.push(AdaptedOp::Measure {
            qubit,
            basis: MeasBasis::Y,
            result,
        });
        self.next_result = self.next_result.max(result.0 + 1);
        self
    }

    /// Measure a qubit in a specific basis.
    #[must_use]
    pub fn meas(mut self, qubit: QubitId, basis: MeasBasis, result: ResultId) -> Self {
        self.ops.push(AdaptedOp::Measure {
            qubit,
            basis,
            result,
        });
        self.next_result = self.next_result.max(result.0 + 1);
        self
    }

    // --- Single-qubit gates ---

    /// Apply an identity gate.
    #[must_use]
    pub fn i(self, qubit: QubitId) -> Self {
        self.gate1(gates::I, qubit)
    }

    /// Apply a Pauli X gate.
    #[must_use]
    pub fn x(self, qubit: QubitId) -> Self {
        self.gate1(gates::X, qubit)
    }

    /// Apply a Pauli Y gate.
    #[must_use]
    pub fn y(self, qubit: QubitId) -> Self {
        self.gate1(gates::Y, qubit)
    }

    /// Apply a Pauli Z gate.
    #[must_use]
    pub fn z(self, qubit: QubitId) -> Self {
        self.gate1(gates::Z, qubit)
    }

    /// Apply a Hadamard gate.
    #[must_use]
    pub fn h(self, qubit: QubitId) -> Self {
        self.gate1(gates::H, qubit)
    }

    /// Apply an S gate (sqrt-Z).
    #[must_use]
    pub fn s(self, qubit: QubitId) -> Self {
        self.gate1(gates::SZ, qubit)
    }

    /// Apply an S-dagger gate.
    #[must_use]
    pub fn sdg(self, qubit: QubitId) -> Self {
        self.gate1(gates::SZdg, qubit)
    }

    /// Apply a T gate.
    #[must_use]
    pub fn t(self, qubit: QubitId) -> Self {
        self.gate1(gates::T, qubit)
    }

    /// Apply a T-dagger gate.
    #[must_use]
    pub fn tdg(self, qubit: QubitId) -> Self {
        self.gate1(gates::Tdg, qubit)
    }

    /// Apply an SX gate (sqrt-X).
    #[must_use]
    pub fn sx(self, qubit: QubitId) -> Self {
        self.gate1(gates::SX, qubit)
    }

    /// Apply an SX-dagger gate.
    #[must_use]
    pub fn sxdg(self, qubit: QubitId) -> Self {
        self.gate1(gates::SXdg, qubit)
    }

    /// Apply an arbitrary single-qubit gate by `GateId`.
    #[must_use]
    pub fn gate1(mut self, gate_id: GateId, qubit: QubitId) -> Self {
        self.ops.push(AdaptedOp::gate1(gate_id, qubit));
        self
    }

    // --- Single-qubit rotations ---

    /// Apply an RX rotation.
    #[must_use]
    pub fn rx(self, qubit: QubitId, angle: Angle64) -> Self {
        self.rotation(gates::RX, qubit, angle)
    }

    /// Apply an RY rotation.
    #[must_use]
    pub fn ry(self, qubit: QubitId, angle: Angle64) -> Self {
        self.rotation(gates::RY, qubit, angle)
    }

    /// Apply an RZ rotation.
    #[must_use]
    pub fn rz(self, qubit: QubitId, angle: Angle64) -> Self {
        self.rotation(gates::RZ, qubit, angle)
    }

    /// Apply a rotation gate.
    #[must_use]
    pub fn rotation(mut self, gate_id: GateId, qubit: QubitId, angle: Angle64) -> Self {
        self.ops.push(AdaptedOp::rotation(gate_id, qubit, angle));
        self
    }

    // --- Two-qubit gates ---

    /// Apply a CNOT (CX) gate.
    #[must_use]
    pub fn cx(self, control: QubitId, target: QubitId) -> Self {
        self.gate2(gates::CX, control, target)
    }

    /// Apply a CY gate.
    #[must_use]
    pub fn cy(self, control: QubitId, target: QubitId) -> Self {
        self.gate2(gates::CY, control, target)
    }

    /// Apply a CZ gate.
    #[must_use]
    pub fn cz(self, q0: QubitId, q1: QubitId) -> Self {
        self.gate2(gates::CZ, q0, q1)
    }

    /// Apply a SWAP gate.
    #[must_use]
    pub fn swap(self, q0: QubitId, q1: QubitId) -> Self {
        self.gate2(gates::SWAP, q0, q1)
    }

    /// Apply an arbitrary two-qubit gate by `GateId`.
    #[must_use]
    pub fn gate2(mut self, gate_id: GateId, q0: QubitId, q1: QubitId) -> Self {
        self.ops.push(AdaptedOp::gate2(gate_id, q0, q1));
        self
    }

    // --- Multi-qubit gates ---

    /// Apply an arbitrary gate with any number of qubits and angles.
    #[must_use]
    pub fn gate(mut self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> Self {
        self.ops.push(AdaptedOp::Gate {
            gate_id,
            qubits: qubits.iter().copied().collect(),
            angles: angles.iter().copied().collect(),
        });
        self
    }

    // --- Control flow ---

    /// Execute operations conditionally based on a measurement result.
    #[must_use]
    pub fn if_then_else<F1, F2>(
        mut self,
        condition: ResultId,
        if_one_fn: F1,
        if_zero_fn: F2,
    ) -> Self
    where
        F1: FnOnce(OpBuilder) -> OpBuilder,
        F2: FnOnce(OpBuilder) -> OpBuilder,
    {
        let if_one = Self::run_branch(if_one_fn);
        let if_zero = Self::run_branch(if_zero_fn);

        self.ops
            .push(AdaptedOp::Conditional(Box::new(ConditionalOp {
                condition,
                if_one,
                if_zero,
            })));
        self
    }

    /// Run a branch closure and extract the resulting ops.
    ///
    /// Factored out to avoid a miscompilation at opt-level >= 2 where
    /// the partial move of `.ops` from the builder temporary within
    /// `if_then_else` causes a double-free during drop.
    #[inline(never)]
    fn run_branch<F>(f: F) -> Vec<AdaptedOp>
    where
        F: FnOnce(OpBuilder) -> OpBuilder,
    {
        f(OpBuilder::new()).build().ops
    }

    /// Execute operations if measurement result is 1.
    #[must_use]
    pub fn if_one<F>(self, condition: ResultId, f: F) -> Self
    where
        F: FnOnce(OpBuilder) -> OpBuilder,
    {
        self.if_then_else(condition, f, |b| b)
    }

    /// Execute operations if measurement result is 0.
    #[must_use]
    pub fn if_zero<F>(self, condition: ResultId, f: F) -> Self
    where
        F: FnOnce(OpBuilder) -> OpBuilder,
    {
        self.if_then_else(condition, |b| b, f)
    }

    /// Apply X gate if measurement result is 1.
    #[must_use]
    pub fn conditional_x(mut self, condition: ResultId, qubit: QubitId) -> Self {
        self.ops.push(AdaptedOp::conditional_x(condition, qubit));
        self
    }

    /// Apply Z gate if measurement result is 1.
    #[must_use]
    pub fn conditional_z(mut self, condition: ResultId, qubit: QubitId) -> Self {
        self.ops.push(AdaptedOp::conditional_z(condition, qubit));
        self
    }

    // --- Result operations ---

    /// XOR one result into another.
    #[must_use]
    pub fn xor_result(mut self, target: ResultId, source: ResultId) -> Self {
        self.ops.push(AdaptedOp::XorResult { target, source });
        self
    }

    /// Mark a result as the output of this operation.
    #[must_use]
    pub fn output(mut self, result: ResultId) -> Self {
        self.ops.push(AdaptedOp::OutputResult { result });
        self
    }

    // --- Subcircuits and custom gates ---

    /// Inline a subcircuit with qubit remapping.
    ///
    /// The mapping is from subcircuit qubit IDs to actual qubit IDs.
    #[must_use]
    pub fn subcircuit(mut self, sub: &Subcircuit, qubit_map: &[(QubitId, QubitId)]) -> Self {
        let map: HashMap<QubitId, QubitId> = qubit_map.iter().copied().collect();

        for op in sub.ops.iter() {
            self.ops.push(remap_op(op, &map));
        }
        self
    }

    /// Inline a subcircuit using positional qubit arguments.
    ///
    /// Maps subcircuit qubits 0, 1, 2, ... to the provided qubits in order.
    ///
    /// # Panics
    /// Panics if `qubits.len()` does not match the subcircuit's qubit count.
    #[must_use]
    pub fn call(mut self, sub: &Subcircuit, qubits: &[QubitId]) -> Self {
        assert_eq!(
            qubits.len(),
            sub.qubit_count,
            "Subcircuit expects {} qubits, got {}",
            sub.qubit_count,
            qubits.len()
        );

        let map: HashMap<QubitId, QubitId> = (0..sub.qubit_count)
            .map(|i| (QubitId(i), qubits[i]))
            .collect();

        for op in sub.ops.iter() {
            self.ops.push(remap_op(op, &map));
        }
        self
    }

    /// Call a named gate from a library.
    ///
    /// # Panics
    /// Panics if the named gate is not found in the library.
    #[must_use]
    pub fn call_named(self, library: &GateLibrary, name: &str, qubits: &[QubitId]) -> Self {
        let sub = library
            .get(name)
            .unwrap_or_else(|| panic!("Unknown gate: {name}"));
        self.call(sub, qubits)
    }

    // --- Stabilizer operations ---

    /// Measure an arbitrary stabilizer (Pauli string).
    ///
    /// # Panics
    /// Panics if the Pauli string is invalid.
    #[must_use]
    pub fn stabilizer_meas(
        mut self,
        pauli: &str,
        qubits: &[QubitId],
        ancilla: QubitId,
        result: ResultId,
    ) -> Self {
        let ps = PauliString::from_str(pauli).expect("Invalid Pauli string");
        let meas = StabilizerMeasurement::new(ps);
        let seq = meas.decompose(qubits, ancilla);

        for op in seq.ops {
            match op {
                AdaptedOp::Measure { qubit, basis, .. } => {
                    self.ops.push(AdaptedOp::Measure {
                        qubit,
                        basis,
                        result,
                    });
                }
                AdaptedOp::OutputResult { .. } => {}
                other => self.ops.push(other),
            }
        }

        self.next_result = self.next_result.max(result.0 + 1);
        self.ancilla_count = self.ancilla_count.max(1);
        self.clean_ancillas = true;
        self
    }

    /// Prepare an arbitrary stabilizer eigenstate.
    ///
    /// # Panics
    /// Panics if the Pauli string is invalid.
    #[must_use]
    pub fn stabilizer_prep(mut self, pauli: &str, qubits: &[QubitId]) -> Self {
        let ps = PauliString::from_str(pauli).expect("Invalid Pauli string");
        let prep = StabilizerPreparation::new(ps);
        let seq = prep.decompose(qubits);
        self.ops.extend(seq.ops);
        self
    }

    /// Prepare Bell state (|00⟩ + |11⟩)/√2.
    #[must_use]
    pub fn prep_bell(self, q0: QubitId, q1: QubitId) -> Self {
        self.pz(q0).pz(q1).h(q0).cx(q0, q1)
    }

    /// Prepare GHZ state on multiple qubits.
    #[must_use]
    pub fn prep_ghz(mut self, qubits: &[QubitId]) -> Self {
        if qubits.is_empty() {
            return self;
        }

        for &q in qubits {
            self.ops.push(AdaptedOp::pz(q));
        }
        self.ops.push(AdaptedOp::gate1(gates::H, qubits[0]));
        for i in 1..qubits.len() {
            self.ops
                .push(AdaptedOp::gate2(gates::CX, qubits[0], qubits[i]));
        }
        self
    }

    // --- Conversion to CommandQueue ---

    /// Convert to a `CommandQueue` for execution.
    ///
    /// Note: This flattens the sequence. Conditional operations are not supported
    /// in `CommandQueue` and will cause this to return an error.
    ///
    /// # Errors
    /// Returns `ConversionError` if an unsupported gate or conditional operation is encountered.
    pub fn to_command_queue(&self) -> Result<CommandQueue, ConversionError> {
        let mut queue = CommandQueue::with_capacity(self.ops.len());

        for (idx, op) in self.ops.iter().enumerate() {
            match op {
                AdaptedOp::Gate {
                    gate_id,
                    qubits,
                    angles,
                } => {
                    let gate_type = gate_id.try_to_gate_type().ok_or({
                        ConversionError::UnsupportedGate {
                            gate_id: *gate_id,
                            position: idx,
                        }
                    })?;

                    let cmd = if angles.is_empty() {
                        GateCommand::new(gate_type, qubits.clone())
                    } else {
                        // Convert SmallVec<[Angle64; 3]> to SmallVec<[Angle64; 2]>
                        let angles2: SmallVec<[Angle64; 2]> = angles.iter().copied().collect();
                        GateCommand::with_angles(gate_type, qubits.clone(), angles2)
                    };
                    queue.push(cmd);
                }

                AdaptedOp::Prep { qubit, basis } => {
                    // Prep in Z is native; other bases need rotation after
                    queue.push(GateCommand::pz(*qubit));
                    match basis {
                        PrepBasis::Z => {}
                        PrepBasis::X => queue.push(GateCommand::h(*qubit)),
                        PrepBasis::Y => {
                            queue.push(GateCommand::h(*qubit));
                            queue.push(GateCommand::sz(*qubit));
                        }
                    }
                }

                AdaptedOp::Measure { qubit, basis, .. } => {
                    // Rotate to Z basis, measure, rotate back
                    match basis {
                        MeasBasis::Z => {}
                        MeasBasis::X => queue.push(GateCommand::h(*qubit)),
                        MeasBasis::Y => {
                            queue.push(GateCommand::new(
                                GateType::SZdg,
                                smallvec::smallvec![*qubit],
                            ));
                            queue.push(GateCommand::h(*qubit));
                        }
                    }
                    queue.push(GateCommand::mz(*qubit));
                    // Rotate back (for non-destructive measurement)
                    match basis {
                        MeasBasis::Z => {}
                        MeasBasis::X => queue.push(GateCommand::h(*qubit)),
                        MeasBasis::Y => {
                            queue.push(GateCommand::h(*qubit));
                            queue.push(GateCommand::sz(*qubit));
                        }
                    }
                }

                AdaptedOp::Conditional(_) => {
                    return Err(ConversionError::ConditionalNotSupported { position: idx });
                }

                AdaptedOp::XorResult { .. } | AdaptedOp::OutputResult { .. } => {
                    // Classical operations - skip in CommandQueue
                }
            }
        }

        Ok(queue)
    }
}

/// Error during conversion to `CommandQueue`.
#[derive(Clone, Debug)]
pub enum ConversionError {
    /// A gate ID has no corresponding `GateType`.
    UnsupportedGate { gate_id: GateId, position: usize },
    /// Conditional operations are not supported in `CommandQueue`.
    ConditionalNotSupported { position: usize },
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedGate { gate_id, position } => {
                write!(
                    f,
                    "Gate ID {} at position {} has no GateType equivalent",
                    gate_id.0, position
                )
            }
            Self::ConditionalNotSupported { position } => {
                write!(
                    f,
                    "Conditional operation at position {position} not supported in CommandQueue"
                )
            }
        }
    }
}

impl std::error::Error for ConversionError {}

/// Remap qubit IDs in an operation.
fn remap_op(op: &AdaptedOp, map: &HashMap<QubitId, QubitId>) -> AdaptedOp {
    let remap = |q: &QubitId| *map.get(q).unwrap_or(q);
    let remap_vec =
        |qs: &SmallVec<[QubitId; 4]>| -> SmallVec<[QubitId; 4]> { qs.iter().map(remap).collect() };

    match op {
        AdaptedOp::Gate {
            gate_id,
            qubits,
            angles,
        } => AdaptedOp::Gate {
            gate_id: *gate_id,
            qubits: remap_vec(qubits),
            angles: angles.clone(),
        },
        AdaptedOp::Prep { qubit, basis } => AdaptedOp::Prep {
            qubit: remap(qubit),
            basis: *basis,
        },
        AdaptedOp::Measure {
            qubit,
            basis,
            result,
        } => AdaptedOp::Measure {
            qubit: remap(qubit),
            basis: *basis,
            result: *result,
        },
        AdaptedOp::Conditional(cond) => AdaptedOp::Conditional(Box::new(ConditionalOp {
            condition: cond.condition,
            if_one: cond.if_one.iter().map(|o| remap_op(o, map)).collect(),
            if_zero: cond.if_zero.iter().map(|o| remap_op(o, map)).collect(),
        })),
        AdaptedOp::XorResult { target, source } => AdaptedOp::XorResult {
            target: *target,
            source: *source,
        },
        AdaptedOp::OutputResult { result } => AdaptedOp::OutputResult { result: *result },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluent_chaining() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = OpBuilder::new()
            .pz(q0)
            .pz(q1)
            .h(q0)
            .cx(q0, q1)
            .mz(q0, ResultId(0))
            .mz(q1, ResultId(1))
            .build();

        assert_eq!(seq.ops.len(), 6);
        assert_eq!(seq.result_count, 2);
    }

    #[test]
    fn test_conditional_chaining() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = OpBuilder::new()
            .pz(q0)
            .h(q0)
            .mz(q0, ResultId(0))
            .if_one(ResultId(0), |b| b.x(q1))
            .build();

        assert_eq!(seq.ops.len(), 4);
    }

    #[test]
    fn test_subcircuit() {
        // Define a Bell state subcircuit
        let bell = Subcircuit::from_builder(
            OpBuilder::new()
                .pz(QubitId(0))
                .pz(QubitId(1))
                .h(QubitId(0))
                .cx(QubitId(0), QubitId(1)),
            2,
        );

        // Use it with different qubits
        let q2 = QubitId(2);
        let q3 = QubitId(3);

        let seq = OpBuilder::new()
            .call(&bell, &[q2, q3])
            .mz(q2, ResultId(0))
            .build();

        // 4 ops from bell + 1 measurement
        assert_eq!(seq.ops.len(), 5);

        // Check that qubits were remapped
        match &seq.ops[0] {
            AdaptedOp::Prep { qubit, .. } => assert_eq!(*qubit, q2),
            _ => panic!("Expected Prep"),
        }
    }

    #[test]
    fn test_gate_library() {
        let mut lib = GateLibrary::new();

        lib.define_with("bell", 2, |b, qs| {
            b.pz(qs[0]).pz(qs[1]).h(qs[0]).cx(qs[0], qs[1])
        });

        let q0 = QubitId(10);
        let q1 = QubitId(11);

        let seq = OpBuilder::new()
            .call_named(&lib, "bell", &[q0, q1])
            .mz(q0, ResultId(0))
            .build();

        assert_eq!(seq.ops.len(), 5);
    }

    #[test]
    fn test_to_command_queue() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let builder = OpBuilder::new()
            .pz(q0)
            .pz(q1)
            .h(q0)
            .cx(q0, q1)
            .mz(q0, ResultId(0));

        let queue = builder.to_command_queue().unwrap();

        // prep, prep, h, cx, meas
        assert_eq!(queue.len(), 5);
    }

    #[test]
    fn test_to_command_queue_conditional_error() {
        let q0 = QubitId(0);

        let builder = OpBuilder::new()
            .pz(q0)
            .mz(q0, ResultId(0))
            .if_one(ResultId(0), |b| b.x(q0));

        let result = builder.to_command_queue();
        assert!(result.is_err());
    }

    #[test]
    fn test_teleportation() {
        let (msg, alice, bob) = (QubitId(0), QubitId(1), QubitId(2));

        let seq = OpBuilder::new()
            // Bell pair between Alice and Bob
            .prep_bell(alice, bob)
            // Teleportation
            .cx(msg, alice)
            .h(msg)
            .mz(msg, ResultId(0))
            .mz(alice, ResultId(1))
            // Corrections
            .conditional_x(ResultId(1), bob)
            .conditional_z(ResultId(0), bob)
            .build();

        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_ghz_state() {
        let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];

        let seq = OpBuilder::new().prep_ghz(&qubits).build();

        // 4 preps + 1 H + 3 CX = 8 ops
        assert_eq!(seq.ops.len(), 8);
    }

    #[test]
    fn test_stabilizer_meas_chaining() {
        let qubits = [QubitId(0), QubitId(1)];
        let anc = QubitId(10);

        let seq = OpBuilder::new()
            .stabilizer_meas("ZZ", &qubits, anc, ResultId(0))
            .if_one(ResultId(0), |b| b.x(qubits[0]))
            .build();

        assert!(!seq.ops.is_empty());
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating random qubit IDs (0-15 range for realistic circuits)
    fn qubit_id() -> impl Strategy<Value = QubitId> {
        (0usize..16).prop_map(QubitId)
    }

    /// Strategy for generating a pair of distinct qubit IDs
    fn qubit_pair() -> impl Strategy<Value = (QubitId, QubitId)> {
        (0usize..16, 0usize..16)
            .prop_filter("qubits must be distinct", |(a, b)| a != b)
            .prop_map(|(a, b)| (QubitId(a), QubitId(b)))
    }

    /// Strategy for generating random angles
    fn angle() -> impl Strategy<Value = Angle64> {
        prop_oneof![
            Just(Angle64::ZERO),
            Just(Angle64::QUARTER_TURN),
            Just(Angle64::HALF_TURN),
            Just(Angle64::THREE_QUARTERS_TURN),
            any::<u64>().prop_map(Angle64::new),
        ]
    }

    /// Enum representing a random operation to apply
    #[derive(Clone, Debug)]
    enum RandomOp {
        // Preparations
        PrepZ(QubitId),
        PrepX(QubitId),
        PrepY(QubitId),
        // Single-qubit gates
        H(QubitId),
        X(QubitId),
        Y(QubitId),
        Z(QubitId),
        S(QubitId),
        Sdg(QubitId),
        T(QubitId),
        Tdg(QubitId),
        Sx(QubitId),
        Sxdg(QubitId),
        // Rotations
        Rx(QubitId, Angle64),
        Ry(QubitId, Angle64),
        Rz(QubitId, Angle64),
        // Two-qubit gates
        Cx(QubitId, QubitId),
        Cy(QubitId, QubitId),
        Cz(QubitId, QubitId),
        Swap(QubitId, QubitId),
    }

    impl RandomOp {
        /// Apply this operation to a builder
        fn apply(self, builder: OpBuilder) -> OpBuilder {
            match self {
                Self::PrepZ(q) => builder.pz(q),
                Self::PrepX(q) => builder.px(q),
                Self::PrepY(q) => builder.prep_y(q),
                Self::H(q) => builder.h(q),
                Self::X(q) => builder.x(q),
                Self::Y(q) => builder.y(q),
                Self::Z(q) => builder.z(q),
                Self::S(q) => builder.s(q),
                Self::Sdg(q) => builder.sdg(q),
                Self::T(q) => builder.t(q),
                Self::Tdg(q) => builder.tdg(q),
                Self::Sx(q) => builder.sx(q),
                Self::Sxdg(q) => builder.sxdg(q),
                Self::Rx(q, a) => builder.rx(q, a),
                Self::Ry(q, a) => builder.ry(q, a),
                Self::Rz(q, a) => builder.rz(q, a),
                Self::Cx(c, t) => builder.cx(c, t),
                Self::Cy(c, t) => builder.cy(c, t),
                Self::Cz(q0, q1) => builder.cz(q0, q1),
                Self::Swap(q0, q1) => builder.swap(q0, q1),
            }
        }
    }

    /// Strategy for generating a random operation
    fn random_op() -> impl Strategy<Value = RandomOp> {
        prop_oneof![
            // Preparations (weight 3)
            qubit_id().prop_map(RandomOp::PrepZ),
            qubit_id().prop_map(RandomOp::PrepX),
            qubit_id().prop_map(RandomOp::PrepY),
            // Single-qubit gates (weight 12)
            qubit_id().prop_map(RandomOp::H),
            qubit_id().prop_map(RandomOp::X),
            qubit_id().prop_map(RandomOp::Y),
            qubit_id().prop_map(RandomOp::Z),
            qubit_id().prop_map(RandomOp::S),
            qubit_id().prop_map(RandomOp::Sdg),
            qubit_id().prop_map(RandomOp::T),
            qubit_id().prop_map(RandomOp::Tdg),
            qubit_id().prop_map(RandomOp::Sx),
            qubit_id().prop_map(RandomOp::Sxdg),
            // Rotations (weight 3)
            (qubit_id(), angle()).prop_map(|(q, a)| RandomOp::Rx(q, a)),
            (qubit_id(), angle()).prop_map(|(q, a)| RandomOp::Ry(q, a)),
            (qubit_id(), angle()).prop_map(|(q, a)| RandomOp::Rz(q, a)),
            // Two-qubit gates (weight 4)
            qubit_pair().prop_map(|(c, t)| RandomOp::Cx(c, t)),
            qubit_pair().prop_map(|(c, t)| RandomOp::Cy(c, t)),
            qubit_pair().prop_map(|(q0, q1)| RandomOp::Cz(q0, q1)),
            qubit_pair().prop_map(|(q0, q1)| RandomOp::Swap(q0, q1)),
        ]
    }

    /// Strategy for generating a sequence of random operations
    fn random_ops(max_len: usize) -> impl Strategy<Value = Vec<RandomOp>> {
        proptest::collection::vec(random_op(), 0..=max_len)
    }

    proptest! {
        /// Building random sequences never panics
        #[test]
        fn building_random_circuits_doesnt_panic(ops in random_ops(50)) {
            let mut builder = OpBuilder::new();
            for op in ops {
                builder = op.apply(builder);
            }
            let _seq = builder.build();
        }

        /// Op count matches the number of operations added
        #[test]
        fn op_count_matches(ops in random_ops(50)) {
            let count = ops.len();
            let mut builder = OpBuilder::new();
            for op in ops {
                builder = op.apply(builder);
            }
            let seq = builder.build();
            prop_assert_eq!(seq.ops.len(), count);
        }

        /// Empty builder produces empty sequence
        #[test]
        fn empty_builder_produces_empty_sequence(_seed in any::<u64>()) {
            let seq = OpBuilder::new().build();
            prop_assert!(seq.ops.is_empty());
            prop_assert_eq!(seq.result_count, 0);
        }

        /// Single operations work correctly
        #[test]
        fn single_op_works(op in random_op()) {
            let seq = op.apply(OpBuilder::new()).build();
            prop_assert_eq!(seq.ops.len(), 1);
        }

        /// Conversion to CommandQueue works for non-conditional circuits
        #[test]
        fn command_queue_conversion_works(ops in random_ops(30)) {
            let mut builder = OpBuilder::new();
            for op in ops {
                builder = op.apply(builder);
            }
            // Should succeed since we're not adding conditionals
            let result = builder.to_command_queue();
            prop_assert!(result.is_ok());
        }

        /// GHZ state preparation produces correct op count
        #[test]
        fn ghz_op_count(n in 1usize..=10) {
            let qubits: Vec<QubitId> = (0..n).map(QubitId).collect();
            let seq = OpBuilder::new().prep_ghz(&qubits).build();
            // n preps + 1 H + (n-1) CX = 2n
            prop_assert_eq!(seq.ops.len(), 2 * n);
        }

        /// Bell state preparation produces 4 ops
        #[test]
        fn bell_op_count(q0 in qubit_id(), q1 in qubit_id()) {
            prop_assume!(q0 != q1);
            let seq = OpBuilder::new().prep_bell(q0, q1).build();
            // prep, prep, h, cx = 4
            prop_assert_eq!(seq.ops.len(), 4);
        }

        /// Measurements increment result count correctly
        #[test]
        fn measurement_result_count(num_meas in 1usize..=10) {
            let mut builder = OpBuilder::new();
            for i in 0..num_meas {
                let q = QubitId(i);
                #[allow(clippy::cast_possible_truncation)] // test index bounded by 10
                { builder = builder.pz(q).mz(q, ResultId(i as u16)); }
            }
            let seq = builder.build();
            prop_assert_eq!(seq.result_count, num_meas);
        }

        /// Chained operations maintain correct length
        #[test]
        fn chained_gates_length(
            n_h in 0usize..10,
            n_x in 0usize..10,
            n_cx in 0usize..5
        ) {
            let q0 = QubitId(0);
            let q1 = QubitId(1);

            let mut builder = OpBuilder::new();

            for _ in 0..n_h {
                builder = builder.h(q0);
            }
            for _ in 0..n_x {
                builder = builder.x(q0);
            }
            for _ in 0..n_cx {
                builder = builder.cx(q0, q1);
            }

            let seq = builder.build();
            prop_assert_eq!(seq.ops.len(), n_h + n_x + n_cx);
        }

        /// All single-qubit gates produce exactly one op
        #[test]
        fn all_single_qubit_gates_are_single_op(q in qubit_id()) {
            prop_assert_eq!(OpBuilder::new().h(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().x(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().y(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().z(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().s(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().sdg(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().t(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().tdg(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().sx(q).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().sxdg(q).build().ops.len(), 1);
        }

        /// All rotation gates produce exactly one op
        #[test]
        fn all_rotation_gates_are_single_op(q in qubit_id(), a in angle()) {
            prop_assert_eq!(OpBuilder::new().rx(q, a).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().ry(q, a).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().rz(q, a).build().ops.len(), 1);
        }

        /// All two-qubit gates produce exactly one op
        #[test]
        fn all_two_qubit_gates_are_single_op((q0, q1) in qubit_pair()) {
            prop_assert_eq!(OpBuilder::new().cx(q0, q1).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().cy(q0, q1).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().cz(q0, q1).build().ops.len(), 1);
            prop_assert_eq!(OpBuilder::new().swap(q0, q1).build().ops.len(), 1);
        }

        /// Conditional operations add exactly one op
        #[test]
        fn conditional_is_single_op(q in qubit_id()) {
            let seq = OpBuilder::new()
                .pz(q)
                .mz(q, ResultId(0))
                .if_one(ResultId(0), |b| b.x(q))
                .build();
            // prep + meas + conditional = 3
            prop_assert_eq!(seq.ops.len(), 3);
        }

        /// Subcircuit inlining produces correct op count
        #[test]
        fn subcircuit_inlining_count(n_ops in 1usize..=10) {
            // Create a subcircuit with n_ops H gates
            let mut sub_builder = OpBuilder::new();
            for _ in 0..n_ops {
                sub_builder = sub_builder.h(QubitId(0));
            }
            let sub = Subcircuit::from_builder(sub_builder, 1);

            // Call it
            let seq = OpBuilder::new()
                .call(&sub, &[QubitId(5)])
                .build();

            prop_assert_eq!(seq.ops.len(), n_ops);
        }

        /// Multiple subcircuit calls accumulate correctly
        #[test]
        fn multiple_subcircuit_calls(n_calls in 1usize..=5) {
            let sub = Subcircuit::from_builder(
                OpBuilder::new().h(QubitId(0)).x(QubitId(0)),
                1
            );

            let mut builder = OpBuilder::new();
            for i in 0..n_calls {
                builder = builder.call(&sub, &[QubitId(i)]);
            }

            let seq = builder.build();
            // Each call adds 2 ops
            prop_assert_eq!(seq.ops.len(), n_calls * 2);
        }
    }
}

/// Negative tests - verifying error conditions are handled correctly
#[cfg(test)]
mod negative_tests {
    use super::*;

    // --- CommandQueue conversion errors ---

    #[test]
    fn conditional_fails_command_queue_conversion() {
        let q = QubitId(0);
        let builder = OpBuilder::new()
            .pz(q)
            .mz(q, ResultId(0))
            .if_one(ResultId(0), |b| b.x(q));

        let result = builder.to_command_queue();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConversionError::ConditionalNotSupported { .. }
        ));
    }

    #[test]
    fn nested_conditional_fails_conversion() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let builder = OpBuilder::new()
            .pz(q0)
            .pz(q1)
            .mz(q0, ResultId(0))
            .if_one(ResultId(0), |b| {
                b.mz(q1, ResultId(1)).if_one(ResultId(1), |b2| b2.x(q0))
            });

        let result = builder.to_command_queue();
        assert!(result.is_err());
    }

    #[test]
    fn if_then_else_fails_conversion() {
        let q = QubitId(0);
        let builder = OpBuilder::new().pz(q).mz(q, ResultId(0)).if_then_else(
            ResultId(0),
            |b| b.x(q),
            |b| b.z(q),
        );

        let result = builder.to_command_queue();
        assert!(result.is_err());
    }

    // --- Subcircuit errors ---

    #[test]
    #[should_panic(expected = "expects 2 qubits, got 1")]
    fn subcircuit_wrong_qubit_count_too_few() {
        let sub = Subcircuit::from_builder(
            OpBuilder::new().cx(QubitId(0), QubitId(1)),
            2, // expects 2 qubits
        );

        // Try to call with only 1 qubit - should panic
        let _ = OpBuilder::new().call(&sub, &[QubitId(5)]).build();
    }

    #[test]
    #[should_panic(expected = "expects 2 qubits, got 3")]
    fn subcircuit_wrong_qubit_count_too_many() {
        let sub = Subcircuit::from_builder(
            OpBuilder::new().cx(QubitId(0), QubitId(1)),
            2, // expects 2 qubits
        );

        // Try to call with 3 qubits - should panic
        let _ = OpBuilder::new()
            .call(&sub, &[QubitId(5), QubitId(6), QubitId(7)])
            .build();
    }

    #[test]
    #[should_panic(expected = "Unknown gate: nonexistent")]
    fn unknown_gate_from_library_panics() {
        let lib = GateLibrary::new();

        // Try to call a gate that doesn't exist
        let _ = OpBuilder::new()
            .call_named(&lib, "nonexistent", &[QubitId(0)])
            .build();
    }

    // --- Invalid stabilizer strings ---

    #[test]
    #[should_panic(expected = "Invalid Pauli string")]
    fn invalid_pauli_character_panics() {
        let qubits = [QubitId(0), QubitId(1)];
        let anc = QubitId(10);

        // 'A' is not a valid Pauli operator
        let _ = OpBuilder::new()
            .stabilizer_meas("ZA", &qubits, anc, ResultId(0))
            .build();
    }

    #[test]
    #[should_panic(expected = "Invalid Pauli string")]
    fn invalid_pauli_number_panics() {
        let qubits = [QubitId(0), QubitId(1)];
        let anc = QubitId(10);

        // Numbers are not valid Pauli operators
        let _ = OpBuilder::new()
            .stabilizer_meas("Z1", &qubits, anc, ResultId(0))
            .build();
    }

    #[test]
    fn lowercase_pauli_is_valid() {
        // Lowercase is actually accepted
        let qubits = [QubitId(0), QubitId(1)];
        let anc = QubitId(10);

        let seq = OpBuilder::new()
            .stabilizer_meas("zz", &qubits, anc, ResultId(0))
            .build();

        assert!(!seq.ops.is_empty());
    }

    #[test]
    #[should_panic(expected = "Qubit count must match Pauli string length")]
    fn stabilizer_meas_qubit_count_mismatch_panics() {
        let qubits = [QubitId(0)]; // Only 1 qubit
        let anc = QubitId(10);

        // "ZZ" needs 2 qubits
        let _ = OpBuilder::new()
            .stabilizer_meas("ZZ", &qubits, anc, ResultId(0))
            .build();
    }

    // --- Edge cases ---

    #[test]
    fn empty_ghz_produces_empty_sequence() {
        let seq = OpBuilder::new().prep_ghz(&[]).build();
        assert!(seq.ops.is_empty());
    }

    #[test]
    fn single_qubit_ghz_works() {
        let seq = OpBuilder::new().prep_ghz(&[QubitId(0)]).build();
        // 1 prep + 1 H + 0 CX = 2
        assert_eq!(seq.ops.len(), 2);
    }

    #[test]
    fn same_qubit_bell_works() {
        // Technically weird but shouldn't panic
        let seq = OpBuilder::new().prep_bell(QubitId(0), QubitId(0)).build();
        assert_eq!(seq.ops.len(), 4);
    }

    #[test]
    fn very_large_qubit_id_works() {
        let large_q = QubitId(usize::MAX - 1);
        let seq = OpBuilder::new().pz(large_q).h(large_q).build();
        assert_eq!(seq.ops.len(), 2);
    }

    #[test]
    fn result_id_can_be_reused() {
        // Not recommended but shouldn't panic
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = OpBuilder::new()
            .pz(q0)
            .pz(q1)
            .mz(q0, ResultId(0))
            .mz(q1, ResultId(0)) // Same result ID
            .build();

        assert_eq!(seq.ops.len(), 4);
        assert_eq!(seq.result_count, 1); // Only counts unique max
    }

    #[test]
    fn result_ids_out_of_order_work() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = OpBuilder::new()
            .pz(q0)
            .pz(q1)
            .mz(q0, ResultId(5)) // Start at 5
            .mz(q1, ResultId(2)) // Then 2
            .build();

        assert_eq!(seq.ops.len(), 4);
        // result_count should be max + 1 = 6
        assert_eq!(seq.result_count, 6);
    }

    #[test]
    fn empty_conditional_branches_work() {
        let q = QubitId(0);

        let seq = OpBuilder::new()
            .pz(q)
            .mz(q, ResultId(0))
            .if_then_else(ResultId(0), |b| b, |b| b) // Empty branches
            .build();

        assert_eq!(seq.ops.len(), 3);
    }

    #[test]
    fn xor_result_with_same_target_and_source() {
        // XOR with itself (clears the result)
        let seq = OpBuilder::new()
            .xor_result(ResultId(0), ResultId(0))
            .build();

        assert_eq!(seq.ops.len(), 1);
    }

    #[test]
    fn output_nonexistent_result() {
        // Outputting a result that was never measured - weird but shouldn't panic
        let seq = OpBuilder::new().output(ResultId(99)).build();

        assert_eq!(seq.ops.len(), 1);
    }

    #[test]
    fn conversion_error_display() {
        let err1 = ConversionError::UnsupportedGate {
            gate_id: GateId(999),
            position: 5,
        };
        let msg1 = format!("{err1}");
        assert!(msg1.contains("999"));
        assert!(msg1.contains('5'));

        let err2 = ConversionError::ConditionalNotSupported { position: 10 };
        let msg2 = format!("{err2}");
        assert!(msg2.contains("10"));
        assert!(msg2.contains("Conditional"));
    }
}
