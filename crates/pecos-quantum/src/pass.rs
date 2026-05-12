// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Circuit transformation passes.
//!
//! Passes are explicit transformations applied to circuits before display or
//! simulation. Each pass implements [`CircuitPass`] and can modify both
//! [`TickCircuit`] and [`DagCircuit`] in place.

use std::collections::{BTreeMap, HashMap, HashSet};

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, GateQubits, QubitId};

use crate::{Attribute, DagCircuit, Tick, TickCircuit};

/// A transformation pass that can be applied to circuits.
///
/// Passes transform circuits in-place. For a copy, clone the circuit first:
///
/// ```no_run
/// # use pecos_quantum::pass::{CircuitPass, SimplifyRotations};
/// # use pecos_quantum::TickCircuit;
/// # let circuit = TickCircuit::new();
/// // In-place
/// let mut tc = circuit;
/// SimplifyRotations.apply_tick(&mut tc);
///
/// // Copy (clone first)
/// # let original = TickCircuit::new();
/// let transformed = SimplifyRotations.transform_tick(original);
/// ```
pub trait CircuitPass {
    /// Apply this pass to a [`TickCircuit`] in-place.
    fn apply_tick(&self, circuit: &mut TickCircuit);

    /// Apply this pass to a [`DagCircuit`] in-place.
    fn apply_dag(&self, circuit: &mut DagCircuit);

    /// Take ownership, transform, return.
    fn transform_tick(&self, mut circuit: TickCircuit) -> TickCircuit {
        self.apply_tick(&mut circuit);
        circuit
    }

    /// Take ownership, transform, return.
    fn transform_dag(&self, mut circuit: DagCircuit) -> DagCircuit {
        self.apply_dag(&mut circuit);
        circuit
    }
}

// ============================================================================
// Free functions: primary user-facing pass API
// ============================================================================

/// Lower Clifford-angle rotations to named Clifford gates.
///
/// RZ(pi/2) -> SZ, RZ(pi) -> Z, RX(pi/2) -> SX, etc.
/// Also decomposes two-qubit rotations: RZZ(pi) -> Z+Z.
pub fn lower_clifford_rotations(circuit: &mut TickCircuit) {
    SimplifyRotations.apply_tick(circuit);
}

/// Insert Idle gates after each two-qubit gate on both of its qubits.
///
/// Adds Idle(duration) on both qubits of each 2q gate. Models idle noise
/// during 2q gate execution.
pub fn insert_idle_after_two_qubit_gates(circuit: &mut TickCircuit, duration: f64) {
    InsertIdleAfterTwoQubitGates(duration).apply_tick(circuit);
}

/// Remove identity gates (I, Idle, zero-angle rotations).
pub fn remove_identity(circuit: &mut TickCircuit) {
    RemoveIdentity.apply_tick(circuit);
}

/// Cancel adjacent inverse gate pairs (H-H, S-Sdg, T-Tdg, etc.).
pub fn cancel_inverses(circuit: &mut TickCircuit) {
    CancelInverses.apply_tick(circuit);
}

/// Merge adjacent rotations on the same qubit into a single rotation.
pub fn merge_adjacent_rotations(circuit: &mut TickCircuit) {
    MergeAdjacentRotations.apply_tick(circuit);
}

/// Peephole optimization (rotation merging + Clifford lowering).
pub fn peephole_optimize(circuit: &mut TickCircuit) {
    PeepholeOptimize.apply_tick(circuit);
}

/// Absorb single-qubit basis gates into adjacent preps/measurements.
pub fn absorb_basis_gates(circuit: &mut TickCircuit) {
    AbsorbBasisGates.apply_tick(circuit);
}

/// Compact ticks by ASAP scheduling (merge gates into earlier ticks).
pub fn compact_ticks(circuit: &mut TickCircuit) {
    CompactTicks.apply_tick(circuit);
}

/// Assign `MeasId` to measurement gates that don't have them.
///
/// Walks the circuit in tick order and assigns sequential `MeasId`s
/// to any MZ/MeasureFree gate with empty `meas_ids`. Existing `MeasId`s
/// are preserved. New IDs continue from the circuit's current counter.
pub fn assign_missing_meas_ids(circuit: &mut TickCircuit) {
    AssignMissingMeasIds.apply_tick(circuit);
}

// ============================================================================
// Pass trait and pipeline
// ============================================================================

/// An ordered collection of passes applied sequentially.
///
/// `PassPipeline` itself implements [`CircuitPass`], so pipelines can be
/// nested inside other pipelines.
///
/// # Examples
///
/// ```
/// use pecos_quantum::pass::*;
///
/// let pipeline = PassPipeline::new()
///     .then(AbsorbBasisGates)
///     .then(MergeAdjacentRotations)
///     .then(RemoveIdentity)
///     .then(SimplifyRotations)
///     .then(CancelInverses)
///     .then(PeepholeOptimize);
/// ```
pub struct PassPipeline {
    passes: Vec<Box<dyn CircuitPass>>,
}

impl PassPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Append a pass to the pipeline and return `self` for chaining.
    #[must_use]
    pub fn then(mut self, pass: impl CircuitPass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }
}

impl Default for PassPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitPass for PassPipeline {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        for pass in &self.passes {
            pass.apply_tick(circuit);
        }
    }

    fn apply_dag(&self, circuit: &mut DagCircuit) {
        for pass in &self.passes {
            pass.apply_dag(circuit);
        }
    }
}

/// Replace rotation gates at special angles with their named equivalents.
///
/// For example, `RZ(pi/2)` becomes `SZ`, `RX(pi)` becomes `X`, and
/// `RZZ(pi)` decomposes into two independent `Z` gates.
///
/// # Single-qubit simplifications (in-place)
///
/// | Rotation | Angle | Result |
/// |----------|-------|--------|
/// | RZ | pi   | Z   |
/// | RZ | pi/2 | SZ  |
/// | RZ | 3pi/2 | `SZdg` |
/// | RZ | pi/4 | T   |
/// | RZ | 7pi/4 | `Tdg` |
/// | RX | pi   | X   |
/// | RX | pi/2 | SX  |
/// | RX | 3pi/2 | `SXdg` |
/// | RY | pi   | Y   |
/// | RY | pi/2 | SY  |
/// | RY | 3pi/2 | `SYdg` |
/// | RZZ | pi/2 | SZZ |
/// | RZZ | 3pi/2 | `SZZdg` |
///
/// # Two-qubit decompositions
///
/// | Rotation | Angle | Result |
/// |----------|-------|--------|
/// | RZZ | pi | Z + Z |
/// | RXX | pi | X + X |
/// | RYY | pi | Y + Y |
pub struct SimplifyRotations;

/// Insert Idle gates after each two-qubit gate on both of its qubits.
///
/// For each tick containing two-qubit gates, adds a new tick immediately
/// after with `Idle(duration)` on each qubit involved in a two-qubit gate.
///
/// This models the idle noise that qubits experience during two-qubit
/// gate execution. The noise model applies `RZ(p_idle * duration)` when
/// it encounters an Idle gate.
///
/// The inner value is the idle duration in time units (typically 1.0).
pub struct InsertIdleAfterTwoQubitGates(pub f64);

impl CircuitPass for InsertIdleAfterTwoQubitGates {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        let duration = self.0;
        let mut new_ticks = Vec::with_capacity(circuit.ticks().len() * 2);

        // Drain ticks from circuit and rebuild with idle insertions
        let old_ticks = circuit.take_ticks();

        for tick in old_ticks {
            let mut idle_qubits: Vec<QubitId> = Vec::new();
            for gate in tick.iter_gate_batches() {
                if gate.is_two_qubit() {
                    for q in &gate.qubits {
                        if !idle_qubits.contains(q) {
                            idle_qubits.push(*q);
                        }
                    }
                }
            }

            new_ticks.push(tick);

            if !idle_qubits.is_empty() {
                let mut idle_tick = crate::Tick::new();
                for q in idle_qubits {
                    idle_tick.add_gate(Gate::idle(duration, vec![q]));
                }
                new_ticks.push(idle_tick);
            }
        }

        circuit.replace_ticks(new_ticks);
    }

    fn apply_dag(&self, _circuit: &mut DagCircuit) {
        // DAG doesn't have tick structure — no-op
    }
}

/// Apply an in-place simplification to a gate. Returns `true` if the gate was
/// simplified (either renamed in place or needs decomposition handling).
fn simplify_gate_in_place(gate: &mut Gate) -> bool {
    // R1XY has two angles — handle separately
    if gate.gate_type == GateType::R1XY && gate.angles.len() == 2 {
        if let Some(named) = pecos_core::try_simplify_r1xy(gate.angles[0], gate.angles[1]) {
            gate.gate_type = named;
            gate.angles.clear();
            return true;
        }
        return false;
    }

    if gate.angles.len() != 1 {
        return false;
    }
    if let Some(named) = pecos_core::try_simplify_rotation(gate.gate_type, gate.angles[0]) {
        gate.gate_type = named;
        gate.angles.clear();
        return true;
    }
    false
}

// === Helper functions for circuit transformation passes ===

/// Returns `true` if the gate is an identity operation (I, Idle, or zero-angle rotation).
fn is_identity_gate(gate: &Gate) -> bool {
    match gate.gate_type {
        GateType::I | GateType::Idle => true,
        gt if is_rotation(gt) => gate.angles.len() == 1 && gate.angles[0].is_zero(),
        _ => false,
    }
}

/// Returns `true` if the gate type is a rotation (parameterized by a single angle).
fn is_rotation(gt: GateType) -> bool {
    matches!(
        gt,
        GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::CRZ
    )
}

/// Returns `true` if the gate type is its own inverse.
fn is_self_inverse(gt: GateType) -> bool {
    matches!(
        gt,
        GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::H
            | GateType::I
            | GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::SWAP
            | GateType::CCX
    )
}

/// Returns the named inverse of a gate type, if one exists.
fn named_inverse(gt: GateType) -> Option<GateType> {
    match gt {
        GateType::SX => Some(GateType::SXdg),
        GateType::SXdg => Some(GateType::SX),
        GateType::SY => Some(GateType::SYdg),
        GateType::SYdg => Some(GateType::SY),
        GateType::SZ => Some(GateType::SZdg),
        GateType::SZdg => Some(GateType::SZ),
        GateType::T => Some(GateType::Tdg),
        GateType::Tdg => Some(GateType::T),
        GateType::SZZ => Some(GateType::SZZdg),
        GateType::SZZdg => Some(GateType::SZZ),
        _ => None,
    }
}

/// Returns `true` if gates `a` and `b` are inverses of each other.
///
/// Checks (in order):
/// 1. Qubits must match exactly.
/// 2. Self-inverse identical gates (e.g., H*H, CX*CX).
/// 3. Named inverse pairs (e.g., `SX*SXdg`, `T*Tdg`).
/// 4. Same rotation type with angles summing to zero (e.g., `RZ(t)*RZ(-t)`).
fn are_inverses(a: &Gate, b: &Gate) -> bool {
    if a.qubits != b.qubits {
        return false;
    }
    // Self-inverse identical gates
    if a.gate_type == b.gate_type && is_self_inverse(a.gate_type) && a.angles == b.angles {
        return true;
    }
    // Named inverse pairs
    if let Some(inv) = named_inverse(a.gate_type)
        && inv == b.gate_type
        && a.angles == b.angles
    {
        return true;
    }
    // Rotation angles summing to zero
    if a.gate_type == b.gate_type
        && is_rotation(a.gate_type)
        && a.angles.len() == 1
        && b.angles.len() == 1
        && (a.angles[0] + b.angles[0]).is_zero()
    {
        return true;
    }
    false
}

/// Check if all qubit stacks agree on the same top-of-stack position.
///
/// Returns `Some((tick_idx, gate_idx))` if every qubit in `qubits` has
/// a non-empty stack whose top entry is the same position, `None` otherwise.
fn check_all_stacks_agree(
    stacks: &HashMap<QubitId, Vec<(usize, usize)>>,
    qubits: &[QubitId],
) -> Option<(usize, usize)> {
    let mut agreed: Option<(usize, usize)> = None;
    for &q in qubits {
        let top = *stacks.get(&q)?.last()?;
        match agreed {
            None => agreed = Some(top),
            Some(prev) => {
                if prev != top {
                    return None;
                }
            }
        }
    }
    agreed
}

/// Check if the successor of `node` on every qubit in `qubits` is the same DAG node.
fn dag_common_successor(circuit: &DagCircuit, node: usize, qubits: &[QubitId]) -> Option<usize> {
    let mut result: Option<usize> = None;
    for &q in qubits {
        let s = circuit.successor_on_qubit(node, q)?;
        match result {
            None => result = Some(s),
            Some(prev) if prev == s => {}
            _ => return None,
        }
    }
    result
}

/// Check if a gate conjugated by H on a specific qubit simplifies.
///
/// Returns `Some((new_gate_type, new_qubits))` if:
/// - H on target of CX -> CZ (same qubits)
/// - H on either qubit of CZ -> CX (other qubit becomes control, H qubit becomes target)
fn peephole_conjugation(middle: &Gate, h_qubit: QubitId) -> Option<(GateType, GateQubits)> {
    match middle.gate_type {
        GateType::CX if middle.qubits.len() == 2 && middle.qubits[1] == h_qubit => {
            // H(target) CX(c,t) H(target) = CZ(c,t)
            Some((GateType::CZ, middle.qubits.clone()))
        }
        GateType::CZ if middle.qubits.len() == 2 && middle.qubits.contains(&h_qubit) => {
            // H(q) CZ(a,b) H(q) = CX(other, q)
            let other = if middle.qubits[0] == h_qubit {
                middle.qubits[1]
            } else {
                middle.qubits[0]
            };
            Some((GateType::CX, smallvec::smallvec![other, h_qubit]))
        }
        _ => None,
    }
}

fn split_batched_tick_commands(circuit: &mut TickCircuit) {
    let old_ticks = circuit.take_ticks();
    let mut new_ticks = Vec::with_capacity(old_ticks.len());

    for old_tick in old_ticks {
        let mut new_tick = Tick::new();
        for (key, value) in old_tick.tick_attrs() {
            new_tick.set_attr(key, value.clone());
        }

        for batch in old_tick.iter_gate_batches() {
            let gate = batch.as_gate();
            let attrs: BTreeMap<String, Attribute> = batch
                .attrs()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();

            let split_gates: Vec<Gate> = if batch.gate_count() == 0 {
                vec![gate.clone()]
            } else {
                batch
                    .iter_gate_instances()
                    .map(super::tick_circuit::GateInstanceRef::to_gate)
                    .collect()
            };

            if split_gates.is_empty() {
                continue;
            }

            if split_gates.len() == 1 {
                let new_idx = new_tick
                    .try_add_gate_preserving_command(split_gates[0].clone())
                    .unwrap_or_else(|err| panic!("{err}"));
                if !attrs.is_empty() {
                    new_tick.set_gate_attrs(new_idx, attrs);
                }
                continue;
            }

            for split_gate in split_gates {
                let new_idx = new_tick
                    .try_add_gate_preserving_command(split_gate)
                    .unwrap_or_else(|err| panic!("{err}"));
                if !attrs.is_empty() {
                    new_tick.set_gate_attrs(new_idx, attrs.clone());
                }
            }
        }

        new_ticks.push(new_tick);
    }

    circuit.replace_ticks(new_ticks);
}

impl CircuitPass for SimplifyRotations {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        for tick in circuit.ticks_mut() {
            // First pass: collect two-qubit decompositions.
            // We need to know which gate indices to remove and what to add.
            let mut decompositions: Vec<(usize, GateType)> = Vec::new();

            for gate in tick.iter_gate_batches() {
                if gate.angles.len() == 1
                    && let Some(pauli) =
                        pecos_core::half_turn_decomposition(gate.gate_type, gate.angles[0])
                {
                    decompositions.push((gate.batch_index(), pauli));
                }
            }

            // Process decompositions in reverse order to keep indices valid.
            for &(idx, pauli) in decompositions.iter().rev() {
                let qubits = tick.gate_batches()[idx].qubits.clone();
                // Remove the two-qubit gate, add two single-qubit gates.
                tick.remove_gate(idx);
                for pair in qubits.chunks(2) {
                    if pair.len() == 2 {
                        tick.add_gate(Gate::simple(pauli, smallvec::smallvec![pair[0]]));
                        tick.add_gate(Gate::simple(pauli, smallvec::smallvec![pair[1]]));
                    }
                }
            }

            // Second pass: in-place simplification of remaining gates.
            for gate_idx in 0..tick.len() {
                tick.update_gate_batch(gate_idx, |gate| {
                    simplify_gate_in_place(gate);
                })
                .unwrap_or_else(|err| panic!("{err}"));
            }
        }
    }

    fn apply_dag(&self, circuit: &mut DagCircuit) {
        let nodes = circuit.nodes();

        for node in nodes {
            let Some(gate) = circuit.gate(node) else {
                continue;
            };

            // Check for two-qubit half-turn decomposition first.
            if gate.angles.len() == 1
                && let Some(pauli) =
                    pecos_core::half_turn_decomposition(gate.gate_type, gate.angles[0])
            {
                let qubits = gate.qubits.clone();

                // Collect predecessor/successor nodes *before* removal
                // (remove_gate deletes edges too).
                let mut pred_map = Vec::new();
                let mut succ_map = Vec::new();
                for &q in &qubits {
                    pred_map.push((q, circuit.predecessor_on_qubit(node, q)));
                    succ_map.push((q, circuit.successor_on_qubit(node, q)));
                }

                // Remove the two-qubit gate (and its edges).
                circuit.remove_gate(node);

                // Add two single-qubit gates and rewire.
                for pair in qubits.chunks(2) {
                    if pair.len() < 2 {
                        continue;
                    }
                    let node_a =
                        circuit.add_gate(Gate::simple(pauli, smallvec::smallvec![pair[0]]));
                    let node_b =
                        circuit.add_gate(Gate::simple(pauli, smallvec::smallvec![pair[1]]));

                    // Rewire predecessors -> new nodes.
                    for &(q, pred) in &pred_map {
                        if let Some(pred) = pred {
                            if q == pair[0] {
                                let _ = circuit.connect(pred, node_a, q);
                            } else if q == pair[1] {
                                let _ = circuit.connect(pred, node_b, q);
                            }
                        }
                    }

                    // Rewire new nodes -> successors.
                    for &(q, succ) in &succ_map {
                        if let Some(succ) = succ {
                            if q == pair[0] {
                                let _ = circuit.connect(node_a, succ, q);
                            } else if q == pair[1] {
                                let _ = circuit.connect(node_b, succ, q);
                            }
                        }
                    }
                }
                continue;
            }

            // In-place simplification for single-qubit and two-qubit named replacements.
            if let Some(gate) = circuit.gate_mut(node) {
                simplify_gate_in_place(gate);
            }
        }
    }
}

/// Remove identity gates (I, Idle, zero-angle rotations) from circuits.
pub struct RemoveIdentity;

impl CircuitPass for RemoveIdentity {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        for tick in circuit.ticks_mut() {
            let to_remove: Vec<usize> = tick
                .gate_batches()
                .iter()
                .enumerate()
                .filter(|(_, g)| is_identity_gate(g))
                .map(|(i, _)| i)
                .collect();
            for &idx in to_remove.iter().rev() {
                tick.remove_gate(idx);
            }
        }
    }

    fn apply_dag(&self, circuit: &mut DagCircuit) {
        let nodes = circuit.nodes();
        for node in nodes {
            let Some(gate) = circuit.gate(node) else {
                continue;
            };
            if !is_identity_gate(gate) {
                continue;
            }
            let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();
            let mut rewire = Vec::new();
            for &q in &qubits {
                let pred = circuit.predecessor_on_qubit(node, q);
                let succ = circuit.successor_on_qubit(node, q);
                rewire.push((q, pred, succ));
            }
            circuit.remove_gate(node);
            for (q, pred, succ) in rewire {
                if let (Some(p), Some(s)) = (pred, succ) {
                    let _ = circuit.connect(p, s, q);
                }
            }
        }
    }
}

/// Cancel adjacent inverse gate pairs (e.g., `H*H`, `SX*SXdg`, `RZ(t)*RZ(-t)`).
///
/// Uses a per-qubit stack to handle nested cancellations (A B B^-1 A^-1)
/// in a single pass over tick circuits.
pub struct CancelInverses;

impl CircuitPass for CancelInverses {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        let mut stacks: HashMap<QubitId, Vec<(usize, usize)>> = HashMap::new();
        let mut to_remove: Vec<(usize, usize)> = Vec::new();

        for (ti, tick) in circuit.iter_ticks() {
            for gate in tick.iter_gate_batches() {
                let gi = gate.batch_index();
                let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();

                if let Some((pred_ti, pred_gi)) = check_all_stacks_agree(&stacks, &qubits) {
                    let pred_gate = &circuit.ticks()[pred_ti].gate_batches()[pred_gi];
                    if are_inverses(pred_gate, gate.as_gate()) {
                        for &q in &qubits {
                            if let Some(stack) = stacks.get_mut(&q) {
                                stack.pop();
                            }
                        }
                        to_remove.push((pred_ti, pred_gi));
                        to_remove.push((ti, gi));
                        continue;
                    }
                }

                for &q in &qubits {
                    stacks.entry(q).or_default().push((ti, gi));
                }
            }
        }

        to_remove.sort_unstable();
        to_remove.dedup();
        for &(ti, gi) in to_remove.iter().rev() {
            if let Some(tick) = circuit.get_tick_mut(ti) {
                tick.remove_gate(gi);
            }
        }
    }

    fn apply_dag(&self, circuit: &mut DagCircuit) {
        let topo = circuit.topological_order();
        for node in topo {
            let Some(gate) = circuit.gate(node) else {
                continue;
            };
            let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();

            let Some(succ) = dag_common_successor(circuit, node, &qubits) else {
                continue;
            };
            let Some(succ_gate) = circuit.gate(succ) else {
                continue;
            };

            if !are_inverses(gate, succ_gate) {
                continue;
            }

            let mut rewire = Vec::new();
            for &q in &qubits {
                let pred = circuit.predecessor_on_qubit(node, q);
                let succ_succ = circuit.successor_on_qubit(succ, q);
                rewire.push((q, pred, succ_succ));
            }

            circuit.remove_gate(node);
            circuit.remove_gate(succ);

            for (q, pred, succ_succ) in rewire {
                if let (Some(p), Some(s)) = (pred, succ_succ) {
                    let _ = circuit.connect(p, s, q);
                }
            }
        }
    }
}

/// Merge consecutive same-axis rotations (e.g., RZ(a)*RZ(b) -> RZ(a+b)).
///
/// Uses a per-qubit stack to handle chains of rotations. After merging,
/// the surviving gate's angle is the sum of all merged angles.
pub struct MergeAdjacentRotations;

impl CircuitPass for MergeAdjacentRotations {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        let mut stacks: HashMap<QubitId, Vec<(usize, usize)>> = HashMap::new();
        let mut angle_adjustments: HashMap<(usize, usize), Angle64> = HashMap::new();
        let mut to_remove: Vec<(usize, usize)> = Vec::new();

        for (ti, tick) in circuit.iter_ticks() {
            for gate in tick.iter_gate_batches() {
                let gi = gate.batch_index();
                let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();

                if is_rotation(gate.gate_type)
                    && gate.angles.len() == 1
                    && let Some((pred_ti, pred_gi)) = check_all_stacks_agree(&stacks, &qubits)
                {
                    let pred_gate = &circuit.ticks()[pred_ti].gate_batches()[pred_gi];
                    if pred_gate.gate_type == gate.gate_type && pred_gate.qubits == gate.qubits {
                        *angle_adjustments
                            .entry((pred_ti, pred_gi))
                            .or_insert(Angle64::ZERO) += gate.angles[0];
                        to_remove.push((ti, gi));
                        // Don't push; predecessor stays on stack for chain merging.
                        continue;
                    }
                }

                // Push to stacks (for rotation or non-rotation gates).
                for &q in &qubits {
                    stacks.entry(q).or_default().push((ti, gi));
                }
            }
        }

        // Apply angle adjustments to surviving gates.
        for (&(ti, gi), &delta) in &angle_adjustments {
            if let Some(tick) = circuit.get_tick_mut(ti) {
                tick.update_gate_batch(gi, |gate| {
                    gate.angles[0] += delta;
                })
                .unwrap_or_else(|err| panic!("{err}"));
            }
        }

        // Remove merged gates in reverse order.
        to_remove.sort_unstable();
        for &(ti, gi) in to_remove.iter().rev() {
            if let Some(tick) = circuit.get_tick_mut(ti) {
                tick.remove_gate(gi);
            }
        }
    }

    fn apply_dag(&self, circuit: &mut DagCircuit) {
        let topo = circuit.topological_order();
        for node in topo {
            while let Some(gate) = circuit.gate(node) {
                if !is_rotation(gate.gate_type) || gate.angles.len() != 1 {
                    break;
                }
                let gate_type = gate.gate_type;
                let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();

                let Some(succ) = dag_common_successor(circuit, node, &qubits) else {
                    break;
                };
                let Some(succ_gate) = circuit.gate(succ) else {
                    break;
                };

                if succ_gate.gate_type != gate_type
                    || succ_gate.qubits[..] != qubits[..]
                    || succ_gate.angles.len() != 1
                {
                    break;
                }

                let succ_angle = succ_gate.angles[0];

                // Save succ-of-successor for rewiring.
                let mut rewire = Vec::new();
                for &q in &qubits {
                    let succ_succ = circuit.successor_on_qubit(succ, q);
                    rewire.push((q, succ_succ));
                }

                // Merge angle and remove successor.
                circuit
                    .gate_mut(node)
                    .expect("node must exist in circuit")
                    .angles[0] += succ_angle;
                circuit.remove_gate(succ);

                for (q, succ_succ) in rewire {
                    if let Some(ss) = succ_succ {
                        let _ = circuit.connect(node, ss, q);
                    }
                }
            }
        }
    }
}

/// Recognize and simplify multi-gate patterns.
///
/// Current rules:
/// - `H(q) CX(c,q) H(q)` -> `CZ(c,q)`
/// - `H(q) CZ(a,b) H(q)` -> `CX(other, q)`
pub struct PeepholeOptimize;

impl CircuitPass for PeepholeOptimize {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        split_batched_tick_commands(circuit);

        // Build per-qubit timeline: Vec of (tick_idx, gate_idx) in order.
        let mut timelines: HashMap<QubitId, Vec<(usize, usize)>> = HashMap::new();
        for (ti, tick) in circuit.iter_ticks() {
            for gate in tick.iter_gate_batches() {
                let gi = gate.batch_index();
                for &q in &gate.qubits {
                    timelines.entry(q).or_default().push((ti, gi));
                }
            }
        }

        let mut replacements: Vec<((usize, usize), GateType, GateQubits)> = Vec::new();
        let mut to_remove: HashSet<(usize, usize)> = HashSet::new();

        // Scan each qubit's timeline for H - middle - H pattern.
        for (q, timeline) in &timelines {
            if timeline.len() < 3 {
                continue;
            }
            let mut i = 0;
            while i + 2 < timeline.len() {
                let (h1_ti, h1_gi) = timeline[i];
                let (mid_ti, mid_gi) = timeline[i + 1];
                let (h2_ti, h2_gi) = timeline[i + 2];

                // Skip if any of these gates are already consumed.
                if to_remove.contains(&(h1_ti, h1_gi))
                    || to_remove.contains(&(mid_ti, mid_gi))
                    || to_remove.contains(&(h2_ti, h2_gi))
                {
                    i += 1;
                    continue;
                }

                let h1 = &circuit.ticks()[h1_ti].gate_batches()[h1_gi];
                let mid = &circuit.ticks()[mid_ti].gate_batches()[mid_gi];
                let h2 = &circuit.ticks()[h2_ti].gate_batches()[h2_gi];

                // Both must be single-qubit H on this qubit.
                if h1.gate_type != GateType::H
                    || h1.qubits.len() != 1
                    || h2.gate_type != GateType::H
                    || h2.qubits.len() != 1
                {
                    i += 1;
                    continue;
                }

                if let Some((new_gt, new_qubits)) = peephole_conjugation(mid, *q) {
                    to_remove.insert((h1_ti, h1_gi));
                    to_remove.insert((h2_ti, h2_gi));
                    replacements.push(((mid_ti, mid_gi), new_gt, new_qubits));
                    i += 3; // skip past the consumed triple
                } else {
                    i += 1;
                }
            }
        }

        // Apply replacements.
        for ((ti, gi), new_gt, new_qubits) in &replacements {
            if let Some(tick) = circuit.get_tick_mut(*ti) {
                tick.update_gate_batch(*gi, |gate| {
                    gate.gate_type = *new_gt;
                    gate.qubits.clone_from(new_qubits);
                })
                .unwrap_or_else(|err| panic!("{err}"));
            }
        }

        // Remove H gates in reverse order to preserve indices.
        let mut remove_list: Vec<(usize, usize)> = to_remove
            .iter()
            .filter(|pos| !replacements.iter().any(|(p, _, _)| p == *pos))
            .copied()
            .collect();
        remove_list.sort_unstable();
        for &(ti, gi) in remove_list.iter().rev() {
            if let Some(tick) = circuit.get_tick_mut(ti) {
                tick.remove_gate(gi);
            }
        }
    }

    fn apply_dag(&self, circuit: &mut DagCircuit) {
        let topo = circuit.topological_order();
        for node in topo {
            let Some(gate) = circuit.gate(node) else {
                continue;
            };
            // Look for two-qubit gates (CX, CZ) where one qubit has H before and after.
            if !matches!(gate.gate_type, GateType::CX | GateType::CZ) || gate.qubits.len() != 2 {
                continue;
            }
            let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();

            // Check each qubit for H-conjugation.
            for &q in &qubits {
                let Some(pred) = circuit.predecessor_on_qubit(node, q) else {
                    continue;
                };
                let Some(succ) = circuit.successor_on_qubit(node, q) else {
                    continue;
                };
                let Some(pred_gate) = circuit.gate(pred) else {
                    continue;
                };
                let Some(succ_gate) = circuit.gate(succ) else {
                    continue;
                };

                // Both must be single-qubit H on this qubit.
                if pred_gate.gate_type != GateType::H
                    || pred_gate.qubits.len() != 1
                    || succ_gate.gate_type != GateType::H
                    || succ_gate.qubits.len() != 1
                {
                    continue;
                }

                let gate = circuit.gate(node).expect("node must exist in circuit");
                if let Some((new_gt, new_qubits)) = peephole_conjugation(gate, q) {
                    // Rewire around the two H gates.
                    let h_pred = circuit.predecessor_on_qubit(pred, q);
                    let h_succ = circuit.successor_on_qubit(succ, q);
                    circuit.remove_gate(pred);
                    circuit.remove_gate(succ);
                    // Update the middle gate in place.
                    let g = circuit.gate_mut(node).expect("node must exist in circuit");
                    g.gate_type = new_gt;
                    g.qubits = new_qubits;
                    // Rewire: h_pred -> node, node -> h_succ
                    if let Some(hp) = h_pred {
                        let _ = circuit.connect(hp, node, q);
                    }
                    if let Some(hs) = h_succ {
                        let _ = circuit.connect(node, hs, q);
                    }
                    break; // gate changed, move to next node
                }
            }
        }
    }
}

// === Helper functions for AbsorbBasisGates ===

/// Returns `true` if the gate is a Z-basis preparation (produces |0>).
fn is_z_prep(gt: GateType) -> bool {
    matches!(gt, GateType::PZ | GateType::QAlloc)
}

/// Returns `true` if the gate is a Z-basis measurement.
fn is_z_measure(gt: GateType) -> bool {
    matches!(gt, GateType::MZ | GateType::MeasureFree)
}

/// Returns `true` if the gate is Z-diagonal (single- or multi-qubit).
///
/// Z-diagonal gates are diagonal in the computational basis: they map each
/// basis state to itself times a phase.  Applying one when every qubit is in
/// a Z eigenstate only adds a global phase (no-op), and it does not change
/// Z-measurement statistics.
fn is_z_diagonal(gate: &Gate) -> bool {
    matches!(
        gate.gate_type,
        GateType::Z
            | GateType::SZ
            | GateType::SZdg
            | GateType::T
            | GateType::Tdg
            | GateType::RZ
            | GateType::CZ
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::RZZ
            | GateType::CRZ
    )
}

/// Remove Z-diagonal gates that are redundant due to adjacent Z-basis
/// preparations or measurements.
///
/// Z-basis preparations (PZ / `QAlloc`) produce |0>, an eigenstate of every
/// Z-diagonal operator.  Applying any Z-diagonal gate (Z, SZ, `SZdg`, T,
/// `Tdg`, RZ, CZ, SZZ, `SZZdg`, RZZ, CRZ) when all its qubits are still
/// in a Z eigenstate only adds a global phase -- a physical no-op.
/// Similarly, Z-diagonal gates immediately before Z-basis measurements
/// (MZ / `MeasureFree`) do not change measurement statistics and can be
/// removed.
pub struct AbsorbBasisGates;

impl CircuitPass for AbsorbBasisGates {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        let mut to_remove: Vec<(usize, usize)> = Vec::new();

        // Forward scan: absorb Z-diagonal gates after Z-preps.
        let mut z_eigenstate: HashSet<QubitId> = HashSet::new();
        for (ti, tick) in circuit.iter_ticks() {
            for gate in tick.iter_gate_batches() {
                let gi = gate.batch_index();
                if is_z_prep(gate.gate_type) {
                    for &q in &gate.qubits {
                        z_eigenstate.insert(q);
                    }
                } else if is_z_diagonal(gate.as_gate())
                    && gate.qubits.iter().all(|q| z_eigenstate.contains(q))
                {
                    to_remove.push((ti, gi));
                } else {
                    for &q in &gate.qubits {
                        z_eigenstate.remove(&q);
                    }
                }
            }
        }

        // Backward scan: absorb Z-diagonal gates before Z-measures.
        let mut before_z_measure: HashSet<QubitId> = HashSet::new();
        for (ti, tick) in circuit.ticks().iter().enumerate().rev() {
            for (gi, gate) in tick.gate_batches().iter().enumerate().rev() {
                if is_z_measure(gate.gate_type) {
                    for &q in &gate.qubits {
                        before_z_measure.insert(q);
                    }
                } else if is_z_diagonal(gate)
                    && gate.qubits.iter().all(|q| before_z_measure.contains(q))
                {
                    to_remove.push((ti, gi));
                } else {
                    for &q in &gate.qubits {
                        before_z_measure.remove(&q);
                    }
                }
            }
        }

        // Deduplicate and remove in reverse order to preserve indices.
        to_remove.sort_unstable();
        to_remove.dedup();
        for &(ti, gi) in to_remove.iter().rev() {
            if let Some(tick) = circuit.get_tick_mut(ti) {
                tick.remove_gate(gi);
            }
        }
    }

    fn apply_dag(&self, circuit: &mut DagCircuit) {
        let topo = circuit.topological_order();
        let mut to_remove: Vec<usize> = Vec::new();

        // Forward: track qubits in Z eigenstates, absorb Z-diagonal gates.
        let mut z_eigenstate: HashSet<QubitId> = HashSet::new();
        for &node in &topo {
            let Some(gate) = circuit.gate(node) else {
                continue;
            };
            if is_z_prep(gate.gate_type) {
                for &q in &gate.qubits {
                    z_eigenstate.insert(q);
                }
            } else if is_z_diagonal(gate) && gate.qubits.iter().all(|q| z_eigenstate.contains(q)) {
                to_remove.push(node);
            } else {
                for &q in &gate.qubits {
                    z_eigenstate.remove(&q);
                }
            }
        }

        // Backward: track qubits whose next operation is a Z-measure.
        let mut before_z_measure: HashSet<QubitId> = HashSet::new();
        for &node in topo.iter().rev() {
            let Some(gate) = circuit.gate(node) else {
                continue;
            };
            if is_z_measure(gate.gate_type) {
                for &q in &gate.qubits {
                    before_z_measure.insert(q);
                }
            } else if is_z_diagonal(gate)
                && gate.qubits.iter().all(|q| before_z_measure.contains(q))
            {
                to_remove.push(node);
            } else {
                for &q in &gate.qubits {
                    before_z_measure.remove(&q);
                }
            }
        }

        // Deduplicate and remove, rewiring around each removed node.
        to_remove.sort_unstable();
        to_remove.dedup();
        for &node in &to_remove {
            let Some(gate) = circuit.gate(node) else {
                continue;
            };
            let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();
            let mut rewire = Vec::new();
            for &q in &qubits {
                let pred = circuit.predecessor_on_qubit(node, q);
                let succ = circuit.successor_on_qubit(node, q);
                rewire.push((q, pred, succ));
            }
            circuit.remove_gate(node);
            for (q, pred, succ) in rewire {
                if let (Some(p), Some(s)) = (pred, succ) {
                    let _ = circuit.connect(p, s, q);
                }
            }
        }
    }
}

/// ASAP-schedule gates to minimise tick count, then drop empty ticks.
///
/// For each gate (processed in original tick order), the pass assigns it to
/// the earliest tick where none of its qubits are still occupied.  The
/// resulting circuit has the same gate order per qubit but fewer ticks.
///
/// This is a `TickCircuit`-only optimisation; `apply_dag` is a no-op because
/// a DAG already represents the dependency graph without fixed time slots.
pub struct CompactTicks;

impl CircuitPass for CompactTicks {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        // Collect every gate together with its per-gate attributes.
        let mut entries: Vec<(Gate, BTreeMap<String, Attribute>)> = Vec::new();
        for tick in circuit.ticks() {
            for gate in tick.iter_gate_batches() {
                let attrs: BTreeMap<String, Attribute> =
                    gate.attrs().map(|(k, v)| (k.clone(), v.clone())).collect();
                entries.push((gate.as_gate().clone(), attrs));
            }
        }

        if entries.is_empty() {
            circuit.clear();
            return;
        }

        // ASAP scheduling: for each gate, find the earliest tick where none
        // of its qubits are busy.
        // `qubit_ready[q]` = the next tick index at which qubit q is free.
        let mut qubit_ready: HashMap<QubitId, usize> = HashMap::new();
        let mut assignments: Vec<usize> = Vec::with_capacity(entries.len());
        let mut num_ticks: usize = 0;

        for (gate, _) in &entries {
            let earliest = gate
                .qubits
                .iter()
                .map(|q| qubit_ready.get(q).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            assignments.push(earliest);
            for &q in &gate.qubits {
                qubit_ready.insert(q, earliest + 1);
            }
            if earliest + 1 > num_ticks {
                num_ticks = earliest + 1;
            }
        }

        // Save and restore circuit-level metadata across the rebuild.
        let saved_attrs: BTreeMap<String, Attribute> = circuit
            .circuit_attrs()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        circuit.clear();
        circuit.reserve_ticks(num_ticks);

        for (i, (gate, attrs)) in entries.into_iter().enumerate() {
            let ti = assignments[i];
            let tick = circuit
                .get_tick_mut(ti)
                .expect("tick index must exist in circuit");
            let gi = tick.add_gate(gate);
            if !attrs.is_empty() {
                tick.set_gate_attrs(gi, attrs);
            }
        }

        if !saved_attrs.is_empty() {
            circuit.set_metas(saved_attrs);
        }
    }

    fn apply_dag(&self, _circuit: &mut DagCircuit) {
        // No-op: a DAG has no fixed time slots to compact.
    }
}

/// Assign [`MeasId`](pecos_core::MeasId) to measurement gates that don't have them.
///
/// Walks the circuit in tick order and assigns sequential IDs to any
/// measurement gate with empty `meas_ids`. Existing IDs are preserved.
/// New IDs continue from the circuit's current measurement counter.
///
/// Use this on circuits from external sources (QIS trace, Stim import)
/// that don't assign `MeasId` during construction.
pub struct AssignMissingMeasIds;

impl CircuitPass for AssignMissingMeasIds {
    fn apply_tick(&self, circuit: &mut TickCircuit) {
        let mut next_id = circuit.num_measurements();
        for tick in circuit.ticks_mut() {
            for gate_idx in 0..tick.len() {
                tick.update_gate_batch(gate_idx, |gate| {
                    let is_measurement =
                        matches!(gate.gate_type, GateType::MZ | GateType::MeasureFree);
                    if is_measurement && gate.meas_ids.is_empty() {
                        for _ in &gate.qubits {
                            gate.meas_ids.push(pecos_core::MeasId(next_id));
                            next_id += 1;
                        }
                    }
                })
                .unwrap_or_else(|err| panic!("{err}"));
            }
        }
        let added = next_id - circuit.num_measurements();
        if added > 0 {
            circuit.advance_meas_counter(added);
        }
    }

    fn apply_dag(&self, _circuit: &mut DagCircuit) {
        // No-op: DagCircuit gates are accessed differently.
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use pecos_core::MeasId;

    // ==================== simplify_rotation unit tests ====================

    #[test]
    fn simplify_rz_quarter_turn_to_sz() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZ, Angle64::QUARTER_TURN),
            Some(GateType::SZ)
        );
    }

    #[test]
    fn simplify_rz_half_turn_to_z() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZ, Angle64::HALF_TURN),
            Some(GateType::Z)
        );
    }

    #[test]
    fn simplify_rz_three_quarters_to_szdg() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZ, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SZdg)
        );
    }

    #[test]
    fn simplify_rz_eighth_turn_to_t() {
        let eighth = Angle64::from_turn_ratio(1, 8);
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZ, eighth),
            Some(GateType::T)
        );
    }

    #[test]
    fn simplify_rz_seven_eighths_to_tdg() {
        let seven_eighths = Angle64::from_turn_ratio(7, 8);
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZ, seven_eighths),
            Some(GateType::Tdg)
        );
    }

    #[test]
    fn simplify_rx_quarter_turn_to_sx() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RX, Angle64::QUARTER_TURN),
            Some(GateType::SX)
        );
    }

    #[test]
    fn simplify_rx_half_turn_to_x() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RX, Angle64::HALF_TURN),
            Some(GateType::X)
        );
    }

    #[test]
    fn simplify_rx_three_quarters_to_sxdg() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RX, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SXdg)
        );
    }

    #[test]
    fn simplify_ry_quarter_turn_to_sy() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RY, Angle64::QUARTER_TURN),
            Some(GateType::SY)
        );
    }

    #[test]
    fn simplify_ry_half_turn_to_y() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RY, Angle64::HALF_TURN),
            Some(GateType::Y)
        );
    }

    #[test]
    fn simplify_ry_three_quarters_to_sydg() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RY, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn simplify_rzz_quarter_turn_to_szz() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZZ, Angle64::QUARTER_TURN),
            Some(GateType::SZZ)
        );
    }

    #[test]
    fn simplify_rzz_three_quarters_to_szzdg() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZZ, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SZZdg)
        );
    }

    #[test]
    fn simplify_non_special_angle_unchanged() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::RZ, Angle64::from_turn_ratio(1, 6)),
            None
        );
    }

    #[test]
    fn simplify_non_rotation_unchanged() {
        assert_eq!(
            pecos_core::try_simplify_rotation(GateType::H, Angle64::QUARTER_TURN),
            None
        );
    }

    // ==================== half_turn_decomposition tests ====================

    #[test]
    fn rzz_half_turn_decomposes_to_z() {
        assert_eq!(
            pecos_core::half_turn_decomposition(GateType::RZZ, Angle64::HALF_TURN),
            Some(GateType::Z)
        );
    }

    #[test]
    fn rxx_half_turn_decomposes_to_x() {
        assert_eq!(
            pecos_core::half_turn_decomposition(GateType::RXX, Angle64::HALF_TURN),
            Some(GateType::X)
        );
    }

    #[test]
    fn ryy_half_turn_decomposes_to_y() {
        assert_eq!(
            pecos_core::half_turn_decomposition(GateType::RYY, Angle64::HALF_TURN),
            Some(GateType::Y)
        );
    }

    #[test]
    fn rzz_non_half_turn_no_decomposition() {
        assert_eq!(
            pecos_core::half_turn_decomposition(GateType::RZZ, Angle64::QUARTER_TURN),
            None
        );
    }

    // ==================== TickCircuit pass tests ====================

    #[test]
    fn tick_simplify_rz_quarter_to_sz() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::SZ);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn tick_simplify_rz_half_to_z() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::HALF_TURN, &[0]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::Z);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn tick_simplify_rx_quarter_to_sx() {
        let mut tc = TickCircuit::new();
        tc.tick().rx(Angle64::QUARTER_TURN, &[0]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::SX);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn tick_simplify_ry_half_to_y() {
        let mut tc = TickCircuit::new();
        tc.tick().ry(Angle64::HALF_TURN, &[0]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::Y);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn tick_simplify_rzz_quarter_to_szz() {
        let mut tc = TickCircuit::new();
        tc.tick().rzz(Angle64::QUARTER_TURN, &[(0, 1)]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::SZZ);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn tick_simplify_rzz_half_to_zz() {
        let mut tc = TickCircuit::new();
        tc.tick().rzz(Angle64::HALF_TURN, &[(0, 1)]);
        SimplifyRotations.apply_tick(&mut tc);
        let gates = tc.ticks()[0].gate_batches();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::Z);
        assert_eq!(
            gates[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(gates[0].num_gates(), 2);
    }

    #[test]
    fn tick_simplify_rxx_half_to_xx() {
        let mut tc = TickCircuit::new();
        tc.tick().rxx(Angle64::HALF_TURN, &[(0, 1)]);
        SimplifyRotations.apply_tick(&mut tc);
        let gates = tc.ticks()[0].gate_batches();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::X);
        assert_eq!(
            gates[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(gates[0].num_gates(), 2);
    }

    #[test]
    fn tick_non_special_angle_unchanged() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::from_turn_ratio(1, 6), &[0]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::RZ);
        assert_eq!(gate.angles.len(), 1);
    }

    #[test]
    fn tick_non_rotation_unchanged() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::H);
    }

    #[test]
    fn tick_simplify_eighth_turn_to_t() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::from_turn_ratio(1, 8), &[0]);
        SimplifyRotations.apply_tick(&mut tc);
        let gate = &tc.ticks()[0].gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::T);
        assert!(gate.angles.is_empty());
    }

    // ==================== DagCircuit pass tests ====================

    #[test]
    fn dag_simplify_rz_quarter_to_sz() {
        let mut dag = DagCircuit::new();
        dag.rz(Angle64::QUARTER_TURN, &[0]);
        let nodes = dag.nodes();
        SimplifyRotations.apply_dag(&mut dag);
        let gate = dag.gate(nodes[0]).unwrap();
        assert_eq!(gate.gate_type, GateType::SZ);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn dag_simplify_rz_half_to_z() {
        let mut dag = DagCircuit::new();
        dag.rz(Angle64::HALF_TURN, &[0]);
        let nodes = dag.nodes();
        SimplifyRotations.apply_dag(&mut dag);
        let gate = dag.gate(nodes[0]).unwrap();
        assert_eq!(gate.gate_type, GateType::Z);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn dag_simplify_rzz_quarter_to_szz() {
        let mut dag = DagCircuit::new();
        dag.rzz(Angle64::QUARTER_TURN, &[(0, 1)]);
        let nodes = dag.nodes();
        SimplifyRotations.apply_dag(&mut dag);
        let gate = dag.gate(nodes[0]).unwrap();
        assert_eq!(gate.gate_type, GateType::SZZ);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn dag_simplify_rzz_half_to_zz() {
        let mut dag = DagCircuit::new();
        dag.rzz(Angle64::HALF_TURN, &[(0, 1)]);
        SimplifyRotations.apply_dag(&mut dag);
        // The old node is removed, two new Z gates are added.
        let nodes = dag.nodes();
        assert_eq!(nodes.len(), 2);
        for &n in &nodes {
            let g = dag.gate(n).unwrap();
            assert_eq!(g.gate_type, GateType::Z);
            assert!(g.angles.is_empty());
            assert_eq!(g.qubits.len(), 1);
        }
    }

    #[test]
    fn dag_non_special_angle_unchanged() {
        let mut dag = DagCircuit::new();
        dag.rz(Angle64::from_turn_ratio(1, 6), &[0]);
        let nodes = dag.nodes();
        SimplifyRotations.apply_dag(&mut dag);
        let gate = dag.gate(nodes[0]).unwrap();
        assert_eq!(gate.gate_type, GateType::RZ);
        assert_eq!(gate.angles.len(), 1);
    }

    // ==================== Matrix equivalence tests (UnitaryRep level) ====================
    //
    // These verify that each simplification mapping preserves the unitary
    // (up to global phase) by comparing the rotation UnitaryRep against the
    // named-gate UnitaryRep using dense matrix comparison.

    use crate::unitary_matrix::{matrices_equiv_up_to_phase, to_matrix_with_size, unitaries_equiv};

    use pecos_core::unitary_rep::{self, UnitaryRep};

    #[test]
    fn matrix_rz_half_equiv_z() {
        assert!(unitaries_equiv(
            &unitary_rep::RZ(Angle64::HALF_TURN, 0),
            &unitary_rep::Z(0),
        ));
    }

    #[test]
    fn matrix_rz_quarter_equiv_sz() {
        assert!(unitaries_equiv(
            &unitary_rep::RZ(Angle64::QUARTER_TURN, 0),
            &unitary_rep::SZ(0),
        ));
    }

    #[test]
    fn matrix_rz_three_quarters_equiv_szdg() {
        assert!(unitaries_equiv(
            &unitary_rep::RZ(Angle64::THREE_QUARTERS_TURN, 0),
            &unitary_rep::SZ(0).dg(),
        ));
    }

    #[test]
    fn matrix_rz_eighth_equiv_t() {
        assert!(unitaries_equiv(
            &unitary_rep::RZ(Angle64::from_turn_ratio(1, 8), 0),
            &unitary_rep::T(0),
        ));
    }

    #[test]
    fn matrix_rz_seven_eighths_equiv_tdg() {
        assert!(unitaries_equiv(
            &unitary_rep::RZ(Angle64::from_turn_ratio(7, 8), 0),
            &unitary_rep::T(0).dg(),
        ));
    }

    #[test]
    fn matrix_rx_half_equiv_x() {
        assert!(unitaries_equiv(
            &unitary_rep::RX(Angle64::HALF_TURN, 0),
            &unitary_rep::X(0),
        ));
    }

    #[test]
    fn matrix_rx_quarter_equiv_sx() {
        assert!(unitaries_equiv(
            &unitary_rep::RX(Angle64::QUARTER_TURN, 0),
            &unitary_rep::SX(0),
        ));
    }

    #[test]
    fn matrix_rx_three_quarters_equiv_sxdg() {
        assert!(unitaries_equiv(
            &unitary_rep::RX(Angle64::THREE_QUARTERS_TURN, 0),
            &unitary_rep::SX(0).dg(),
        ));
    }

    #[test]
    fn matrix_ry_half_equiv_y() {
        assert!(unitaries_equiv(
            &unitary_rep::RY(Angle64::HALF_TURN, 0),
            &unitary_rep::Y(0),
        ));
    }

    #[test]
    fn matrix_ry_quarter_equiv_sy() {
        assert!(unitaries_equiv(
            &unitary_rep::RY(Angle64::QUARTER_TURN, 0),
            &unitary_rep::SY(0),
        ));
    }

    #[test]
    fn matrix_ry_three_quarters_equiv_sydg() {
        assert!(unitaries_equiv(
            &unitary_rep::RY(Angle64::THREE_QUARTERS_TURN, 0),
            &unitary_rep::SY(0).dg(),
        ));
    }

    #[test]
    fn matrix_rzz_quarter_equiv_szz() {
        assert!(unitaries_equiv(
            &unitary_rep::RZZ(Angle64::QUARTER_TURN, 0, 1),
            &unitary_rep::SZZ(0, 1),
        ));
    }

    #[test]
    fn matrix_rzz_three_quarters_equiv_szzdg() {
        assert!(unitaries_equiv(
            &unitary_rep::RZZ(Angle64::THREE_QUARTERS_TURN, 0, 1),
            &unitary_rep::SZZ(0, 1).dg(),
        ));
    }

    #[test]
    fn matrix_rzz_half_equiv_z_tensor_z() {
        let rzz_pi = unitary_rep::RZZ(Angle64::HALF_TURN, 0, 1);
        let z_z = unitary_rep::Z(0) & unitary_rep::Z(1);
        assert!(unitaries_equiv(&rzz_pi, &z_z));
    }

    #[test]
    fn matrix_rxx_half_equiv_x_tensor_x() {
        let rxx_pi = unitary_rep::RXX(Angle64::HALF_TURN, 0, 1);
        let x_x = unitary_rep::X(0) & unitary_rep::X(1);
        assert!(unitaries_equiv(&rxx_pi, &x_x));
    }

    #[test]
    fn matrix_ryy_half_equiv_y_tensor_y() {
        let ryy_pi = unitary_rep::RYY(Angle64::HALF_TURN, 0, 1);
        let y_y = unitary_rep::Y(0) & unitary_rep::Y(1);
        assert!(unitaries_equiv(&ryy_pi, &y_y));
    }

    // ==================== Full-circuit matrix equivalence tests ====================
    //
    // Convert a TickCircuit to an UnitaryRep chain, compute its unitary,
    // apply SimplifyRotations, compute the new unitary, and compare.

    /// Convert a `TickCircuit` to an `UnitaryRep` by composing gates in order.
    ///
    /// Each tick's gates are tensored (parallel), then ticks are composed
    /// (sequential). Returns `None` for an empty circuit.
    fn tick_circuit_to_unitary(tc: &TickCircuit) -> Option<UnitaryRep> {
        let mut tick_ops: Vec<UnitaryRep> = Vec::new();

        for tick in tc.ticks() {
            if tick.is_empty() {
                continue;
            }
            let mut gate_ops: Vec<UnitaryRep> = Vec::new();
            for gate in tick.iter_gate_batches() {
                let op = gate_to_unitary(gate.as_gate())?;
                gate_ops.push(op);
            }
            // Tensor all gates in this tick (they act on disjoint qubits).
            let tick_op = gate_ops.into_iter().reduce(|a, b| a & b).unwrap();
            tick_ops.push(tick_op);
        }

        if tick_ops.is_empty() {
            return None;
        }

        // Compose ticks: last tick is outermost in matrix multiplication.
        // UnitaryRep::Compose applies in reverse (like matrix multiplication),
        // so we reverse to get time-ordering right.
        tick_ops.reverse();
        Some(tick_ops.into_iter().reduce(|a, b| a * b).unwrap())
    }

    /// Convert a single `Gate` to an `UnitaryRep`.
    fn gate_to_unitary(gate: &pecos_core::Gate) -> Option<UnitaryRep> {
        let arity = gate.gate_type.quantum_arity();
        let mut ops = Vec::new();
        for qubits in gate.qubits.chunks(arity) {
            if qubits.len() != arity {
                return None;
            }
            ops.push(gate_instance_to_unitary(gate, qubits)?);
        }
        ops.into_iter().reduce(|a, b| a & b)
    }

    fn gate_instance_to_unitary(gate: &pecos_core::Gate, qubits: &[QubitId]) -> Option<UnitaryRep> {
        let q0 = qubits.first().copied()?;
        match gate.gate_type {
            GateType::H => Some(unitary_rep::H(q0)),
            GateType::X => Some(unitary_rep::X(q0)),
            GateType::Y => Some(unitary_rep::Y(q0)),
            GateType::Z => Some(unitary_rep::Z(q0)),
            GateType::SX => Some(unitary_rep::SX(q0)),
            GateType::SXdg => Some(unitary_rep::SX(q0).dg()),
            GateType::SY => Some(unitary_rep::SY(q0)),
            GateType::SYdg => Some(unitary_rep::SY(q0).dg()),
            GateType::SZ => Some(unitary_rep::SZ(q0)),
            GateType::SZdg => Some(unitary_rep::SZ(q0).dg()),
            GateType::T => Some(unitary_rep::T(q0)),
            GateType::Tdg => Some(unitary_rep::T(q0).dg()),
            GateType::RX => {
                let angle = *gate.angles.first()?;
                Some(unitary_rep::RX(angle, q0))
            }
            GateType::RY => {
                let angle = *gate.angles.first()?;
                Some(unitary_rep::RY(angle, q0))
            }
            GateType::RZ => {
                let angle = *gate.angles.first()?;
                Some(unitary_rep::RZ(angle, q0))
            }
            GateType::CX => {
                let q1 = qubits.get(1).copied()?;
                Some(unitary_rep::CX(q0, q1))
            }
            GateType::CY => {
                let q1 = qubits.get(1).copied()?;
                Some(unitary_rep::CY(q0, q1))
            }
            GateType::CZ => {
                let q1 = qubits.get(1).copied()?;
                Some(unitary_rep::CZ(q0, q1))
            }
            GateType::RXX => {
                let q1 = qubits.get(1).copied()?;
                let angle = *gate.angles.first()?;
                Some(unitary_rep::RXX(angle, q0, q1))
            }
            GateType::RYY => {
                let q1 = qubits.get(1).copied()?;
                let angle = *gate.angles.first()?;
                Some(unitary_rep::RYY(angle, q0, q1))
            }
            GateType::RZZ => {
                let q1 = qubits.get(1).copied()?;
                let angle = *gate.angles.first()?;
                Some(unitary_rep::RZZ(angle, q0, q1))
            }
            GateType::SZZ => {
                let q1 = qubits.get(1).copied()?;
                Some(unitary_rep::SZZ(q0, q1))
            }
            GateType::SZZdg => {
                let q1 = qubits.get(1).copied()?;
                Some(unitary_rep::SZZ(q0, q1).dg())
            }
            GateType::I | GateType::Idle => Some(unitary_rep::I(q0)),
            _ => None,
        }
    }

    /// Assert that two `TickCircuit`s produce the same unitary (up to global phase).
    fn assert_circuits_equiv(a: &TickCircuit, b: &TickCircuit) {
        let op_a = tick_circuit_to_unitary(a).expect("circuit A should be non-empty");
        let op_b = tick_circuit_to_unitary(b).expect("circuit B should be non-empty");

        // Determine qubit count from both operators.
        let nq_a = op_a.qubits().into_iter().max().map_or(1, |q| q + 1);
        let nq_b = op_b.qubits().into_iter().max().map_or(1, |q| q + 1);
        let num_qubits = nq_a.max(nq_b);

        let mat_a = to_matrix_with_size(&op_a, num_qubits);
        let mat_b = to_matrix_with_size(&op_b, num_qubits);

        assert!(
            matrices_equiv_up_to_phase(&mat_a, &mat_b, 1e-10),
            "circuits are not unitarily equivalent (up to global phase)",
        );
    }

    #[test]
    fn circuit_equiv_single_rz_quarter() {
        let mut original = TickCircuit::new();
        original.tick().rz(Angle64::QUARTER_TURN, &[0]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_single_rz_half() {
        let mut original = TickCircuit::new();
        original.tick().rz(Angle64::HALF_TURN, &[0]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_single_rx_quarter() {
        let mut original = TickCircuit::new();
        original.tick().rx(Angle64::QUARTER_TURN, &[0]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_single_ry_half() {
        let mut original = TickCircuit::new();
        original.tick().ry(Angle64::HALF_TURN, &[0]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_rzz_quarter() {
        let mut original = TickCircuit::new();
        original.tick().rzz(Angle64::QUARTER_TURN, &[(0, 1)]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_rzz_half_decomposition() {
        let mut original = TickCircuit::new();
        original.tick().rzz(Angle64::HALF_TURN, &[(0, 1)]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_rxx_half_decomposition() {
        let mut original = TickCircuit::new();
        original.tick().rxx(Angle64::HALF_TURN, &[(0, 1)]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_ryy_half_decomposition() {
        let mut original = TickCircuit::new();
        original.tick().ryy(Angle64::HALF_TURN, &[(0, 1)]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_multi_gate_mixed() {
        // A circuit with multiple rotation gates, some simplifiable, some not.
        let mut original = TickCircuit::new();
        original
            .tick()
            .rz(Angle64::HALF_TURN, &[0])
            .rx(Angle64::QUARTER_TURN, &[1]);
        original.tick().cx(&[(0, 1)]);
        original
            .tick()
            .rz(Angle64::from_turn_ratio(1, 8), &[0])
            .ry(Angle64::THREE_QUARTERS_TURN, &[1]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_mixed_with_non_special_angles() {
        // Mix of simplifiable and non-simplifiable rotations.
        let mut original = TickCircuit::new();
        original
            .tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::from_turn_ratio(1, 6), &[1]);
        original.tick().h(&[0, 1]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_rzz_half_in_larger_circuit() {
        // RZZ decomposition embedded in a multi-tick circuit.
        let mut original = TickCircuit::new();
        original.tick().h(&[0, 1]);
        original.tick().rzz(Angle64::HALF_TURN, &[(0, 1)]);
        original.tick().h(&[0, 1]);
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_all_single_qubit_simplifications() {
        // One gate for every single-qubit entry in the mapping table.
        let seventh_eighth = Angle64::from_turn_ratio(7, 8);
        let eighth = Angle64::from_turn_ratio(1, 8);
        let mut original = TickCircuit::new();
        original
            .tick()
            .rz(Angle64::HALF_TURN, &[0]) // -> Z
            .rz(Angle64::QUARTER_TURN, &[1]) // -> SZ
            .rz(Angle64::THREE_QUARTERS_TURN, &[2]) // -> SZdg
            .rz(eighth, &[3]); // -> T
        original
            .tick()
            .rz(seventh_eighth, &[0]) // -> Tdg
            .rx(Angle64::HALF_TURN, &[1]) // -> X
            .rx(Angle64::QUARTER_TURN, &[2]) // -> SX
            .rx(Angle64::THREE_QUARTERS_TURN, &[3]); // -> SXdg
        original
            .tick()
            .ry(Angle64::HALF_TURN, &[0]) // -> Y
            .ry(Angle64::QUARTER_TURN, &[1]) // -> SY
            .ry(Angle64::THREE_QUARTERS_TURN, &[2]); // -> SYdg
        let mut simplified = original.clone();
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    // ==================== is_identity_gate tests ====================

    #[test]
    fn identity_gate_i() {
        let gate = Gate::i(&[0]);
        assert!(is_identity_gate(&gate));
    }

    #[test]
    fn identity_gate_idle() {
        let gate = Gate::idle(1.0, vec![QubitId::from(0)]);
        assert!(is_identity_gate(&gate));
    }

    #[test]
    fn identity_gate_rz_zero() {
        let gate = Gate::rz(Angle64::ZERO, &[0]);
        assert!(is_identity_gate(&gate));
    }

    #[test]
    fn identity_gate_rxx_zero() {
        let gate = Gate::rxx(Angle64::ZERO, &[(0, 1)]);
        assert!(is_identity_gate(&gate));
    }

    #[test]
    fn not_identity_gate_rz_nonzero() {
        let gate = Gate::rz(Angle64::QUARTER_TURN, &[0]);
        assert!(!is_identity_gate(&gate));
    }

    #[test]
    fn not_identity_gate_h() {
        let gate = Gate::h(&[0]);
        assert!(!is_identity_gate(&gate));
    }

    // ==================== is_self_inverse tests ====================

    #[test]
    fn self_inverse_x() {
        assert!(is_self_inverse(GateType::X));
    }

    #[test]
    fn self_inverse_cx() {
        assert!(is_self_inverse(GateType::CX));
    }

    #[test]
    fn not_self_inverse_sx() {
        assert!(!is_self_inverse(GateType::SX));
    }

    // ==================== named_inverse tests ====================

    #[test]
    fn named_inverse_sx_sxdg() {
        assert_eq!(named_inverse(GateType::SX), Some(GateType::SXdg));
        assert_eq!(named_inverse(GateType::SXdg), Some(GateType::SX));
    }

    #[test]
    fn named_inverse_t_tdg() {
        assert_eq!(named_inverse(GateType::T), Some(GateType::Tdg));
        assert_eq!(named_inverse(GateType::Tdg), Some(GateType::T));
    }

    #[test]
    fn named_inverse_szz_szzdg() {
        assert_eq!(named_inverse(GateType::SZZ), Some(GateType::SZZdg));
        assert_eq!(named_inverse(GateType::SZZdg), Some(GateType::SZZ));
    }

    #[test]
    fn named_inverse_h_none() {
        assert_eq!(named_inverse(GateType::H), None);
    }

    // ==================== are_inverses tests ====================

    #[test]
    fn inverses_x_x() {
        let a = Gate::x(&[0]);
        let b = Gate::x(&[0]);
        assert!(are_inverses(&a, &b));
    }

    #[test]
    fn inverses_cx_cx() {
        let a = Gate::cx(&[(0, 1)]);
        let b = Gate::cx(&[(0, 1)]);
        assert!(are_inverses(&a, &b));
    }

    #[test]
    fn inverses_sx_sxdg() {
        let a = Gate::sx(&[0]);
        let b = Gate::sxdg(&[0]);
        assert!(are_inverses(&a, &b));
    }

    #[test]
    fn inverses_rz_neg() {
        let angle = Angle64::QUARTER_TURN;
        let a = Gate::rz(angle, &[0]);
        let b = Gate::rz(-angle, &[0]);
        assert!(are_inverses(&a, &b));
    }

    #[test]
    fn not_inverses_different_qubits() {
        let a = Gate::x(&[0]);
        let b = Gate::x(&[1]);
        assert!(!are_inverses(&a, &b));
    }

    // ==================== RemoveIdentity tick tests ====================

    #[test]
    fn tick_remove_identity_i() {
        let mut tc = TickCircuit::new();
        tc.tick();
        tc.ticks_mut()[0].add_gate(Gate::i(&[0]));
        RemoveIdentity.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
    }

    #[test]
    fn tick_remove_identity_rz_zero() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::ZERO, &[0]);
        RemoveIdentity.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
    }

    #[test]
    fn tick_remove_identity_preserves_nonzero() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        RemoveIdentity.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].gate_batches().len(), 1);
        assert_eq!(tc.ticks()[0].gate_batches()[0].gate_type, GateType::RZ);
    }

    #[test]
    fn tick_remove_identity_mixed() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.ticks_mut()[0].add_gate(Gate::i(&[1]));
        RemoveIdentity.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].gate_batches().len(), 1);
        assert_eq!(tc.ticks()[0].gate_batches()[0].gate_type, GateType::H);
    }

    // ==================== RemoveIdentity DAG tests ====================

    #[test]
    fn dag_remove_identity_i() {
        let mut dag = DagCircuit::new();
        dag.add_gate(Gate::i(&[0]));
        RemoveIdentity.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn dag_remove_identity_rz_zero() {
        let mut dag = DagCircuit::new();
        dag.rz(Angle64::ZERO, &[0]);
        RemoveIdentity.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn dag_remove_identity_preserves_h() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        RemoveIdentity.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 1);
    }

    #[test]
    fn dag_remove_identity_rewires() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.add_gate(Gate::i(&[0]));
        dag.z(&[0]);
        let nodes_before = dag.nodes();
        assert_eq!(nodes_before.len(), 3);
        RemoveIdentity.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 2);
    }

    // ==================== CancelInverses tick tests ====================

    #[test]
    fn tick_cancel_h_h() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().h(&[0]);
        CancelInverses.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
        assert!(tc.ticks()[1].gate_batches().is_empty());
    }

    #[test]
    fn tick_cancel_x_x() {
        let mut tc = TickCircuit::new();
        tc.tick().x(&[0]);
        tc.tick().x(&[0]);
        CancelInverses.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
        assert!(tc.ticks()[1].gate_batches().is_empty());
    }

    #[test]
    fn tick_cancel_sx_sxdg() {
        let mut tc = TickCircuit::new();
        tc.tick().sx(&[0]);
        tc.tick();
        tc.ticks_mut()[1].add_gate(Gate::sxdg(&[0]));
        CancelInverses.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
        assert!(tc.ticks()[1].gate_batches().is_empty());
    }

    #[test]
    fn tick_cancel_t_tdg() {
        let mut tc = TickCircuit::new();
        tc.tick().t(&[0]);
        tc.tick();
        tc.ticks_mut()[1].add_gate(Gate::tdg(&[0]));
        CancelInverses.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
        assert!(tc.ticks()[1].gate_batches().is_empty());
    }

    #[test]
    fn tick_cancel_cx_cx() {
        let mut tc = TickCircuit::new();
        tc.tick().cx(&[(0, 1)]);
        tc.tick().cx(&[(0, 1)]);
        CancelInverses.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
        assert!(tc.ticks()[1].gate_batches().is_empty());
    }

    #[test]
    fn tick_cancel_rz_neg() {
        let angle = Angle64::QUARTER_TURN;
        let mut tc = TickCircuit::new();
        tc.tick().rz(angle, &[0]);
        tc.tick().rz(-angle, &[0]);
        CancelInverses.apply_tick(&mut tc);
        assert!(tc.ticks()[0].gate_batches().is_empty());
        assert!(tc.ticks()[1].gate_batches().is_empty());
    }

    #[test]
    fn tick_cancel_nested() {
        // H T Tdg H -> all cancel
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().t(&[0]);
        tc.tick();
        tc.ticks_mut()[2].add_gate(Gate::tdg(&[0]));
        tc.tick().h(&[0]);
        CancelInverses.apply_tick(&mut tc);
        for tick in tc.ticks() {
            assert!(tick.gate_batches().is_empty());
        }
    }

    #[test]
    fn tick_no_cancel_with_intervening_gate() {
        // H X H -> no cancellation (X on same qubit between the H gates)
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().x(&[0]);
        tc.tick().h(&[0]);
        CancelInverses.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].gate_batches().len(), 1);
        assert_eq!(tc.ticks()[1].gate_batches().len(), 1);
        assert_eq!(tc.ticks()[2].gate_batches().len(), 1);
    }

    #[test]
    fn tick_no_cancel_different_qubits() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().h(&[1]);
        CancelInverses.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].gate_batches().len(), 1);
        assert_eq!(tc.ticks()[1].gate_batches().len(), 1);
    }

    // ==================== CancelInverses DAG tests ====================

    #[test]
    fn dag_cancel_h_h() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]).h(&[0]);
        CancelInverses.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn dag_cancel_cx_cx() {
        let mut dag = DagCircuit::new();
        dag.cx(&[(0, 1)]).cx(&[(0, 1)]);
        CancelInverses.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn dag_cancel_rz_neg() {
        let angle = Angle64::QUARTER_TURN;
        let mut dag = DagCircuit::new();
        dag.rz(angle, &[0]).rz(-angle, &[0]);
        CancelInverses.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn dag_no_cancel_with_intervening_gate() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]).x(&[0]).h(&[0]);
        CancelInverses.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 3);
    }

    // ==================== MergeAdjacentRotations tick tests ====================

    #[test]
    fn tick_merge_rz_rz() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        MergeAdjacentRotations.apply_tick(&mut tc);
        let gate = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .next()
            .unwrap();
        assert_eq!(gate.gate_type, GateType::RZ);
        assert_eq!(gate.angles[0], Angle64::HALF_TURN);
    }

    #[test]
    fn tick_merge_chain_of_three() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        MergeAdjacentRotations.apply_tick(&mut tc);
        let gate = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .next()
            .unwrap();
        assert_eq!(gate.gate_type, GateType::RZ);
        assert_eq!(gate.angles[0], Angle64::THREE_QUARTERS_TURN);
    }

    #[test]
    fn tick_merge_to_zero() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().rz(Angle64::THREE_QUARTERS_TURN, &[0]);
        MergeAdjacentRotations.apply_tick(&mut tc);
        let gate = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .next()
            .unwrap();
        assert_eq!(gate.gate_type, GateType::RZ);
        assert!(gate.angles[0].is_zero());
    }

    #[test]
    fn tick_merge_rzz() {
        let mut tc = TickCircuit::new();
        tc.tick().rzz(Angle64::QUARTER_TURN, &[(0, 1)]);
        tc.tick().rzz(Angle64::QUARTER_TURN, &[(0, 1)]);
        MergeAdjacentRotations.apply_tick(&mut tc);
        let gate = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .next()
            .unwrap();
        assert_eq!(gate.gate_type, GateType::RZZ);
        assert_eq!(gate.angles[0], Angle64::HALF_TURN);
    }

    #[test]
    fn tick_no_merge_different_types() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().rx(Angle64::QUARTER_TURN, &[0]);
        MergeAdjacentRotations.apply_tick(&mut tc);
        assert_eq!(tc.gate_count(), 2);
    }

    #[test]
    fn tick_no_merge_with_intervening_gate() {
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().h(&[0]);
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        MergeAdjacentRotations.apply_tick(&mut tc);
        assert_eq!(tc.gate_count(), 3);
    }

    // ==================== MergeAdjacentRotations DAG tests ====================

    #[test]
    fn dag_merge_rz_rz() {
        let mut dag = DagCircuit::new();
        dag.rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[0]);
        MergeAdjacentRotations.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 1);
        let node = dag.nodes()[0];
        let gate = dag.gate(node).unwrap();
        assert_eq!(gate.gate_type, GateType::RZ);
        assert_eq!(gate.angles[0], Angle64::HALF_TURN);
    }

    #[test]
    fn dag_merge_chain_of_three() {
        let mut dag = DagCircuit::new();
        dag.rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[0]);
        MergeAdjacentRotations.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 1);
        let node = dag.nodes()[0];
        let gate = dag.gate(node).unwrap();
        assert_eq!(gate.angles[0], Angle64::THREE_QUARTERS_TURN);
    }

    #[test]
    fn dag_no_merge_with_intervening_gate() {
        let mut dag = DagCircuit::new();
        dag.rz(Angle64::QUARTER_TURN, &[0])
            .h(&[0])
            .rz(Angle64::QUARTER_TURN, &[0]);
        MergeAdjacentRotations.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 3);
    }

    // ==================== New pass matrix equivalence tests ====================

    #[test]
    fn circuit_equiv_remove_identity() {
        let mut original = TickCircuit::new();
        original.tick().h(&[0]);
        original.ticks_mut()[0].add_gate(Gate::i(&[1]));
        original.tick().cx(&[(0, 1)]);
        let mut simplified = original.clone();
        RemoveIdentity.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_cancel_inverses() {
        let mut original = TickCircuit::new();
        original.tick().h(&[0, 1]);
        original.tick().sx(&[0]).t(&[1]);
        original.tick();
        original.ticks_mut()[2].add_gate(Gate::sxdg(&[0]));
        original.ticks_mut()[2].add_gate(Gate::tdg(&[1]));
        original.tick().cx(&[(0, 1)]);
        let mut simplified = original.clone();
        CancelInverses.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_merge_adjacent() {
        let mut original = TickCircuit::new();
        original.tick().rz(Angle64::QUARTER_TURN, &[0]);
        original.tick().rz(Angle64::QUARTER_TURN, &[0]);
        let mut simplified = original.clone();
        MergeAdjacentRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_merge_then_simplify() {
        let mut original = TickCircuit::new();
        original.tick().rz(Angle64::QUARTER_TURN, &[0]).h(&[1]);
        original.tick().rz(Angle64::QUARTER_TURN, &[0]);
        original.tick().cx(&[(0, 1)]);
        let mut simplified = original.clone();
        MergeAdjacentRotations.apply_tick(&mut simplified);
        SimplifyRotations.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_merge_then_remove_identity() {
        let mut original = TickCircuit::new();
        original.tick().h(&[0]);
        original.tick().rz(Angle64::QUARTER_TURN, &[0]);
        original.tick().rz(Angle64::THREE_QUARTERS_TURN, &[0]);
        original.tick().h(&[0]);
        let mut simplified = original.clone();
        MergeAdjacentRotations.apply_tick(&mut simplified);
        RemoveIdentity.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    #[test]
    fn circuit_equiv_full_pipeline() {
        let mut original = TickCircuit::new();
        original.tick().rz(Angle64::QUARTER_TURN, &[0]).h(&[1]);
        original.tick().rz(Angle64::QUARTER_TURN, &[0]).h(&[1]);
        original.tick().cx(&[(0, 1)]);
        let mut simplified = original.clone();
        MergeAdjacentRotations.apply_tick(&mut simplified);
        RemoveIdentity.apply_tick(&mut simplified);
        SimplifyRotations.apply_tick(&mut simplified);
        CancelInverses.apply_tick(&mut simplified);
        assert_circuits_equiv(&original, &simplified);
    }

    // ==================== Pass effectiveness analysis ====================

    /// Count stored gate batches across all ticks.
    fn count_gate_batches(tc: &TickCircuit) -> usize {
        tc.ticks().iter().map(|t| t.gate_batches().len()).sum()
    }

    /// Apply the full pipeline and return (before, after) gate-batch counts.
    fn pipeline_stats(tc: &mut TickCircuit) -> (usize, usize) {
        let before = count_gate_batches(tc);
        MergeAdjacentRotations.apply_tick(tc);
        RemoveIdentity.apply_tick(tc);
        SimplifyRotations.apply_tick(tc);
        CancelInverses.apply_tick(tc);
        PeepholeOptimize.apply_tick(tc);
        let after = count_gate_batches(tc);
        (before, after)
    }

    #[test]
    fn analysis_pass_effectiveness() {
        // -- Circuit 1: Redundant basis changes (common in compiled circuits) --
        // Pattern: H-CX-H on target qubit is equivalent to CZ
        let mut c1 = TickCircuit::new();
        c1.tick().h(&[1]);
        c1.tick().cx(&[(0, 1)]);
        c1.tick().h(&[1]);
        // PeepholeOptimize: H(target) CX(c,t) H(target) -> CZ(c,t)
        let (b1, a1) = pipeline_stats(&mut c1);

        // -- Circuit 2: Rotation accumulation (variational / compiled) --
        let mut c2 = TickCircuit::new();
        c2.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::from_turn_ratio(1, 8), &[1]);
        c2.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::from_turn_ratio(1, 8), &[1]);
        c2.tick().cx(&[(0, 1)]);
        c2.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::from_turn_ratio(3, 8), &[1]);
        c2.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::from_turn_ratio(3, 8), &[1]);
        // Merge: RZ(pi/2)+RZ(pi/2)->RZ(pi) on q0, RZ(1/8)+RZ(1/8)->RZ(1/4) on q1
        // Simplify: RZ(pi)->Z, RZ(pi/4)->T, etc.
        // After CX: same pattern again
        let (b2, a2) = pipeline_stats(&mut c2);

        // -- Circuit 3: Inverse cancellation (from circuit composition) --
        let mut c3 = TickCircuit::new();
        c3.tick().h(&[0, 1]);
        c3.tick().t(&[0]).sx(&[1]);
        c3.tick().cx(&[(0, 1)]);
        c3.tick().cx(&[(0, 1)]); // CX*CX = I
        c3.tick();
        c3.ticks_mut()[4].add_gate(Gate::tdg(&[0]));
        c3.ticks_mut()[4].add_gate(Gate::sxdg(&[1]));
        c3.tick().h(&[0, 1]); // H*H = I (but intervening gates block)
        c3.tick().z(&[0]); // actual operation
        let (b3, a3) = pipeline_stats(&mut c3);

        // -- Circuit 4: Zero-angle rotations (from parameterized circuits at theta=0) --
        let mut c4 = TickCircuit::new();
        c4.tick().h(&[0, 1, 2]);
        c4.tick()
            .rz(Angle64::ZERO, &[0])
            .rx(Angle64::ZERO, &[1])
            .ry(Angle64::ZERO, &[2]);
        c4.tick().cx(&[(0, 1)]);
        c4.tick().cz(&[(1, 2)]);
        c4.tick().rz(Angle64::ZERO, &[0]).rz(Angle64::ZERO, &[1]);
        c4.tick().h(&[0, 1, 2]);
        let (b4, a4) = pipeline_stats(&mut c4);

        // -- Circuit 5: Mixed redundancies (realistic compiled output) --
        let mut c5 = TickCircuit::new();
        c5.tick().h(&[0, 1, 2, 3]);
        // Rotation chain on q0
        c5.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[1]);
        c5.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[1]);
        c5.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[1]);
        c5.tick()
            .rz(Angle64::QUARTER_TURN, &[0])
            .rz(Angle64::QUARTER_TURN, &[1]);
        // Identity rotations on q2, q3
        c5.tick().rz(Angle64::ZERO, &[2]).rx(Angle64::ZERO, &[3]);
        // Two-qubit rotation merge
        c5.tick().rzz(Angle64::QUARTER_TURN, &[(0, 1)]);
        c5.tick().rzz(Angle64::QUARTER_TURN, &[(0, 1)]);
        // Self-inverse pair
        c5.tick().h(&[2, 3]);
        c5.tick().h(&[2, 3]);
        c5.tick().cx(&[(0, 1)]).cz(&[(2, 3)]);
        let (b5, a5) = pipeline_stats(&mut c5);

        // -- Circuit 6: Steane-style syndrome extraction fragment --
        let mut c6 = TickCircuit::new();
        // Ancilla prep
        c6.tick().h(&[4, 5, 6]);
        // CNOT fan-out
        c6.tick().cx(&[(4, 0)]);
        c6.tick().cx(&[(4, 1)]);
        c6.tick().cx(&[(5, 1)]);
        c6.tick().cx(&[(5, 2)]);
        c6.tick().cx(&[(6, 2)]);
        c6.tick().cx(&[(6, 3)]);
        // Ancilla readout
        c6.tick().h(&[4, 5, 6]);
        // No redundancy here -- well-optimized QEC circuit
        let (b6, a6) = pipeline_stats(&mut c6);

        println!();
        println!("=== Pass Pipeline Effectiveness ===");
        println!(
            "Pipeline: MergeAdjacentRotations -> RemoveIdentity -> SimplifyRotations -> CancelInverses -> PeepholeOptimize"
        );
        println!();
        println!(
            "{:<45} {:>6} {:>6} {:>7}",
            "Circuit", "Before", "After", "Saved"
        );
        println!("{:-<45} {:->6} {:->6} {:->7}", "", "", "", "");
        for (name, b, a) in [
            ("1. Basis change (H-CX-H)", b1, a1),
            ("2. Rotation accumulation", b2, a2),
            ("3. Inverse cancellation (composed)", b3, a3),
            ("4. Zero-angle rotations (theta=0)", b4, a4),
            ("5. Mixed redundancies (compiled)", b5, a5),
            ("6. QEC syndrome extraction", b6, a6),
        ] {
            let saved = b.saturating_sub(a);
            let pct = if b > 0 {
                saved as f64 / b as f64 * 100.0
            } else {
                0.0
            };
            println!("{name:<45} {b:>6} {a:>6} {saved:>4} ({pct:.0}%)");
        }
        println!();
    }

    // ==================== peephole_conjugation helper tests ====================

    #[test]
    fn peephole_h_cx_target_to_cz() {
        // H on CX target -> CZ
        let gate = Gate::cx(&[(0, 1)]);
        let result = peephole_conjugation(&gate, QubitId::from(1));
        assert!(result.is_some());
        let (gt, qubits) = result.unwrap();
        assert_eq!(gt, GateType::CZ);
        assert_eq!(qubits[0], QubitId::from(0));
        assert_eq!(qubits[1], QubitId::from(1));
    }

    #[test]
    fn peephole_h_cz_to_cx() {
        // H on CZ qubit -> CX
        let gate = Gate::cz(&[(0, 1)]);
        let result = peephole_conjugation(&gate, QubitId::from(0));
        assert!(result.is_some());
        let (gt, qubits) = result.unwrap();
        assert_eq!(gt, GateType::CX);
        assert_eq!(qubits[0], QubitId::from(1)); // other qubit becomes control
        assert_eq!(qubits[1], QubitId::from(0)); // H qubit becomes target
    }

    #[test]
    fn peephole_h_cx_control_none() {
        // H on CX control -> None (not a valid simplification)
        let gate = Gate::cx(&[(0, 1)]);
        assert!(peephole_conjugation(&gate, QubitId::from(0)).is_none());
    }

    #[test]
    fn peephole_non_matching_none() {
        // H with non-CX/CZ gate -> None
        let gate = Gate::h(&[0]);
        assert!(peephole_conjugation(&gate, QubitId::from(0)).is_none());
    }

    // ==================== PeepholeOptimize TickCircuit tests ====================

    #[test]
    fn peephole_tick_h_cx_h_to_cz() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[1]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().h(&[1]);
        PeepholeOptimize.apply_tick(&mut tc);
        // Should have 1 gate total: CZ(0,1)
        let gates: Vec<&Gate> = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .collect();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::CZ);
        assert_eq!(gates[0].qubits[0], QubitId::from(0));
        assert_eq!(gates[0].qubits[1], QubitId::from(1));
    }

    #[test]
    fn peephole_tick_h_cz_h_to_cx() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cz(&[(0, 1)]);
        tc.tick().h(&[0]);
        PeepholeOptimize.apply_tick(&mut tc);
        let gates: Vec<&Gate> = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .collect();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::CX);
        assert_eq!(gates[0].qubits[0], QubitId::from(1)); // other is control
        assert_eq!(gates[0].qubits[1], QubitId::from(0)); // H qubit is target
    }

    #[test]
    fn peephole_tick_no_match_wrong_qubit() {
        // H on CX control qubit does not trigger
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().h(&[0]);
        PeepholeOptimize.apply_tick(&mut tc);
        let gates: Vec<&Gate> = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .collect();
        assert_eq!(gates.len(), 3); // unchanged
    }

    #[test]
    fn peephole_tick_preserves_other_gates() {
        // Surrounding gates are untouched
        let mut tc = TickCircuit::new();
        tc.tick().x(&[2]); // unrelated gate
        tc.tick().h(&[1]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().h(&[1]);
        tc.tick().z(&[2]); // unrelated gate
        PeepholeOptimize.apply_tick(&mut tc);
        let gates: Vec<&Gate> = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .collect();
        assert_eq!(gates.len(), 3); // X, CZ, Z
        assert_eq!(gates[0].gate_type, GateType::X);
        assert_eq!(gates[1].gate_type, GateType::CZ);
        assert_eq!(gates[2].gate_type, GateType::Z);
    }

    #[test]
    fn peephole_tick_preserves_metadata_when_splitting_batched_commands() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[1]);
        tc.tick()
            .cx(&[(0, 1), (2, 3)])
            .meta("calibration", Attribute::String("entangler".into()));
        tc.tick().h(&[1]);

        PeepholeOptimize.apply_tick(&mut tc);

        let middle = tc
            .ticks()
            .iter()
            .find(|tick| !tick.gate_batches().is_empty())
            .expect("peephole result should keep the middle tick");
        assert_eq!(middle.len(), 2);
        assert_eq!(middle.gate_count(), 2);

        let mut saw_rewritten = false;
        let mut saw_untouched = false;
        for gate in middle.iter_gate_batches() {
            assert_eq!(
                gate.get_attr("calibration"),
                Some(&Attribute::String("entangler".into()))
            );
            match gate.gate_type {
                GateType::CZ => {
                    saw_rewritten = true;
                    assert_eq!(
                        gate.qubits.as_slice(),
                        &[QubitId::from(0), QubitId::from(1)]
                    );
                }
                GateType::CX => {
                    saw_untouched = true;
                    assert_eq!(
                        gate.qubits.as_slice(),
                        &[QubitId::from(2), QubitId::from(3)]
                    );
                }
                other => panic!("unexpected gate after peephole optimization: {other:?}"),
            }
        }
        assert!(saw_rewritten);
        assert!(saw_untouched);
    }

    #[test]
    fn split_batched_tick_commands_preserves_payloads_attrs_and_counters() {
        let mut tc = TickCircuit::new();
        let initial_refs = tc.tick().mz(&[0, 1]);
        assert_eq!(initial_refs[0].record_idx, 0);
        assert_eq!(initial_refs[1].record_idx, 1);
        tc.get_tick_mut(0).unwrap().set_gate_attr(
            0,
            "role",
            Attribute::String("measurement".into()),
        );
        tc.tick()
            .cx(&[(2, 3), (4, 5)])
            .meta("role", Attribute::String("entangler".into()));

        split_batched_tick_commands(&mut tc);

        assert_eq!(tc.num_ticks(), 2);
        assert_eq!(tc.next_tick_index(), 2);
        assert_eq!(tc.num_measurements(), 2);

        let meas_tick = tc.get_tick(0).unwrap();
        assert_eq!(meas_tick.len(), 2);
        assert_eq!(meas_tick.gate_count(), 2);
        assert_eq!(
            meas_tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0)]
        );
        assert_eq!(
            meas_tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0)]
        );
        assert_eq!(
            meas_tick.gate_batches()[1].qubits.as_slice(),
            &[QubitId::from(1)]
        );
        assert_eq!(
            meas_tick.gate_batches()[1].meas_ids.as_slice(),
            &[MeasId(1)]
        );
        for batch in meas_tick.iter_gate_batches() {
            assert_eq!(
                batch.get_attr("role"),
                Some(&Attribute::String("measurement".into()))
            );
        }

        let entangler_tick = tc.get_tick(1).unwrap();
        assert_eq!(entangler_tick.len(), 2);
        assert_eq!(entangler_tick.gate_count(), 2);
        assert_eq!(
            entangler_tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(2), QubitId::from(3)]
        );
        assert_eq!(
            entangler_tick.gate_batches()[1].qubits.as_slice(),
            &[QubitId::from(4), QubitId::from(5)]
        );
        for batch in entangler_tick.iter_gate_batches() {
            assert_eq!(
                batch.get_attr("role"),
                Some(&Attribute::String("entangler".into()))
            );
        }

        let later_refs = tc.tick().mz(&[6]);
        assert_eq!(later_refs[0].record_idx, 2);
        assert_eq!(later_refs[0].meas_id, MeasId(2));
        assert_eq!(tc.next_tick_index(), 3);
        assert_eq!(tc.num_measurements(), 3);
    }

    #[test]
    fn peephole_tick_multiple_patterns() {
        // Two independent H-CX-H patterns
        let mut tc = TickCircuit::new();
        tc.tick().h(&[1]).h(&[3]);
        tc.tick().cx(&[(0, 1)]).cx(&[(2, 3)]);
        tc.tick().h(&[1]).h(&[3]);
        PeepholeOptimize.apply_tick(&mut tc);
        let gates: Vec<&Gate> = tc
            .ticks()
            .iter()
            .flat_map(super::super::tick_circuit::Tick::gate_batches)
            .collect();
        assert_eq!(gates.len(), 2);
        assert!(gates.iter().all(|g| g.gate_type == GateType::CZ));
    }

    // ==================== PeepholeOptimize DagCircuit tests ====================

    #[test]
    fn peephole_dag_h_cx_h_to_cz() {
        let mut dag = DagCircuit::new();
        dag.h(&[1]).cx(&[(0, 1)]).h(&[1]);
        PeepholeOptimize.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 1);
        let node = dag.nodes()[0];
        let gate = dag.gate(node).unwrap();
        assert_eq!(gate.gate_type, GateType::CZ);
    }

    #[test]
    fn peephole_dag_h_cz_h_to_cx() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]).cz(&[(0, 1)]).h(&[0]);
        PeepholeOptimize.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 1);
        let node = dag.nodes()[0];
        let gate = dag.gate(node).unwrap();
        assert_eq!(gate.gate_type, GateType::CX);
        assert_eq!(gate.qubits[0], QubitId::from(1)); // other is control
        assert_eq!(gate.qubits[1], QubitId::from(0)); // H qubit is target
    }

    #[test]
    fn peephole_dag_no_match() {
        // H on CX control qubit does not trigger
        let mut dag = DagCircuit::new();
        dag.h(&[0]).cx(&[(0, 1)]).h(&[0]);
        PeepholeOptimize.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 3); // unchanged
    }

    // ==================== Peephole matrix equivalence tests ====================

    #[test]
    fn peephole_preserves_unitary_h_cx_h() {
        // H(1) CX(0,1) H(1) should equal CZ(0,1)
        let mut original = TickCircuit::new();
        original.tick().h(&[1]);
        original.tick().cx(&[(0, 1)]);
        original.tick().h(&[1]);
        let mut optimized = original.clone();
        PeepholeOptimize.apply_tick(&mut optimized);
        assert_circuits_equiv(&original, &optimized);
    }

    #[test]
    fn peephole_preserves_unitary_h_cz_h() {
        // H(0) CZ(0,1) H(0) should equal CX(1,0)
        let mut original = TickCircuit::new();
        original.tick().h(&[0]);
        original.tick().cz(&[(0, 1)]);
        original.tick().h(&[0]);
        let mut optimized = original.clone();
        PeepholeOptimize.apply_tick(&mut optimized);
        assert_circuits_equiv(&original, &optimized);
    }

    #[test]
    fn peephole_pipeline_with_peephole() {
        // Full pipeline on a circuit combining rotation merging and peephole.
        let mut original = TickCircuit::new();
        original.tick().rz(Angle64::QUARTER_TURN, &[0]).h(&[1]);
        original.tick().rz(Angle64::QUARTER_TURN, &[0]);
        original.tick().cx(&[(0, 1)]);
        original.tick().h(&[1]);
        let mut optimized = original.clone();
        MergeAdjacentRotations.apply_tick(&mut optimized);
        RemoveIdentity.apply_tick(&mut optimized);
        SimplifyRotations.apply_tick(&mut optimized);
        CancelInverses.apply_tick(&mut optimized);
        PeepholeOptimize.apply_tick(&mut optimized);
        assert_circuits_equiv(&original, &optimized);
    }

    // ==================== AbsorbBasisGates tick tests ====================

    #[test]
    fn tick_absorb_z_after_prep() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().z(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1); // PZ stays
        assert_eq!(tc.ticks()[1].len(), 0); // Z removed
    }

    #[test]
    fn tick_absorb_rz_after_prep() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().rz(Angle64::from_turn_ratio(3, 7), &[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1);
        assert_eq!(tc.ticks()[1].len(), 0);
    }

    #[test]
    fn tick_absorb_chain_after_prep() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().t(&[0]);
        tc.tick().sz(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1);
        assert_eq!(tc.ticks()[1].len(), 0);
        assert_eq!(tc.ticks()[2].len(), 0);
    }

    #[test]
    fn tick_no_absorb_x_after_prep() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().x(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1);
        assert_eq!(tc.ticks()[1].len(), 1); // X stays
    }

    #[test]
    fn tick_absorb_before_measure() {
        let mut tc = TickCircuit::new();
        tc.tick().sz(&[0]);
        tc.tick().mz(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 0); // SZ removed
        assert_eq!(tc.ticks()[1].len(), 1); // MZ stays
    }

    #[test]
    fn tick_absorb_chain_before_measure() {
        let mut tc = TickCircuit::new();
        tc.tick().t(&[0]);
        tc.tick().sz(&[0]);
        tc.tick().mz(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 0);
        assert_eq!(tc.ticks()[1].len(), 0);
        assert_eq!(tc.ticks()[2].len(), 1);
    }

    #[test]
    fn tick_no_absorb_h_before_measure() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().mz(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1); // H stays
        assert_eq!(tc.ticks()[1].len(), 1);
    }

    #[test]
    fn tick_absorb_both_ends() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().t(&[0]); // absorbed by prep
        tc.tick().x(&[0]); // breaks eigenstate
        tc.tick().sz(&[0]); // absorbed by measure
        tc.tick().mz(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1); // PZ
        assert_eq!(tc.ticks()[1].len(), 0); // T removed
        assert_eq!(tc.ticks()[2].len(), 1); // X stays
        assert_eq!(tc.ticks()[3].len(), 0); // SZ removed
        assert_eq!(tc.ticks()[4].len(), 1); // MZ
    }

    #[test]
    fn tick_z_diagonal_between_non_z_preserved() {
        // PZ -> X -> T -> MZ
        // Forward: T not absorbed (X breaks eigenstate)
        // Backward: T absorbed (before MZ)
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().x(&[0]);
        tc.tick().t(&[0]);
        tc.tick().mz(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1); // PZ
        assert_eq!(tc.ticks()[1].len(), 1); // X stays
        assert_eq!(tc.ticks()[2].len(), 0); // T removed (before MZ)
        assert_eq!(tc.ticks()[3].len(), 1); // MZ
    }

    // ==================== AbsorbBasisGates DAG tests ====================

    #[test]
    fn dag_absorb_z_after_prep() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.z(&[0]);
        dag.h(&[0]); // non-Z-diagonal anchor
        AbsorbBasisGates.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 2); // PZ + H remain
        let topo = dag.topological_order();
        assert_eq!(dag.gate(topo[0]).unwrap().gate_type, GateType::PZ);
        assert_eq!(dag.gate(topo[1]).unwrap().gate_type, GateType::H);
    }

    #[test]
    fn dag_absorb_before_measure() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]); // non-Z-diagonal anchor
        dag.sz(&[0]);
        dag.mz(&[0]);
        AbsorbBasisGates.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 2); // H + MZ remain
        let topo = dag.topological_order();
        assert_eq!(dag.gate(topo[0]).unwrap().gate_type, GateType::H);
        assert_eq!(dag.gate(topo[1]).unwrap().gate_type, GateType::MZ);
    }

    // ==================== AbsorbBasisGates multi-qubit tests ====================

    #[test]
    fn tick_absorb_cz_after_two_preps() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().cz(&[(0, 1)]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1); // PZ(0,1) stays
        assert_eq!(tc.ticks()[1].len(), 0); // CZ removed
    }

    #[test]
    fn tick_no_absorb_cz_after_one_prep() {
        // Only qubit 0 is prepped; qubit 1 is not in Z eigenstate.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().cz(&[(0, 1)]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1); // PZ stays
        assert_eq!(tc.ticks()[1].len(), 1); // CZ stays
    }

    #[test]
    fn tick_absorb_cz_before_two_measures() {
        let mut tc = TickCircuit::new();
        tc.tick().cz(&[(0, 1)]);
        tc.tick().mz(&[0, 1]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 0); // CZ removed
        assert_eq!(tc.ticks()[1].len(), 1); // MZ(0,1) stays
    }

    #[test]
    fn tick_no_absorb_cz_before_one_measure() {
        // Only qubit 0 is measured; qubit 1 continues.
        let mut tc = TickCircuit::new();
        tc.tick().cz(&[(0, 1)]);
        tc.tick().mz(&[0]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1); // CZ stays
        assert_eq!(tc.ticks()[1].len(), 1);
    }

    #[test]
    fn tick_absorb_szz_after_two_preps() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().szz(&[(0, 1)]);
        AbsorbBasisGates.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[1].len(), 0); // SZZ removed
    }

    #[test]
    fn dag_absorb_cz_after_two_preps() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.cz(&[(0, 1)]);
        dag.h(&[0]); // anchor
        dag.h(&[1]); // anchor
        AbsorbBasisGates.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 4); // 2 PZ + 2 H, CZ removed
    }

    #[test]
    fn dag_absorb_cz_before_two_measures() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]); // anchor
        dag.h(&[1]); // anchor
        dag.cz(&[(0, 1)]);
        dag.mz(&[0]);
        dag.mz(&[1]);
        AbsorbBasisGates.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 4); // 2 H + 2 MZ, CZ removed
    }

    // ==================== PassPipeline tests ====================

    #[test]
    fn pipeline_empty_is_noop() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().x(&[0]);
        let pipeline = PassPipeline::new();
        pipeline.apply_tick(&mut tc);
        assert_eq!(tc.ticks()[0].len(), 1);
        assert_eq!(tc.ticks()[1].len(), 1);
    }

    #[test]
    fn pipeline_applies_passes_in_order() {
        // RZ(pi/4) RZ(pi/4) -> merge to RZ(pi/2) -> simplify to SZ
        let mut tc = TickCircuit::new();
        tc.tick().rz(Angle64::from_turn_ratio(1, 8), &[0]);
        tc.tick().rz(Angle64::from_turn_ratio(1, 8), &[0]);
        let pipeline = PassPipeline::new()
            .then(MergeAdjacentRotations)
            .then(SimplifyRotations);
        pipeline.apply_tick(&mut tc);
        assert_eq!(count_gate_batches(&tc), 1);
        assert_eq!(tc.ticks()[0].gate_batches()[0].gate_type, GateType::SZ);
    }

    #[test]
    fn pipeline_full_tick() {
        // PZ -> T -> RZ(q) -> RZ(q) -> H -> H -> MZ
        // AbsorbBasisGates removes T (after prep), merging combines RZs,
        // CancelInverses removes H-H pair.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0]);
        tc.tick().t(&[0]);
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().rz(Angle64::QUARTER_TURN, &[0]);
        tc.tick().h(&[0]);
        tc.tick().h(&[0]);
        tc.tick().mz(&[0]);
        let pipeline = PassPipeline::new()
            .then(AbsorbBasisGates)
            .then(MergeAdjacentRotations)
            .then(RemoveIdentity)
            .then(SimplifyRotations)
            .then(CancelInverses);
        pipeline.apply_tick(&mut tc);
        // PZ stays, T and both RZs absorbed (after PZ), H+H cancelled, MZ stays
        assert_eq!(count_gate_batches(&tc), 2); // PZ + MZ
    }

    #[test]
    fn pipeline_full_dag() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.z(&[0]); // absorbed after prep
        dag.h(&[0]);
        dag.h(&[0]); // cancel with previous H
        dag.mz(&[0]);
        let pipeline = PassPipeline::new()
            .then(AbsorbBasisGates)
            .then(CancelInverses);
        pipeline.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 2); // PZ + MZ
    }

    #[test]
    fn pipeline_default_is_empty() {
        let pipeline = PassPipeline::default();
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        pipeline.apply_tick(&mut tc);
        assert_eq!(count_gate_batches(&tc), 1);
    }

    // ==================== CompactTicks tests ====================

    #[test]
    fn compact_independent_gates_merge_into_one_tick() {
        // H(0) and X(1) are on different qubits -- can be parallel.
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().x(&[1]);
        assert_eq!(tc.num_ticks(), 2);
        CompactTicks.apply_tick(&mut tc);
        assert_eq!(tc.num_ticks(), 1);
        assert_eq!(tc.ticks()[0].len(), 2);
    }

    #[test]
    fn compact_dependent_gates_stay_sequential() {
        // H(0) then X(0) -- same qubit, must stay in order.
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().x(&[0]);
        CompactTicks.apply_tick(&mut tc);
        assert_eq!(tc.num_ticks(), 2);
        assert_eq!(tc.ticks()[0].gate_batches()[0].gate_type, GateType::H);
        assert_eq!(tc.ticks()[1].gate_batches()[0].gate_type, GateType::X);
    }

    #[test]
    fn compact_removes_empty_ticks() {
        // After CancelInverses there may be empty ticks.
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().h(&[0]); // will be cancelled
        tc.tick().x(&[0]);
        CancelInverses.apply_tick(&mut tc);
        // Now ticks 0 and 1 are empty, tick 2 has X.
        assert_eq!(tc.num_ticks(), 3);
        CompactTicks.apply_tick(&mut tc);
        assert_eq!(tc.num_ticks(), 1);
        assert_eq!(tc.ticks()[0].gate_batches()[0].gate_type, GateType::X);
    }

    #[test]
    fn compact_empty_circuit() {
        let mut tc = TickCircuit::new();
        tc.tick(); // empty tick
        tc.tick(); // another empty tick
        CompactTicks.apply_tick(&mut tc);
        assert_eq!(tc.num_ticks(), 0);
    }

    #[test]
    fn compact_already_optimal() {
        // All gates on different qubits in one tick -- already optimal.
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]).z(&[2]);
        CompactTicks.apply_tick(&mut tc);
        assert_eq!(tc.num_ticks(), 1);
        assert_eq!(tc.ticks()[0].len(), 3);
    }

    #[test]
    fn compact_diamond_pattern() {
        // PZ(0,1) -> H(0), X(1) -> CX(0,1) -> MZ(0,1)
        // Spread across 4 ticks but H and X can share a tick.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().h(&[0]);
        tc.tick().x(&[1]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().mz(&[0, 1]);
        assert_eq!(tc.num_ticks(), 5);
        CompactTicks.apply_tick(&mut tc);
        // PZ(0,1) | H(0)+X(1) | CX(0,1) | MZ(0,1) = 4 ticks
        assert_eq!(tc.num_ticks(), 4);
        assert_eq!(tc.ticks()[1].len(), 2); // H and X merged
    }

    #[test]
    fn compact_preserves_gate_order_per_qubit() {
        // Ensure per-qubit ordering is maintained.
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().t(&[0]);
        tc.tick().sz(&[0]);
        CompactTicks.apply_tick(&mut tc);
        assert_eq!(tc.num_ticks(), 3); // all same qubit, no compaction
        assert_eq!(tc.ticks()[0].gate_batches()[0].gate_type, GateType::H);
        assert_eq!(tc.ticks()[1].gate_batches()[0].gate_type, GateType::T);
        assert_eq!(tc.ticks()[2].gate_batches()[0].gate_type, GateType::SZ);
    }

    #[test]
    fn compact_in_pipeline() {
        // Full pipeline: absorb + cancel + compact.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        tc.tick().t(&[0]); // absorbed after prep
        tc.tick().h(&[1]);
        tc.tick().x(&[0]);
        tc.tick().h(&[1]); // H-H cancel
        tc.tick().mz(&[0, 1]);
        let pipeline = PassPipeline::new()
            .then(AbsorbBasisGates)
            .then(CancelInverses)
            .then(CompactTicks);
        pipeline.apply_tick(&mut tc);
        // After absorb+cancel: PZ(0,1), X(0), MZ(0,1)
        // X(0) can't merge with PZ (qubit 0 busy) or MZ (qubit 0 busy).
        assert_eq!(tc.num_ticks(), 3);
        assert_eq!(count_gate_batches(&tc), 3);
    }
}
