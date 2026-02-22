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

//! Conditional control flow handling.
//!
//! Conditionals are HUGR's branching construct. They select one of multiple
//! Case branches based on a Sum type control input (typically from a measurement).
//!
//! # Structure
//!
//! A Conditional node has:
//! - Input port 0: Sum type control value (determines which Case to execute)
//! - Input ports 1+: Data inputs passed to the selected Case
//! - Output ports: Results from the selected Case
//! - Case children: Each Case contains a dataflow subgraph
//!
//! # Execution Flow
//!
//! 1. Conditional encountered during traversal
//! 2. Control value checked (may need measurement results)
//! 3. If known, appropriate Case is expanded
//! 4. Case operations execute
//! 5. Case outputs propagate to Conditional outputs

use std::collections::BTreeSet;

use log::debug;
use pecos_core::gate_type::GateType;
use pecos_quantum::hugr_convert::{
    hugr_op_to_gate_type, is_rotation_gate, try_extract_rotation_angle,
};
use tket::hugr::ops::OpType;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::engine::GuppyHugrEngine;
use crate::engine::analysis::find_input_node;
use crate::engine::types::{ActiveCaseInfo, QuantumOp};

impl GuppyHugrEngine {
    /// Try to resolve the control value for a Conditional node.
    /// Returns `Some(branch_index)` if the control value is known, None otherwise.
    pub(crate) fn try_resolve_conditional_control(
        &self,
        hugr: &Hugr,
        cond_node: Node,
    ) -> Option<usize> {
        // The first input to a Conditional is the Sum type that determines the branch
        let control_port = IncomingPort::from(0);

        if let Some((src_node, src_port)) = hugr.single_linked_output(cond_node, control_port) {
            let wire_key = (src_node, src_port.index());
            let src_op = hugr.get_optype(src_node);

            // Check if we have a classical value for this wire
            if let Some(value) = self.wire_state.classical_values.get(&wire_key)
                && let Some(v) = value.to_u32()
            {
                debug!("Conditional {cond_node:?} control value resolved to {v}");
                return Some(v as usize);
            }

            // Check if the source is a Tag node (creates Sum type from a bool)
            if let OpType::Tag(tag_op) = src_op {
                // Tag has a "tag" field indicating which variant
                // For a bool->Sum conversion, tag 0 = false, tag 1 = true
                let tag_value = tag_op.tag;

                // Check if the Tag's input is a known value
                let tag_input_port = IncomingPort::from(0);
                if let Some((tag_src_node, tag_src_port)) =
                    hugr.single_linked_output(src_node, tag_input_port)
                {
                    let tag_src_wire = (tag_src_node, tag_src_port.index());
                    if let Some(input_value) = self.wire_state.classical_values.get(&tag_src_wire) {
                        // The branch depends on the input value and tag
                        // For bool inputs: tag determines which Sum variant
                        debug!(
                            "Conditional {cond_node:?} resolved via Tag: tag={tag_value}, input={input_value:?}"
                        );
                        return Some(tag_value);
                    }
                }

                // If the Tag has a constant tag value and no dynamic input,
                // the branch is just the tag value
                let num_inputs = hugr.num_inputs(src_node);
                if num_inputs == 0 {
                    return Some(tag_value);
                }
            }
        }

        None
    }

    /// Try to resolve any pending conditionals that were waiting for measurement results.
    pub(crate) fn try_resolve_pending_conditionals(&mut self) {
        let hugr = match &self.hugr {
            Some(h) => h.clone(),
            None => return,
        };

        // Collect conditionals that can now be resolved
        let mut to_resolve = Vec::new();
        for &cond_node in self.pending_conditionals.keys() {
            if let Some(branch_index) = self.try_resolve_conditional_control(&hugr, cond_node) {
                to_resolve.push((cond_node, branch_index));
            }
        }

        // Resolve them
        for (cond_node, branch_index) in to_resolve {
            self.pending_conditionals.remove(&cond_node);

            let entry_nodes = self.expand_conditional(&hugr, cond_node, branch_index);
            let num_entry_nodes = entry_nodes.len();
            for entry_node in entry_nodes {
                if !self.work_queue.contains(&entry_node) && !self.processed.contains(&entry_node) {
                    self.work_queue.push_back(entry_node);
                }
            }

            debug!(
                "Resolved pending Conditional {cond_node:?}, branch {branch_index} selected, added {num_entry_nodes} entry nodes"
            );
        }
    }

    /// Check if any active Case is complete after processing an operation.
    /// If complete, propagate the Case's outputs to the parent Conditional.
    pub(crate) fn check_case_completion(&mut self, hugr: &Hugr, processed_node: Node) {
        // Find which Case (if any) this node belongs to
        let mut completed_cases = Vec::new();

        for (case_node, case_info) in &self.active_cases {
            if case_info.ops_in_case.contains(&processed_node) {
                // Check if all ops in this Case are now processed
                let all_done = case_info
                    .ops_in_case
                    .iter()
                    .all(|op| self.processed.contains(op));

                if all_done {
                    completed_cases.push((*case_node, case_info.conditional_node));
                }
            }
        }

        // Propagate outputs for completed cases
        for (case_node, cond_node) in completed_cases {
            debug!("Case {case_node:?} complete, propagating outputs to Conditional {cond_node:?}");
            self.propagate_conditional_outputs(hugr, cond_node, case_node);
            self.active_cases.remove(&case_node);
        }
    }

    /// Propagate wire mappings from a Case's Output node to the Conditional's outputs.
    ///
    /// After Case operations execute, we need to copy the wire mappings from
    /// the Case Output node's inputs to the Conditional's output ports.
    /// This includes both qubit mappings and classical values (for Sum types from Tag nodes).
    pub(crate) fn propagate_conditional_outputs(
        &mut self,
        hugr: &Hugr,
        cond_node: Node,
        case_node: Node,
    ) {
        use crate::engine::types::ClassicalValue;

        let Some(output_node) = crate::engine::analysis::find_output_node(hugr, case_node) else {
            debug!("No Output node found in Case {case_node:?}");
            return;
        };

        // The Case Output node's inputs correspond to the Conditional's outputs
        let num_outputs = hugr.num_inputs(output_node);
        debug!(
            "Propagating {num_outputs} outputs from Case {case_node:?} Output {output_node:?} to Conditional {cond_node:?}"
        );

        for port_idx in 0..num_outputs {
            let out_in_port = IncomingPort::from(port_idx);

            // Find what's connected to this Output node input
            if let Some((src_node, src_port)) = hugr.single_linked_output(output_node, out_in_port)
            {
                let src_wire = (src_node, src_port.index());

                // Check if we have a qubit mapping for this wire
                if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                    // Map to the Conditional's output port
                    self.wire_state
                        .wire_to_qubit
                        .insert((cond_node, port_idx), qubit_id);
                    debug!(
                        "Mapped Conditional {cond_node:?} output {port_idx} to qubit {qubit_id:?} (from {src_wire:?})"
                    );
                }

                // Check if we have a classical value for this wire
                if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                    self.wire_state
                        .classical_values
                        .insert((cond_node, port_idx), value.clone());
                    debug!(
                        "Mapped Conditional {cond_node:?} output {port_idx} to classical value {value:?} (from {src_wire:?})"
                    );
                }

                // Check if the source is a Tag node - store its tag value and payload
                let src_op = hugr.get_optype(src_node);
                if let OpType::Tag(tag_op) = src_op {
                    let tag_value = tag_op.tag;
                    #[allow(clippy::cast_possible_wrap)] // Tag indices are small
                    self.wire_state
                        .classical_values
                        .insert((cond_node, port_idx), ClassicalValue::Int(tag_value as i64));

                    // Also extract and store the Tag's inputs (Sum payload values)
                    // Store them at "virtual" output ports (1, 2, ...) on the Conditional
                    // These will be used during CFG block transitions
                    let num_tag_inputs = hugr.num_inputs(src_node);
                    for payload_idx in 0..num_tag_inputs {
                        let tag_in_port = IncomingPort::from(payload_idx);
                        if let Some((payload_src_node, payload_src_port)) =
                            hugr.single_linked_output(src_node, tag_in_port)
                        {
                            let payload_src_wire = (payload_src_node, payload_src_port.index());
                            if let Some(payload_value) = self
                                .wire_state
                                .classical_values
                                .get(&payload_src_wire)
                                .cloned()
                            {
                                // Store at virtual output port (port_idx + 1 + payload_idx)
                                // This allows CFG block transitions to find the payload values
                                let virtual_port = port_idx + 1 + payload_idx;
                                debug!(
                                    "Conditional {cond_node:?} Tag payload {payload_idx}: {payload_value:?} at virtual port {virtual_port}"
                                );
                                self.wire_state
                                    .classical_values
                                    .insert((cond_node, virtual_port), payload_value);
                            } else {
                                debug!("No payload value at {payload_src_wire:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Expand a Conditional by selecting the appropriate Case branch.
    /// Returns the entry nodes of the selected Case that should be added to the work queue.
    pub(crate) fn expand_conditional(
        &mut self,
        hugr: &Hugr,
        cond_node: Node,
        branch_index: usize,
    ) -> Vec<Node> {
        let Some(cond_info) = self.conditionals.get(&cond_node).cloned() else {
            debug!("Conditional {cond_node:?} not found in conditionals map");
            return Vec::new();
        };

        if branch_index >= cond_info.cases.len() {
            debug!(
                "Branch index {} out of range for Conditional {:?} with {} cases",
                branch_index,
                cond_node,
                cond_info.cases.len()
            );
            return Vec::new();
        }

        let selected_case = cond_info.cases[branch_index];
        debug!(
            "Expanding Conditional {cond_node:?} branch {branch_index} -> Case {selected_case:?}"
        );

        // Find the Input node inside the selected Case
        // Operations inside the Case connect to this Input node, not to the Case node itself
        let input_node = find_input_node(hugr, selected_case);

        if let Some(input_node) = input_node {
            debug!("Case {selected_case:?} has Input node {input_node:?}");

            // Propagate ALL wires (qubit and classical) from Conditional inputs to the Case's Input node
            // Port 0 is the control (Sum type), ports 1+ are data inputs
            // The Case's Input node outputs correspond to the Conditional's non-control inputs
            let num_cond_inputs = hugr.num_inputs(cond_node);

            // Start from port 1 (skip control), propagate all inputs
            for port_idx in 1..num_cond_inputs {
                let cond_in_port = IncomingPort::from(port_idx);
                if let Some((src_node, src_port)) =
                    hugr.single_linked_output(cond_node, cond_in_port)
                {
                    let src_wire = (src_node, src_port.index());
                    let input_output_idx = port_idx - 1;

                    // Propagate qubit mappings
                    if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                        self.wire_state
                            .wire_to_qubit
                            .insert((input_node, input_output_idx), qubit_id);
                        debug!(
                            "Propagated qubit {qubit_id:?} to Input node {input_node:?} port {input_output_idx}"
                        );
                    }

                    // Also propagate classical values (integers, bools, etc.)
                    if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                        debug!(
                            "Propagated classical value {value:?} from {src_wire:?} to Case Input ({input_node:?}, {input_output_idx})"
                        );
                        self.wire_state
                            .classical_values
                            .insert((input_node, input_output_idx), value);
                    }
                }
            }
        } else {
            debug!("No Input node found in Case {selected_case:?}");
        }

        // Extract operations from the selected Case
        let entry_nodes = self.extract_case_ops(hugr, selected_case);

        // Collect all quantum ops in this Case for tracking completion
        let mut ops_in_case = BTreeSet::new();
        for &node in &entry_nodes {
            ops_in_case.insert(node);
        }
        // Also collect any non-entry ops that were extracted
        for child in hugr.children(selected_case) {
            if self.quantum_ops.contains_key(&child) {
                ops_in_case.insert(child);
            }
        }

        // Register this Case as active so we can propagate outputs when complete
        if ops_in_case.is_empty() {
            // No ops in this Case - propagate outputs immediately
            debug!("Case {selected_case:?} has no quantum ops, propagating outputs immediately");
            self.propagate_conditional_outputs(hugr, cond_node, selected_case);
        } else {
            self.active_cases.insert(
                selected_case,
                ActiveCaseInfo {
                    conditional_node: cond_node,
                    ops_in_case,
                },
            );
            debug!(
                "Registered Case {:?} as active with {} ops",
                selected_case,
                self.active_cases
                    .get(&selected_case)
                    .map_or(0, |c| c.ops_in_case.len())
            );
        }

        // Mark the Conditional as processed
        self.processed.insert(cond_node);

        entry_nodes
    }

    /// Extract quantum operations from inside a Case node (a branch of a Conditional).
    /// This adds the operations to `quantum_ops` and returns the entry nodes (roots) of the Case.
    pub(crate) fn extract_case_ops(&mut self, hugr: &Hugr, case_node: Node) -> Vec<Node> {
        let mut entry_nodes = Vec::new();

        // Iterate over children of the Case node
        for child in hugr.children(case_node) {
            let op = hugr.get_optype(child);

            // Check if this is an extension operation from tket.quantum
            let Some(ext_op) = op.as_extension_op() else {
                continue;
            };

            let ext_id = ext_op.extension_id();
            if ext_id.as_ref() as &str != "tket.quantum" {
                continue;
            }

            let op_name = ext_op.unqualified_id().to_string();
            let Some(gate_type) = hugr_op_to_gate_type(&op_name) else {
                debug!("Unknown quantum operation in Case: {op_name}");
                continue;
            };

            // Determine number of qubit inputs/outputs
            // Use quantum_arity() for most gates to correctly handle CRZ, SWAP, CCX, etc.
            let (num_qubit_inputs, num_qubit_outputs) = match gate_type {
                GateType::QAlloc => (0, 1),
                GateType::QFree | GateType::MeasureFree => (1, 0),
                _ => {
                    let arity = gate_type.quantum_arity();
                    (arity, arity)
                }
            };

            // Extract rotation parameters
            let params = if is_rotation_gate(gate_type) {
                if let Some(angle_turns) = try_extract_rotation_angle(hugr, child, num_qubit_inputs)
                {
                    vec![angle_turns * std::f64::consts::TAU]
                } else {
                    vec![]
                }
            } else {
                vec![]
            };

            // Check if this is an entry node (no quantum predecessors inside the Case)
            let is_entry = hugr.input_neighbours(child).all(|pred| {
                // Entry if predecessor is not a quantum op or is outside this Case
                !self.quantum_ops.contains_key(&pred) || hugr.get_parent(pred) != Some(case_node)
            });

            if is_entry {
                entry_nodes.push(child);
            }

            self.quantum_ops.insert(
                child,
                QuantumOp {
                    node: child,
                    gate_type,
                    num_qubit_inputs,
                    num_qubit_outputs,
                    params,
                },
            );
        }

        entry_nodes
    }
}
