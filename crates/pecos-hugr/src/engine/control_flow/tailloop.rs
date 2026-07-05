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

//! `TailLoop` control flow handling.
//!
//! `TailLoops` are HUGR's looping construct. They repeatedly execute their body
//! until the body outputs a "break" Sum variant (tag 1). The body can also
//! output a "continue" variant (tag 0) to loop again with new values.
//!
//! # Loop Structure
//!
//! A `TailLoop` has:
//! - Input ports: `just_inputs` + `rest` values
//! - Output ports: `just_outputs` + `rest` values
//! - Body Input node: receives iteration values
//! - Body Output node: produces `Sum(continue_values` | `break_values`) + rest
//!
//! # Iteration Flow
//!
//! 1. First iteration: external inputs -> body Input
//! 2. Body executes quantum/classical operations
//! 3. Body Output produces Sum tag:
//!    - Tag 0 (CONTINUE): loop again with new `just_inputs` + rest
//!    - Tag 1 (BREAK): exit loop with `just_outputs` + rest

use log::debug;
use tket::hugr::ops::OpType;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::engine::HugrEngine;
use crate::engine::activation::{ContainerActivation, QueuePolicy};
use crate::engine::types::{ActiveTailLoopInfo, TailLoopInfo};

impl HugrEngine {
    /// Iteration ceiling for `TailLoop`s: a loop that never breaks would
    /// otherwise spin the processing loop forever with no yield.
    const MAX_LOOP_ITERATIONS: usize = 10_000_000;

    /// Try to resolve the control value for a `TailLoop`'s current iteration.
    /// Returns `Some(0)` for `CONTINUE_TAG` (continue looping) or `Some(1)` for `BREAK_TAG` (exit loop).
    pub(crate) fn try_resolve_tailloop_control(
        &self,
        hugr: &Hugr,
        tailloop_node: Node,
    ) -> Option<usize> {
        let tailloop_info = self.tailloops.get(&tailloop_node)?;

        // The Output node's first input port (port 0) receives the Sum type (control)
        let output_node = tailloop_info.output_node;
        let control_port = IncomingPort::from(0);

        if let Some((src_node, src_port)) = hugr.single_linked_output(output_node, control_port) {
            let wire_key = (src_node, src_port.index());

            // Check if we have a classical value for this wire
            if let Some(value) = self.wire_state.classical_values.get(&wire_key)
                && let Some(v) = value.to_u32()
            {
                debug!("TailLoop {tailloop_node:?} control value resolved to {v}");
                return Some(v as usize);
            }

            // Check if the source is a Tag node
            let src_op = hugr.get_optype(src_node);
            if let OpType::Tag(tag_op) = src_op {
                let tag_value = tag_op.tag;

                // Check Tag's input for dynamic value
                let tag_input_port = IncomingPort::from(0);
                if let Some((tag_src_node, tag_src_port)) =
                    hugr.single_linked_output(src_node, tag_input_port)
                {
                    let tag_src_wire = (tag_src_node, tag_src_port.index());
                    if self.wire_state.classical_values.contains_key(&tag_src_wire) {
                        // The tag itself determines CONTINUE (0) or BREAK (1)
                        debug!(
                            "TailLoop {tailloop_node:?} resolved via Tag with known input: tag={tag_value}"
                        );
                        return Some(tag_value);
                    }
                }

                // Static tag with no dynamic input
                if hugr.num_inputs(src_node) == 0 {
                    debug!("TailLoop {tailloop_node:?} resolved via static Tag: tag={tag_value}");
                    return Some(tag_value);
                }
            }

            // Check for tket.bool.read converting to Sum
            if let Some(ext_op) = src_op.as_extension_op() {
                let ext_id = ext_op.extension_id();
                let op_name = ext_op.unqualified_id();
                if ext_id.as_ref() as &str == "tket.bool" && op_name == "read" {
                    let bool_input_port = IncomingPort::from(0);
                    if let Some((bool_src_node, bool_src_port)) =
                        hugr.single_linked_output(src_node, bool_input_port)
                    {
                        let bool_wire = (bool_src_node, bool_src_port.index());
                        if let Some(bool_value) = self.wire_state.classical_values.get(&bool_wire)
                            && let Some(v) = bool_value.to_u32()
                        {
                            debug!(
                                "TailLoop {tailloop_node:?} resolved via tket.bool.read: value={v}"
                            );
                            return Some(v as usize);
                        }
                    }
                }
            }
        }

        None
    }

    /// Expand a `TailLoop` by activating its body for the first iteration.
    /// Returns the entry nodes that should be added to the work queue.
    pub(crate) fn expand_tailloop(&mut self, hugr: &Hugr, tailloop_node: Node) {
        let Some(tailloop_info) = self.tailloops.get(&tailloop_node).cloned() else {
            debug!("TailLoop {tailloop_node:?} not found in tailloops map");
            return;
        };

        debug!("Expanding TailLoop {tailloop_node:?} for iteration 0");

        self.executed_containers
            .insert(tailloop_node, "TailLoop body");

        // Two-phase discipline via the shared activation mechanism (parity
        // with continue_tailloop_iteration and CFG/case re-activation): all
        // processed flags and stale output wires clear BEFORE any readiness
        // check. A TailLoop nested in a re-expanded Case/CFG otherwise
        // inherits the previous execution's state and its consumers pass
        // readiness against stale flags. Queueing also releases every
        // container gate (a body node may carry OUTER gates too -- a
        // TailLoop inside a CFG block leaves its internals in
        // nodes_inside_cfg_blocks): entry pushes bypass the gates, but the
        // retry path (queue_ready_successors) checks all of them.
        let mut act = ContainerActivation::new();
        for &op_node in tailloop_info
            .quantum_ops
            .iter()
            .chain(&tailloop_info.call_nodes)
            .chain(&tailloop_info.extension_ops)
            .chain(&tailloop_info.classical_ops)
            .chain(&tailloop_info.bool_ops)
        {
            act.reset(op_node);
            act.queue(op_node, QueuePolicy::IfReady);
        }
        // Conditionals defer internally until their control resolves.
        for &cond_node in &tailloop_info.conditional_nodes {
            act.reset(cond_node);
            act.queue(cond_node, QueuePolicy::Always);
        }
        // Nested containers (TailLoop-in-TailLoop, CFG-in-body): nothing
        // else queues them once the body is gated. Loops defer internally
        // until their inputs resolve; CFGs copy their inputs one-shot at
        // activation, so they must wait for readiness like Calls do.
        for &nested in &tailloop_info.tailloop_nodes {
            act.reset(nested);
            act.queue(nested, QueuePolicy::Always);
        }
        for &nested in &tailloop_info.cfg_nodes {
            act.reset(nested);
            act.queue(nested, QueuePolicy::IfReady);
        }
        // Body LoadConstants are gated behind nodes_inside_tailloops (no
        // other path queues them) while readiness gates their consumers on
        // the processed flag, so skipping them starves the body's classical
        // dataflow.
        for child in hugr.children(tailloop_node) {
            if matches!(hugr.get_optype(child), OpType::LoadConstant(_)) {
                act.reset_processed(child);
                act.queue(child, QueuePolicy::Always);
            }
        }

        // Propagate input wires from TailLoop inputs to body Input node
        // outputs, then apply the batch (reset -> ungate -> queue).
        self.propagate_tailloop_inputs(hugr, tailloop_node, &tailloop_info, 0);
        self.run_activation(hugr, &act);

        // Register as active TailLoop
        self.active_tailloops.insert(
            tailloop_node,
            ActiveTailLoopInfo {
                tailloop_node,
                iteration: 0,
                body_active: true,
            },
        );

        debug!("TailLoop {tailloop_node:?}: activated body");
    }

    /// Propagate wire mappings from `TailLoop` inputs to body Input node.
    pub(crate) fn propagate_tailloop_inputs(
        &mut self,
        hugr: &Hugr,
        tailloop_node: Node,
        tailloop_info: &TailLoopInfo,
        iteration: usize,
    ) {
        let input_node = tailloop_info.input_node;

        if iteration == 0 {
            // First iteration: inputs come from TailLoop's external inputs
            for port_idx in 0..tailloop_info.num_inputs {
                let tailloop_in_port = IncomingPort::from(port_idx);
                if let Some((src_node, src_port)) =
                    hugr.single_linked_output(tailloop_node, tailloop_in_port)
                {
                    let src_wire = (src_node, src_port.index());
                    if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                        self.wire_state
                            .wire_to_qubit
                            .insert((input_node, port_idx), qubit_id);
                        debug!(
                            "TailLoop {tailloop_node:?} iter {iteration}: propagated qubit {qubit_id:?} to Input port {port_idx}"
                        );
                    }
                    // Also propagate classical values
                    if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                        self.wire_state
                            .classical_values
                            .insert((input_node, port_idx), value);
                    }
                }
            }
        }
        // For subsequent iterations, propagate_continue_values handles this
    }

    /// Continue a `TailLoop` with a new iteration after receiving `CONTINUE_TAG`.
    #[allow(clippy::too_many_lines)] // Loop iteration control flow is inherently complex
    pub(crate) fn continue_tailloop_iteration(&mut self, hugr: &Hugr, tailloop_node: Node) {
        let Some(tailloop_info) = self.tailloops.get(&tailloop_node).cloned() else {
            return;
        };

        // Get current iteration count first
        let new_iteration = match self.active_tailloops.get(&tailloop_node) {
            Some(info) => info.iteration + 1,
            None => return,
        };

        debug!("TailLoop {tailloop_node:?}: continuing to iteration {new_iteration}");

        // Safety ceiling: a loop whose control never breaks (a program bug,
        // or an engine bug feeding a stale control value) would otherwise
        // spin forever inside the processing loop with no yield -- the batch
        // cap counts only quantum ops, so a purely classical loop hangs the
        // host process.
        if new_iteration > Self::MAX_LOOP_ITERATIONS {
            self.execution_error = Some(format!(
                "TailLoop {tailloop_node:?} exceeded {} iterations without breaking",
                Self::MAX_LOOP_ITERATIONS
            ));
            return;
        }

        // Propagate iteration values from Output to Input. This reads the
        // PREVIOUS iteration's body-output wires, so it must happen before
        // the batch below clears them. (It touches only wire maps, never
        // the processed set, so running it first is order-equivalent.)
        self.propagate_continue_values(hugr, tailloop_node, &tailloop_info);

        // Re-activate the body through the shared two-phase mechanism:
        // clear processed flags for every tracked body op (and body
        // LoadConstants -- readiness gates consumers on their processed
        // flag and no other path queues them once the body is gated) and
        // stale wires for every child except the Input node, which holds
        // the fresh continue values; then queue.
        let mut act = ContainerActivation::new();
        for &op_node in tailloop_info
            .quantum_ops
            .iter()
            .chain(&tailloop_info.call_nodes)
            .chain(&tailloop_info.extension_ops)
            .chain(&tailloop_info.classical_ops)
            .chain(&tailloop_info.bool_ops)
        {
            act.reset_processed(op_node);
            act.queue(op_node, QueuePolicy::IfReady);
        }
        for &cond_node in &tailloop_info.conditional_nodes {
            act.reset_processed(cond_node);
            act.queue(cond_node, QueuePolicy::Always);
        }
        for &nested in &tailloop_info.tailloop_nodes {
            act.reset_processed(nested);
            act.queue(nested, QueuePolicy::Always);
        }
        for &nested in &tailloop_info.cfg_nodes {
            act.reset_processed(nested);
            act.queue(nested, QueuePolicy::IfReady);
        }
        for child in hugr.children(tailloop_node) {
            if matches!(hugr.get_optype(child), OpType::LoadConstant(_)) {
                act.reset_processed(child);
                act.queue(child, QueuePolicy::Always);
            }
            act.reset_wires(child);
        }
        act.keep_wires(tailloop_info.input_node);

        // Update iteration counter
        if let Some(active_info) = self.active_tailloops.get_mut(&tailloop_node) {
            active_info.iteration = new_iteration;
            active_info.body_active = true;
        }

        self.run_activation(hugr, &act);
    }

    /// Propagate values from CONTINUE tag to next iteration's inputs.
    pub(crate) fn propagate_continue_values(
        &mut self,
        hugr: &Hugr,
        _tailloop_node: Node,
        tailloop_info: &TailLoopInfo,
    ) {
        let output_node = tailloop_info.output_node;
        let input_node = tailloop_info.input_node;

        // Output node layout: port 0 = Sum (control), ports 1.. = rest values
        // For CONTINUE, the Sum's variant 0 contains just_inputs values for next iteration
        // The Input node receives: just_inputs + rest

        let just_inputs_count = tailloop_info.just_inputs_count;

        // Clear EVERY Input port before writing fresh values: the
        // activation batch exempts the Input node from wire clearing
        // (keep_wires) so the fresh values survive, which means any port
        // this propagation cannot source would otherwise silently keep the
        // PREVIOUS iteration's value. Missing sources must starve loudly
        // instead (same discipline as case-input propagation).
        for port_idx in 0..(just_inputs_count + tailloop_info.rest_count) {
            self.wire_state
                .classical_values
                .remove(&(input_node, port_idx));
            self.wire_state
                .wire_to_qubit
                .remove(&(input_node, port_idx));
        }

        // Propagate the "rest" values from Output ports 1.. to Input ports (after just_inputs)
        for rest_idx in 0..tailloop_info.rest_count {
            let output_port_idx = rest_idx + 1; // Skip Sum port
            let input_port_idx = just_inputs_count + rest_idx;

            let output_in_port = IncomingPort::from(output_port_idx);
            if let Some((src_node, src_port)) =
                hugr.single_linked_output(output_node, output_in_port)
            {
                let src_wire = (src_node, src_port.index());

                if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                    self.wire_state
                        .wire_to_qubit
                        .insert((input_node, input_port_idx), qubit_id);
                    debug!(
                        "TailLoop continue: propagated rest qubit {qubit_id:?} from Output:{output_port_idx} to Input:{input_port_idx}"
                    );
                }
                if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                    self.wire_state
                        .classical_values
                        .insert((input_node, input_port_idx), value);
                }
            }
        }

        // The just_inputs values come from unpacking the Sum (CONTINUE variant).
        // If the control wire carries an executed Sum VALUE (e.g. built by a
        // Tag inside a Conditional case and routed through the Conditional's
        // output), its payload elements ARE the next iteration's just_inputs.
        let control_port = IncomingPort::from(0);
        if let Some((ctrl_src, ctrl_port)) = hugr.single_linked_output(output_node, control_port)
            && let Some(crate::engine::types::ClassicalValue::Sum { tag: 0, values }) = self
                .wire_state
                .classical_values
                .get(&(ctrl_src, ctrl_port.index()))
                .cloned()
        {
            for (port_idx, value) in values.into_iter().enumerate() {
                if port_idx >= just_inputs_count {
                    break;
                }
                debug!(
                    "TailLoop continue: propagated just_input value {value:?} to Input:{port_idx}"
                );
                if let crate::engine::types::ClassicalValue::QubitRef(qubit_id) = &value {
                    self.wire_state
                        .wire_to_qubit
                        .insert((input_node, port_idx), *qubit_id);
                }
                self.wire_state
                    .classical_values
                    .insert((input_node, port_idx), value);
            }
            return;
        }

        // Otherwise trace through the Tag node that created the Sum structurally
        if let Some((tag_node, _)) = hugr.single_linked_output(output_node, control_port)
            && let OpType::Tag(tag_op) = hugr.get_optype(tag_node)
            && tag_op.tag == 0
        {
            // CONTINUE tag - its inputs become just_inputs for next iteration
            for port_idx in 0..just_inputs_count {
                let tag_in_port = IncomingPort::from(port_idx);
                if let Some((src_node, src_port)) = hugr.single_linked_output(tag_node, tag_in_port)
                {
                    let src_wire = (src_node, src_port.index());
                    if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                        self.wire_state
                            .wire_to_qubit
                            .insert((input_node, port_idx), qubit_id);
                        debug!(
                            "TailLoop continue: propagated just_input qubit {qubit_id:?} to Input:{port_idx}"
                        );
                    }
                    if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                        debug!(
                            "TailLoop continue: propagated just_input value {value:?} to Input:{port_idx}"
                        );
                        self.wire_state
                            .classical_values
                            .insert((input_node, port_idx), value);
                    }
                }
            }
        }
    }

    /// Complete a `TailLoop` after receiving `BREAK_TAG`.
    pub(crate) fn complete_tailloop(&mut self, hugr: &Hugr, tailloop_node: Node) {
        let Some(tailloop_info) = self.tailloops.get(&tailloop_node).cloned() else {
            return;
        };

        debug!("Completing TailLoop {tailloop_node:?}");

        // Propagate outputs from body Output node to TailLoop output ports
        self.propagate_tailloop_outputs(hugr, tailloop_node, &tailloop_info);

        // Mark TailLoop as processed
        self.processed.insert(tailloop_node);
        self.active_tailloops.remove(&tailloop_node);
        self.pending_tailloop_control.remove(&tailloop_node);

        // Add TailLoop successors to work queue (use unified method)
        self.queue_ready_successors(hugr, tailloop_node);

        // Check if this TailLoop completion allows a CFG block to complete
        // A parent Case whose final completion event is this loop must
        // observe it too.
        self.check_scan_frame_completion(hugr, tailloop_node);
        self.check_case_completion(hugr, tailloop_node);
        self.check_cfg_block_completion(hugr, tailloop_node);
        // A completed loop may be the last op of an ENCLOSING loop's body
        // (TailLoop-in-TailLoop).
        self.check_tailloop_body_completion(hugr, tailloop_node);
    }

    /// Propagate outputs from `TailLoop` body to `TailLoop` node outputs.
    pub(crate) fn propagate_tailloop_outputs(
        &mut self,
        hugr: &Hugr,
        tailloop_node: Node,
        tailloop_info: &TailLoopInfo,
    ) {
        let output_node = tailloop_info.output_node;

        // TailLoop outputs = just_outputs (from BREAK Sum) + rest (from Output ports 1..)
        let just_outputs_count = tailloop_info.just_outputs_count;

        // Propagate rest values from Output ports 1..
        for rest_idx in 0..tailloop_info.rest_count {
            let output_port_idx = rest_idx + 1; // Skip Sum port
            let tailloop_output_idx = just_outputs_count + rest_idx;

            let output_in_port = IncomingPort::from(output_port_idx);
            if let Some((src_node, src_port)) =
                hugr.single_linked_output(output_node, output_in_port)
            {
                let src_wire = (src_node, src_port.index());

                // Map qubits
                if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                    self.wire_state
                        .wire_to_qubit
                        .insert((tailloop_node, tailloop_output_idx), qubit_id);
                    debug!(
                        "TailLoop {tailloop_node:?} output {tailloop_output_idx}: mapped rest qubit {qubit_id:?}"
                    );
                }
                // Map classical values (including arrays)
                if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                    self.wire_state
                        .classical_values
                        .insert((tailloop_node, tailloop_output_idx), value);
                    debug!(
                        "TailLoop {tailloop_node:?} output {tailloop_output_idx}: mapped rest classical value"
                    );
                }
            }
        }

        // Extract just_outputs from BREAK Sum variant (tag 1). If the
        // control wire carries an executed Sum VALUE (e.g. routed through a
        // Conditional's output rather than a direct Tag node), its payload
        // elements ARE the just_outputs.
        let control_port = IncomingPort::from(0);
        if let Some((ctrl_src, ctrl_port)) = hugr.single_linked_output(output_node, control_port)
            && let Some(crate::engine::types::ClassicalValue::Sum { tag: 1, values }) = self
                .wire_state
                .classical_values
                .get(&(ctrl_src, ctrl_port.index()))
                .cloned()
        {
            for (port_idx, value) in values.into_iter().enumerate() {
                if port_idx >= just_outputs_count {
                    break;
                }
                debug!(
                    "TailLoop {tailloop_node:?} output {port_idx}: mapped just_output payload {value:?}"
                );
                if let crate::engine::types::ClassicalValue::QubitRef(qubit_id) = &value {
                    self.wire_state
                        .wire_to_qubit
                        .insert((tailloop_node, port_idx), *qubit_id);
                }
                self.wire_state
                    .classical_values
                    .insert((tailloop_node, port_idx), value);
            }
            return;
        }

        // Otherwise trace the BREAK Tag structurally
        if let Some((tag_node, _)) = hugr.single_linked_output(output_node, control_port)
            && let OpType::Tag(tag_op) = hugr.get_optype(tag_node)
            && tag_op.tag == 1
        {
            // BREAK tag - its inputs are just_outputs
            for port_idx in 0..just_outputs_count {
                let tag_in_port = IncomingPort::from(port_idx);
                if let Some((src_node, src_port)) = hugr.single_linked_output(tag_node, tag_in_port)
                {
                    let src_wire = (src_node, src_port.index());
                    // Map qubits
                    if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                        self.wire_state
                            .wire_to_qubit
                            .insert((tailloop_node, port_idx), qubit_id);
                        debug!(
                            "TailLoop {tailloop_node:?} output {port_idx}: mapped just_output qubit {qubit_id:?}"
                        );
                    }
                    // Map classical values
                    if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                        self.wire_state
                            .classical_values
                            .insert((tailloop_node, port_idx), value);
                        debug!(
                            "TailLoop {tailloop_node:?} output {port_idx}: mapped just_output classical value"
                        );
                    }
                }
            }
        }
    }

    /// Check if a `TailLoop` body is complete after processing an operation.
    pub(crate) fn check_tailloop_body_completion(&mut self, hugr: &Hugr, processed_node: Node) {
        let mut completions = Vec::new();

        for (tailloop_node, active_info) in &self.active_tailloops {
            if !active_info.body_active {
                continue;
            }

            let Some(tailloop_info) = self.tailloops.get(tailloop_node) else {
                continue;
            };

            // Check if processed node is in this TailLoop
            let is_in_loop = tailloop_info.quantum_ops.contains(&processed_node)
                || tailloop_info.call_nodes.contains(&processed_node)
                || tailloop_info.extension_ops.contains(&processed_node)
                || tailloop_info.classical_ops.contains(&processed_node)
                || tailloop_info.bool_ops.contains(&processed_node)
                || tailloop_info.conditional_nodes.contains(&processed_node)
                || tailloop_info.tailloop_nodes.contains(&processed_node)
                || tailloop_info.cfg_nodes.contains(&processed_node);

            if is_in_loop {
                // Check if all ops are processed
                let all_quantum_done = tailloop_info
                    .quantum_ops
                    .iter()
                    .all(|op| self.processed.contains(op));
                let all_calls_done = tailloop_info
                    .call_nodes
                    .iter()
                    .all(|call| self.processed.contains(call));
                let all_extension_done = tailloop_info
                    .extension_ops
                    .iter()
                    .all(|op| self.processed.contains(op));
                let all_classical_done = tailloop_info
                    .classical_ops
                    .iter()
                    .all(|op| self.processed.contains(op));
                let all_bool_done = tailloop_info
                    .bool_ops
                    .iter()
                    .all(|op| self.processed.contains(op));
                // A Conditional is marked processed at EXPANSION; its
                // outputs exist only once its selected case completes, so
                // an active case must also block body completion.
                let all_conditionals_done = tailloop_info.conditional_nodes.iter().all(|cond| {
                    self.processed.contains(cond)
                        && !self
                            .active_cases
                            .values()
                            .any(|case| case.conditional_node == *cond)
                });

                // Nested containers count as done only once processed AND
                // no longer active (a nested loop marks processed at its
                // own completion; a nested CFG at complete_cfg_execution).
                let all_nested_done = tailloop_info.tailloop_nodes.iter().all(|tl| {
                    self.processed.contains(tl) && !self.active_tailloops.contains_key(tl)
                }) && tailloop_info
                    .cfg_nodes
                    .iter()
                    .all(|c| self.processed.contains(c) && !self.active_cfgs.contains_key(c));

                if all_quantum_done
                    && all_calls_done
                    && all_extension_done
                    && all_classical_done
                    && all_bool_done
                    && all_conditionals_done
                    && all_nested_done
                {
                    completions.push(*tailloop_node);
                }
            }
        }

        for tailloop_node in completions {
            debug!("TailLoop {tailloop_node:?} body iteration complete");

            // Mark body as inactive (waiting for control resolution)
            if let Some(active_info) = self.active_tailloops.get_mut(&tailloop_node) {
                active_info.body_active = false;
            }

            // Try to resolve control immediately
            if let Some(tag) = self.try_resolve_tailloop_control(hugr, tailloop_node) {
                if tag > 1 {
                    // The control Sum has exactly two variants (0=continue,
                    // 1=break); anything else is an upstream propagation
                    // bug that must not complete cleanly as a break.
                    self.execution_error = Some(format!(
                        "TailLoop {tailloop_node:?}: control tag {tag} out of range"
                    ));
                } else if tag == 0 {
                    // CONTINUE
                    self.continue_tailloop_iteration(hugr, tailloop_node);
                } else {
                    // BREAK
                    self.complete_tailloop(hugr, tailloop_node);
                }
            } else {
                // Add to pending
                self.pending_tailloop_control.insert(tailloop_node);
                // Re-add to work queue for resolution after measurements
                if !self.work_queue.contains(&tailloop_node) {
                    self.work_queue.push_back(tailloop_node);
                }
            }
        }
    }
}
