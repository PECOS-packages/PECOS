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

//! CFG (Control Flow Graph) handling.
//!
//! CFGs contain `DataflowBlocks` connected by control edges. Each block can
//! branch to multiple successors based on a Sum output value.
//!
//! # Structure
//!
//! A CFG node has:
//! - Entry block: First `DataflowBlock` to execute
//! - Exit block: Terminal block that produces CFG outputs
//! - `DataflowBlocks`: Each contains operations and branches to successors
//!
//! # Execution Flow
//!
//! 1. CFG encountered during traversal
//! 2. Entry block activated with CFG inputs propagated
//! 3. Block operations execute
//! 4. On block completion, branch value resolved
//! 5. Transition to successor block (or exit)
//! 6. Repeat until exit block reached
//! 7. CFG outputs propagated from final block

use log::debug;
use tket::hugr::ops::OpType;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::engine::HugrEngine;
use crate::engine::analysis::{
    all_predecessors_ready, find_extension_ops_in_block, find_input_node, find_output_node,
};
use crate::engine::types::ClassicalValue;

impl HugrEngine {
    /// Try to resolve the branch value for a CFG `DataflowBlock`.
    /// Returns `Some(branch_index)` if the Sum tag value is known, None otherwise.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn try_resolve_cfg_block_branch(
        &self,
        hugr: &Hugr,
        block_node: Node,
    ) -> Option<usize> {
        // Find the Output node of this block
        let output_node = hugr.get_io(block_node).map(|[_, o]| o)?;
        debug!(
            "[TRACE] try_resolve_cfg_block_branch: block {block_node:?}, output_node {output_node:?}"
        );

        // The first output of the block (port 0) is the Sum type that determines the branch
        // Trace back from Output port 0 to find where the Sum value comes from
        let output_port = IncomingPort::from(0);

        if let Some((src_node, src_port)) = hugr.single_linked_output(output_node, output_port) {
            let wire_key = (src_node, src_port.index());

            // Check if we have a classical value for this wire
            if let Some(value) = self.wire_state.classical_values.get(&wire_key)
                && let Some(v) = value.to_u32()
            {
                return Some(v as usize);
            }

            // Check if the source is a Tag node (creates Sum type from a bool)
            let src_op = hugr.get_optype(src_node);
            if let OpType::Tag(tag_op) = src_op {
                let tag_value = tag_op.tag;

                // Check if the Tag's input is a known value
                let tag_input_port = IncomingPort::from(0);
                if let Some((tag_src_node, tag_src_port)) =
                    hugr.single_linked_output(src_node, tag_input_port)
                {
                    let tag_src_wire = (tag_src_node, tag_src_port.index());
                    if let Some(input_value) = self.wire_state.classical_values.get(&tag_src_wire)
                        && let Some(v) = input_value.to_u32()
                    {
                        debug!(
                            "CFG block {block_node:?} resolved via Tag: tag={tag_value}, input={v}"
                        );
                        // For booleans converted to Sum: input_value determines the branch
                        // The Tag wraps the value - we use the input value as the branch
                        return Some(v as usize);
                    }
                }

                // If the Tag has a constant tag value and no dynamic input
                if hugr.num_inputs(src_node) == 0 {
                    return Some(tag_value);
                }
            }

            // Check for extension op that converts bool to Sum (like tket.bool.read)
            if let Some(ext_op) = src_op.as_extension_op() {
                let ext_id = ext_op.extension_id();
                let op_name = ext_op.unqualified_id();
                if ext_id.as_ref() as &str == "tket.bool" && op_name == "read" {
                    // tket.bool.read converts bool to Sum(Unit, Unit)
                    // The input is the bool value
                    let bool_input_port = IncomingPort::from(0);
                    if let Some((bool_src_node, bool_src_port)) =
                        hugr.single_linked_output(src_node, bool_input_port)
                    {
                        let bool_wire = (bool_src_node, bool_src_port.index());
                        if let Some(bool_value) = self.wire_state.classical_values.get(&bool_wire)
                            && let Some(v) = bool_value.to_u32()
                        {
                            debug!(
                                "CFG block {block_node:?} resolved via tket.bool.read: value={v}"
                            );
                            return Some(v as usize);
                        }

                        // Try to trace through LoadConstant to Const
                        if let Some(const_value) = Self::try_resolve_const_bool(hugr, bool_src_node)
                        {
                            debug!(
                                "CFG block {block_node:?} resolved via constant bool: value={const_value}"
                            );
                            return Some(usize::from(const_value));
                        }
                    }
                }
            }

            // Check if the source is a Conditional node (inside the block)
            // The Conditional's output is a Sum type - we need to trace its control input
            if matches!(src_op, OpType::Conditional(_)) {
                debug!(
                    "[TRACE] Block {block_node:?} output from Conditional {src_node:?}, tracing control input"
                );
                // Conditional's control input is port 0
                let control_port = IncomingPort::from(0);
                if let Some((ctrl_src_node, ctrl_src_port)) =
                    hugr.single_linked_output(src_node, control_port)
                {
                    // The control input might be from tket.bool.read
                    let ctrl_op = hugr.get_optype(ctrl_src_node);
                    if let Some(ext_op) = ctrl_op.as_extension_op() {
                        let ext_id = ext_op.extension_id();
                        let op_name = ext_op.unqualified_id();
                        if ext_id.as_ref() as &str == "tket.bool" && op_name == "read" {
                            // Trace the bool input to tket.bool.read
                            let bool_input_port = IncomingPort::from(0);
                            if let Some((bool_src_node, bool_src_port)) =
                                hugr.single_linked_output(ctrl_src_node, bool_input_port)
                            {
                                let bool_wire = (bool_src_node, bool_src_port.index());
                                debug!(
                                    "[TRACE] tket.bool.read input comes from {bool_wire:?}, checking classical_values"
                                );

                                // First check if we have a classical value for this wire
                                if let Some(bool_value) =
                                    self.wire_state.classical_values.get(&bool_wire)
                                    && let Some(v) = bool_value.to_u32()
                                {
                                    debug!(
                                        "[TRACE] Found classical value {v} for Conditional control"
                                    );
                                    // The bool value (0 or 1) determines which Case
                                    // Case 0 = false, Case 1 = true
                                    // Each Case outputs a Tag that determines the successor
                                    // For while loop: false -> Case 0 -> Tag 0 -> continue
                                    //                 true -> Case 1 -> Tag 1 -> exit
                                    return Some(v as usize);
                                }

                                // Try to resolve constant bool
                                if let Some(const_value) =
                                    Self::try_resolve_const_bool(hugr, bool_src_node)
                                {
                                    debug!(
                                        "CFG block {block_node:?} Conditional control resolved from const: {const_value}"
                                    );
                                    return Some(usize::from(const_value));
                                }

                                debug!(
                                    "[TRACE] Could not resolve bool value for wire {bool_wire:?}"
                                );
                            }
                        }
                    }

                    // Check classical_values for the control wire
                    let ctrl_wire = (ctrl_src_node, ctrl_src_port.index());
                    if let Some(ctrl_value) = self.wire_state.classical_values.get(&ctrl_wire)
                        && let Some(v) = ctrl_value.to_u32()
                    {
                        debug!(
                            "CFG block {block_node:?} Conditional control from classical value: {v}"
                        );
                        return Some(v as usize);
                    }
                }
            }
        }

        None
    }

    /// Try to resolve a constant boolean value by tracing through `LoadConstant` to Const.
    pub(crate) fn try_resolve_const_bool(hugr: &Hugr, node: Node) -> Option<bool> {
        use tket::extension::bool::ConstBool;

        let op = hugr.get_optype(node);
        debug!(
            "[TRACE] try_resolve_const_bool: node {:?}, op type: {:?}",
            node,
            std::mem::discriminant(op)
        );

        // Check if this is a LoadConstant
        if matches!(op, OpType::LoadConstant(_)) {
            debug!("[TRACE] Found LoadConstant at {node:?}");
            // LoadConstant has a static edge from a Const node
            for pred_node in hugr.input_neighbours(node) {
                let pred_op = hugr.get_optype(pred_node);
                debug!(
                    "[TRACE] LoadConstant predecessor {:?}: {:?}",
                    pred_node,
                    std::mem::discriminant(pred_op)
                );
                if let OpType::Const(const_op) = pred_op {
                    // Try to extract bool value from the constant
                    let value = const_op.value();
                    debug!("[TRACE] Found Const, value type: {:?}", value.get_type());
                    // The value is stored as a ConstBool for tket.bool
                    if let Some(const_bool) = value.get_custom_value::<ConstBool>() {
                        let bool_value = const_bool.value();
                        debug!("[TRACE] Found ConstBool: {bool_value}");
                        return Some(bool_value);
                    }
                    debug!("[TRACE] Not a ConstBool, checking other patterns");
                }
            }
        }

        // Check if this is directly a Const node
        if let OpType::Const(const_op) = op {
            use tket::extension::bool::ConstBool;
            let value = const_op.value();
            if let Some(const_bool) = value.get_custom_value::<ConstBool>() {
                return Some(const_bool.value());
            }
        }

        None
    }

    /// Try to resolve pending CFG blocks that were waiting for measurement results.
    pub(crate) fn try_resolve_pending_cfg_branches(&mut self) {
        let hugr = match &self.hugr {
            Some(h) => h.clone(),
            None => return,
        };

        debug!(
            "[TRACE] try_resolve_pending_cfg_branches: {} pending",
            self.pending_cfg_branches.len()
        );

        // Collect blocks that can now be resolved
        let mut to_resolve = Vec::new();
        for (&(cfg_node, block_node), successors) in &self.pending_cfg_branches {
            let branch_result = self.try_resolve_cfg_block_branch(&hugr, block_node);
            debug!(
                "[TRACE] Checking pending block {block_node:?}: branch result = {branch_result:?}"
            );
            if let Some(branch_idx) = branch_result {
                to_resolve.push((cfg_node, block_node, branch_idx, successors.clone()));
            }
        }

        // Resolve them
        for (cfg_node, block_node, branch_idx, successors) in to_resolve {
            self.pending_cfg_branches.remove(&(cfg_node, block_node));

            if branch_idx < successors.len() {
                let next_block = successors[branch_idx];
                debug!(
                    "[TRACE] Resolving pending: {block_node:?} taking branch {branch_idx} to {next_block:?}"
                );
                self.transition_to_cfg_successor(&hugr, cfg_node, block_node, next_block);
            } else {
                debug!(
                    "[TRACE] Resolving pending: {block_node:?} branch {branch_idx} out of range, using first"
                );
                if !successors.is_empty() {
                    self.transition_to_cfg_successor(&hugr, cfg_node, block_node, successors[0]);
                }
            }
        }
    }

    /// Check if a CFG block is complete after processing an operation.
    #[allow(clippy::too_many_lines)] // CFG control flow logic is inherently complex
    pub(crate) fn check_cfg_block_completion(&mut self, hugr: &Hugr, processed_node: Node) {
        // Find which CFG block (if any) this node belongs to
        let mut block_completions = Vec::new();

        for (cfg_node, active_cfg) in &self.active_cfgs {
            let cfg_info = match self.cfgs.get(cfg_node) {
                Some(info) => info.clone(),
                None => continue,
            };

            // Check the current block
            if let Some(block_info) = cfg_info.blocks.get(&active_cfg.current_block) {
                // Check if this block has tracked ops that drive completion
                // Quantum, calls, conditionals, bool, extension, and tailloops are tracked
                // Classical_ops are not tracked (they complete when their inputs are ready)
                let has_tracked_ops = !block_info.quantum_ops.is_empty()
                    || !block_info.call_nodes.is_empty()
                    || !block_info.conditional_nodes.is_empty()
                    || !block_info.bool_ops.is_empty()
                    || !block_info.extension_ops.is_empty()
                    || !block_info.tailloop_nodes.is_empty();

                // Check if the processed node is in this block
                let is_in_block = if has_tracked_ops {
                    block_info.quantum_ops.contains(&processed_node)
                        || block_info.call_nodes.contains(&processed_node)
                        || block_info.conditional_nodes.contains(&processed_node)
                        || block_info.bool_ops.contains(&processed_node)
                        || block_info.extension_ops.contains(&processed_node)
                        || block_info.tailloop_nodes.contains(&processed_node)
                } else {
                    // Block has only classical ops - track those for completion
                    block_info.classical_ops.contains(&processed_node)
                };

                if is_in_block {
                    // Check completion based on block type
                    let block_complete = if has_tracked_ops {
                        // Block with tracked ops: wait for all tracked op types
                        let all_quantum_done = block_info
                            .quantum_ops
                            .iter()
                            .all(|op| self.processed.contains(op));
                        let all_calls_done = block_info
                            .call_nodes
                            .iter()
                            .all(|call| self.processed.contains(call));
                        let all_conditionals_done = block_info
                            .conditional_nodes
                            .iter()
                            .all(|cond| self.processed.contains(cond));
                        let all_bools_done = block_info
                            .bool_ops
                            .iter()
                            .all(|op| self.processed.contains(op));
                        let all_extensions_done = block_info
                            .extension_ops
                            .iter()
                            .all(|op| self.processed.contains(op));
                        let all_tailloops_done = block_info
                            .tailloop_nodes
                            .iter()
                            .all(|tl| self.processed.contains(tl));

                        all_quantum_done
                            && all_calls_done
                            && all_conditionals_done
                            && all_bools_done
                            && all_extensions_done
                            && all_tailloops_done
                    } else {
                        // Classical-only block: wait for classical ops
                        block_info
                            .classical_ops
                            .iter()
                            .all(|op| self.processed.contains(op))
                    };

                    if block_complete {
                        block_completions.push((
                            *cfg_node,
                            active_cfg.current_block,
                            block_info.successors.clone(),
                        ));
                    }
                }
            }
        }

        // Handle block completions
        for (cfg_node, completed_block, successors) in block_completions {
            debug!(
                "CFG {:?} block {:?} complete, {} successors",
                cfg_node,
                completed_block,
                successors.len()
            );
            debug!(
                "[TRACE] Block {:?} complete, {} successors: {:?}",
                completed_block,
                successors.len(),
                successors
            );

            if successors.is_empty() {
                // No successors - this block leads to exit
                self.complete_cfg_execution(hugr, cfg_node, completed_block);
            } else if successors.len() == 1 {
                // Single successor - no branching needed
                debug!(" Single successor, transitioning to {:?}", successors[0]);
                self.transition_to_cfg_successor(hugr, cfg_node, completed_block, successors[0]);
            } else {
                // Multiple successors - need to resolve branch
                let branch_result = self.try_resolve_cfg_block_branch(hugr, completed_block);
                debug!(" Resolving branch for {completed_block:?}: {branch_result:?}");
                if let Some(branch_idx) = branch_result {
                    if branch_idx < successors.len() {
                        let next_block = successors[branch_idx];
                        self.transition_to_cfg_successor(
                            hugr,
                            cfg_node,
                            completed_block,
                            next_block,
                        );
                    } else {
                        debug!(
                            "CFG {:?} block {:?}: branch {} out of range ({}), defaulting to first",
                            cfg_node,
                            completed_block,
                            branch_idx,
                            successors.len()
                        );
                        self.transition_to_cfg_successor(
                            hugr,
                            cfg_node,
                            completed_block,
                            successors[0],
                        );
                    }
                } else {
                    // Branch value not yet known - store as pending
                    debug!(
                        "[TRACE] Adding block {completed_block:?} to pending_cfg_branches (branch not resolved)"
                    );
                    let block_key = (cfg_node, completed_block);
                    self.pending_cfg_branches
                        .insert(block_key, successors.clone());
                }
            }
        }
    }

    /// Transition to a successor block in a CFG.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn transition_to_cfg_successor(
        &mut self,
        hugr: &Hugr,
        cfg_node: Node,
        from_block: Node,
        to_block: Node,
    ) {
        let Some(cfg_info) = self.cfgs.get(&cfg_node).cloned() else {
            return;
        };

        // If to_block is the ExitBlock, complete the CFG.
        // ExitBlock has no operations - it's just a marker node. The from_block (a DataflowBlock)
        // should have already executed any result operations before this transition.
        if to_block == cfg_info.exit_block {
            debug!("CFG {cfg_node:?}: transitioning to exit block {to_block:?}");
            self.complete_cfg_execution(hugr, cfg_node, from_block);
            return;
        }

        debug!("CFG {cfg_node:?}: transitioning from block {from_block:?} to {to_block:?}");

        // Propagate wire mappings from completed block to successor block
        self.propagate_block_outputs_to_successor(hugr, from_block, to_block);

        // Record this propagation for re-propagation after measurement results
        // are available (measurement results may not be stored yet when we transition)
        self.pending_measurement_propagations
            .push((cfg_node, from_block, to_block));

        // Update active CFG state
        if let Some(active_cfg) = self.active_cfgs.get_mut(&cfg_node) {
            active_cfg.completed_blocks.insert(from_block);
            active_cfg.current_block = to_block;
        }

        // Activate successor block's quantum ops and Call nodes
        if let Some(block_info) = cfg_info.blocks.get(&to_block) {
            // Clear stale classical values for all operations in this block.
            // This is critical for loops: without this, nodes like tket.bool.read
            // retain values from the previous iteration. Since Conditionals are
            // added to the work queue before bool_ops, the Conditional would read
            // the stale value and select the wrong branch.
            let block_input_node = find_input_node(hugr, to_block);
            for child in hugr.children(to_block) {
                // Don't clear the Input node - it has fresh values from propagation
                if Some(child) == block_input_node {
                    continue;
                }
                let num_outputs = hugr.num_outputs(child);
                for port_idx in 0..num_outputs {
                    self.wire_state.classical_values.remove(&(child, port_idx));
                }
            }

            // Clear processed state for quantum ops first so they can be re-executed in loops
            for &op_node in &block_info.quantum_ops {
                self.processed.remove(&op_node);
            }
            for &op_node in &block_info.quantum_ops {
                self.nodes_inside_cfg_blocks.remove(&op_node);
                // Skip ops inside TailLoops - they'll be added when the loop expands
                if self.nodes_inside_tailloops.contains(&op_node) {
                    continue;
                }
                if !self.work_queue.contains(&op_node) && !self.processed.contains(&op_node) {
                    self.work_queue.push_back(op_node);
                }
            }
            // Also activate Call nodes in this block
            for &call_node in &block_info.call_nodes {
                self.nodes_inside_cfg_blocks.remove(&call_node);
                // Skip Call nodes inside TailLoops
                if self.nodes_inside_tailloops.contains(&call_node) {
                    continue;
                }
                if !self.work_queue.contains(&call_node)
                    && !self.processed.contains(&call_node)
                    && all_predecessors_ready(
                        hugr,
                        call_node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.processed,
                    )
                {
                    self.work_queue.push_back(call_node);
                }
            }

            // Also activate Conditional nodes in this block
            // Clear processed state first so they can be re-executed in loops
            for &cond_node in &block_info.conditional_nodes {
                self.processed.remove(&cond_node);
            }
            for &cond_node in &block_info.conditional_nodes {
                self.nodes_inside_cfg_blocks.remove(&cond_node);
                // Skip Conditional nodes inside TailLoops
                if self.nodes_inside_tailloops.contains(&cond_node) {
                    continue;
                }
                if !self.work_queue.contains(&cond_node) && !self.processed.contains(&cond_node) {
                    self.work_queue.push_back(cond_node);
                }
            }

            // Also activate other extension ops in this block (like tket.result)
            // IMPORTANT: Process extension/classical ops FIRST so their results are available for bool_ops
            // Find all extension ops that are children of this block
            let extension_ops: Vec<Node> = find_extension_ops_in_block(hugr, to_block);
            for &op_node in &extension_ops {
                self.processed.remove(&op_node);
                self.nodes_inside_cfg_blocks.remove(&op_node);
                // Skip extension ops inside TailLoops
                if self.nodes_inside_tailloops.contains(&op_node) {
                    continue;
                }
                if !self.work_queue.contains(&op_node) && !self.processed.contains(&op_node) {
                    self.work_queue.push_back(op_node);
                }
            }

            // Also activate LoadConstant and classical ops in this block
            for child in hugr.children(to_block) {
                let op = hugr.get_optype(child);
                if matches!(op, OpType::LoadConstant(_)) {
                    self.processed.remove(&child);
                    self.nodes_inside_cfg_blocks.remove(&child);
                    // Skip nodes inside TailLoops
                    if self.nodes_inside_tailloops.contains(&child) {
                        continue;
                    }
                    if !self.work_queue.contains(&child) && !self.processed.contains(&child) {
                        self.work_queue.push_back(child);
                    }
                }
                // Check for classical ops
                if self.classical_ops.contains_key(&child) {
                    self.processed.remove(&child);
                    self.nodes_inside_cfg_blocks.remove(&child);
                    // Skip nodes inside TailLoops
                    if self.nodes_inside_tailloops.contains(&child) {
                        continue;
                    }
                    if !self.work_queue.contains(&child)
                        && !self.processed.contains(&child)
                        && all_predecessors_ready(
                            hugr,
                            child,
                            &self.quantum_ops,
                            &self.conditionals,
                            &self.cfgs,
                            &self.processed,
                        )
                    {
                        self.work_queue.push_back(child);
                    }
                }
            }

            // Now activate bool ops in this block
            // Clear processed state first so they can be re-executed in loops
            for &op_node in &block_info.bool_ops {
                self.processed.remove(&op_node);
            }
            for &op_node in &block_info.bool_ops {
                self.nodes_inside_cfg_blocks.remove(&op_node);
                // Skip bool ops inside TailLoops
                if self.nodes_inside_tailloops.contains(&op_node) {
                    continue;
                }
                if !self.work_queue.contains(&op_node) && !self.processed.contains(&op_node) {
                    self.work_queue.push_back(op_node);
                }
            }

            // Also activate TailLoop nodes in this block
            for &tl_node in &block_info.tailloop_nodes {
                self.processed.remove(&tl_node);
                self.nodes_inside_cfg_blocks.remove(&tl_node);
                if !self.work_queue.contains(&tl_node) && !self.processed.contains(&tl_node) {
                    self.work_queue.push_back(tl_node);
                }
            }

            let num_ops = block_info.quantum_ops.len();
            let num_calls = block_info.call_nodes.len();
            let num_conditionals = block_info.conditional_nodes.len();
            let num_bool_ops = block_info.bool_ops.len();
            let num_tailloops = block_info.tailloop_nodes.len();
            debug!(
                "[TRACE] Activated block {to_block:?} with {num_ops} ops, {num_calls} calls, {num_conditionals} conditionals, {num_bool_ops} bool_ops, {num_tailloops} tailloops"
            );

            // Handle blocks with no operations - immediately complete and transition
            // IMPORTANT: Also check for extension_ops and classical_ops, not just quantum/bool/conditional
            let has_extension_ops = !extension_ops.is_empty();
            let has_classical_ops = !block_info.classical_ops.is_empty();
            let has_tailloops = !block_info.tailloop_nodes.is_empty();

            if num_ops == 0
                && num_calls == 0
                && num_conditionals == 0
                && num_bool_ops == 0
                && !has_extension_ops
                && !has_classical_ops
                && !has_tailloops
            {
                debug!(
                    "[TRACE] Block {to_block:?} has 0 ops and 0 calls, trying to resolve branch"
                );
                debug!("[TRACE] Block {to_block:?} has no quantum ops, checking for successors");
                // Mark this block as complete in the active CFG
                if let Some(active_cfg) = self.active_cfgs.get_mut(&cfg_node) {
                    active_cfg.completed_blocks.insert(to_block);
                }

                // Get successors for this block
                let successors = block_info.successors.clone();
                if successors.is_empty() {
                    // No successors - exit block
                    self.complete_cfg_execution(hugr, cfg_node, to_block);
                } else if successors.len() == 1 {
                    // Single successor - transition immediately
                    let next_block = successors[0];
                    // Check if successor is exit block
                    if next_block == cfg_info.exit_block {
                        self.complete_cfg_execution(hugr, cfg_node, to_block);
                    } else {
                        debug!(
                            "[TRACE] Empty block {to_block:?} transitioning to single successor {next_block:?}"
                        );
                        self.propagate_block_outputs_to_successor(hugr, to_block, next_block);

                        // Update current block
                        if let Some(active_cfg) = self.active_cfgs.get_mut(&cfg_node) {
                            active_cfg.current_block = next_block;
                        }

                        // Recursively activate the next block - add all ops to work queue
                        let next_block_info = cfg_info.blocks.get(&next_block).cloned();
                        if let Some(next_info) = next_block_info {
                            // Quantum ops
                            for &op_node in &next_info.quantum_ops {
                                self.nodes_inside_cfg_blocks.remove(&op_node);
                                if !self.work_queue.contains(&op_node)
                                    && !self.processed.contains(&op_node)
                                {
                                    self.work_queue.push_back(op_node);
                                }
                            }
                            // Bool ops
                            for &op_node in &next_info.bool_ops {
                                self.processed.remove(&op_node);
                                self.nodes_inside_cfg_blocks.remove(&op_node);
                                if !self.work_queue.contains(&op_node)
                                    && !self.processed.contains(&op_node)
                                {
                                    self.work_queue.push_back(op_node);
                                }
                            }
                            // Conditional nodes
                            for &cond_node in &next_info.conditional_nodes {
                                self.processed.remove(&cond_node);
                                self.nodes_inside_cfg_blocks.remove(&cond_node);
                                if !self.work_queue.contains(&cond_node)
                                    && !self.processed.contains(&cond_node)
                                {
                                    self.work_queue.push_back(cond_node);
                                }
                            }
                            // TailLoop nodes
                            for &tl_node in &next_info.tailloop_nodes {
                                self.processed.remove(&tl_node);
                                self.nodes_inside_cfg_blocks.remove(&tl_node);
                                if !self.work_queue.contains(&tl_node)
                                    && !self.processed.contains(&tl_node)
                                {
                                    self.work_queue.push_back(tl_node);
                                }
                            }
                            // Call nodes
                            for &call_node in &next_info.call_nodes {
                                self.processed.remove(&call_node);
                                self.nodes_inside_cfg_blocks.remove(&call_node);
                                if !self.work_queue.contains(&call_node)
                                    && !self.processed.contains(&call_node)
                                    && all_predecessors_ready(
                                        hugr,
                                        call_node,
                                        &self.quantum_ops,
                                        &self.conditionals,
                                        &self.cfgs,
                                        &self.processed,
                                    )
                                {
                                    self.work_queue.push_back(call_node);
                                }
                            }
                            // Also find and add classical ops and extension ops
                            for child in hugr.children(next_block) {
                                let op = hugr.get_optype(child);
                                if matches!(op, OpType::LoadConstant(_))
                                    || self.classical_ops.contains_key(&child)
                                    || op.as_extension_op().is_some()
                                {
                                    self.processed.remove(&child);
                                    self.nodes_inside_cfg_blocks.remove(&child);
                                    if !self.work_queue.contains(&child)
                                        && !self.processed.contains(&child)
                                    {
                                        self.work_queue.push_back(child);
                                    }
                                }
                            }
                            debug!(
                                "[TRACE] Activated next block {:?} with {} quantum ops, {} bool_ops",
                                next_block,
                                next_info.quantum_ops.len(),
                                next_info.bool_ops.len()
                            );

                            // Check if the next block is also empty - if so, we need to handle it recursively
                            // Find extension ops in this block
                            let next_extension_ops: Vec<Node> =
                                find_extension_ops_in_block(hugr, next_block);
                            let next_has_extension_ops = !next_extension_ops.is_empty();
                            let next_has_classical_ops = !next_info.classical_ops.is_empty();

                            if next_info.quantum_ops.is_empty()
                                && next_info.call_nodes.is_empty()
                                && next_info.conditional_nodes.is_empty()
                                && next_info.bool_ops.is_empty()
                                && !next_has_extension_ops
                                && !next_has_classical_ops
                                && next_info.tailloop_nodes.is_empty()
                            {
                                // Next block is also empty - need to continue transitioning
                                let next_successors = next_info.successors.clone();
                                if next_successors.len() == 1 {
                                    let next_next_block = next_successors[0];
                                    if next_next_block == cfg_info.exit_block {
                                        self.complete_cfg_execution(hugr, cfg_node, next_block);
                                    } else {
                                        // Recursively transition
                                        self.transition_to_cfg_successor(
                                            hugr,
                                            cfg_node,
                                            next_block,
                                            next_next_block,
                                        );
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Multiple successors - need to resolve branch
                    debug!(
                        "[TRACE] Block {:?} has {} successors, resolving branch",
                        to_block,
                        successors.len()
                    );
                    if let Some(branch_idx) = self.try_resolve_cfg_block_branch(hugr, to_block) {
                        debug!("[TRACE] Branch resolved to {branch_idx} for block {to_block:?}");
                        if branch_idx < successors.len() {
                            let next_block = successors[branch_idx];
                            debug!(
                                "[TRACE] Empty block {to_block:?} resolved branch {branch_idx} to {next_block:?}"
                            );
                            // Recursively transition
                            self.transition_to_cfg_successor(hugr, cfg_node, to_block, next_block);
                        }
                    } else {
                        debug!(
                            "[TRACE] Branch NOT resolved for block {to_block:?}, adding to pending"
                        );
                        // Branch not resolved - add to pending
                        let block_key = (cfg_node, to_block);
                        self.pending_cfg_branches.insert(block_key, successors);
                    }
                }
            }
        }
    }

    /// Complete CFG execution and propagate outputs.
    pub(crate) fn complete_cfg_execution(
        &mut self,
        hugr: &Hugr,
        cfg_node: Node,
        final_block: Node,
    ) {
        debug!("complete_cfg_execution: CFG {cfg_node:?} from block {final_block:?}");

        // Propagate outputs from final block to CFG output ports
        self.propagate_cfg_outputs(hugr, cfg_node, final_block);

        // Mark CFG as processed
        self.processed.insert(cfg_node);
        self.active_cfgs.remove(&cfg_node);

        // Check if this CFG is inside a FuncDefn that's being called
        self.complete_func_call_if_needed(hugr, cfg_node);

        // Add CFG successors to work queue
        let successors: Vec<_> = hugr.output_neighbours(cfg_node).collect();
        debug!(
            "CFG {:?} has {} successors: {:?}",
            cfg_node,
            successors.len(),
            successors
        );
        for succ_node in successors {
            // Include classical ops (like tket.result) as well as quantum ops
            let is_relevant = self.quantum_ops.contains_key(&succ_node)
                || self.classical_ops.contains_key(&succ_node)
                || self.conditionals.contains_key(&succ_node)
                || self.cfgs.contains_key(&succ_node)
                || self.tailloops.contains_key(&succ_node);

            // Also check if it's an extension op by looking at the optype
            let succ_op = hugr.get_optype(succ_node);
            let is_extension = succ_op.as_extension_op().is_some();

            debug!(
                "Successor {:?}: op={:?}, is_relevant={}, is_extension={}, processed={}, in_queue={}",
                succ_node,
                succ_op,
                is_relevant,
                is_extension,
                self.processed.contains(&succ_node),
                self.work_queue.contains(&succ_node)
            );

            if (is_relevant || is_extension)
                && !self.processed.contains(&succ_node)
                && !self.work_queue.contains(&succ_node)
                && all_predecessors_ready(
                    hugr,
                    succ_node,
                    &self.quantum_ops,
                    &self.conditionals,
                    &self.cfgs,
                    &self.processed,
                )
            {
                debug!("CFG complete: adding successor {succ_node:?} to work queue");
                self.work_queue.push_back(succ_node);
            }
        }
    }

    /// Propagate wire mappings from a completed block to a successor block.
    #[allow(clippy::too_many_lines)] // Wire propagation logic is inherently complex
    pub(crate) fn propagate_block_outputs_to_successor(
        &mut self,
        hugr: &Hugr,
        from_block: Node,
        to_block: Node,
    ) {
        debug!("[TRACE] propagate_block_outputs_to_successor: from {from_block:?} to {to_block:?}");
        let from_output = find_output_node(hugr, from_block);
        let to_input = find_input_node(hugr, to_block);
        debug!("[TRACE] from_output={from_output:?}, to_input={to_input:?}");

        let (Some(from_output), Some(to_input)) = (from_output, to_input) else {
            debug!("[TRACE] Cannot propagate: from_output={from_output:?}, to_input={to_input:?}");
            return;
        };

        // Block Output ports: [Sum (port 0), data1, data2, ...]
        // For CFG blocks with branching, the successor Input ports are:
        //   [payload from Sum (if any), other_outputs...]
        // So we need to:
        // 1. Check if port 0's source is a Conditional/Tag with payload
        // 2. Extract payload values and map them to successor Input ports 0..payload_len-1
        // 3. Map other_outputs (port 1+) to successor Input ports payload_len+

        let num_output_ports = hugr.num_inputs(from_output);

        // First, check if port 0 (Sum) has payload values from a Conditional
        let sum_port = IncomingPort::from(0);
        let mut payload_len = 0;

        if let Some((sum_src_node, _)) = hugr.single_linked_output(from_output, sum_port) {
            let sum_src_op = hugr.get_optype(sum_src_node);

            // Check if it's a Conditional - extract payload from virtual output ports
            if matches!(sum_src_op, OpType::Conditional(_)) {
                // Look for payload values at virtual output ports (1, 2, ...)
                let mut idx = 1;
                while let Some(value) = self
                    .wire_state
                    .classical_values
                    .get(&(sum_src_node, idx))
                    .cloned()
                {
                    let to_wire = (to_input, idx - 1);
                    self.wire_state.classical_values.insert(to_wire, value);
                    payload_len += 1;
                    idx += 1;
                }
            }
        }

        // Now map other_outputs (port 1+) to successor Input ports
        // Check if the target Input node has enough outputs to accommodate the payload offset
        let target_num_outputs = hugr.num_outputs(to_input);
        let num_data_outputs = num_output_ports.saturating_sub(1);
        // Only apply payload offset if the target has enough outputs
        // This handles exit blocks which don't expect payloads
        let effective_payload_len = if payload_len + num_data_outputs <= target_num_outputs {
            payload_len
        } else {
            0 // Target doesn't have room for payloads, don't offset
        };
        debug!("[TRACE] num_data_outputs={num_data_outputs}");
        debug!(
            "[TRACE] propagate_block_outputs: from_block={from_block:?}, to_block={to_block:?}, num_data_outputs={num_data_outputs}"
        );

        for port_idx in 0..num_data_outputs {
            let from_port = IncomingPort::from(port_idx + 1); // Skip Sum port
            let to_port_idx = effective_payload_len + port_idx; // Offset by effective payload length
            debug!("[TRACE] port_idx={port_idx}, from_port={from_port:?}");

            if let Some((src_node, src_port)) = hugr.single_linked_output(from_output, from_port) {
                let src_op = hugr.get_optype(src_node);
                debug!(
                    "[TRACE] linked to src_node={:?}, src_port={:?}, op={:?}",
                    src_node,
                    src_port.index(),
                    std::mem::discriminant(src_op)
                );
                let src_wire = (src_node, src_port.index());

                if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                    self.wire_state
                        .wire_to_qubit
                        .insert((to_input, to_port_idx), qubit_id);
                    debug!(
                        "[TRACE] Block transition: mapped qubit {:?} from {:?}:{} to {:?}:{}",
                        qubit_id,
                        from_output,
                        port_idx + 1,
                        to_input,
                        to_port_idx
                    );
                }

                // Also propagate classical values
                if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                    let to_wire = (to_input, to_port_idx);
                    debug!(
                        "[TRACE] Block transition: propagated classical value {value:?} from {src_wire:?} to {to_wire:?}"
                    );
                    self.wire_state.classical_values.insert(to_wire, value);
                } else {
                    // Try to resolve constant value at source
                    if let Some(const_value) = Self::try_resolve_const_bool(hugr, src_node) {
                        let to_wire = (to_input, to_port_idx);
                        self.wire_state
                            .classical_values
                            .insert(to_wire, ClassicalValue::Bool(const_value));
                        debug!(
                            "[TRACE] Block transition: resolved constant bool {const_value} for {to_wire:?}"
                        );
                    } else if !self.wire_state.wire_to_qubit.contains_key(&src_wire) {
                        debug!(
                            "[TRACE] No qubit or classical mapping for wire {:?} (from_output {:?} port {})",
                            src_wire,
                            from_output,
                            port_idx + 1
                        );
                    }
                }
            } else {
                debug!(
                    "[TRACE] No linked output for {:?} port {}",
                    from_output,
                    port_idx + 1
                );

                // Fallback: if the Output node has no connection for this port,
                // try to get the value from the Input node's corresponding output.
                // This handles cases where values are "passed through" without explicit wiring.
                let from_input = find_input_node(hugr, from_block);
                if let Some(from_input_node) = from_input {
                    let input_wire = (from_input_node, port_idx);
                    if let Some(value) = self.wire_state.classical_values.get(&input_wire).cloned()
                    {
                        let to_wire = (to_input, to_port_idx);
                        debug!(
                            "[TRACE] Fallback: propagating {value:?} from input {input_wire:?} to {to_wire:?}"
                        );
                        self.wire_state.classical_values.insert(to_wire, value);
                    }
                }
            }
        }
    }

    /// Re-propagate measurement values to successor blocks.
    ///
    /// When a CFG block completes and transitions to a successor, the propagation
    /// happens before measurement results are available. This function re-propagates
    /// values after measurement results are stored.
    pub(crate) fn repropagate_measurement_values(&mut self, hugr: &Hugr) {
        // Take ownership of the pending list to avoid borrow issues
        let pending: Vec<_> = std::mem::take(&mut self.pending_measurement_propagations);

        for (_cfg_node, from_block, to_block) in pending {
            self.propagate_block_outputs_to_successor(hugr, from_block, to_block);
        }
    }

    /// Propagate wire mappings from final block to CFG outputs.
    pub(crate) fn propagate_cfg_outputs(&mut self, hugr: &Hugr, cfg_node: Node, final_block: Node) {
        let Some(output_node) = find_output_node(hugr, final_block) else {
            debug!("No Output node found in final block {final_block:?}");
            return;
        };

        // Block Output: port 0 = Sum (control), ports 1+ = data
        // CFG outputs correspond to data ports (skip the Sum)
        let num_data_outputs = hugr.num_inputs(output_node).saturating_sub(1);

        for port_idx in 0..num_data_outputs {
            let block_port = IncomingPort::from(port_idx + 1); // Skip Sum port

            if let Some((src_node, src_port)) = hugr.single_linked_output(output_node, block_port) {
                let src_wire = (src_node, src_port.index());

                if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                    self.wire_state
                        .wire_to_qubit
                        .insert((cfg_node, port_idx), qubit_id);
                    debug!("CFG {cfg_node:?} output {port_idx}: mapped qubit {qubit_id:?}");
                }
            }
        }
    }

    /// Propagate wire mappings from CFG inputs to the entry block's Input node.
    ///
    /// When a CFG is activated, qubits flowing into the CFG need to be mapped
    /// to the entry block's Input node outputs, so operations inside the block
    /// can resolve their qubit inputs.
    pub(crate) fn propagate_cfg_inputs_to_entry_block(
        &mut self,
        hugr: &Hugr,
        cfg_node: Node,
        entry_block: Node,
    ) {
        // Find the Input node inside the entry block
        let Some(input_node) = find_input_node(hugr, entry_block) else {
            debug!("No Input node found in entry block {entry_block:?}");
            return;
        };

        // Get number of CFG inputs
        let num_cfg_inputs = hugr.num_inputs(cfg_node);
        debug!(
            "Propagating {num_cfg_inputs} CFG inputs from {cfg_node:?} to entry block {entry_block:?} Input {input_node:?}"
        );

        // Map each CFG input to the corresponding entry block Input node output
        for port_idx in 0..num_cfg_inputs {
            let cfg_in_port = IncomingPort::from(port_idx);

            if let Some((src_node, src_port)) = hugr.single_linked_output(cfg_node, cfg_in_port) {
                let src_wire = (src_node, src_port.index());

                // Check for qubit mapping
                if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                    // Map to entry block's Input node output
                    self.wire_state
                        .wire_to_qubit
                        .insert((input_node, port_idx), qubit_id);
                    debug!(
                        "CFG {cfg_node:?}: mapped input {port_idx} qubit {qubit_id:?} to entry Input {input_node:?}:{port_idx}"
                    );
                }

                // Also propagate classical values
                if let Some(value) = self.wire_state.classical_values.get(&src_wire).cloned() {
                    debug!(
                        "CFG {cfg_node:?}: propagated classical value {value:?} to entry Input {input_node:?}:{port_idx}"
                    );
                    self.wire_state
                        .classical_values
                        .insert((input_node, port_idx), value);
                }
            }
        }
    }
}
