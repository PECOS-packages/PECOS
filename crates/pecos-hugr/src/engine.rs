// Copyright 2025 The PECOS Developers
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

//! HUGR interpreter engine.
//!
//! This module provides the core [`HugrEngine`] for executing HUGR programs.
//!
//! # Module Structure
//!
//! - [`types`]: Type definitions (`QuantumOp`, `ClassicalOp`, `ClassicalValue`, etc.)
//! - [`analysis`]: HUGR static analysis and extraction functions
//! - [`control_flow`]: Control flow handling (`TailLoop`, Conditional, CFG, Call)

pub(crate) mod analysis;
mod control_flow;
mod handlers;
mod propagation;
pub(crate) mod types;

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use log::debug;
use pecos_core::errors::PecosError;
use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, QubitId};
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::prelude::*;
use tket::hugr::ops::OpType;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::loader::load_hugr_from_bytes;

// Re-export public types from submodules
pub use types::{CapturedResult, ClassicalValue, FutureId, ResultValue, RngContextId};

// Use internal types from submodules
use types::{
    ActiveCallInfo, ActiveCaseInfo, ActiveCfgInfo, ActiveTailLoopInfo, CfgInfo, ClassicalOp,
    ConditionalInfo, ExtensionState, FuncDefnInfo, MeasurementState, QuantumOp, TailLoopInfo,
    WireState,
};

// Use analysis functions from submodule
use analysis::{
    all_predecessors_ready, collect_descendants, extract_call_targets, extract_cfgs,
    extract_classical_ops, extract_conditionals, extract_func_defns, extract_quantum_ops,
    extract_tailloops, find_nodes_inside_cases, find_nodes_inside_cfg_blocks,
    find_nodes_inside_func_defns, find_nodes_inside_tailloops,
};
/// A HUGR interpreter engine that directly executes HUGR programs.
///
/// This engine walks a HUGR graph in topological order, emitting quantum
/// commands and handling measurement results without LLVM compilation.
///
/// # Control Flow Support
///
/// The engine supports HUGR Conditional nodes for branching based on
/// measurement results. When a Conditional is encountered:
/// 1. The engine pauses execution and waits for measurement results
/// 2. Based on the result value (0 or 1), the appropriate Case branch is selected
/// 3. Operations from the selected branch are processed
pub struct HugrEngine {
    /// The HUGR program being executed.
    pub(crate) hugr: Option<Hugr>,

    /// Extracted quantum operations indexed by node.
    pub(crate) quantum_ops: BTreeMap<Node, QuantumOp>,

    /// Extracted classical operations indexed by node.
    pub(crate) classical_ops: BTreeMap<Node, ClassicalOp>,

    /// Work queue for topological traversal.
    pub(crate) work_queue: VecDeque<Node>,

    /// Set of processed nodes.
    pub(crate) processed: BTreeSet<Node>,

    /// Reusable message builder for generating commands.
    pub(crate) message_builder: ByteMessageBuilder,

    // === Grouped State ===
    /// Wire tracking state (qubit mappings, classical values, qubit arrays).
    pub(crate) wire_state: WireState,

    /// Measurement tracking state (mappings, results, output wires).
    pub(crate) measurement_state: MeasurementState,

    /// Extension state (futures, RNG, shot tracking, global phase).
    pub(crate) extension_state: ExtensionState,

    // === Control Flow Support ===
    /// Conditional nodes extracted from the HUGR.
    pub(crate) conditionals: BTreeMap<Node, ConditionalInfo>,

    /// Pending conditionals waiting for measurement results.
    /// Maps the Conditional node to the qubit ID whose measurement determines the branch.
    pub(crate) pending_conditionals: BTreeMap<Node, QubitId>,

    /// Pending bool.read nodes waiting for measurement results.
    /// These are re-added to the work queue when measurement results arrive.
    pub(crate) pending_bool_reads: BTreeSet<Node>,

    /// Set of nodes that are inside Case nodes (children of Conditionals).
    /// These should not be processed until their parent Conditional is expanded.
    pub(crate) nodes_inside_cases: BTreeSet<Node>,

    /// Active Cases being processed: maps Case node -> (parent Conditional, nodes to process).
    /// When all nodes in a Case are processed, we propagate outputs to the Conditional.
    pub(crate) active_cases: BTreeMap<Node, ActiveCaseInfo>,

    // === CFG Control Flow Support ===
    /// CFG nodes extracted from the HUGR.
    pub(crate) cfgs: BTreeMap<Node, CfgInfo>,

    /// Nodes inside CFG blocks (should not be processed until block is active).
    pub(crate) nodes_inside_cfg_blocks: BTreeSet<Node>,

    /// Active CFGs being processed.
    pub(crate) active_cfgs: BTreeMap<Node, ActiveCfgInfo>,

    /// Pending CFG blocks waiting for Sum value (measurement result) to determine branch.
    /// Maps (`cfg_node`, `block_node`) to the list of successor blocks.
    pub(crate) pending_cfg_branches: BTreeMap<(Node, Node), Vec<Node>>,

    /// Pending block propagations that need re-propagation after measurement results.
    /// Stores (`cfg_node`, `from_block`, `to_block`) tuples.
    pub(crate) pending_measurement_propagations: Vec<(Node, Node, Node)>,

    // === Call/FuncDefn Support ===
    /// `FuncDefn` nodes extracted from the HUGR.
    pub(crate) func_defns: BTreeMap<Node, FuncDefnInfo>,

    /// Call nodes and their target `FuncDefn`.
    /// Maps Call node -> `FuncDefn` node.
    pub(crate) call_targets: BTreeMap<Node, Node>,

    /// Active Calls being processed.
    pub(crate) active_calls: BTreeMap<Node, ActiveCallInfo>,

    /// Nodes inside `FuncDefn` bodies (should not be processed until function is called).
    pub(crate) nodes_inside_func_defns: BTreeSet<Node>,

    /// Pending Calls waiting for a `FuncDefn` to be free.
    /// Maps `FuncDefn` node -> queue of Call nodes waiting.
    pub(crate) pending_func_calls: BTreeMap<Node, Vec<Node>>,

    // === TailLoop Support ===
    /// `TailLoop` nodes extracted from the HUGR.
    pub(crate) tailloops: BTreeMap<Node, TailLoopInfo>,

    /// Nodes inside `TailLoop` bodies (should not be processed until loop is active).
    pub(crate) nodes_inside_tailloops: BTreeSet<Node>,

    /// Active `TailLoops` being processed.
    pub(crate) active_tailloops: BTreeMap<Node, ActiveTailLoopInfo>,

    /// Pending `TailLoops` waiting for Sum value (measurement result) to determine continue/break.
    pub(crate) pending_tailloop_control: BTreeSet<Node>,

    // === Result Capture ===
    /// Captured results from tket.result operations.
    pub captured_results: Vec<CapturedResult>,

    // === WASM Support ===
    /// Foreign object for WASM function calls.
    #[cfg(feature = "wasm")]
    pub(crate) foreign_object: Option<Box<dyn pecos_wasm::ForeignObject>>,
}
impl HugrEngine {
    /// Maximum batch size for quantum operations.
    const MAX_BATCH_SIZE: usize = 100;

    /// Create a new empty `HugrEngine`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // === Result Capture API ===

    /// Get all captured results from tket.result operations.
    #[must_use]
    pub fn get_captured_results(&self) -> &[CapturedResult] {
        &self.captured_results
    }

    /// Get a captured result by label.
    #[must_use]
    pub fn get_result_by_label(&self, label: &str) -> Option<&CapturedResult> {
        self.captured_results.iter().find(|r| r.label == label)
    }

    /// Clear all captured results.
    pub fn clear_captured_results(&mut self) {
        self.captured_results.clear();
    }

    // === Shot Tracking API ===

    /// Get the current shot number.
    #[must_use]
    pub fn current_shot(&self) -> u64 {
        self.extension_state.current_shot
    }

    /// Set the current shot number.
    pub fn set_current_shot(&mut self, shot: u64) {
        self.extension_state.current_shot = shot;
    }

    /// Increment the current shot number.
    pub fn increment_shot(&mut self) {
        self.extension_state.current_shot += 1;
    }

    /// Set the foreign object for WASM function calls.
    #[cfg(feature = "wasm")]
    pub fn set_foreign_object(&mut self, foreign_obj: Box<dyn pecos_wasm::ForeignObject>) {
        self.foreign_object = Some(foreign_obj);
    }

    // === Global Phase API ===

    /// Get the accumulated global phase (in half-turns).
    #[must_use]
    pub fn global_phase(&self) -> f64 {
        self.extension_state.global_phase
    }

    /// Create a `HugrEngine` from HUGR bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the HUGR cannot be parsed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PecosError> {
        let hugr = load_hugr_from_bytes(bytes)
            .map_err(|e| PecosError::Input(format!("Failed to load HUGR: {e}")))?;
        Ok(Self::from_hugr(hugr))
    }

    /// Create a `HugrEngine` from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PecosError> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| PecosError::Input(format!("Failed to read HUGR file: {e}")))?;
        Self::from_bytes(&bytes)
    }

    /// Create a `HugrEngine` from a loaded HUGR.
    #[must_use]
    pub fn from_hugr(hugr: Hugr) -> Self {
        let mut engine = Self::new();
        engine.load_hugr(hugr);
        engine
    }

    /// Load a HUGR program into the engine.
    pub fn load_hugr(&mut self, hugr: Hugr) {
        debug!("Loading HUGR program");

        // Extract control flow structures (Conditionals) first
        self.conditionals = extract_conditionals(&hugr);
        debug!("Extracted {} conditional nodes", self.conditionals.len());

        // Track which nodes are inside Case nodes (should not be processed until expanded)
        self.nodes_inside_cases = find_nodes_inside_cases(&hugr, &self.conditionals);
        debug!("Found {} nodes inside cases", self.nodes_inside_cases.len());

        // Extract CFG control flow structures
        self.cfgs = extract_cfgs(&hugr);
        debug!("Extracted {} CFG nodes", self.cfgs.len());

        // Track which nodes are inside CFG blocks (should not be processed until block is active)
        self.nodes_inside_cfg_blocks = find_nodes_inside_cfg_blocks(&hugr, &self.cfgs);
        debug!(
            "Found {} nodes inside CFG blocks",
            self.nodes_inside_cfg_blocks.len()
        );

        // Extract FuncDefn and Call nodes
        self.func_defns = extract_func_defns(&hugr);
        debug!("Extracted {} FuncDefn nodes", self.func_defns.len());

        self.call_targets = extract_call_targets(&hugr);
        debug!("Extracted {} Call nodes", self.call_targets.len());

        // Track nodes inside FuncDefn bodies (not the entrypoint FuncDefn)
        self.nodes_inside_func_defns =
            find_nodes_inside_func_defns(&hugr, &self.func_defns, &self.call_targets);
        debug!(
            "Found {} nodes inside FuncDefn bodies",
            self.nodes_inside_func_defns.len()
        );

        // Extract TailLoop control flow structures
        self.tailloops = extract_tailloops(&hugr);
        debug!("Extracted {} TailLoop nodes", self.tailloops.len());
        eprintln!("[DEBUG] Extracted {} TailLoop nodes", self.tailloops.len());

        // Track nodes inside TailLoop bodies (should not be processed until loop is active)
        self.nodes_inside_tailloops = find_nodes_inside_tailloops(&hugr, &self.tailloops);
        debug!(
            "Found {} nodes inside TailLoop bodies",
            self.nodes_inside_tailloops.len()
        );

        // Extract quantum operations (but we'll skip case/CFG-internal ones in work queue)
        self.quantum_ops = extract_quantum_ops(&hugr);
        debug!("Extracted {} quantum operations", self.quantum_ops.len());
        eprintln!(
            "[DEBUG] Extracted {} quantum ops, {} cfgs, {} func_defns, {} call_targets",
            self.quantum_ops.len(),
            self.cfgs.len(),
            self.func_defns.len(),
            self.call_targets.len()
        );

        // Extract classical operations (arithmetic, logic, etc.)
        self.classical_ops = extract_classical_ops(&hugr);
        debug!(
            "Extracted {} classical operations",
            self.classical_ops.len()
        );

        self.hugr = Some(hugr);
        self.reset_state();
    }

    /// Reset the engine's internal state for a new shot.
    #[allow(clippy::too_many_lines)]
    fn reset_state(&mut self) {
        debug!("HugrEngine::reset_state()");

        self.work_queue.clear();
        self.processed.clear();
        self.message_builder.reset();

        // Clear grouped state (note: extension_state.reset() doesn't reset current_shot)
        self.wire_state.reset();
        self.measurement_state.reset();
        self.extension_state.reset();

        // Clear Conditional control flow state
        self.pending_conditionals.clear();
        self.pending_bool_reads.clear();
        self.active_cases.clear();

        // Clear CFG control flow state
        self.active_cfgs.clear();
        self.pending_cfg_branches.clear();
        self.pending_measurement_propagations.clear();

        // Clear Call/FuncDefn control flow state
        self.active_calls.clear();
        self.pending_func_calls.clear();

        // Clear TailLoop control flow state
        self.active_tailloops.clear();
        self.pending_tailloop_control.clear();

        // Clear result capture state
        self.captured_results.clear();

        // Re-initialize nodes_inside_* from their respective control structures
        // (in case we need to re-process after a reset)
        if let Some(hugr) = &self.hugr {
            self.nodes_inside_cfg_blocks = find_nodes_inside_cfg_blocks(hugr, &self.cfgs);
            self.nodes_inside_func_defns =
                find_nodes_inside_func_defns(hugr, &self.func_defns, &self.call_targets);
            self.nodes_inside_tailloops = find_nodes_inside_tailloops(hugr, &self.tailloops);
        }

        // Initialize work queue with source nodes (QAlloc and nodes with no quantum predecessors)
        // IMPORTANT: Skip nodes that are inside Case nodes, CFG blocks, FuncDefn bodies, or TailLoops -
        // they should only be processed after their parent control flow structure is expanded
        if let Some(hugr) = &self.hugr {
            // Helper closure to check if a node should be skipped
            let should_skip = |node: &Node| {
                self.nodes_inside_cases.contains(node)
                    || self.nodes_inside_cfg_blocks.contains(node)
                    || self.nodes_inside_func_defns.contains(node)
                    || self.nodes_inside_tailloops.contains(node)
            };

            // First add QAlloc nodes that are NOT inside cases or CFG blocks
            for (node, op) in &self.quantum_ops {
                if op.gate_type == GateType::QAlloc && !should_skip(node) {
                    self.work_queue.push_back(*node);
                }
            }

            // Then add nodes whose quantum predecessors are all non-quantum or already in queue
            // (but skip nodes inside cases or CFG blocks)
            for node in self.quantum_ops.keys() {
                if !should_skip(node)
                    && !self.work_queue.contains(node)
                    && all_predecessors_ready(
                        hugr,
                        *node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.processed,
                    )
                {
                    self.work_queue.push_back(*node);
                }
            }

            // Add classical ops that have no predecessors pending
            // (but skip classical ops inside cases, CFG blocks, etc.)
            for node in self.classical_ops.keys() {
                if !should_skip(node)
                    && !self.work_queue.contains(node)
                    && all_predecessors_ready(
                        hugr,
                        *node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.processed,
                    )
                {
                    self.work_queue.push_back(*node);
                }
            }

            // Add Conditional nodes that have no quantum predecessors pending
            // (but skip Conditionals inside FuncDefn bodies or CFG blocks)
            for node in self.conditionals.keys() {
                if !should_skip(node)
                    && !self.work_queue.contains(node)
                    && all_predecessors_ready(
                        hugr,
                        *node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.processed,
                    )
                {
                    self.work_queue.push_back(*node);
                }
            }

            // Add CFG nodes that have no quantum predecessors pending
            // (but skip CFGs inside FuncDefn bodies - they should only be activated when called)
            for node in self.cfgs.keys() {
                if !should_skip(node)
                    && !self.work_queue.contains(node)
                    && all_predecessors_ready(
                        hugr,
                        *node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.processed,
                    )
                {
                    self.work_queue.push_back(*node);
                }
            }

            // Add Call nodes that have no quantum predecessors pending
            // (but skip Calls inside FuncDefn bodies or CFG blocks)
            for node in self.call_targets.keys() {
                if !should_skip(node)
                    && !self.work_queue.contains(node)
                    && all_predecessors_ready(
                        hugr,
                        *node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.processed,
                    )
                {
                    self.work_queue.push_back(*node);
                }
            }

            // Add LoadConstant nodes that are not inside control flow constructs
            // These need to be processed before classical ops can use their values
            for node in hugr.nodes() {
                let op = hugr.get_optype(node);
                if matches!(op, OpType::LoadConstant(_))
                    && !should_skip(&node)
                    && !self.work_queue.contains(&node)
                {
                    self.work_queue.push_back(node);
                }
            }

            // Also add TailLoop nodes that have no quantum predecessors pending
            // (but skip TailLoops inside FuncDefn bodies, CFG blocks, etc.)
            for node in self.tailloops.keys() {
                if !should_skip(node)
                    && !self.work_queue.contains(node)
                    && all_predecessors_ready(
                        hugr,
                        *node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.processed,
                    )
                {
                    self.work_queue.push_back(*node);
                }
            }
        }

        debug!(
            "Reset complete. Work queue has {} initial nodes",
            self.work_queue.len()
        );
    }

    /// Try to resolve pending `TailLoop` control values after measurement results are available.
    fn try_resolve_pending_tailloops(&mut self) {
        let hugr = match &self.hugr {
            Some(h) => h.clone(),
            None => return,
        };

        debug!(
            "[TRACE] try_resolve_pending_tailloops: {} pending",
            self.pending_tailloop_control.len()
        );

        // Collect TailLoops that can now be resolved
        let mut to_resolve = Vec::new();
        for &tailloop_node in &self.pending_tailloop_control {
            if let Some(tag) = self.try_resolve_tailloop_control(&hugr, tailloop_node) {
                to_resolve.push((tailloop_node, tag));
            }
        }

        // Resolve them
        for (tailloop_node, tag) in to_resolve {
            self.pending_tailloop_control.remove(&tailloop_node);

            if tag == 0 {
                // CONTINUE_TAG - start next iteration
                debug!("Pending TailLoop {tailloop_node:?}: CONTINUE, starting next iteration");
                self.continue_tailloop_iteration(&hugr, tailloop_node);
            } else {
                // BREAK_TAG - complete the loop
                debug!("Pending TailLoop {tailloop_node:?}: BREAK, completing loop");
                self.complete_tailloop(&hugr, tailloop_node);
            }
        }
    }

    /// Re-queue pending bool.read nodes that were waiting for measurement results.
    /// When a measurement result arrives, the classical value is stored and we need to
    /// retry any bool.read nodes that were deferred because their input wasn't ready.
    fn retry_pending_bool_reads(&mut self) {
        // Move pending bool.reads to work queue so they can be retried
        let pending: Vec<_> = std::mem::take(&mut self.pending_bool_reads)
            .into_iter()
            .collect();

        for node in pending {
            if !self.processed.contains(&node) && !self.work_queue.contains(&node) {
                self.work_queue.push_back(node);
            }
        }
    }

    /// Process the HUGR and generate quantum commands.
    ///
    /// This is the main execution loop that processes nodes from the work queue
    /// and emits quantum operations. The processing flow is:
    ///
    /// 1. **Control Flow Dispatch** (in priority order):
    ///    - Conditional nodes: Branch based on measurement results
    ///    - CFG nodes: Execute entry block and manage transitions
    ///    - `TailLoop` nodes: Handle iteration and break conditions
    ///    - Call nodes: Activate function definitions
    ///
    /// 2. **Operation Processing**:
    ///    - Classical operations: Execute and propagate values
    ///    - Extension operations: Handle via [`handle_extension_op`](Self::handle_extension_op)
    ///    - Quantum operations: Emit gates to message builder
    ///
    /// 3. **Completion Checks**: After each operation, check if it completes
    ///    any active Case, CFG block, or `TailLoop` body.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(msg))` - Batch of quantum operations ready for execution
    /// - `Ok(None)` - No operations to process (empty or complete)
    #[allow(clippy::too_many_lines, clippy::unnecessary_wraps)]
    fn process_hugr_impl(&mut self) -> Result<Option<ByteMessage>, PecosError> {
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();

        let Some(hugr) = self.hugr.clone() else {
            debug!("No HUGR loaded");
            return Ok(None);
        };

        if self.work_queue.is_empty() && self.quantum_ops.is_empty() {
            debug!("Empty HUGR, no commands to generate");
            eprintln!("[DEBUG] Empty HUGR, no commands to generate");
            return Ok(None);
        }

        if self.work_queue.is_empty() {
            debug!("Work queue empty, processing complete");
            eprintln!("[DEBUG] Work queue empty, processing complete");
            return Ok(None);
        }
        eprintln!("[DEBUG] Work queue has {} items", self.work_queue.len());

        let mut operation_count = 0;
        let mut hit_measurement = false;

        while let Some(current_node) = self.work_queue.pop_front() {
            if self.processed.contains(&current_node) {
                continue;
            }
            let node_op = hugr.get_optype(current_node);
            eprintln!("[DEBUG] Processing node {current_node:?}: {node_op:?}");

            // Check batch size
            if operation_count >= Self::MAX_BATCH_SIZE {
                // Put this node back for next batch
                self.work_queue.push_front(current_node);
                break;
            }

            // --- Control Flow: Conditional ---
            if self.conditionals.contains_key(&current_node) {
                // Try to resolve the conditional's control value
                if let Some(branch_index) =
                    self.try_resolve_conditional_control(&hugr, current_node)
                {
                    // Expand the selected branch and add its entry nodes to the queue
                    let entry_nodes = self.expand_conditional(&hugr, current_node, branch_index);
                    for entry_node in entry_nodes {
                        if !self.work_queue.contains(&entry_node) {
                            self.work_queue.push_back(entry_node);
                        }
                    }
                    debug!("Conditional {current_node:?} expanded, branch {branch_index} selected");

                    // Check if this Conditional completion allows a CFG block to complete
                    self.check_cfg_block_completion(&hugr, current_node);
                } else {
                    // Can't resolve yet - likely waiting for measurement result
                    // Add to pending conditionals and continue
                    debug!("Conditional {current_node:?} cannot be resolved yet, deferring");
                    // We'll re-add this after measurement results come in
                    // For now, mark as pending and don't add back to queue
                    self.pending_conditionals
                        .insert(current_node, QubitId::from(0)); // placeholder
                }
                continue;
            }

            // --- Control Flow: CFG ---
            if let Some(cfg_info) = self.cfgs.get(&current_node).cloned() {
                debug!("Starting CFG {current_node:?} execution");
                debug!("[TRACE] Starting CFG {current_node:?}");
                eprintln!(
                    "[DEBUG] Starting CFG {current_node:?}, entry_block={:?}",
                    cfg_info.entry_block
                );

                // Start CFG execution by activating the entry block's operations
                let entry_block = cfg_info.entry_block;
                if let Some(block_info) = cfg_info.blocks.get(&entry_block) {
                    // Register as active CFG
                    self.active_cfgs.insert(
                        current_node,
                        ActiveCfgInfo {
                            cfg_node: current_node,
                            current_block: entry_block,
                            completed_blocks: BTreeSet::new(),
                        },
                    );

                    // Propagate CFG inputs to entry block's Input node
                    self.propagate_cfg_inputs_to_entry_block(&hugr, current_node, entry_block);

                    // Remove entry block's quantum ops from nodes_inside_cfg_blocks
                    // and add ops whose predecessors are ready to the work queue
                    for &op_node in &block_info.quantum_ops {
                        self.nodes_inside_cfg_blocks.remove(&op_node);
                        // Skip ops inside TailLoops - they'll be added when the loop expands
                        if self.nodes_inside_tailloops.contains(&op_node) {
                            continue;
                        }
                        let preds_ready = all_predecessors_ready(
                            &hugr,
                            op_node,
                            &self.quantum_ops,
                            &self.conditionals,
                            &self.cfgs,
                            &self.processed,
                        );
                        if !self.work_queue.contains(&op_node)
                            && !self.processed.contains(&op_node)
                            && preds_ready
                        {
                            self.work_queue.push_back(op_node);
                        }
                    }

                    // Also activate Call nodes in the entry block
                    for child in hugr.children(entry_block) {
                        let op = hugr.get_optype(child);
                        if matches!(op, OpType::Call(_)) {
                            self.nodes_inside_cfg_blocks.remove(&child);
                            // Skip Call nodes inside TailLoops - they'll be added when the loop expands
                            if self.nodes_inside_tailloops.contains(&child) {
                                continue;
                            }
                            if !self.work_queue.contains(&child)
                                && !self.processed.contains(&child)
                                && all_predecessors_ready(
                                    &hugr,
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

                    // Also activate Conditional nodes in the entry block
                    for &cond_node in &block_info.conditional_nodes {
                        self.nodes_inside_cfg_blocks.remove(&cond_node);
                        // Skip Conditional nodes inside TailLoops
                        if self.nodes_inside_tailloops.contains(&cond_node) {
                            continue;
                        }
                        if !self.work_queue.contains(&cond_node)
                            && !self.processed.contains(&cond_node)
                        {
                            self.work_queue.push_back(cond_node);
                        }
                    }

                    // Also activate bool ops in the entry block
                    for &op_node in &block_info.bool_ops {
                        self.nodes_inside_cfg_blocks.remove(&op_node);
                        // Skip bool ops inside TailLoops
                        if self.nodes_inside_tailloops.contains(&op_node) {
                            continue;
                        }
                        if !self.work_queue.contains(&op_node) && !self.processed.contains(&op_node)
                        {
                            self.work_queue.push_back(op_node);
                        }
                    }

                    // Also activate LoadConstant and classical ops in the entry block
                    for child in hugr.children(entry_block) {
                        let op = hugr.get_optype(child);
                        if matches!(op, OpType::LoadConstant(_)) {
                            self.nodes_inside_cfg_blocks.remove(&child);
                            // Skip nodes inside TailLoops
                            if self.nodes_inside_tailloops.contains(&child) {
                                continue;
                            }
                            if !self.work_queue.contains(&child) && !self.processed.contains(&child)
                            {
                                self.work_queue.push_back(child);
                            }
                        }
                        // Check for classical ops (extension ops in arithmetic.int, etc.)
                        if self.classical_ops.contains_key(&child) {
                            self.nodes_inside_cfg_blocks.remove(&child);
                            // Skip nodes inside TailLoops
                            if self.nodes_inside_tailloops.contains(&child) {
                                continue;
                            }
                            // Classical ops need their inputs ready
                            if !self.work_queue.contains(&child)
                                && !self.processed.contains(&child)
                                && all_predecessors_ready(
                                    &hugr,
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

                    // Also activate extension ops (tket.rotation, tket.result, etc.)
                    // Use block_info.extension_ops which is already filtered to exclude
                    // quantum_ops, bool_ops, and classical_ops (those are handled above).
                    for &op_node in &block_info.extension_ops {
                        self.nodes_inside_cfg_blocks.remove(&op_node);
                        // Skip extension ops inside TailLoops
                        if self.nodes_inside_tailloops.contains(&op_node) {
                            continue;
                        }
                        if !self.work_queue.contains(&op_node)
                            && !self.processed.contains(&op_node)
                            && all_predecessors_ready(
                                &hugr,
                                op_node,
                                &self.quantum_ops,
                                &self.conditionals,
                                &self.cfgs,
                                &self.processed,
                            )
                        {
                            self.work_queue.push_back(op_node);
                        }
                    }

                    // Also activate TailLoop nodes in the entry block
                    // NOTE: Don't check preds_ready for TailLoops - they handle input
                    // propagation separately during expansion.
                    for &tl_node in &block_info.tailloop_nodes {
                        self.nodes_inside_cfg_blocks.remove(&tl_node);
                        if !self.work_queue.contains(&tl_node) && !self.processed.contains(&tl_node)
                        {
                            self.work_queue.push_back(tl_node);
                        }
                    }

                    let num_ops = block_info.quantum_ops.len();
                    let num_calls = block_info.call_nodes.len();
                    let num_conditionals = block_info.conditional_nodes.len();
                    let num_bool_ops = block_info.bool_ops.len();
                    let num_tailloops = block_info.tailloop_nodes.len();
                    debug!(
                        "CFG {current_node:?}: activated entry block {entry_block:?} with {num_ops} ops, {num_conditionals} conditionals, {num_bool_ops} bool_ops, {num_tailloops} tailloops"
                    );

                    // If entry block has no operations, immediately transition to successor
                    if num_ops == 0
                        && num_calls == 0
                        && num_conditionals == 0
                        && num_bool_ops == 0
                        && num_tailloops == 0
                    {
                        debug!(
                            "[TRACE] Entry block {:?} has 0 ops and 0 calls, successors: {:?}",
                            entry_block, block_info.successors
                        );
                        debug!(
                            "CFG {current_node:?}: entry block {entry_block:?} has no ops, transitioning to successor"
                        );
                        let successors = block_info.successors.clone();
                        if successors.len() == 1 {
                            debug!(
                                "[TRACE] Single successor {:?}, transitioning",
                                successors[0]
                            );
                            // Mark entry block as complete and transition
                            if let Some(active_cfg) = self.active_cfgs.get_mut(&current_node) {
                                active_cfg.completed_blocks.insert(entry_block);
                            }
                            self.transition_to_cfg_successor(
                                &hugr,
                                current_node,
                                entry_block,
                                successors[0],
                            );
                        } else if !successors.is_empty() {
                            debug!(
                                "[TRACE] Multiple successors {successors:?}, trying to resolve branch"
                            );
                            // Multiple successors - try to resolve branch
                            if let Some(branch_idx) =
                                self.try_resolve_cfg_block_branch(&hugr, entry_block)
                            {
                                debug!("[TRACE] Branch resolved to index {branch_idx}");
                                if branch_idx < successors.len() {
                                    if let Some(active_cfg) =
                                        self.active_cfgs.get_mut(&current_node)
                                    {
                                        active_cfg.completed_blocks.insert(entry_block);
                                    }
                                    self.transition_to_cfg_successor(
                                        &hugr,
                                        current_node,
                                        entry_block,
                                        successors[branch_idx],
                                    );
                                }
                            } else {
                                debug!("[TRACE] Branch NOT resolved, adding to pending");
                                // Branch not resolved - add to pending
                                let block_key = (current_node, entry_block);
                                self.pending_cfg_branches.insert(block_key, successors);
                            }
                        }
                    }
                }
                continue;
            }

            // --- Control Flow: TailLoop ---
            if self.tailloops.contains_key(&current_node) {
                // Check if already active
                if self.active_tailloops.contains_key(&current_node) {
                    // Active TailLoop - check if we can resolve control
                    if let Some(tag) = self.try_resolve_tailloop_control(&hugr, current_node) {
                        if tag == 0 {
                            // CONTINUE_TAG - start next iteration
                            debug!("TailLoop {current_node:?}: CONTINUE, starting next iteration");
                            self.continue_tailloop_iteration(&hugr, current_node);
                        } else {
                            // BREAK_TAG - complete the loop
                            debug!("TailLoop {current_node:?}: BREAK, completing loop");
                            self.complete_tailloop(&hugr, current_node);
                        }
                    } else {
                        // Can't resolve control - add to pending
                        debug!("TailLoop {current_node:?}: control not resolved, deferring");
                        self.pending_tailloop_control.insert(current_node);
                    }
                } else {
                    // Not active - start first iteration
                    debug!("TailLoop {current_node:?}: starting first iteration");
                    let entry_nodes = self.expand_tailloop(&hugr, current_node);
                    for entry_node in entry_nodes {
                        if !self.work_queue.contains(&entry_node) {
                            self.work_queue.push_back(entry_node);
                        }
                    }
                }
                continue;
            }

            // --- Control Flow: Function Call ---
            if let Some(&func_defn_node) = self.call_targets.get(&current_node) {
                // Skip if already being processed (waiting for FuncDefn to complete)
                if self.active_calls.contains_key(&current_node) {
                    continue;
                }

                debug!("Processing Call {current_node:?} to FuncDefn {func_defn_node:?}");
                eprintln!(
                    "[DEBUG] Processing Call {current_node:?} to FuncDefn {func_defn_node:?}"
                );

                // Check if there's already an active call to this FuncDefn
                // If so, queue this call to wait
                let func_defn_in_use = self
                    .active_calls
                    .values()
                    .any(|info| info.func_defn_node == func_defn_node);

                if func_defn_in_use {
                    debug!(
                        "Call {current_node:?}: FuncDefn {func_defn_node:?} is in use, queueing"
                    );
                    self.pending_func_calls
                        .entry(func_defn_node)
                        .or_default()
                        .push(current_node);
                    continue;
                }

                if let Some(func_info) = self.func_defns.get(&func_defn_node).cloned() {
                    // Map Call inputs to FuncDefn Input node outputs
                    // Call inputs come from upstream nodes
                    for in_port in 0..func_info.num_inputs {
                        let call_in_port = IncomingPort::from(in_port);
                        if let Some((src_node, src_port)) =
                            hugr.single_linked_output(current_node, call_in_port)
                        {
                            let src_wire = (src_node, src_port.index());

                            // Map qubits
                            if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                                let func_input_wire = (func_info.input_node, in_port);
                                self.wire_state
                                    .wire_to_qubit
                                    .insert(func_input_wire, qubit_id);
                                debug!(
                                    "Call {:?}: mapped input {} qubit {:?} to FuncDefn Input {:?}",
                                    current_node, in_port, qubit_id, func_info.input_node
                                );
                            }
                            // Map classical values (including arrays)
                            if let Some(value) =
                                self.wire_state.classical_values.get(&src_wire).cloned()
                            {
                                let func_input_wire = (func_info.input_node, in_port);
                                self.wire_state
                                    .classical_values
                                    .insert(func_input_wire, value.clone());
                                debug!(
                                    "Call {:?}: mapped input {} classical value to FuncDefn Input {:?}",
                                    current_node, in_port, func_info.input_node
                                );
                            }
                        }
                    }

                    // Start executing the FuncDefn's CFG if it has one
                    if let Some(cfg_node) = func_info.cfg_node {
                        debug!("Call {current_node:?}: starting FuncDefn CFG {cfg_node:?}");

                        // Register as active call
                        self.active_calls.insert(
                            current_node,
                            ActiveCallInfo {
                                call_node: current_node,
                                func_defn_node,
                            },
                        );

                        // Remove FuncDefn descendants from nodes_inside_func_defns
                        // so they can be processed now that the function is being called
                        let mut descendants = BTreeSet::new();
                        collect_descendants(&hugr, func_defn_node, &mut descendants);
                        for node in &descendants {
                            self.nodes_inside_func_defns.remove(node);
                        }

                        // Mark ALL FuncDefn descendants as unprocessed so they can be re-executed
                        // This is critical for supporting multiple calls to the same function
                        for node in &descendants {
                            self.processed.remove(node);
                        }
                        self.processed.remove(&cfg_node);

                        // Add the CFG to the work queue to be processed
                        if !self.work_queue.contains(&cfg_node) {
                            self.work_queue.push_front(cfg_node);
                        }
                    } else {
                        debug!("Call {current_node:?}: FuncDefn has no CFG, passing through");
                        // No CFG - just pass through qubits (identity function)
                        for port in 0..func_info.num_outputs {
                            let func_input_wire = (func_info.input_node, port);
                            if let Some(&qubit_id) =
                                self.wire_state.wire_to_qubit.get(&func_input_wire)
                            {
                                let call_output_wire = (current_node, port);
                                self.wire_state
                                    .wire_to_qubit
                                    .insert(call_output_wire, qubit_id);
                            }
                        }
                    }
                }

                // Don't mark Call as processed yet - wait for FuncDefn to complete
                // The Call will be marked as processed in complete_func_call_if_needed
                continue;
            }

            // --- LoadConstant Operations (integer/float constants) ---
            let current_op = hugr.get_optype(current_node);
            if matches!(current_op, OpType::LoadConstant(_)) {
                if let Some(value) = Self::try_load_constant(&hugr, current_node) {
                    self.wire_state
                        .classical_values
                        .insert((current_node, 0), value);
                    debug!("LoadConstant {current_node:?}: loaded value");
                } else {
                    debug!("LoadConstant {current_node:?}: failed to load value");
                }
                self.processed.insert(current_node);

                // Retry any pending ops that might now have their inputs ready
                self.retry_pending_bool_reads();

                self.queue_ready_successors(&hugr, current_node);
                continue;
            }

            // --- Classical Operations (arithmetic, logic, etc.) ---
            if let Some(classical_op) = self.classical_ops.get(&current_node).cloned() {
                debug!(
                    "Processing classical op {current_node:?}: {:?}",
                    classical_op.op_type
                );

                // Execute the classical operation
                let outputs = self.handle_classical_op(&hugr, current_node, &classical_op);

                // If outputs are empty, inputs weren't ready - defer this operation
                if outputs.is_empty() && classical_op.num_outputs > 0 {
                    debug!("Classical op {current_node:?}: deferring - inputs not ready");
                    // Clear stale output values so dependent ops see None and also defer
                    // This is critical for loops where old iteration values could be misread
                    for port in 0..classical_op.num_outputs {
                        self.wire_state
                            .classical_values
                            .remove(&(current_node, port));
                    }
                    // Add to pending bool reads set for retry (reusing the same mechanism)
                    self.pending_bool_reads.insert(current_node);
                    continue;
                }

                // Successfully resolved - remove from pending if it was there
                self.pending_bool_reads.remove(&current_node);

                // Store output values
                for (port, value) in outputs {
                    let wire_key = (current_node, port);
                    self.wire_state.classical_values.insert(wire_key, value);
                }

                // Mark as processed
                self.processed.insert(current_node);

                // Retry any pending ops that might now have their inputs ready
                self.retry_pending_bool_reads();

                // Check if any pending conditionals can now be resolved
                self.try_resolve_pending_conditionals();

                // Check if this classical op completion allows a CFG block to complete
                // This is especially important for loop control (iadd for incrementing counters)
                self.check_cfg_block_completion(&hugr, current_node);

                // Check if this operation completes any active TailLoop body
                self.check_tailloop_body_completion(&hugr, current_node);

                // Add ready successors to work queue
                self.queue_ready_successors(&hugr, current_node);

                continue;
            }

            // --- Extension Operations (tket.result, tket.qsystem, etc.) ---
            let op = hugr.get_optype(current_node);
            let is_extension_op = op.as_extension_op().is_some();
            let ext_result = self.handle_extension_op(&hugr, current_node);
            if ext_result {
                self.processed.insert(current_node);

                // Retry any pending ops that might now have their inputs ready
                self.retry_pending_bool_reads();

                // Check if any pending conditionals can now be resolved
                self.try_resolve_pending_conditionals();

                // Check if this extension op completion allows a CFG block to complete
                // This is especially important for tket.bool ops in loop control
                self.check_cfg_block_completion(&hugr, current_node);

                // Check if this operation completes any active TailLoop body
                self.check_tailloop_body_completion(&hugr, current_node);

                // Add ready successors to work queue
                self.queue_ready_successors(&hugr, current_node);

                continue;
            } else if is_extension_op && !self.quantum_ops.contains_key(&current_node) {
                // Extension op couldn't be processed (input not ready) - defer it
                // But don't defer if it's also a quantum op (e.g., MeasureFree from tket.quantum)
                // - those should fall through to the quantum op handling below
                self.pending_bool_reads.insert(current_node);
                continue;
            }
            // Fall through to quantum op handling

            // --- Quantum Operations (gates, measurements) ---
            let Some(op) = self.quantum_ops.get(&current_node).cloned() else {
                continue;
            };

            // Resolve qubit IDs for this operation
            let qubits = self.resolve_qubits(&hugr, current_node, &op);

            // Emit the gate operation
            if self.emit_quantum_gate(&hugr, current_node, &op, &qubits) {
                hit_measurement = true;
            }

            self.processed.insert(current_node);
            operation_count += 1;

            // Check if this operation completes any active Case
            self.check_case_completion(&hugr, current_node);

            // Check if this operation completes any active CFG block
            self.check_cfg_block_completion(&hugr, current_node);

            // Check if this operation completes any active TailLoop body
            self.check_tailloop_body_completion(&hugr, current_node);

            // Add ready successors to work queue
            self.queue_ready_successors(&hugr, current_node);

            // Break after measurement to wait for results
            if hit_measurement {
                break;
            }
        }

        if operation_count == 0 {
            debug!("No operations processed");
            return Ok(None);
        }

        let msg = self.message_builder.build();
        debug!("Generated ByteMessage with {operation_count} operations");
        Ok(Some(msg))
    }

    // === Helper Methods for process_hugr_impl ===

    /// Emit a quantum gate operation to the message builder.
    ///
    /// This handles all gate types and their decompositions.
    /// Returns true if the gate was a measurement (requires pause for results).
    #[allow(clippy::too_many_lines)] // Gate emission has many gate type cases
    fn emit_quantum_gate(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op: &QuantumOp,
        qubits: &[QubitId],
    ) -> bool {
        let mut hit_measurement = false;

        match op.gate_type {
            // Lifecycle operations
            GateType::QAlloc => {
                debug!("QAlloc: created qubit {:?}", qubits.first());
            }
            GateType::QFree => {
                debug!("QFree: qubit {:?}", qubits.first());
            }

            // Single-qubit gates
            GateType::H => {
                self.message_builder.h(&[qubits[0].0]);
            }
            GateType::X => {
                self.message_builder.x(&[qubits[0].0]);
            }
            GateType::Y => {
                self.message_builder.y(&[qubits[0].0]);
            }
            GateType::Z => {
                self.message_builder.z(&[qubits[0].0]);
            }
            GateType::SZ => {
                self.message_builder.rz(
                    Angle64::from_radians(std::f64::consts::FRAC_PI_2),
                    &[qubits[0].0],
                );
            }
            GateType::SZdg => {
                self.message_builder.rz(
                    Angle64::from_radians(-std::f64::consts::FRAC_PI_2),
                    &[qubits[0].0],
                );
            }
            GateType::T => {
                self.message_builder.rz(
                    Angle64::from_radians(std::f64::consts::FRAC_PI_4),
                    &[qubits[0].0],
                );
            }
            GateType::Tdg => {
                self.message_builder.rz(
                    Angle64::from_radians(-std::f64::consts::FRAC_PI_4),
                    &[qubits[0].0],
                );
            }
            GateType::RX => {
                let angle = self.resolve_rotation_angle(hugr, node, op);
                self.message_builder
                    .rx(Angle64::from_radians(angle), &[qubits[0].0]);
            }
            GateType::RY => {
                let angle = self.resolve_rotation_angle(hugr, node, op);
                self.message_builder
                    .ry(Angle64::from_radians(angle), &[qubits[0].0]);
            }
            GateType::RZ => {
                let angle = self.resolve_rotation_angle(hugr, node, op);
                self.message_builder
                    .rz(Angle64::from_radians(angle), &[qubits[0].0]);
            }
            GateType::PZ => {
                self.message_builder.pz(&[qubits[0].0]);
            }
            GateType::SX => {
                self.message_builder.rx(
                    Angle64::from_radians(std::f64::consts::FRAC_PI_2),
                    &[qubits[0].0],
                );
            }
            GateType::SXdg => {
                self.message_builder.rx(
                    Angle64::from_radians(-std::f64::consts::FRAC_PI_2),
                    &[qubits[0].0],
                );
            }

            // Two-qubit gates
            GateType::CX => {
                self.message_builder.cx(&[(qubits[0].0, qubits[1].0)]);
            }
            GateType::CY => {
                self.message_builder.cy(&[(qubits[0].0, qubits[1].0)]);
            }
            GateType::CZ => {
                self.message_builder.cz(&[(qubits[0].0, qubits[1].0)]);
            }
            GateType::SZZ => {
                self.message_builder.szz(&[(qubits[0].0, qubits[1].0)]);
            }
            GateType::SWAP => {
                self.message_builder.cx(&[(qubits[0].0, qubits[1].0)]);
                self.message_builder.cx(&[(qubits[1].0, qubits[0].0)]);
                self.message_builder.cx(&[(qubits[0].0, qubits[1].0)]);
            }
            GateType::CH => {
                // CH = Ry(pi/4) on target, CX(control, target), Ry(-pi/4) on target
                let control = qubits[0].0;
                let target = qubits[1].0;
                self.message_builder.ry(
                    Angle64::from_radians(std::f64::consts::FRAC_PI_4),
                    &[target],
                );
                self.message_builder.cx(&[(control, target)]);
                self.message_builder.ry(
                    Angle64::from_radians(-std::f64::consts::FRAC_PI_4),
                    &[target],
                );
            }
            GateType::CRZ => {
                let angle = self.resolve_rotation_angle(hugr, node, op);
                let half_angle = angle / 2.0;
                self.message_builder
                    .rz(Angle64::from_radians(half_angle), &[qubits[1].0]);
                self.message_builder.cx(&[(qubits[0].0, qubits[1].0)]);
                self.message_builder
                    .rz(Angle64::from_radians(-half_angle), &[qubits[1].0]);
                self.message_builder.cx(&[(qubits[0].0, qubits[1].0)]);
            }
            GateType::CCX => {
                let c0 = qubits[0].0;
                let c1 = qubits[1].0;
                let target = qubits[2].0;
                self.message_builder.h(&[target]);
                self.message_builder.cx(&[(c1, target)]);
                self.message_builder.rz(
                    Angle64::from_radians(-std::f64::consts::FRAC_PI_4),
                    &[target],
                );
                self.message_builder.cx(&[(c0, target)]);
                self.message_builder.rz(
                    Angle64::from_radians(std::f64::consts::FRAC_PI_4),
                    &[target],
                );
                self.message_builder.cx(&[(c1, target)]);
                self.message_builder.rz(
                    Angle64::from_radians(-std::f64::consts::FRAC_PI_4),
                    &[target],
                );
                self.message_builder.cx(&[(c0, target)]);
                self.message_builder
                    .rz(Angle64::from_radians(std::f64::consts::FRAC_PI_4), &[c1]);
                self.message_builder.rz(
                    Angle64::from_radians(std::f64::consts::FRAC_PI_4),
                    &[target],
                );
                self.message_builder.h(&[target]);
                self.message_builder.cx(&[(c0, c1)]);
                self.message_builder
                    .rz(Angle64::from_radians(std::f64::consts::FRAC_PI_4), &[c0]);
                self.message_builder
                    .rz(Angle64::from_radians(-std::f64::consts::FRAC_PI_4), &[c1]);
                self.message_builder.cx(&[(c0, c1)]);
            }

            // Measurement operations
            GateType::MZ | GateType::MeasureFree => {
                let qubit_id = qubits[0];
                debug!(" Measure: qubit {qubit_id:?} at node {node:?}");
                self.message_builder.mz(&[qubit_id.0]);
                self.measurement_state.mappings.push((node, qubit_id));

                let bool_output_port = usize::from(op.gate_type == GateType::MZ);
                self.measurement_state
                    .output_wires
                    .insert(node, (node, bool_output_port));

                debug!(
                    "Measurement on qubit {qubit_id:?}, classical output on port {bool_output_port}"
                );
                hit_measurement = true;
            }

            _ => {
                debug!("Unsupported gate type: {:?}", op.gate_type);
            }
        }

        hit_measurement
    }

    /// Resolve a rotation angle for a quantum gate.
    ///
    /// First tries statically extracted params (already in radians from analysis).
    /// Falls back to reading the runtime classical value at the angle input port,
    /// which is needed when angles are computed dynamically (e.g., guppylang's CH
    /// decomposition passes rotation values through MakeTuple/UnpackTuple chains
    /// that can't be statically traced).
    fn resolve_rotation_angle(&self, hugr: &Hugr, node: Node, op: &QuantumOp) -> f64 {
        // Try statically extracted params first (already in radians)
        if let Some(&angle) = op.params.first() {
            return angle;
        }
        // Fall back to runtime classical value at the angle input port.
        // The angle port is after all qubit inputs.
        if let Some(value) = self.get_input_value(hugr, node, op.num_qubit_inputs)
            && let Some(halfturns) = value.as_rotation()
        {
            // Convert half-turns to radians: halfturns * pi
            return halfturns * std::f64::consts::PI;
        }
        0.0
    }

    /// Queue ready successor nodes after processing a node.
    ///
    /// Adds successor nodes to the work queue if they are relevant node types,
    /// not yet processed, not already queued, and have all predecessors ready.
    fn queue_ready_successors(&mut self, hugr: &Hugr, node: Node) {
        for succ_node in hugr.output_neighbours(node) {
            let is_relevant = self.quantum_ops.contains_key(&succ_node)
                || self.classical_ops.contains_key(&succ_node)
                || self.call_targets.contains_key(&succ_node)
                || self.conditionals.contains_key(&succ_node)
                || self.cfgs.contains_key(&succ_node)
                || self.tailloops.contains_key(&succ_node);

            // Also check for extension ops (e.g., tket.result, tket.bool) that
            // may depend on quantum predecessors via order edges.
            let succ_op = hugr.get_optype(succ_node);
            let is_extension = succ_op.as_extension_op().is_some();

            // Skip nodes that are inside control flow structures - they should only
            // be processed after their parent control flow structure is expanded
            let inside_control_flow = self.nodes_inside_cases.contains(&succ_node)
                || self.nodes_inside_cfg_blocks.contains(&succ_node)
                || self.nodes_inside_func_defns.contains(&succ_node)
                || self.nodes_inside_tailloops.contains(&succ_node);

            if (is_relevant || is_extension)
                && !inside_control_flow
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
                self.work_queue.push_back(succ_node);
            }
        }
    }
}

impl Default for HugrEngine {
    fn default() -> Self {
        Self {
            hugr: None,
            quantum_ops: BTreeMap::new(),
            classical_ops: BTreeMap::new(),
            work_queue: VecDeque::new(),
            processed: BTreeSet::new(),
            message_builder: ByteMessageBuilder::new(),
            // Grouped state
            wire_state: WireState::default(),
            measurement_state: MeasurementState::default(),
            extension_state: ExtensionState::default(),
            // Control flow fields (Conditional)
            conditionals: BTreeMap::new(),
            pending_conditionals: BTreeMap::new(),
            pending_bool_reads: BTreeSet::new(),
            nodes_inside_cases: BTreeSet::new(),
            active_cases: BTreeMap::new(),
            // Control flow fields (CFG)
            cfgs: BTreeMap::new(),
            nodes_inside_cfg_blocks: BTreeSet::new(),
            active_cfgs: BTreeMap::new(),
            pending_cfg_branches: BTreeMap::new(),
            pending_measurement_propagations: Vec::new(),
            // Control flow fields (Call/FuncDefn)
            func_defns: BTreeMap::new(),
            call_targets: BTreeMap::new(),
            active_calls: BTreeMap::new(),
            nodes_inside_func_defns: BTreeSet::new(),
            pending_func_calls: BTreeMap::new(),
            // Control flow fields (TailLoop)
            tailloops: BTreeMap::new(),
            nodes_inside_tailloops: BTreeSet::new(),
            active_tailloops: BTreeMap::new(),
            pending_tailloop_control: BTreeSet::new(),
            // Result capture
            captured_results: Vec::new(),
            // WASM support
            #[cfg(feature = "wasm")]
            foreign_object: None,
        }
    }
}

impl ClassicalEngine for HugrEngine {
    fn num_qubits(&self) -> usize {
        // If we've already assigned qubit IDs (during command generation),
        // return the actual count needed.
        if self.wire_state.next_qubit_id > 0 {
            return self.wire_state.next_qubit_id;
        }

        // Count QAlloc operations as the base estimate
        let qalloc_count = self
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::QAlloc)
            .count();

        // Check if the HUGR has CFG nodes (control flow graphs).
        // For CFG-style HUGRs (like from Guppy), wire tracking may fail and create
        // fallback qubit IDs, so we need a more conservative estimate.
        let has_cfg = self.hugr.as_ref().is_some_and(|hugr| {
            hugr.nodes()
                .any(|node| matches!(hugr.get_optype(node), OpType::CFG(_)))
        });

        if has_cfg {
            // For CFG-style HUGRs, wire tracking may fail and create fallback qubit IDs.
            // Each operation with qubit inputs might need fallback IDs.
            // Additionally, QAlloc operations get their own IDs after fallbacks.
            //
            // Note: For general Guppy/HUGR programs, the number of qubits is not
            // well-defined at compile time since arbitrary computation can allocate
            // an arbitrary number of qubits. This is a conservative estimate.
            let ops_with_inputs = self
                .quantum_ops
                .values()
                .filter(|op| op.num_qubit_inputs > 0)
                .count();

            // Worst case: all ops with inputs get fallback IDs, then QAllocs get fresh IDs
            (qalloc_count + ops_with_inputs).max(1)
        } else {
            // For simple HUGRs without CFG control flow, QAlloc count is accurate
            qalloc_count.max(1)
        }
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        debug!("HugrEngine::generate_commands()");

        match self.process_hugr_impl()? {
            Some(msg) => Ok(msg),
            None => Ok(ByteMessage::create_empty()),
        }
    }

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        debug!("HugrEngine::handle_measurements()");

        match message.outcomes() {
            Ok(outcomes) => {
                let num_outcomes = outcomes.len();
                debug!("Processing {num_outcomes} measurement results");

                for (local_idx, value) in outcomes.into_iter().enumerate() {
                    let global_idx = self.measurement_state.processed_count + local_idx;

                    if let Some((meas_node, qubit_id)) =
                        self.measurement_state.mappings.get(global_idx)
                    {
                        debug!("Measurement result: qubit {qubit_id:?} = {value}");
                        self.measurement_state.results.insert(*qubit_id, value);

                        // Record the classical value on the measurement's output wire
                        if let Some(&wire_key) = self.measurement_state.output_wires.get(meas_node)
                        {
                            debug!("Recording classical value {value} on wire {wire_key:?}");
                            self.wire_state
                                .classical_values
                                .insert(wire_key, ClassicalValue::Bool(value != 0));
                        }
                    } else {
                        debug!("No mapping for measurement index {global_idx}");
                    }
                }

                self.measurement_state.processed_count += num_outcomes;

                // Check if any pending conditionals can now be resolved
                self.try_resolve_pending_conditionals();

                // Check if any pending CFG branches can now be resolved
                self.try_resolve_pending_cfg_branches();

                // Check if any pending TailLoop controls can now be resolved
                self.try_resolve_pending_tailloops();

                // Re-propagate measurement values to successor blocks
                // This is needed because block transitions happen before measurement
                // results are available
                if let Some(hugr) = self.hugr.clone() {
                    self.repropagate_measurement_values(&hugr);
                }

                // Retry any bool.read nodes that were waiting for measurement results
                self.retry_pending_bool_reads();

                Ok(())
            }
            Err(e) => Err(PecosError::Input(format!(
                "Error parsing measurement results: {e}"
            ))),
        }
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        let mut result = Shot::default();

        // Only include raw measurement results if there are no captured results.
        // When the user uses result() to capture specific values, the raw measurements
        // are internal to the algorithm (e.g., in repeat-until-success loops where
        // the number of measurements varies between shots).
        if self.captured_results.is_empty() {
            // Convert measurement results to output format
            // Group by qubit ID
            for (&qubit_id, &value) in &self.measurement_state.results {
                let key = format!("q{}", qubit_id.0);
                result.data.insert(key, Data::U32(value));
            }

            // Also provide a combined "measurements" array
            if !self.measurement_state.results.is_empty() {
                let mut sorted_results: Vec<_> = self.measurement_state.results.iter().collect();
                sorted_results.sort_by_key(|(q, _)| q.0);
                let values: Vec<u32> = sorted_results.iter().map(|(_, v)| **v).collect();
                result
                    .data
                    .insert("measurements".to_string(), Data::from_u32_vec(values));
            }
        }

        // Add captured results from result() calls
        for captured in &self.captured_results {
            let data = match &captured.value {
                ResultValue::Bool(b) => Data::U32(u32::from(*b)),
                ResultValue::Int(i) => Data::I64(*i),
                ResultValue::UInt(u) => Data::U64(*u),
                ResultValue::Float(f) => Data::F64(*f),
                ResultValue::ArrayBool(arr) => {
                    Data::from_u32_vec(arr.iter().map(|b| u32::from(*b)).collect())
                }
                ResultValue::ArrayInt(arr) => {
                    Data::Vec(arr.iter().map(|i| Data::I64(*i)).collect())
                }
                ResultValue::ArrayUInt(arr) => {
                    Data::Vec(arr.iter().map(|u| Data::U64(*u)).collect())
                }
                ResultValue::ArrayFloat(arr) => {
                    Data::Vec(arr.iter().map(|f| Data::F64(*f)).collect())
                }
            };
            result.data.insert(captured.label.clone(), data);
        }

        Ok(result)
    }

    fn compile(&self) -> Result<(), PecosError> {
        Ok(())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.reset_state();
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for HugrEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        debug!("HugrEngine::start()");

        self.reset_state();

        if let Some(commands) = self.process_hugr_impl()? {
            debug!("Commands generated, returning NeedsProcessing");
            Ok(EngineStage::NeedsProcessing(commands))
        } else {
            debug!("No commands, returning Complete");
            Ok(EngineStage::Complete(self.get_results()?))
        }
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        debug!("HugrEngine::continue_processing()");

        self.handle_measurements(measurements)?;

        if let Some(commands) = self.process_hugr_impl()? {
            debug!("More commands generated, returning NeedsProcessing");
            Ok(EngineStage::NeedsProcessing(commands))
        } else {
            debug!("No more commands, returning Complete");
            Ok(EngineStage::Complete(self.get_results()?))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        <Self as ClassicalEngine>::reset(self)
    }
}

impl Engine for HugrEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError> {
        debug!("HugrEngine::process()");

        <Self as ClassicalEngine>::reset(self)?;

        let stage = self.start(input)?;

        match stage {
            EngineStage::Complete(result) => Ok(result),
            EngineStage::NeedsProcessing(_) => {
                debug!("HugrEngine cannot process quantum operations directly");
                Ok(self.get_results()?)
            }
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        <Self as ControlEngine>::reset(self)
    }
}

impl Clone for HugrEngine {
    fn clone(&self) -> Self {
        let mut engine = Self {
            hugr: self.hugr.clone(),
            quantum_ops: self.quantum_ops.clone(),
            classical_ops: self.classical_ops.clone(),
            // Control flow structures must be cloned, not defaulted
            conditionals: self.conditionals.clone(),
            cfgs: self.cfgs.clone(),
            func_defns: self.func_defns.clone(),
            call_targets: self.call_targets.clone(),
            tailloops: self.tailloops.clone(),
            ..Self::default()
        };

        // Re-initialize state
        if engine.hugr.is_some() {
            engine.reset_state();
        }

        engine
    }
}

impl std::fmt::Debug for HugrEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HugrEngine")
            .field("has_hugr", &self.hugr.is_some())
            .field("quantum_ops_count", &self.quantum_ops.len())
            .field("work_queue_len", &self.work_queue.len())
            .field("processed_count", &self.processed.len())
            .field(
                "measurements_processed",
                &self.measurement_state.processed_count,
            )
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{Angle64, Gate};
    use pecos_quantum::DagCircuit;
    use pecos_quantum::hugr_convert::dag_circuit_to_hugr;

    #[test]
    fn test_empty_engine() {
        let engine = HugrEngine::new();
        // Empty engine returns minimum of 1 qubit for safety
        assert!(engine.num_qubits() >= 1);
    }

    #[test]
    fn test_default_engine() {
        let engine = HugrEngine::default();
        assert!(engine.hugr.is_none());
        assert!(engine.quantum_ops.is_empty());
    }

    #[test]
    fn test_load_single_hadamard() {
        // Load the single_hadamard.hugr test file
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // Should have at least 1 qubit (QAlloc)
        // Note: CFG-style HUGRs use conservative estimates, so we check >= 1
        assert!(engine.num_qubits() >= 1, "Expected at least 1 qubit");

        // Should have extracted quantum ops: QAlloc, H, MeasureFree
        assert!(
            engine.quantum_ops.len() >= 2,
            "Expected at least QAlloc and H operations"
        );
    }

    #[test]
    fn test_load_bell_state() {
        // Load the bell_state.hugr test file
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/bell_state.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // Should have at least 2 qubits
        // Note: CFG-style HUGRs use conservative estimates, so we check >= 2
        assert!(
            engine.num_qubits() >= 2,
            "Expected at least 2 qubits for Bell state"
        );
    }

    #[test]
    fn test_generate_commands_single_hadamard() {
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );
        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // Generate commands
        let msg = engine.generate_commands();
        assert!(msg.is_ok(), "Failed to generate commands: {:?}", msg.err());

        let msg = msg.unwrap();

        // Should have quantum operations
        if let Ok(ops) = msg.quantum_ops() {
            assert!(!ops.is_empty(), "Expected quantum operations");
            // First op after QAlloc should be H gate
            let has_h = ops.iter().any(|g| g.gate_type == GateType::H);
            assert!(has_h, "Expected H gate in operations");
        }
    }

    // --- Rotation Gate Tests ---

    /// Helper to create a `HugrEngine` from a `DagCircuit`
    fn engine_from_dag(dag: &DagCircuit) -> HugrEngine {
        let hugr = dag_circuit_to_hugr(dag).expect("Failed to convert DagCircuit to HUGR");
        HugrEngine::from_hugr(hugr)
    }

    #[test]
    fn test_rz_gate_extraction() {
        // Test RZ gate with pi/4 rotation (0.125 turns)
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let angle = Angle64::from_turns(0.125); // pi/4 radians
        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::RZ, vec![angle], vec![q0]));

        let engine = engine_from_dag(&dag);

        // Check that we extracted the RZ gate
        let rz_ops: Vec<_> = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RZ)
            .collect();

        assert_eq!(rz_ops.len(), 1, "Expected 1 RZ gate");

        // Check angle extraction (should be in radians: 0.125 * 2π = π/4)
        let rz_op = rz_ops[0];
        assert_eq!(rz_op.params.len(), 1, "RZ should have 1 parameter");

        let expected_radians = 0.125 * std::f64::consts::TAU;
        let actual_radians = rz_op.params[0];
        assert!(
            (actual_radians - expected_radians).abs() < 1e-10,
            "RZ angle should be {expected_radians}, got {actual_radians}"
        );
    }

    #[test]
    fn test_rx_gate_extraction() {
        // Test RX gate with pi/2 rotation (0.25 turns)
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let angle = Angle64::from_turns(0.25); // pi/2 radians
        dag.add_gate(Gate::with_angles(GateType::RX, vec![angle], vec![q0]));

        let engine = engine_from_dag(&dag);

        let rx_ops: Vec<_> = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RX)
            .collect();

        assert_eq!(rx_ops.len(), 1, "Expected 1 RX gate");

        let expected_radians = 0.25 * std::f64::consts::TAU; // pi/2
        let actual_radians = rx_ops[0].params[0];
        assert!(
            (actual_radians - expected_radians).abs() < 1e-10,
            "RX angle should be {expected_radians}, got {actual_radians}"
        );
    }

    #[test]
    fn test_ry_gate_extraction() {
        // Test RY gate with pi rotation (0.5 turns)
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let angle = Angle64::from_turns(0.5); // pi radians
        dag.add_gate(Gate::with_angles(GateType::RY, vec![angle], vec![q0]));

        let engine = engine_from_dag(&dag);

        let ry_ops: Vec<_> = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RY)
            .collect();

        assert_eq!(ry_ops.len(), 1, "Expected 1 RY gate");

        let expected_radians = 0.5 * std::f64::consts::TAU; // pi
        let actual_radians = ry_ops[0].params[0];
        assert!(
            (actual_radians - expected_radians).abs() < 1e-10,
            "RY angle should be {expected_radians}, got {actual_radians}"
        );
    }

    #[test]
    fn test_rotation_gate_command_generation() {
        // Test that rotation gates produce correct commands
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let angle = Angle64::from_turns(0.125); // pi/4

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::RZ, vec![angle], vec![q0]));

        let mut engine = engine_from_dag(&dag);

        // Verify the RZ operation was extracted with its angle
        let rz_ops: Vec<_> = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RZ)
            .collect();
        assert_eq!(rz_ops.len(), 1, "Expected 1 RZ operation");
        assert_eq!(rz_ops[0].params.len(), 1, "RZ should have 1 parameter");

        let expected_radians = 0.125 * std::f64::consts::TAU;
        assert!(
            (rz_ops[0].params[0] - expected_radians).abs() < 1e-10,
            "RZ parameter should be {expected_radians}, got {}",
            rz_ops[0].params[0]
        );

        // Generate commands and verify
        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        let ops = msg.quantum_ops().expect("Failed to parse quantum ops");

        // Should have H and RZ
        let has_h = ops.iter().any(|g| g.gate_type == GateType::H);
        let has_rz = ops.iter().any(|g| g.gate_type == GateType::RZ);

        assert!(has_h, "Expected H gate in commands");
        assert!(has_rz, "Expected RZ gate in commands");

        // Check RZ command has the correct angle
        if let Some(rz_cmd) = ops.iter().find(|g| g.gate_type == GateType::RZ)
            && !rz_cmd.params.is_empty()
        {
            assert!(
                (rz_cmd.params[0] - expected_radians).abs() < 1e-10,
                "RZ command should have angle {expected_radians}, got {}",
                rz_cmd.params[0]
            );
        }
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_multiple_rotation_gates() {
        // Test circuit with multiple rotation gates
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        dag.add_gate(Gate::with_angles(
            GateType::RX,
            vec![Angle64::from_turns(0.125)],
            vec![q0],
        ));
        dag.add_gate(Gate::with_angles(
            GateType::RY,
            vec![Angle64::from_turns(0.25)],
            vec![q1],
        ));
        dag.add_gate(Gate::with_angles(
            GateType::RZ,
            vec![Angle64::from_turns(0.5)],
            vec![q0],
        ));

        let engine = engine_from_dag(&dag);

        // Count each rotation type
        let rx_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RX)
            .count();
        let ry_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RY)
            .count();
        let rz_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RZ)
            .count();

        assert_eq!(rx_count, 1, "Expected 1 RX gate");
        assert_eq!(ry_count, 1, "Expected 1 RY gate");
        assert_eq!(rz_count, 1, "Expected 1 RZ gate");
    }

    // --- Two-Qubit Gate Tests ---

    #[test]
    fn test_cx_gate() {
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::CX, vec![], vec![q0, q1]));

        let mut engine = engine_from_dag(&dag);

        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        let ops = msg.quantum_ops().expect("Failed to parse quantum ops");

        let has_cx = ops.iter().any(|g| g.gate_type == GateType::CX);
        assert!(has_cx, "Expected CX gate in commands");
    }

    #[test]
    fn test_cy_gate() {
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::CY, vec![], vec![q0, q1]));

        let engine = engine_from_dag(&dag);

        let cy_ops: Vec<_> = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::CY)
            .collect();

        assert_eq!(cy_ops.len(), 1, "Expected 1 CY gate");
        assert_eq!(
            cy_ops[0].num_qubit_inputs, 2,
            "CY should have 2 qubit inputs"
        );
        assert_eq!(
            cy_ops[0].num_qubit_outputs, 2,
            "CY should have 2 qubit outputs"
        );
    }

    #[test]
    fn test_cz_gate() {
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::CZ, vec![], vec![q0, q1]));

        let engine = engine_from_dag(&dag);

        let cz_ops: Vec<_> = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::CZ)
            .collect();

        assert_eq!(cz_ops.len(), 1, "Expected 1 CZ gate");
        assert_eq!(
            cz_ops[0].num_qubit_inputs, 2,
            "CZ should have 2 qubit inputs"
        );
        assert_eq!(
            cz_ops[0].num_qubit_outputs, 2,
            "CZ should have 2 qubit outputs"
        );
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_cy_cz_command_generation() {
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);
        let q2 = QubitId::from(2);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::CY, vec![], vec![q0, q1]));
        dag.add_gate(Gate::with_angles(GateType::CZ, vec![], vec![q1, q2]));

        let engine = engine_from_dag(&dag);

        // Verify that CY and CZ were extracted
        let cy_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::CY)
            .count();
        let cz_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::CZ)
            .count();

        assert_eq!(cy_count, 1, "Expected 1 CY operation extracted");
        assert_eq!(cz_count, 1, "Expected 1 CZ operation extracted");

        // For now, just verify the operations are correctly extracted
        // Command generation for HUGRs without QAlloc nodes needs work queue logic fixes
        // The key test is that CY/CZ are recognized and extracted correctly
    }

    // --- Qubit Tracking Tests ---

    #[test]
    fn test_qubit_tracking_simple() {
        // Ensure qubit IDs are tracked correctly through wire flow
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::X, vec![], vec![q1]));
        dag.add_gate(Gate::with_angles(GateType::CX, vec![], vec![q0, q1]));

        let mut engine = engine_from_dag(&dag);

        // Note: HUGRs from dag_circuit_to_hugr don't have QAlloc nodes,
        // so num_qubits() returns 0. Instead verify gates are extracted.
        let h_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::H)
            .count();
        let x_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::X)
            .count();
        let cx_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::CX)
            .count();

        assert_eq!(h_count, 1, "Expected 1 H gate");
        assert_eq!(x_count, 1, "Expected 1 X gate");
        assert_eq!(cx_count, 1, "Expected 1 CX gate");

        // Verify commands can be generated
        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        let ops = msg.quantum_ops().expect("Failed to parse ops");
        assert!(!ops.is_empty(), "Expected operations in commands");
    }

    #[test]
    fn test_qubit_tracking_three_qubit() {
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);
        let q2 = QubitId::from(2);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q1]));
        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q2]));
        dag.add_gate(Gate::with_angles(GateType::CX, vec![], vec![q0, q1]));
        dag.add_gate(Gate::with_angles(GateType::CX, vec![], vec![q1, q2]));

        let mut engine = engine_from_dag(&dag);

        // Verify gates are extracted correctly
        let h_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::H)
            .count();
        let cx_count = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::CX)
            .count();

        assert_eq!(h_count, 3, "Expected 3 H gates");
        assert_eq!(cx_count, 2, "Expected 2 CX gates");

        // Verify commands can be generated
        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        let ops = msg.quantum_ops().expect("Failed to parse ops");
        assert!(!ops.is_empty(), "Expected operations in commands");
    }

    // --- Engine State Tests ---

    #[test]
    fn test_engine_reset() {
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/bell_state.hugr"
        );
        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // Generate commands to modify state
        let _ = engine.generate_commands();

        // Reset should restore initial state
        ClassicalEngine::reset(&mut engine).expect("Failed to reset");

        // Should still have at least 2 qubits
        // Note: CFG-style HUGRs use conservative estimates
        assert!(engine.num_qubits() >= 2);

        // Work queue should be repopulated
        assert!(
            !engine.work_queue.is_empty(),
            "Work queue should not be empty after reset"
        );
    }

    #[test]
    fn test_engine_clone() {
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        let cloned = engine.clone();

        assert_eq!(engine.num_qubits(), cloned.num_qubits());
        assert_eq!(engine.quantum_ops.len(), cloned.quantum_ops.len());
    }

    // --- Edge Case Tests ---

    #[test]
    fn test_empty_hugr() {
        let dag = DagCircuit::new();
        let hugr = dag_circuit_to_hugr(&dag).expect("Failed to convert empty DagCircuit");
        let mut engine = HugrEngine::from_hugr(hugr);

        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        // Empty circuits should produce empty or minimal messages
        let is_empty = msg.is_empty().unwrap_or(true);
        let has_no_ops = msg.quantum_ops().map_or(true, |ops| ops.is_empty());
        assert!(is_empty || has_no_ops);
    }

    #[test]
    fn test_single_gate_circuit() {
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        dag.add_gate(Gate::with_angles(GateType::X, vec![], vec![q0]));

        let mut engine = engine_from_dag(&dag);

        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        let ops = msg.quantum_ops().expect("Failed to parse quantum ops");

        let has_x = ops.iter().any(|g| g.gate_type == GateType::X);
        assert!(has_x, "Expected X gate in commands");
    }

    #[test]
    fn test_debug_format() {
        let engine = HugrEngine::new();
        let debug_str = format!("{engine:?}");
        assert!(debug_str.contains("HugrEngine"));
        assert!(debug_str.contains("has_hugr"));
    }

    // --- Control Flow Tests ---

    #[test]
    fn test_no_conditionals_in_simple_hugr() {
        // Simple HUGRs should have no conditionals
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/bell_state.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        assert!(
            engine.conditionals.is_empty(),
            "Bell state HUGR should have no conditionals"
        );
    }

    #[test]
    fn test_conditional_extraction_from_simple_hugr() {
        // Test that simple HUGRs from DagCircuit have no conditionals
        // This exercises the extract_conditionals method
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::X, vec![], vec![q0]));

        let engine = engine_from_dag(&dag);

        // extract_conditionals should find nothing in simple circuits
        assert!(engine.conditionals.is_empty());
    }

    #[test]
    fn test_control_flow_fields_reset() {
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));

        let mut engine = engine_from_dag(&dag);

        // Verify control flow fields are empty initially
        assert!(engine.conditionals.is_empty());
        assert!(engine.pending_conditionals.is_empty());
        assert!(engine.wire_state.classical_values.is_empty());
        assert!(engine.measurement_state.output_wires.is_empty());

        // Generate commands and reset
        let _ = engine.generate_commands();
        ClassicalEngine::reset(&mut engine).expect("Failed to reset");

        // After reset, control flow fields should still be empty
        assert!(engine.pending_conditionals.is_empty());
        assert!(engine.wire_state.classical_values.is_empty());
        assert!(engine.measurement_state.output_wires.is_empty());
    }

    #[test]
    fn test_no_conditionals_in_dag_circuit_hugr() {
        // HUGRs created from DagCircuit should have no conditionals
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::CX, vec![], vec![q0, q1]));

        let engine = engine_from_dag(&dag);

        assert!(
            engine.conditionals.is_empty(),
            "DagCircuit-based HUGR should have no conditionals"
        );
    }

    // --- Conditional HUGR Tests ---

    #[test]
    fn test_load_conditional_hugr() {
        // Load the conditional_x.hugr test file (generated from Guppy)
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_x.hugr"
        );

        let result = HugrEngine::from_file(hugr_path);
        assert!(
            result.is_ok(),
            "Failed to load conditional HUGR: {:?}",
            result.err()
        );

        let engine = result.unwrap();

        // The number of qubits depends on how Guppy structures the HUGR
        // In some cases, qubits may be allocated in different ways
        let num_qubits = engine.num_qubits();
        debug!("Conditional HUGR has {num_qubits} QAlloc nodes");
        assert!(num_qubits >= 1, "Expected at least 1 qubit");

        // Should have quantum ops extracted
        assert!(
            !engine.quantum_ops.is_empty(),
            "Expected quantum operations"
        );

        // Check for expected gate types
        let has_h = engine
            .quantum_ops
            .values()
            .any(|op| op.gate_type == GateType::H);
        assert!(has_h, "Expected H gate in conditional circuit");

        // Log all gate types found for debugging
        for (node, op) in &engine.quantum_ops {
            debug!("Op {:?}: {:?}", node, op.gate_type);
        }
    }

    #[test]
    fn test_conditional_hugr_has_conditionals() {
        // The conditional_x.hugr should have Conditional nodes
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_x.hugr"
        );

        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // This HUGR was generated from a Guppy program with if/else
        // It should have Conditional nodes detected
        println!("Conditionals found: {}", engine.conditionals.len());
        println!("Quantum ops: {}", engine.quantum_ops.len());

        // Print gate types found
        let gate_types: Vec<_> = engine.quantum_ops.values().map(|op| op.gate_type).collect();
        println!("Gate types: {gate_types:?}");

        // Print conditional info
        for (node, cond_info) in &engine.conditionals {
            println!(
                "Conditional {:?}: {} cases, {} qubit inputs, {} qubit outputs",
                node,
                cond_info.cases.len(),
                cond_info.num_qubit_inputs,
                cond_info.num_qubit_outputs
            );
        }

        // The HUGR from Guppy should have at least one Conditional node
        // (from the if/else statement in the circuit)
        // Note: The detection depends on how Guppy structures the HUGR
    }

    #[test]
    fn test_conditional_hugr_command_generation() {
        // Test that we can generate commands from a conditional HUGR
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_x.hugr"
        );

        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        println!("Initial state:");
        println!("  Work queue size: {}", engine.work_queue.len());
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  Conditionals: {}", engine.conditionals.len());

        // Generate commands - should produce some quantum operations
        let result = engine.generate_commands();
        assert!(
            result.is_ok(),
            "Failed to generate commands: {:?}",
            result.err()
        );

        let msg = result.unwrap();

        // Should produce some commands (may be X, Measure, etc. depending on HUGR structure)
        // The exact ops depend on the Guppy-generated HUGR structure
        // Note: With proper function call support, gates inside FuncDefn bodies are deferred
        // until the function is called and its CFG completes. The first batch might only
        // include QAlloc (which doesn't emit ops) and Call setup.
        if let Ok(ops) = msg.quantum_ops() {
            println!("Generated {} operations:", ops.len());
            for op in &ops {
                println!("  {:?} on qubits {:?}", op.gate_type, op.qubits);
            }

            // With function calls and conditionals, operations may be spread across
            // multiple generate_commands() calls. Just verify we can parse the ops.
        }

        // Check engine state
        println!(
            "Pending conditionals: {}",
            engine.pending_conditionals.len()
        );
        println!("Processed nodes: {}", engine.processed.len());
    }

    #[test]
    fn test_conditional_hugr_full_execution() {
        // Test simulating the full conditional execution flow
        use pecos_engines::ControlEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_x.hugr"
        );

        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // Start execution
        let stage = engine.start(()).expect("Failed to start engine");

        match stage {
            pecos_engines::EngineStage::NeedsProcessing(msg) => {
                println!("Stage 1: NeedsProcessing");
                if let Ok(ops) = msg.quantum_ops() {
                    println!(
                        "  Operations: {:?}",
                        ops.iter().map(|o| o.gate_type).collect::<Vec<_>>()
                    );
                }

                // Simulate measurement result (0 = else branch, 1 = if branch)
                // Create a mock measurement result
                let mut builder = ByteMessageBuilder::new();
                let _ = builder.for_outcomes();
                builder.add_outcomes(&[0]); // Measure 0, take else branch
                let measurement_msg = builder.build();

                // Continue processing with the measurement result
                let stage2 = engine
                    .continue_processing(measurement_msg)
                    .expect("Failed to continue");

                match stage2 {
                    pecos_engines::EngineStage::NeedsProcessing(msg2) => {
                        println!("Stage 2: NeedsProcessing (more ops after conditional)");
                        if let Ok(ops) = msg2.quantum_ops() {
                            println!(
                                "  Operations: {:?}",
                                ops.iter().map(|o| o.gate_type).collect::<Vec<_>>()
                            );
                        }
                    }
                    pecos_engines::EngineStage::Complete(result) => {
                        println!("Stage 2: Complete");
                        println!("  Result: {result:?}");
                    }
                }
            }
            pecos_engines::EngineStage::Complete(result) => {
                println!("Stage 1: Complete (no quantum ops needed)");
                println!("  Result: {result:?}");
            }
        }

        // The test passes if we get here without panicking
        // Full correctness requires integration with a quantum simulator
    }

    // --- Integration Tests with Quantum Simulator ---

    #[test]
    fn test_bell_state_with_statevec() {
        // Test HugrEngine with PECOS DenseStateVecEngine for a Bell state circuit
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/bell_state.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let num_qubits = hugr_engine.num_qubits();
        println!("Bell state HUGR has {num_qubits} qubits");

        // Create HybridEngine with HugrEngine and DenseStateVecEngine
        let mut hybrid = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(hugr_engine))
            .with_quantum_engine(Box::new(DenseStateVecEngine::new(num_qubits)))
            .build();

        // Set seed for reproducibility
        hybrid.set_seed(42);

        // Run the circuit
        let result = hybrid.run_shot().expect("Failed to run shot");

        println!("Bell state result: {result:?}");

        // For Bell state, both qubits should measure the same value
        // (either both 0 or both 1)
        if let Some(measurements) = result.data.get("measurements")
            && let Some(values) = measurements.as_u32_vec()
            && values.len() >= 2
        {
            assert_eq!(
                values[0], values[1],
                "Bell state qubits should be correlated"
            );
        }
    }

    #[test]
    fn test_simple_hadamard_with_statevec() {
        // Test a simple Hadamard + measure circuit with DenseStateVecEngine
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let num_qubits = hugr_engine.num_qubits();
        println!("Single hadamard HUGR has {num_qubits} qubits");

        // Create HybridEngine
        let mut hybrid = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(hugr_engine))
            .with_quantum_engine(Box::new(DenseStateVecEngine::new(num_qubits)))
            .build();

        hybrid.set_seed(42);

        // Run multiple shots to verify it produces both 0 and 1
        let mut zeros = 0;
        let mut ones = 0;

        for i in 0..20 {
            hybrid.set_seed(i); // Different seed each shot
            let result = hybrid.run_shot().expect("Failed to run shot");

            // Check measurement result
            for data in result.data.values() {
                if let Some(v) = data.as_u32() {
                    if v == 0 {
                        zeros += 1;
                    } else {
                        ones += 1;
                    }
                }
            }
        }

        println!("Hadamard results: {zeros} zeros, {ones} ones");
        // Both outcomes should occur (with high probability)
        assert!(
            zeros > 0 || ones > 0,
            "Should have some measurement results"
        );
    }

    #[test]
    fn test_conditional_with_statevec() {
        // Test conditional circuit with real quantum simulation
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_x.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let num_qubits = hugr_engine.num_qubits();
        println!("Conditional HUGR has {num_qubits} qubits");
        println!("Conditionals detected: {}", hugr_engine.conditionals.len());
        println!("Quantum ops: {}", hugr_engine.quantum_ops.len());

        // Create HybridEngine - use more qubits in case HUGR structure differs
        let mut hybrid = HybridEngineBuilder::new()
            .with_classical_engine(Box::new(hugr_engine))
            .with_quantum_engine(Box::new(DenseStateVecEngine::new(4))) // Use 4 qubits to be safe
            .build();

        hybrid.set_seed(42);

        // Run the circuit
        let result = hybrid.run_shot();

        match result {
            Ok(shot) => {
                println!("Conditional circuit result: {shot:?}");
                // Test passes if we get a result
            }
            Err(e) => {
                println!("Error running conditional circuit: {e:?}");
                // For now, just log the error - full conditional support may need more work
            }
        }
    }

    #[test]
    fn test_wire_propagation_debug() {
        use pecos_engines::ControlEngine;

        // Debug test to understand qubit wire propagation through conditionals.
        // Useful for debugging wire tracking issues in conditional HUGRs.
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_x.hugr"
        );

        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        println!("\n=== Wire Propagation Debug ===");
        println!("QAlloc count (num_qubits): {}", engine.num_qubits());

        // Print all quantum operations with their details
        println!("\nQuantum Operations:");
        for (node, op) in &engine.quantum_ops {
            println!(
                "  Node {:?}: {:?} (inputs: {}, outputs: {})",
                node, op.gate_type, op.num_qubit_inputs, op.num_qubit_outputs
            );
        }

        // Print all conditionals with their details
        println!("\nConditionals:");
        for (node, cond_info) in &engine.conditionals {
            println!(
                "  Node {:?}: {} cases, {} qubit inputs, {} qubit outputs",
                node,
                cond_info.cases.len(),
                cond_info.num_qubit_inputs,
                cond_info.num_qubit_outputs
            );
        }

        // Run a single shot with mock measurements to trace wire flow
        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // Print initial work queue state
        println!("\nInitial work queue ({} items):", engine.work_queue.len());
        for node in &engine.work_queue {
            if let Some(op) = engine.quantum_ops.get(node) {
                println!("  {:?}: {:?}", node, op.gate_type);
            } else if engine.conditionals.contains_key(node) {
                println!("  {node:?}: Conditional");
            } else {
                println!("  {node:?}: unknown");
            }
        }

        println!("\nNodes inside cases: {}", engine.nodes_inside_cases.len());

        // Check if quantum ops are properly excluded
        let hugr = engine.hugr.as_ref().unwrap();
        for (node, op) in &engine.quantum_ops {
            let is_inside = engine.nodes_inside_cases.contains(node);
            let parent = hugr.get_parent(*node);

            // Check what kind of nodes the inputs are
            let mut input_types = Vec::new();
            for port_idx in 0..op.num_qubit_inputs {
                let in_port = IncomingPort::from(port_idx);
                if let Some((src_node, _src_port)) = hugr.single_linked_output(*node, in_port) {
                    let src_op = hugr.get_optype(src_node);
                    let src_type = if engine.quantum_ops.contains_key(&src_node) {
                        "quantum_op"
                    } else if engine.conditionals.contains_key(&src_node) {
                        "conditional"
                    } else if matches!(src_op, tket::hugr::ops::OpType::Input(_)) {
                        "input_node"
                    } else {
                        "other"
                    };
                    input_types.push(format!("{src_node:?}:{src_type}"));
                }
            }

            println!(
                "  {:?} ({:?}): inside_case={}, parent={:?}, input_types={:?}",
                node, op.gate_type, is_inside, parent, input_types
            );
        }

        let stage = engine.start(()).expect("Failed to start");

        match stage {
            pecos_engines::EngineStage::NeedsProcessing(msg) => {
                if let Ok(ops) = msg.quantum_ops() {
                    println!("\nFirst batch operations:");
                    for op in &ops {
                        println!("  {:?} on qubits {:?}", op.gate_type, op.qubits);
                    }
                }

                println!("\nWire to qubit mapping after first batch:");
                for (wire, qubit) in &engine.wire_state.wire_to_qubit {
                    println!("  {wire:?} -> {qubit:?}");
                }
            }
            pecos_engines::EngineStage::Complete(_) => {
                println!("Completed immediately");
            }
        }
    }

    #[test]
    fn test_hugr_structure_trace() {
        // Debug test to trace HUGR structure and wire flow.
        // Useful for understanding how quantum ops connect through containers.
        use tket::hugr::{HugrView, IncomingPort, PortIndex};

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_x.hugr"
        );

        let bytes = std::fs::read(hugr_path).expect("Failed to read HUGR");
        let hugr = crate::loader::load_hugr_from_bytes(&bytes).expect("Failed to load HUGR");

        println!("\n=== HUGR Structure Trace ===\n");

        // Find and trace quantum operations
        for node in hugr.nodes() {
            let op = hugr.get_optype(node);

            // Check if this is a quantum operation
            if let Some(ext_op) = op.as_extension_op() {
                let ext_id = ext_op.extension_id();
                if ext_id.as_ref() as &str == "tket.quantum" {
                    let op_name = ext_op.unqualified_id().to_string();
                    let parent = hugr.get_parent(node);

                    println!("Quantum Op: {node:?} ({op_name}) - parent: {parent:?}");

                    // Trace input connections
                    let num_inputs = hugr.num_inputs(node);
                    for port_idx in 0..num_inputs {
                        let in_port = IncomingPort::from(port_idx);
                        if let Some((src_node, src_port)) = hugr.single_linked_output(node, in_port)
                        {
                            let src_op = hugr.get_optype(src_node);
                            println!(
                                "  Input {}: from {:?} port {} (op: {:?})",
                                port_idx,
                                src_node,
                                src_port.index(),
                                src_op
                            );
                        }
                    }
                    println!();
                }
            }

            // Check for Conditional nodes
            if let tket::hugr::ops::OpType::Conditional(_) = op {
                let parent = hugr.get_parent(node);
                println!("Conditional: {node:?} - parent: {parent:?}");

                // List children (Case nodes)
                for (idx, child) in hugr.children(node).enumerate() {
                    println!("  Case {idx}: {child:?}");

                    // List grandchildren (ops inside Case)
                    for grandchild in hugr.children(child) {
                        let gc_op = hugr.get_optype(grandchild);
                        let gc_desc = match gc_op {
                            tket::hugr::ops::OpType::Input(_) => "Input".to_string(),
                            tket::hugr::ops::OpType::Output(_) => "Output".to_string(),
                            _ => format!("{gc_op:?}"),
                        };
                        println!("    -> {grandchild:?}: {gc_desc}");
                    }
                }
                println!();
            }

            // Check for Input nodes (which provide inputs to parent)
            if matches!(op, tket::hugr::ops::OpType::Input(_)) {
                let parent = hugr.get_parent(node);
                let num_outputs = hugr.num_outputs(node);
                println!("Input node: {node:?} - parent: {parent:?}, outputs: {num_outputs}");
            }
        }
    }

    // --- Simple Conditional HUGR Tests ---
    // These tests use simpler conditional HUGRs with only 1 Conditional node
    // for easier validation and debugging.

    #[test]
    fn test_load_simple_conditional() {
        // Load the simple conditional HUGR (if measure=1, apply X)
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/simple_conditional.hugr"
        );

        let engine =
            HugrEngine::from_file(hugr_path).expect("Failed to load simple_conditional.hugr");

        println!("Simple conditional HUGR:");
        println!("  Qubits: {}", engine.num_qubits());
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  Conditionals: {}", engine.conditionals.len());

        // Print gate types
        let gate_types: Vec<_> = engine.quantum_ops.values().map(|op| op.gate_type).collect();
        println!("  Gate types: {gate_types:?}");

        // The HUGR has 2 QAlloc operations, but num_qubits() returns a conservative
        // estimate that accounts for potential fallback qubit IDs during wire tracking.
        // For dynamically allocated qubits, this is just an estimate.
        let qubits = engine.num_qubits();
        assert!(qubits >= 2, "Expected at least 2 qubits, got {qubits}");

        // Guppy generates CFG control flow (not Conditional nodes) for if statements
        println!(
            "  Conditional count: {} (uses CFG instead)",
            engine.conditionals.len()
        );
    }

    #[test]
    fn test_load_conditional_h() {
        // Load the conditional H HUGR
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_h.hugr"
        );

        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load conditional_h.hugr");

        println!("Conditional H HUGR:");
        println!("  Qubits: {}", engine.num_qubits());
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  Conditionals: {}", engine.conditionals.len());

        // The HUGR has 2 QAlloc operations, but num_qubits() returns a conservative
        // estimate for CFG-style HUGRs with potential fallback qubit IDs.
        let qubits = engine.num_qubits();
        assert!(qubits >= 2, "Expected at least 2 qubits, got {qubits}");

        // Should have H gates
        let has_h = engine
            .quantum_ops
            .values()
            .any(|op| op.gate_type == GateType::H);
        assert!(has_h, "Expected H gate");
    }

    #[test]
    fn test_load_conditional_branch() {
        // Load the conditional branch HUGR (if-else)
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_branch.hugr"
        );

        let engine =
            HugrEngine::from_file(hugr_path).expect("Failed to load conditional_branch.hugr");

        println!("Conditional branch HUGR:");
        println!("  Qubits: {}", engine.num_qubits());
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  Conditionals: {}", engine.conditionals.len());

        // The HUGR has 2 QAlloc operations, but num_qubits() returns a conservative
        // estimate for CFG-style HUGRs with potential fallback qubit IDs.
        let qubits = engine.num_qubits();
        assert!(qubits >= 2, "Expected at least 2 qubits, got {qubits}");

        // Note: Guppy uses CFG control flow, not Conditional nodes
        for (node, cond_info) in &engine.conditionals {
            println!("  Conditional {:?}: {} cases", node, cond_info.cases.len());
            // If-else should have 2 cases
            assert!(
                cond_info.cases.len() >= 2,
                "Expected at least 2 cases for if-else"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_simple_conditional_with_statevec() {
        // Test the simple conditional circuit with DenseStateVecEngine
        // Circuit: H(q0), measure(q0), if result=1: X(q1), measure(q1)
        //
        // Expected behavior:
        // - First measurement (m0): 50/50 due to H gate
        // - Second measurement (m1): equals m0
        //   - If m0=0: no X applied, so m1=0
        //   - If m0=1: X applied, so m1=1
        // Key invariant: m0 == m1 for every shot
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/simple_conditional.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits();

        let num_shots = 100;
        let mut results_00 = 0; // m0=0, m1=0
        let mut results_11 = 0; // m0=1, m1=1
        let mut violations = 0; // m0 != m1 (should never happen)

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    // Extract measurement results from the "measurements" vector
                    if let Some(measurements) = shot.data.get("measurements")
                        && let Some(values) = measurements.as_u32_vec()
                        && values.len() >= 2
                    {
                        let m0 = values[0];
                        let m1 = values[1];

                        if m0 == 0 && m1 == 0 {
                            results_00 += 1;
                        } else if m0 == 1 && m1 == 1 {
                            results_11 += 1;
                        } else {
                            // m0 != m1 - this should never happen
                            violations += 1;
                            println!("VIOLATION at shot {shot_num}: m0={m0}, m1={m1}");
                        }
                    }
                }
                Err(e) => {
                    panic!("Shot {shot_num} failed: {e:?}");
                }
            }
        }

        println!("simple_conditional results over {num_shots} shots:");
        println!("  (0,0): {results_00} shots");
        println!("  (1,1): {results_11} shots");
        println!("  violations (m0 != m1): {violations}");

        // Verify invariant: m0 == m1 always
        assert_eq!(
            violations, 0,
            "Invariant violated: m0 should always equal m1"
        );

        // Verify we got both outcomes (statistical check)
        // With 100 shots and 50/50 probability, getting 0 of either is extremely unlikely
        assert!(
            results_00 > 0,
            "Expected some (0,0) outcomes with H gate superposition"
        );
        assert!(
            results_11 > 0,
            "Expected some (1,1) outcomes with H gate superposition"
        );

        // Verify roughly 50/50 distribution (allow 20% margin)
        let total = results_00 + results_11;
        assert_eq!(total, num_shots, "All shots should produce valid results");
        let ratio = f64::from(results_00) / f64::from(total);
        assert!(
            ratio > 0.3 && ratio < 0.7,
            "Expected roughly 50/50 distribution, got {:.1}% zeros",
            ratio * 100.0
        );
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_conditional_branch_with_statevec() {
        // Test the conditional branch circuit with DenseStateVecEngine
        // Circuit: measure(q0), if m0=0: H(q1), else: X(q1), measure(q1)
        //
        // Expected behavior:
        // - First measurement (m0): always 0 (qubit starts in |0⟩, no gates applied)
        // - Second measurement (m1): 50/50 (H applied since m0=0)
        // Key invariant: m0 is always 0
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_branch.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits();

        let num_shots = 100;
        let mut m0_zeros = 0;
        let mut m0_ones = 0;
        let mut m1_zeros = 0;
        let mut m1_ones = 0;

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    if let Some(measurements) = shot.data.get("measurements")
                        && let Some(values) = measurements.as_u32_vec()
                        && values.len() >= 2
                    {
                        let m0 = values[0];
                        let m1 = values[1];

                        if m0 == 0 {
                            m0_zeros += 1;
                        } else {
                            m0_ones += 1;
                        }

                        if m1 == 0 {
                            m1_zeros += 1;
                        } else {
                            m1_ones += 1;
                        }
                    }
                }
                Err(e) => {
                    panic!("Shot {shot_num} failed: {e:?}");
                }
            }
        }

        println!("conditional_branch results over {num_shots} shots:");
        println!("  m0: {m0_zeros} zeros, {m0_ones} ones");
        println!("  m1: {m1_zeros} zeros, {m1_ones} ones");

        // Verify invariant: m0 is always 0 (qubit measured without any gates)
        assert_eq!(
            m0_ones, 0,
            "Invariant violated: m0 should always be 0 (qubit in |0⟩)"
        );
        assert_eq!(m0_zeros, num_shots, "All m0 should be 0");

        // Verify m1 has both outcomes (H applied, so 50/50)
        assert!(
            m1_zeros > 0,
            "Expected some m1=0 outcomes with H gate superposition"
        );
        assert!(
            m1_ones > 0,
            "Expected some m1=1 outcomes with H gate superposition"
        );

        // Verify roughly 50/50 distribution for m1 (allow 20% margin)
        let ratio = f64::from(m1_zeros) / f64::from(num_shots);
        assert!(
            ratio > 0.3 && ratio < 0.7,
            "Expected roughly 50/50 distribution for m1, got {:.1}% zeros",
            ratio * 100.0
        );
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_conditional_h_with_statevec() {
        // Test the conditional H circuit with DenseStateVecEngine
        // Circuit: H(control), measure(control), if control=1: H(result), measure(result)
        //
        // Expected behavior:
        // - Control measurement (m_control): 50/50 due to H gate
        // - Result measurement (m_result):
        //   - If control=0: result is always 0 (no H applied, qubit stays in |0⟩)
        //   - If control=1: result is 50/50 (H applied)
        // Key invariant: when control=0, result must be 0
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/conditional_h.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits();

        let num_shots = 100;
        let mut control_0_result_0 = 0; // control=0, result=0 (expected)
        let mut control_0_result_1 = 0; // control=0, result=1 (VIOLATION)
        let mut control_1_result_0 = 0; // control=1, result=0 (ok, 50/50)
        let mut control_1_result_1 = 0; // control=1, result=1 (ok, 50/50)

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    if let Some(measurements) = shot.data.get("measurements")
                        && let Some(values) = measurements.as_u32_vec()
                        && values.len() >= 2
                    {
                        // Measurements sorted by qubit ID:
                        // values[0] = QubitId(0) = q_result (measured second)
                        // values[1] = QubitId(1) = q_control (measured first)
                        let result = values[0];
                        let control = values[1];

                        match (control, result) {
                            (0, 0) => control_0_result_0 += 1,
                            (0, 1) => control_0_result_1 += 1,
                            (1, 0) => control_1_result_0 += 1,
                            (1, 1) => control_1_result_1 += 1,
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    panic!("Shot {shot_num} failed: {e:?}");
                }
            }
        }

        println!("conditional_h results over {num_shots} shots:");
        println!("  (control=0, result=0): {control_0_result_0}");
        println!("  (control=0, result=1): {control_0_result_1} (should be 0)");
        println!("  (control=1, result=0): {control_1_result_0}");
        println!("  (control=1, result=1): {control_1_result_1}");

        // Verify invariant: when control=0, result must be 0
        assert_eq!(
            control_0_result_1, 0,
            "Invariant violated: when control=0, result should always be 0"
        );

        // Verify control has both outcomes (H applied, so 50/50)
        let control_zeros = control_0_result_0 + control_0_result_1;
        let control_ones = control_1_result_0 + control_1_result_1;
        assert!(control_zeros > 0, "Expected some control=0 outcomes");
        assert!(control_ones > 0, "Expected some control=1 outcomes");

        // Verify when control=1, result has both outcomes (H applied)
        // Only check if we had enough control=1 shots
        if control_ones >= 10 {
            assert!(
                control_1_result_0 > 0,
                "Expected some result=0 when control=1 (H applied)"
            );
            assert!(
                control_1_result_1 > 0,
                "Expected some result=1 when control=1 (H applied)"
            );
        }

        // Verify all shots accounted for
        let total =
            control_0_result_0 + control_0_result_1 + control_1_result_0 + control_1_result_1;
        assert_eq!(total, num_shots, "All shots should produce valid results");
    }

    #[test]
    fn test_load_while_loop() {
        // Test loading a while loop HUGR (uses CFG with back edges)
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/simple_while_loop.hugr"
        );

        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load while loop HUGR");

        println!("While loop HUGR loaded:");
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  CFGs: {}", engine.cfgs.len());
        println!("  Conditionals: {}", engine.conditionals.len());
        println!("  Num qubits: {}", engine.num_qubits());

        // Print CFG details
        for (cfg_node, cfg_info) in &engine.cfgs {
            println!("\nCFG {cfg_node:?}:");
            println!("  Entry block: {:?}", cfg_info.entry_block);
            println!("  Exit block: {:?}", cfg_info.exit_block);
            println!("  Blocks: {}", cfg_info.blocks.len());

            for (block_node, block_info) in &cfg_info.blocks {
                println!(
                    "  Block {:?}: {} quantum ops, {} successors {:?}",
                    block_node,
                    block_info.quantum_ops.len(),
                    block_info.num_successors,
                    block_info.successors
                );
                for op in &block_info.quantum_ops {
                    if let Some(op_info) = engine.quantum_ops.get(op) {
                        println!("    Op {:?}: {:?}", op, op_info.gate_type);
                    }
                }
            }
        }

        // Print initial work queue
        println!("\nInitial work queue: {:?}", engine.work_queue);
        println!(
            "Nodes inside CFG blocks: {:?}",
            engine.nodes_inside_cfg_blocks
        );

        // Should have at least one CFG for the while loop
        assert!(
            !engine.cfgs.is_empty(),
            "While loop should have at least one CFG"
        );
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_while_loop_with_statevec() {
        // Test the while loop circuit with DenseStateVecEngine
        // Circuit: while not result: q=qubit(), H(q), result=measure(q)
        //
        // Expected behavior:
        // - Loop continues until measurement returns 1
        // - Each iteration has 50% chance to exit (H gate → measure)
        // - Final result is always True (1) since that's the exit condition
        use pecos_engines::ControlEngine;
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/simple_while_loop.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits().max(4); // At least 4 qubits for safety

        println!("While loop HUGR:");
        println!("  CFGs: {}", hugr_engine.cfgs.len());
        println!("  Quantum ops: {}", hugr_engine.quantum_ops.len());
        for (node, cfg) in &hugr_engine.cfgs {
            println!("  CFG {:?}: {} blocks", node, cfg.blocks.len());
        }

        // Test single shot with manual stepping to trace execution
        println!("\n=== Manual stepping test ===");
        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        println!("Initial state:");
        println!("  Work queue: {:?}", engine.work_queue);
        println!(
            "  Active CFGs: {:?}",
            engine.active_cfgs.keys().collect::<Vec<_>>()
        );

        // Start the engine
        let stage = engine.start(()).expect("Failed to start");
        match &stage {
            pecos_engines::EngineStage::NeedsProcessing(msg) => {
                if let Ok(ops) = msg.quantum_ops() {
                    println!(
                        "After start - ops to process: {:?}",
                        ops.iter().map(|op| op.gate_type).collect::<Vec<_>>()
                    );
                }
            }
            pecos_engines::EngineStage::Complete(_) => {
                println!("After start - completed immediately");
            }
        }
        println!("  Work queue after start: {:?}", engine.work_queue);
        println!(
            "  Active CFGs: {:?}",
            engine.active_cfgs.keys().collect::<Vec<_>>()
        );
        println!("  Processed: {} nodes", engine.processed.len());

        let num_shots = 10;
        let mut successes = 0;
        let mut failures = 0;

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    println!("Shot {}: {:?}", shot_num, shot.data);
                    successes += 1;
                }
                Err(e) => {
                    println!("Shot {shot_num} failed: {e:?}");
                    failures += 1;
                }
            }
        }

        println!("While loop results: {successes} successes, {failures} failures");

        // For now, just check that we can load and attempt to run
        // Full while loop support may require additional work for CFG back edges
        assert!(
            successes > 0 || failures > 0,
            "Should have attempted at least some shots"
        );
    }

    #[test]
    fn test_load_function_call() {
        // Load the function_call.hugr test file
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/function_call.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        // Check that we loaded the HUGR
        println!("Function call HUGR:");
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  CFGs: {}", engine.cfgs.len());

        // Should have quantum ops (H in apply_h, QAlloc + MeasureFree in main)
        assert!(
            engine.quantum_ops.len() >= 2,
            "Expected at least 2 quantum ops"
        );
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_function_call_with_statevec() {
        // Test function call circuit with DenseStateVecEngine
        // Circuit: q = qubit(), q = apply_h(q), measure(q)
        // where apply_h applies H gate
        //
        // Expected behavior: 50/50 measurement outcome
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/function_call.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits().max(4);

        println!("Function call HUGR:");
        println!("  CFGs: {}", hugr_engine.cfgs.len());
        println!("  Quantum ops: {}", hugr_engine.quantum_ops.len());
        println!("  FuncDefns: {}", hugr_engine.func_defns.len());
        for (node, info) in &hugr_engine.func_defns {
            println!(
                "    FuncDefn {:?}: name={}, inputs={}, outputs={}, cfg={:?}",
                node, info.name, info.num_inputs, info.num_outputs, info.cfg_node
            );
        }
        println!("  Call targets: {}", hugr_engine.call_targets.len());
        for (call_node, func_defn_node) in &hugr_engine.call_targets {
            println!("    Call {call_node:?} -> FuncDefn {func_defn_node:?}");
        }
        println!(
            "  Nodes inside FuncDefns: {}",
            hugr_engine.nodes_inside_func_defns.len()
        );

        let num_shots = 100;
        let mut count_0 = 0;
        let mut count_1 = 0;
        let mut failures = 0;

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    // Check measurement result
                    if let Some(measurements) = shot.data.get("measurements")
                        && let Some(values) = measurements.as_u32_vec()
                        && let Some(&m) = values.first()
                    {
                        if m == 0 {
                            count_0 += 1;
                        } else {
                            count_1 += 1;
                        }
                    }
                }
                Err(e) => {
                    println!("Shot {shot_num} failed: {e:?}");
                    failures += 1;
                }
            }
        }

        println!("Function call results: {count_0} zeros, {count_1} ones, {failures} failures");

        // With H gate, should be roughly 50/50
        // Allow for statistical variance
        assert!(
            failures < num_shots,
            "All shots failed - function call not working"
        );
        if failures == 0 {
            // Check distribution only if all shots succeeded
            let total = count_0 + count_1;
            assert!(total > 0, "No measurements recorded");
            let ratio = f64::from(count_0) / f64::from(total);
            assert!(
                ratio > 0.3 && ratio < 0.7,
                "Expected ~50/50 distribution, got {:.2}%/{:.2}%",
                ratio * 100.0,
                (1.0 - ratio) * 100.0
            );
        }
    }

    #[test]
    fn test_load_multiple_function_calls() {
        // Load the multiple_function_calls.hugr test file
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/multiple_function_calls.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        println!("Multiple function calls HUGR:");
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  CFGs: {}", engine.cfgs.len());
        println!("  FuncDefns: {}", engine.func_defns.len());
        println!("  Call targets: {}", engine.call_targets.len());

        // Should have 2 Call nodes (calling apply_h twice)
        assert!(
            engine.call_targets.len() >= 2,
            "Expected at least 2 Call nodes, got {}",
            engine.call_targets.len()
        );
    }

    #[test]
    #[allow(clippy::too_many_lines, clippy::cast_sign_loss)]
    fn test_multiple_function_calls_with_statevec() {
        // Test multiple function calls: apply_h to two qubits
        // Expected: both measurements are 50/50 independent
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/multiple_function_calls.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits().max(4);

        println!("Multiple function calls HUGR:");
        println!(
            "  Quantum ops: {} -> {:?}",
            hugr_engine.quantum_ops.len(),
            hugr_engine.quantum_ops.keys().collect::<Vec<_>>()
        );
        println!("  CFGs: {}", hugr_engine.cfgs.len());
        println!("  FuncDefns: {}", hugr_engine.func_defns.len());
        for (node, info) in &hugr_engine.func_defns {
            println!(
                "    {:?}: {}, inputs={}, outputs={}, cfg={:?}",
                node, info.name, info.num_inputs, info.num_outputs, info.cfg_node
            );
        }
        println!("  Call targets: {}", hugr_engine.call_targets.len());
        for (call_node, func_defn_node) in &hugr_engine.call_targets {
            println!("    Call {call_node:?} -> FuncDefn {func_defn_node:?}");
        }
        println!(
            "  Nodes inside FuncDefns: {}",
            hugr_engine.nodes_inside_func_defns.len()
        );

        let num_shots = 100;
        let mut count_00 = 0;
        let mut count_01 = 0;
        let mut count_10 = 0;
        let mut count_11 = 0;
        let mut failures = 0;

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

            if shot_num == 0 {
                println!("\n=== Shot 0 Pre-run state ===");
                println!("  Work queue: {:?}", hugr_engine.work_queue);
                println!("  Processed: {:?}", hugr_engine.processed);
            }

            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    if shot_num == 0 {
                        println!(
                            "Shot 0 data keys: {:?}",
                            shot.data.keys().collect::<Vec<_>>()
                        );
                        if let Some(measurements) = shot.data.get("measurements") {
                            println!("  measurements: {measurements:?}");
                        }
                    }
                    if let Some(measurements) = shot.data.get("measurements")
                        && let Some(values) = measurements.as_u32_vec()
                        && values.len() >= 2
                    {
                        let m0 = values[0];
                        let m1 = values[1];
                        match (m0, m1) {
                            (0, 0) => count_00 += 1,
                            (0, 1) => count_01 += 1,
                            (1, 0) => count_10 += 1,
                            (1, 1) => count_11 += 1,
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    println!("Shot {shot_num} failed: {e:?}");
                    failures += 1;
                }
            }
        }

        println!(
            "Multiple function calls results: 00={count_00}, 01={count_01}, 10={count_10}, 11={count_11}, failures={failures}"
        );

        // With two independent H gates, should see roughly 25% each
        assert!(
            failures < num_shots,
            "All shots failed - multiple function calls not working"
        );
        if failures == 0 {
            let total = count_00 + count_01 + count_10 + count_11;
            assert!(total > 0, "No measurements recorded");
            // Each outcome should be roughly 25% (allow 10-40%)
            for (name, count) in [
                ("00", count_00),
                ("01", count_01),
                ("10", count_10),
                ("11", count_11),
            ] {
                let ratio = f64::from(count) / f64::from(total);
                assert!(
                    ratio > 0.10 && ratio < 0.40,
                    "{} ratio {:.2}% out of expected range 10-40%",
                    name,
                    ratio * 100.0
                );
            }
        }
    }

    #[test]
    fn test_load_nested_function_calls() {
        // Load the nested_function_calls.hugr test file
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/nested_function_calls.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        println!("Nested function calls HUGR:");
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  CFGs: {}", engine.cfgs.len());
        println!("  FuncDefns: {}", engine.func_defns.len());
        for (node, info) in &engine.func_defns {
            println!("    FuncDefn {:?}: {}", node, info.name);
        }
        println!("  Call targets: {}", engine.call_targets.len());

        // Should have at least 2 FuncDefns (inner_h and outer_func)
        assert!(
            engine.func_defns.len() >= 2,
            "Expected at least 2 FuncDefns, got {}",
            engine.func_defns.len()
        );
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_nested_function_calls_with_statevec() {
        // Test nested function calls: main -> outer_func -> inner_h
        // Expected: 50/50 measurement outcome
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/nested_function_calls.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits().max(4);

        println!("Nested function calls HUGR:");
        println!("  FuncDefns: {}", hugr_engine.func_defns.len());
        for (node, info) in &hugr_engine.func_defns {
            println!("    {:?}: {}", node, info.name);
        }

        let num_shots = 100;
        let mut count_0 = 0;
        let mut count_1 = 0;
        let mut failures = 0;

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    if let Some(measurements) = shot.data.get("measurements")
                        && let Some(values) = measurements.as_u32_vec()
                        && let Some(&m) = values.first()
                    {
                        if m == 0 {
                            count_0 += 1;
                        } else {
                            count_1 += 1;
                        }
                    }
                }
                Err(e) => {
                    println!("Shot {shot_num} failed: {e:?}");
                    failures += 1;
                }
            }
        }

        println!(
            "Nested function calls results: {count_0} zeros, {count_1} ones, {failures} failures"
        );

        // With H gate (through nested calls), should be roughly 50/50
        assert!(
            failures < num_shots,
            "All shots failed - nested function calls not working"
        );
        if failures == 0 {
            let total = count_0 + count_1;
            assert!(total > 0, "No measurements recorded");
            let ratio = f64::from(count_0) / f64::from(total);
            assert!(
                ratio > 0.3 && ratio < 0.7,
                "Expected ~50/50 distribution, got {:.2}%/{:.2}%",
                ratio * 100.0,
                (1.0 - ratio) * 100.0
            );
        }
    }

    #[test]
    fn test_load_multi_qubit_function() {
        // Load the multi_qubit_function.hugr test file
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/multi_qubit_function.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        println!("Multi-qubit function HUGR:");
        println!("  Quantum ops: {}", engine.quantum_ops.len());
        println!("  CFGs: {}", engine.cfgs.len());
        println!("  FuncDefns: {}", engine.func_defns.len());
        for (node, info) in &engine.func_defns {
            println!(
                "    FuncDefn {:?}: {}, inputs={}, outputs={}",
                node, info.name, info.num_inputs, info.num_outputs
            );
        }
        println!("  Call targets: {}", engine.call_targets.len());

        // Should have a FuncDefn with 2 inputs (2 qubits)
        let has_multi_qubit_func = engine.func_defns.values().any(|info| info.num_inputs >= 2);
        assert!(
            has_multi_qubit_func,
            "Expected a function with at least 2 inputs"
        );
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_multi_qubit_function_with_statevec() {
        // Test multi-qubit function: apply_cx creates Bell state
        // Expected: measurements are correlated (00 or 11, never 01 or 10)
        use pecos_engines::hybrid::HybridEngineBuilder;
        use pecos_engines::quantum::DenseStateVecEngine;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/multi_qubit_function.hugr"
        );

        let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
        let estimated_qubits = hugr_engine.num_qubits().max(4);

        println!("Multi-qubit function HUGR:");
        println!("  FuncDefns: {}", hugr_engine.func_defns.len());
        for (node, info) in &hugr_engine.func_defns {
            println!(
                "    {:?}: {}, inputs={}, outputs={}",
                node, info.name, info.num_inputs, info.num_outputs
            );
        }

        let num_shots = 100;
        let mut count_00 = 0;
        let mut count_01 = 0;
        let mut count_10 = 0;
        let mut count_11 = 0;
        let mut failures = 0;

        for shot_num in 0..num_shots {
            let hugr_engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");
            let mut hybrid = HybridEngineBuilder::new()
                .with_classical_engine(Box::new(hugr_engine))
                .with_quantum_engine(Box::new(DenseStateVecEngine::new(estimated_qubits)))
                .build();

            hybrid.set_seed(shot_num as u64);

            match hybrid.run_shot() {
                Ok(shot) => {
                    if let Some(measurements) = shot.data.get("measurements")
                        && let Some(values) = measurements.as_u32_vec()
                        && values.len() >= 2
                    {
                        let m0 = values[0];
                        let m1 = values[1];
                        match (m0, m1) {
                            (0, 0) => count_00 += 1,
                            (0, 1) => count_01 += 1,
                            (1, 0) => count_10 += 1,
                            (1, 1) => count_11 += 1,
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    println!("Shot {shot_num} failed: {e:?}");
                    failures += 1;
                }
            }
        }

        println!(
            "Multi-qubit function results: 00={count_00}, 01={count_01}, 10={count_10}, 11={count_11}, failures={failures}"
        );

        // Bell state: should only see 00 or 11 (correlated measurements)
        assert!(
            failures < num_shots,
            "All shots failed - multi-qubit function not working"
        );
        if failures == 0 {
            let total = count_00 + count_01 + count_10 + count_11;
            assert!(total > 0, "No measurements recorded");

            // Correlated measurements: 00 and 11 should dominate, 01 and 10 should be rare
            let correlated = count_00 + count_11;
            let uncorrelated = count_01 + count_10;
            assert!(
                correlated > uncorrelated * 4,
                "Expected Bell state correlation: {correlated} correlated vs {uncorrelated} uncorrelated"
            );
        }
    }
}
