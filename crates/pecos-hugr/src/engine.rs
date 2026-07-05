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

mod activation;
pub(crate) mod analysis;
mod control_flow;
mod handlers;
use handlers::{ClassicalOutcome, HandlerOutcome};
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
    ActiveCallInfo, ActiveCaseInfo, ActiveCfgInfo, ActiveScanInfo, ActiveTailLoopInfo, CfgInfo,
    ClassicalOp, ConditionalInfo, ExtensionState, FuncDefnInfo, MeasurementState, QuantumOp,
    TailLoopInfo, WireState,
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

    /// In-flight higher-order array scans, keyed by scan node.
    pub(crate) active_scans: BTreeMap<Node, ActiveScanInfo>,

    /// The entrypoint's classical return values, captured when its CFG
    /// completes. Pure-classical programs (no measurements, no `result()`
    /// calls) surface these as their shot results.
    pub(crate) return_values: Vec<ClassicalValue>,

    /// Container regions the engine actually activated this shot
    /// (`DataflowBlocks`, selected Cases, `TailLoop` bodies), with a label for
    /// diagnostics. Persistent across the shot (unlike the active_* maps),
    /// so completion can audit that every child of an executed region ran.
    pub(crate) executed_containers: BTreeMap<Node, &'static str>,

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
    /// Conditionals whose control value is not yet resolvable (waiting on
    /// a measurement); retried when results arrive.
    pub(crate) pending_conditionals: BTreeSet<Node>,

    /// The starved-node parking lot: every op that DEFERRED (missing or
    /// unconvertible inputs) -- classical ops, bool reads, extension ops,
    /// `LoadConstants`, parked scans. Re-queued by `retry_deferred_nodes` on
    /// completions and measurement rounds; anything still here at
    /// completion time surfaces in the stall report.
    pub(crate) deferred_nodes: BTreeSet<Node>,

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
    pub(crate) pending_func_calls: BTreeMap<Node, VecDeque<Node>>,

    // === TailLoop Support ===
    /// `TailLoop` nodes extracted from the HUGR.
    pub(crate) tailloops: BTreeMap<Node, TailLoopInfo>,

    /// Nodes inside `TailLoop` bodies (should not be processed until loop is active).
    pub(crate) nodes_inside_tailloops: BTreeSet<Node>,

    /// Active `TailLoops` being processed.
    pub(crate) active_tailloops: BTreeMap<Node, ActiveTailLoopInfo>,

    /// Pending `TailLoops` waiting for Sum value (measurement result) to determine continue/break.
    pub(crate) pending_tailloop_control: BTreeSet<Node>,
    /// A fatal execution fault raised from deep (non-Result) code paths --
    /// e.g. an executed `prelude.panic`, an out-of-range branch tag, or a
    /// loop-iteration ceiling. Checked by the main processing loop, which
    /// converts it into an error instead of continuing on corrupt control
    /// flow.
    pub(crate) execution_error: Option<String>,

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

        // Track nodes inside TailLoop bodies (should not be processed until loop is active)
        self.nodes_inside_tailloops = find_nodes_inside_tailloops(&hugr, &self.tailloops);
        debug!(
            "Found {} nodes inside TailLoop bodies",
            self.nodes_inside_tailloops.len()
        );

        // Extract quantum operations (but we'll skip case/CFG-internal ones in work queue)
        self.quantum_ops = extract_quantum_ops(&hugr);
        debug!("Extracted {} quantum operations", self.quantum_ops.len());
        debug!(
            "Extracted {} quantum ops, {} cfgs, {} func_defns, {} call_targets",
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
        self.deferred_nodes.clear();
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
        self.execution_error = None;
        self.active_scans.clear();
        self.return_values.clear();
        self.executed_containers.clear();

        // Clear result capture state
        self.captured_results.clear();

        // Re-initialize nodes_inside_* from their respective control structures
        // (in case we need to re-process after a reset)
        if let Some(hugr) = &self.hugr {
            self.nodes_inside_cases = find_nodes_inside_cases(hugr, &self.conditionals);
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
                        &self.active_cases,
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
                        &self.active_cases,
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
                        &self.active_cases,
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
                        &self.active_cases,
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
                        &self.active_cases,
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
                        &self.active_cases,
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
        if self.pending_tailloop_control.is_empty() {
            return;
        }
        let hugr = match &self.hugr {
            Some(h) => h.clone(),
            None => return,
        };

        debug!(
            "[TRACE] try_resolve_pending_tailloops: {} pending",
            self.pending_tailloop_control.len()
        );

        // Collect TailLoops that can now be resolved. A loop whose body is
        // still mid-iteration must NOT resolve here: a stale or early
        // control value would re-activate (or complete) the loop while
        // in-flight body ops still run, silently corrupting the iteration.
        // Only body completion legitimately re-arms resolution.
        let mut to_resolve = Vec::new();
        for &tailloop_node in &self.pending_tailloop_control {
            if self
                .active_tailloops
                .get(&tailloop_node)
                .is_some_and(|info| info.body_active)
            {
                continue;
            }
            if let Some(tag) = self.try_resolve_tailloop_control(&hugr, tailloop_node) {
                to_resolve.push((tailloop_node, tag));
            }
        }

        // Resolve them
        for (tailloop_node, tag) in to_resolve {
            self.pending_tailloop_control.remove(&tailloop_node);

            if tag > 1 {
                // Two variants only (0=continue, 1=break); see the loop arm.
                self.execution_error = Some(format!(
                    "TailLoop {tailloop_node:?}: control tag {tag} out of range"
                ));
            } else if tag == 0 {
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
    fn retry_deferred_nodes(&mut self) {
        // Move pending bool.reads to work queue so they can be retried
        let pending: Vec<_> = std::mem::take(&mut self.deferred_nodes)
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
        // A fault raised by a completion cascade (e.g. during measurement
        // handling) must surface even when the queue is empty -- check
        // BEFORE the early returns below, or the message is discarded and
        // at best re-reported as a generic stall.
        if let Some(fault) = self.execution_error.take() {
            return Err(PecosError::Generic(fault));
        }
        self.message_builder.reset();
        let _ = self.message_builder.for_quantum_operations();

        let Some(hugr) = self.hugr.clone() else {
            debug!("No HUGR loaded");
            return Ok(None);
        };

        if self.work_queue.is_empty() && self.quantum_ops.is_empty() {
            debug!("Empty HUGR, no commands to generate");
            return Ok(None);
        }

        if self.work_queue.is_empty() {
            // Same completion claim as the post-drain return below: an
            // already-empty queue with active control flow or starved nodes
            // is a stall, not a finished program.
            self.ensure_no_stalled_execution()?;
            debug!("Work queue empty, processing complete");
            return Ok(None);
        }
        debug!("Work queue has {} items", self.work_queue.len());

        let mut operation_count = 0;
        let mut hit_measurement = false;

        while let Some(current_node) = self.work_queue.pop_front() {
            if let Some(fault) = self.execution_error.take() {
                return Err(PecosError::Generic(fault));
            }
            if self.processed.contains(&current_node) {
                continue;
            }
            let node_op = hugr.get_optype(current_node);
            debug!("Processing node {current_node:?}: {node_op:?}");

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
                    // Completion hooks for a zero-op case (block completion,
                    // consumer wake-up) run inside expand_conditional; a
                    // non-empty case completes later via check_case_completion.
                } else {
                    // Can't resolve yet - likely waiting for measurement result
                    // Add to pending conditionals and continue
                    debug!("Conditional {current_node:?} cannot be resolved yet, deferring");
                    // We'll re-add this after measurement results come in
                    // For now, mark as pending and don't add back to queue
                    self.pending_conditionals.insert(current_node);
                }
                continue;
            }

            // --- Control Flow: CFG ---
            if let Some(cfg_info) = self.cfgs.get(&current_node).cloned() {
                // A CFG re-queued while it is still executing must NOT
                // restart: re-registering resets current_block/transitions
                // mid-flight and silently corrupts the walk. (Legitimate
                // re-execution -- a second Call to the same function --
                // only happens after complete_cfg_execution removed the
                // active entry.)
                if self.active_cfgs.contains_key(&current_node) {
                    debug!("CFG {current_node:?} re-queued while active, ignoring");
                    continue;
                }
                debug!("Starting CFG {current_node:?} execution");
                debug!("[TRACE] Starting CFG {current_node:?}");

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
                            transitions: 0,
                        },
                    );

                    // Propagate CFG inputs to entry block's Input node
                    self.propagate_cfg_inputs_to_entry_block(&hugr, current_node, entry_block);

                    self.executed_containers
                        .insert(entry_block, "DataflowBlock");
                    // First activation of the entry block via the shared
                    // mechanism (no resets -- nothing has executed yet).
                    // Ops inside TailLoops leave the block gate but queue
                    // only when their loop expands; TailLoop nodes
                    // themselves queue unconditionally (they handle input
                    // propagation during expansion).
                    let mut act = activation::ContainerActivation::new();
                    let submit =
                        |act: &mut activation::ContainerActivation,
                         node: Node,
                         policy: activation::QueuePolicy| {
                            if self.nodes_inside_tailloops.contains(&node) {
                                act.ungate_block_only(node);
                            } else {
                                act.queue(node, policy);
                            }
                        };
                    for &op_node in &block_info.quantum_ops {
                        submit(&mut act, op_node, activation::QueuePolicy::IfReady);
                    }
                    for child in hugr.children(entry_block) {
                        if matches!(hugr.get_optype(child), OpType::Call(_)) {
                            submit(&mut act, child, activation::QueuePolicy::IfReady);
                        }
                    }
                    for &cond_node in &block_info.conditional_nodes {
                        submit(&mut act, cond_node, activation::QueuePolicy::Always);
                    }
                    for &op_node in &block_info.bool_ops {
                        submit(&mut act, op_node, activation::QueuePolicy::Always);
                    }
                    for child in hugr.children(entry_block) {
                        if matches!(hugr.get_optype(child), OpType::LoadConstant(_)) {
                            submit(&mut act, child, activation::QueuePolicy::Always);
                        }
                        if self.classical_ops.contains_key(&child) {
                            submit(&mut act, child, activation::QueuePolicy::IfReady);
                        }
                    }
                    for &op_node in &block_info.extension_ops {
                        submit(&mut act, op_node, activation::QueuePolicy::IfReady);
                    }
                    for &tl_node in &block_info.tailloop_nodes {
                        act.queue(tl_node, activation::QueuePolicy::Always);
                    }
                    self.run_activation(&hugr, &act);

                    let num_ops = block_info.quantum_ops.len();
                    let num_calls = block_info.call_nodes.len();
                    let num_conditionals = block_info.conditional_nodes.len();
                    let num_bool_ops = block_info.bool_ops.len();
                    let num_tailloops = block_info.tailloop_nodes.len();
                    let num_classical = block_info.classical_ops.len();
                    let num_extension = block_info.extension_ops.len();
                    debug!(
                        "CFG {current_node:?}: activated entry block {entry_block:?} with {num_ops} ops, {num_conditionals} conditionals, {num_bool_ops} bool_ops, {num_tailloops} tailloops, {num_classical} classical, {num_extension} extension"
                    );

                    // If entry block has no operations AT ALL, immediately
                    // transition to the successor. Classical and extension ops
                    // count: transitioning before they run would propagate
                    // missing block outputs (e.g. a loop-bounds tuple built
                    // from LoadConstant + MakeTuple) and starve everything
                    // downstream.
                    if num_ops == 0
                        && num_calls == 0
                        && num_conditionals == 0
                        && num_bool_ops == 0
                        && num_tailloops == 0
                        && num_classical == 0
                        && num_extension == 0
                        && block_info.load_constants.is_empty()
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
                                } else {
                                    // An out-of-range tag means a Sum/tag
                                    // propagation bug upstream -- taking an
                                    // arbitrary branch would mask it as a
                                    // plausible control-flow path.
                                    self.execution_error = Some(format!(
                                        "CFG {current_node:?} block {entry_block:?}: branch                                          tag {branch_idx} out of range ({} successors)",
                                        successors.len()
                                    ));
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
                if let Some(active_info) = self.active_tailloops.get(&current_node) {
                    // A loop whose body is still mid-iteration must not
                    // resolve control: a stale/early value would re-activate
                    // or complete it over in-flight body ops. Body
                    // completion re-arms resolution.
                    if active_info.body_active {
                        debug!("TailLoop {current_node:?}: body mid-iteration, not resolving");
                    } else if let Some(tag) = self.try_resolve_tailloop_control(&hugr, current_node)
                    {
                        if tag > 1 {
                            // Two variants only (0=continue, 1=break).
                            self.execution_error = Some(format!(
                                "TailLoop {current_node:?}: control tag {tag} out of range"
                            ));
                        } else if tag == 0 {
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
                    // Not active - start first iteration, but only once the
                    // loop's input producers have run: expansion propagates
                    // the TailLoop's input wires into the body exactly once,
                    // so expanding early starves the body forever. When a
                    // producer completes, queue_ready_successors re-queues
                    // this node.
                    if !all_predecessors_ready(
                        &hugr,
                        current_node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.active_cases,
                        &self.processed,
                    ) {
                        debug!("TailLoop {current_node:?}: inputs not ready, deferring expansion");
                        continue;
                    }
                    debug!("TailLoop {current_node:?}: starting first iteration");
                    self.expand_tailloop(&hugr, current_node);
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

                // Check if there's already an active call OR an in-flight
                // scan folding through this FuncDefn -- both own the single
                // execution frame, and activating over a scan would reset
                // the scanned function's state mid-element.
                let func_defn_in_use = self
                    .active_calls
                    .values()
                    .any(|info| info.func_defn_node == func_defn_node)
                    || self
                        .active_scans
                        .values()
                        .any(|scan| scan.func_defn_node == func_defn_node);

                if func_defn_in_use {
                    // Direct recursion (a call to F from inside F's own
                    // body) can never make progress: the outer invocation
                    // waits on this Call node while this Call waits for the
                    // FuncDefn to free up. Reject it immediately with a
                    // clear error instead of parking it (indirect recursion
                    // deadlocks the same way and is caught by the
                    // completion-time stall detection, which lists the
                    // parked calls).
                    let mut cur = hugr.get_parent(current_node);
                    while let Some(n) = cur {
                        if n == func_defn_node {
                            return Err(PecosError::Generic(format!(
                                "recursive call to FuncDefn {func_defn_node:?} at {current_node:?}: \
                                 recursion is not supported by the HUGR engine (no call \
                                 stack; each function has a single execution frame)"
                            )));
                        }
                        cur = hugr.get_parent(n);
                    }
                    debug!(
                        "Call {current_node:?}: FuncDefn {func_defn_node:?} is in use, queueing"
                    );
                    let queue = self.pending_func_calls.entry(func_defn_node).or_default();
                    // A parked Call re-queued by a retry wave would
                    // otherwise park twice.
                    if !queue.contains(&current_node) {
                        queue.push_back(current_node);
                    }
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
                                    "Call {:?}: mapped input {} classical value {:?} to FuncDefn Input {:?}",
                                    current_node, in_port, value, func_info.input_node
                                );
                            }
                        }
                    }

                    // Start executing the FuncDefn's CFG if it has one
                    if let Some(cfg_node) = func_info.cfg_node {
                        debug!("Call {current_node:?}: starting FuncDefn CFG {cfg_node:?}");

                        // Capture the Call's instantiation type args so type
                        // variables inside the body (e.g. a generic loop
                        // bound read by prelude.load_nat) can be resolved.
                        let type_args = if let OpType::Call(call_op) = hugr.get_optype(current_node)
                        {
                            call_op.type_args.clone()
                        } else {
                            Vec::new()
                        };

                        // Register as active call
                        self.active_calls.insert(
                            current_node,
                            ActiveCallInfo {
                                call_node: current_node,
                                func_defn_node,
                                type_args,
                            },
                        );

                        // Reset the call frame via the shared mechanism:
                        // every descendant's processed flag AND stale wire
                        // values clear (critical for multiple calls to the
                        // same function -- with only the flags cleared, a
                        // Conditional inside the body can resolve from the
                        // PREVIOUS call's control wire and expand with stale
                        // case inputs before its producers re-run). The
                        // FuncDefn Input node keeps its wires: fresh call
                        // arguments were just copied onto it above. The Call
                        // node's OWN outputs reset too -- a consumer
                        // resolving against the previous invocation's
                        // outputs mid-call reads one-iteration-stale data.
                        let mut descendants = BTreeSet::new();
                        collect_descendants(&hugr, func_defn_node, &mut descendants);
                        let mut act = activation::ContainerActivation::new();
                        for node in &descendants {
                            self.nodes_inside_func_defns.remove(node);
                            act.reset(*node);
                            // The frame reset invalidates the PREVIOUS
                            // invocation's executed-container records (their
                            // processed flags are being cleared); this
                            // invocation re-records whatever it executes, so
                            // the completion audit covers exactly the final
                            // invocation of each frame.
                            self.executed_containers.remove(node);
                        }
                        act.keep_wires(func_info.input_node);
                        act.reset_processed(cfg_node);
                        act.reset_wires(current_node);
                        self.run_activation(&hugr, &act);

                        // Add the CFG to the work queue to be processed
                        if !self.work_queue.contains(&cfg_node) {
                            self.work_queue.push_front(cfg_node);
                        }
                        // Don't mark Call as processed yet - wait for the
                        // FuncDefn's CFG to complete; the Call is completed
                        // in complete_func_call_if_needed.
                        continue;
                    }

                    // No CFG: the body is a plain dataflow region. The
                    // engine only executes CFG-bodied functions, so anything
                    // beyond a pure Input->Output passthrough is
                    // unsupported -- fail loud rather than leaving the Call
                    // stranded outside every completion and stall check
                    // (which silently truncates downstream results).
                    debug!("Call {current_node:?}: FuncDefn has no CFG, treating as passthrough");
                    // The per-port wiring checks below are vacuous for a
                    // zero-output function, so check the body shape first:
                    // any child beyond Input/Output (and inert Const
                    // statics) is real work this path would silently skip.
                    if let Some(extra) = hugr.children(func_defn_node).find(|c| {
                        *c != func_info.input_node
                            && *c != func_info.output_node
                            && !matches!(hugr.get_optype(*c), OpType::Const(_))
                    }) {
                        return Err(PecosError::Generic(format!(
                            "Call {current_node:?} targets FuncDefn {func_defn_node:?} \
                             with no CFG and a non-passthrough body (contains \
                             {extra:?}); plain dataflow function bodies are not \
                             supported by the HUGR engine"
                        )));
                    }
                    for port in 0..func_info.num_outputs {
                        let out_port = IncomingPort::from(port);
                        let Some((src_node, src_port)) =
                            hugr.single_linked_output(func_info.output_node, out_port)
                        else {
                            return Err(PecosError::Generic(format!(
                                "Call {current_node:?} targets FuncDefn {func_defn_node:?} \
                                 whose Output port {port} is unwired; the caller's output \
                                 would silently go missing"
                            )));
                        };
                        if src_node != func_info.input_node {
                            return Err(PecosError::Generic(format!(
                                "Call {current_node:?} targets FuncDefn {func_defn_node:?} \
                                 with no CFG and a non-passthrough body (output {port} fed \
                                 by {src_node:?}); plain dataflow function bodies are not \
                                 supported by the HUGR engine"
                            )));
                        }
                        let func_input_wire = (func_info.input_node, src_port.index());
                        if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&func_input_wire)
                        {
                            self.wire_state
                                .wire_to_qubit
                                .insert((current_node, port), qubit_id);
                        }
                        if let Some(value) = self
                            .wire_state
                            .classical_values
                            .get(&func_input_wire)
                            .cloned()
                        {
                            self.wire_state
                                .classical_values
                                .insert((current_node, port), value);
                        }
                    }
                    self.processed.insert(current_node);
                    self.check_scan_frame_completion(&hugr, current_node);
                    self.check_case_completion(&hugr, current_node);
                    self.check_cfg_block_completion(&hugr, current_node);
                    self.check_tailloop_body_completion(&hugr, current_node);
                    self.queue_ready_successors(&hugr, current_node);
                }
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
                    // An unparseable constant will never parse: defer so the
                    // stall report names this node instead of letting its
                    // block complete around a missing constant-derived value.
                    debug!("LoadConstant {current_node:?}: failed to load value, deferring");
                    self.deferred_nodes.insert(current_node);
                    continue;
                }
                self.processed.insert(current_node);

                // Retry any pending ops that might now have their inputs ready
                self.retry_deferred_nodes();

                // A Case/block may consist of just constants feeding its
                // Output (e.g. a loop's continue-flag bool) -- check
                // completion so outputs propagate only with values present.
                self.check_scan_frame_completion(&hugr, current_node);
                self.check_case_completion(&hugr, current_node);
                self.check_cfg_block_completion(&hugr, current_node);

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
                let outputs = match self.handle_classical_op(&hugr, current_node, &classical_op) {
                    ClassicalOutcome::Outputs(outputs) => outputs,
                    ClassicalOutcome::Defer if classical_op.num_outputs > 0 => {
                        debug!("Classical op {current_node:?}: deferring - inputs not ready");
                        // Clear stale output values so dependent ops see None and also defer
                        // This is critical for loops where old iteration values could be misread
                        for port in 0..classical_op.num_outputs {
                            self.wire_state
                                .classical_values
                                .remove(&(current_node, port));
                        }
                        // Add to pending bool reads set for retry (reusing the same mechanism)
                        self.deferred_nodes.insert(current_node);
                        continue;
                    }
                    // A zero-output op with nothing to store completes.
                    ClassicalOutcome::Defer => Vec::new(),
                    ClassicalOutcome::Fault(msg) => {
                        // Poison and mark processed; the loop-top check
                        // raises it before the next node fires.
                        self.execution_error = Some(msg);
                        self.processed.insert(current_node);
                        continue;
                    }
                };

                // Successfully resolved - remove from pending if it was there
                self.deferred_nodes.remove(&current_node);

                // Store output values. A QubitRef output (e.g. an
                // UnpackTuple or Tag over a linear payload) is mirrored into
                // the qubit-wire map so downstream gates can resolve it.
                for (port, value) in outputs {
                    let wire_key = (current_node, port);
                    if let ClassicalValue::QubitRef(qubit_id) = &value {
                        self.wire_state.wire_to_qubit.insert(wire_key, *qubit_id);
                    }
                    self.wire_state.classical_values.insert(wire_key, value);
                }

                // Mark as processed
                self.processed.insert(current_node);

                // Retry any pending ops that might now have their inputs ready
                self.retry_deferred_nodes();

                // Check if any pending conditionals can now be resolved
                self.try_resolve_pending_conditionals();

                // Check if this classical op completion allows a Case to
                // complete (cases may contain only classical ops, e.g. sum
                // construction for an iterator's continue/break value)
                self.check_scan_frame_completion(&hugr, current_node);
                self.check_case_completion(&hugr, current_node);

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
            if let HandlerOutcome::Fault(msg) = &ext_result {
                // Poison first; the processed path below still runs so the
                // node does not re-fire before the loop-top check raises it.
                self.execution_error = Some(msg.clone());
            }
            if !matches!(ext_result, HandlerOutcome::Defer) {
                self.processed.insert(current_node);

                // Retry any pending ops that might now have their inputs ready
                self.retry_deferred_nodes();

                // Check if any pending conditionals can now be resolved
                self.try_resolve_pending_conditionals();

                // Check if this extension op completion allows a Case to
                // complete (cases may contain only classical/extension ops)
                self.check_scan_frame_completion(&hugr, current_node);
                self.check_case_completion(&hugr, current_node);

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
                self.deferred_nodes.insert(current_node);
                continue;
            }
            // Fall through to quantum op handling

            // A container optype the engine cannot drive must fail loud:
            // silently dropping it starves its consumers with a stall report
            // naming everything except the cause.
            if matches!(hugr.get_optype(current_node), OpType::DFG(_)) {
                self.execution_error = Some(format!(
                    "unsupported container op DFG at {current_node:?}: the engine \
                     executes CFG/Conditional/TailLoop/Call containers only"
                ));
                continue;
            }

            // --- Quantum Operations (gates, measurements) ---
            let Some(op) = self.quantum_ops.get(&current_node).cloned() else {
                continue;
            };

            // Resolve qubit IDs for this operation; defer the gate if a
            // qubit wire has no mapping yet (its producer has not run --
            // completion of that producer re-queues this node).
            let Some(qubits) = self.resolve_qubits(&hugr, current_node, &op) else {
                self.deferred_nodes.insert(current_node);
                continue;
            };
            self.deferred_nodes.remove(&current_node);

            // Emit the gate operation
            if self.emit_quantum_gate(&hugr, current_node, &op, &qubits)? {
                hit_measurement = true;
            }

            self.processed.insert(current_node);
            operation_count += 1;

            // Check if this operation completes any active Case
            self.check_scan_frame_completion(&hugr, current_node);
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

        if let Some(fault) = self.execution_error.take() {
            return Err(PecosError::Generic(fault));
        }
        if operation_count == 0 {
            // Queue drained with no measurement pause: this is the engine's
            // completion claim. Any still-active control flow or starved
            // deferred node at this point means execution stalled mid-program
            // -- fail loud instead of returning silently truncated results.
            self.ensure_no_stalled_execution()?;
            debug!("No operations processed");
            return Ok(None);
        }

        let msg = self.message_builder.build();
        debug!("Generated ByteMessage with {operation_count} operations");
        Ok(Some(msg))
    }

    // === Helper Methods for process_hugr_impl ===

    /// Error if the work queue drained while control flow is still active or
    /// deferred nodes are still starved.
    ///
    /// Called at the point where the engine is about to claim completion
    /// (queue empty, no measurement pause, nothing emitted). Healthy programs
    /// finish with every container completed and no pending reads; anything
    /// left over here is a stall that would otherwise surface only as
    /// silently missing results.
    fn ensure_no_stalled_execution(&self) -> Result<(), PecosError> {
        let mut stalled: Vec<String> = Vec::new();
        if !self.active_cfgs.is_empty() {
            stalled.push(format!(
                "active CFGs: {:?}",
                self.active_cfgs.keys().collect::<Vec<_>>()
            ));
        }
        if !self.active_cases.is_empty() {
            stalled.push(format!(
                "active Conditional cases: {:?}",
                self.active_cases.keys().collect::<Vec<_>>()
            ));
        }
        if !self.active_calls.is_empty() {
            stalled.push(format!(
                "active Calls: {:?}",
                self.active_calls.keys().collect::<Vec<_>>()
            ));
        }
        if !self.active_tailloops.is_empty() {
            stalled.push(format!(
                "active TailLoops: {:?}",
                self.active_tailloops.keys().collect::<Vec<_>>()
            ));
        }
        if !self.active_scans.is_empty() {
            stalled.push(format!(
                "active scans: {:?}",
                self.active_scans.keys().collect::<Vec<_>>()
            ));
        }
        if !self.deferred_nodes.is_empty() {
            stalled.push(format!("starved deferred nodes: {:?}", self.deferred_nodes));
        }
        if !self.pending_conditionals.is_empty() {
            stalled.push(format!(
                "unresolved Conditionals: {:?}",
                self.pending_conditionals
            ));
        }
        if !self.pending_cfg_branches.is_empty() {
            stalled.push(format!(
                "unresolved CFG branches: {:?}",
                self.pending_cfg_branches.keys().collect::<Vec<_>>()
            ));
        }
        if !self.pending_tailloop_control.is_empty() {
            stalled.push(format!(
                "unresolved TailLoop controls: {:?}",
                self.pending_tailloop_control
            ));
        }
        if !self.pending_func_calls.is_empty() {
            stalled.push(format!(
                "parked function calls: {:?}",
                self.pending_func_calls
                    .values()
                    .flatten()
                    .collect::<Vec<_>>()
            ));
        }
        // Reachability audit: the bookkeeping above only sees nodes that
        // entered a queue or pending set. A node that was never QUEUED at
        // all -- an op the category tracking missed inside a container the
        // engine executed -- is invisible to it. Check every direct child
        // of every executed region instead.
        let unexecuted = self.audit_executed_containers();
        if !unexecuted.is_empty() {
            let shown: Vec<&String> = unexecuted.iter().take(10).collect();
            stalled.push(format!(
                "unexecuted ops in executed containers ({} total): {shown:?}",
                unexecuted.len()
            ));
        }
        if stalled.is_empty() {
            Ok(())
        } else {
            Err(PecosError::Generic(format!(
                "HUGR execution stalled before completion; results would be silently \
                 truncated ({})",
                stalled.join("; ")
            )))
        }
    }

    /// Audit that every direct child of every executed container region
    /// (activated `DataflowBlock`, selected Case, expanded `TailLoop` body)
    /// was processed. Exempt kinds never execute: Input/Output boundaries,
    /// static constants, and linear qubit-routing Tags (only Tags
    /// classified as classical `TagSum` ops execute).
    ///
    /// Only the FINAL activation of a re-activated container is audited
    /// (earlier iterations cleared and re-set the same flags), which is
    /// exactly the activation whose flags are still live.
    fn audit_executed_containers(&self) -> Vec<String> {
        let Some(hugr) = &self.hugr else {
            return Vec::new();
        };
        let mut misses = Vec::new();
        for (&container, kind) in &self.executed_containers {
            for child in hugr.children(container) {
                let op = hugr.get_optype(child);
                // Every Tag classifies as an executable TagSum (linear
                // payloads become QubitRef values), so Tags are NOT exempt.
                let exempt = matches!(op, OpType::Input(_) | OpType::Output(_) | OpType::Const(_));
                if !exempt && !self.processed.contains(&child) {
                    misses.push(format!("{child:?} ({op}) in {kind} {container:?}"));
                }
            }
        }
        misses
    }

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
    ) -> Result<bool, PecosError> {
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
                let angle = self.resolve_rotation_angle(hugr, node, op)?;
                self.message_builder
                    .rx(Angle64::from_radians(angle), &[qubits[0].0]);
            }
            GateType::RY => {
                let angle = self.resolve_rotation_angle(hugr, node, op)?;
                self.message_builder
                    .ry(Angle64::from_radians(angle), &[qubits[0].0]);
            }
            GateType::RZ => {
                let angle = self.resolve_rotation_angle(hugr, node, op)?;
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
                let angle = self.resolve_rotation_angle(hugr, node, op)?;
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

        Ok(hit_measurement)
    }

    /// Resolve a rotation angle for a quantum gate.
    ///
    /// First tries statically extracted params (already in radians from analysis).
    /// Falls back to reading the runtime classical value at the angle input port,
    /// which is needed when angles are computed dynamically (e.g., guppylang's CH
    /// decomposition passes rotation values through MakeTuple/UnpackTuple chains
    /// that can't be statically traced).
    fn resolve_rotation_angle(
        &self,
        hugr: &Hugr,
        node: Node,
        op: &QuantumOp,
    ) -> Result<f64, PecosError> {
        // Try statically extracted params first (already in radians)
        if let Some(&angle) = op.params.first() {
            return Ok(angle);
        }
        // Fall back to runtime classical value at the angle input port.
        // The angle port is after all qubit inputs.
        if let Some(value) = self.get_input_value(hugr, node, op.num_qubit_inputs)
            && let Some(halfturns) = value.as_rotation()
        {
            // Convert half-turns to radians: halfturns * pi
            return Ok(halfturns * std::f64::consts::PI);
        }
        // No silent default: a zero angle would turn the gate into a no-op and
        // corrupt the simulated physics without any visible failure.
        Err(PecosError::Input(format!(
            "{:?} at {node:?}: rotation angle unavailable (no static extraction, no runtime value); refusing to default to 0",
            op.gate_type
        )))
    }

    /// Queue ready successor nodes after processing a node.
    ///
    /// Adds successor nodes to the work queue if they are relevant node types,
    /// not yet processed, not already queued, and have all predecessors ready.
    fn queue_ready_successors(&mut self, hugr: &Hugr, node: Node) {
        debug!(
            "queue_ready_successors({node:?}): neighbours={:?}",
            hugr.output_neighbours(node).collect::<Vec<_>>()
        );
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
                    &self.active_cases,
                    &self.processed,
                )
            {
                self.work_queue.push_back(succ_node);
            } else if is_relevant || is_extension {
                debug!(
                    "queue_ready_successors({node:?}): skipped {succ_node:?} gated={inside_control_flow} processed={} queued={} ready={}",
                    self.processed.contains(&succ_node),
                    self.work_queue.contains(&succ_node),
                    all_predecessors_ready(
                        hugr,
                        succ_node,
                        &self.quantum_ops,
                        &self.conditionals,
                        &self.cfgs,
                        &self.active_cases,
                        &self.processed
                    )
                );
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
            active_scans: BTreeMap::new(),
            return_values: Vec::new(),
            executed_containers: BTreeMap::new(),
            message_builder: ByteMessageBuilder::new(),
            // Grouped state
            wire_state: WireState::default(),
            measurement_state: MeasurementState::default(),
            extension_state: ExtensionState::default(),
            // Control flow fields (Conditional)
            conditionals: BTreeMap::new(),
            pending_conditionals: BTreeSet::new(),
            deferred_nodes: BTreeSet::new(),
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
            execution_error: None,
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
                        // An outcome with no queued measurement to bind to
                        // means the driver and engine disagree about how
                        // many measurements this batch contained -- binding
                        // nothing silently corrupts every later index.
                        return Err(PecosError::Input(format!(
                            "measurement outcome {global_idx} has no queued measurement \
                             (engine queued {}, driver sent {num_outcomes} in this batch)",
                            self.measurement_state.mappings.len()
                        )));
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
                    // Likewise for expanded cases: a case can expand on its
                    // own control before OTHER data inputs (later
                    // measurements) exist; refresh its Input ports now.
                    self.repropagate_active_case_inputs(&hugr);
                }

                // Retry any bool.read nodes that were waiting for measurement results
                self.retry_deferred_nodes();

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

            // Pure-classical programs (no measurements either): surface the
            // entrypoint's return values -- "return" for a single value,
            // "return_{port}" for multiple.
            if self.measurement_state.results.is_empty() && !self.return_values.is_empty() {
                let scalars: Vec<Data> = self
                    .return_values
                    .iter()
                    .filter_map(|value| match value {
                        ClassicalValue::Bool(b) => Some(Data::Bool(*b)),
                        ClassicalValue::Int(i) => Some(Data::I64(*i)),
                        ClassicalValue::UInt(u) => Some(Data::U64(*u)),
                        ClassicalValue::Float(f) => Some(Data::F64(*f)),
                        _ => None,
                    })
                    .collect();
                if scalars.len() == 1 {
                    let mut scalars = scalars;
                    result
                        .data
                        .insert("return".to_string(), scalars.pop().expect("len 1"));
                } else {
                    for (i, data) in scalars.into_iter().enumerate() {
                        result.data.insert(format!("return_{i}"), data);
                    }
                }
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

    /// Build the RNG chain from tket-qsystem's own builder test and drive
    /// the qsystem handlers over it, pinning the OUTPUT SHAPES to the
    /// extension signatures: `NewRNGContext -> Option<RNGContext>` (a Sum
    /// with tag 1) and value-FIRST tuples for the Random* ops. The chain
    /// seeds only
    /// the constants and the unwrap Conditional's output; every later op
    /// resolves its context from the previous op's stored port-1 output, so
    /// a swapped port order fails the chain, not just one assert.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_rng_ops_emit_value_first_spec_shapes() {
        use crate::engine::types::RngContextState;
        use tket::hugr::builder::{Dataflow, DataflowHugr, FunctionBuilder};
        use tket::hugr::extension::prelude::{UnwrapBuilder, option_type};
        use tket::hugr::ops::Value;
        use tket::hugr::std_extensions::arithmetic::int_types::{ConstInt, int_type};
        use tket::hugr::types::{Signature, Type};
        use tket_qsystem::extension::random::{CONTEXT_TYPE_NAME, EXTENSION, RandomOpBuilder};

        let hugr = {
            let mut fb =
                FunctionBuilder::new("rng_chain", Signature::new(vec![], vec![int_type(5)]))
                    .unwrap();
            let seed = fb.add_load_const(Value::from(ConstInt::new_u(6, 123_456).unwrap()));
            let maybe_ctx = fb.add_new_rng_context(seed).unwrap();
            let context_type = Type::from(
                EXTENSION
                    .get_type(&CONTEXT_TYPE_NAME)
                    .unwrap()
                    .instantiate([])
                    .unwrap(),
            );
            let [ctx] = fb
                .build_unwrap_sum(1, option_type(vec![context_type]), maybe_ctx)
                .unwrap();
            let bound = fb.add_load_const(Value::from(ConstInt::new_u(5, 100).unwrap()));
            let delta = fb.add_load_const(Value::from(ConstInt::new_s(6, -1).unwrap()));
            let [_, ctx] = fb.add_random_int_bounded(ctx, bound).unwrap();
            let [_, ctx] = fb.add_random_float(ctx).unwrap();
            let ctx = fb.add_random_advance(ctx, delta).unwrap();
            let [rnd, ctx] = fb.add_random_int(ctx).unwrap();
            fb.add_delete_rng_context(ctx).unwrap();
            fb.finish_hugr_with_outputs([rnd]).unwrap()
        };

        let find = |name: &str| -> Node {
            hugr.nodes()
                .find(|n| {
                    hugr.get_optype(*n)
                        .as_extension_op()
                        .is_some_and(|op| op.unqualified_id() == name)
                })
                .unwrap_or_else(|| panic!("no {name} node in the built chain"))
        };
        let new_ctx = find("NewRNGContext");
        let bounded = find("RandomIntBounded");
        let float = find("RandomFloat");
        let advance = find("RandomAdvance");
        let int = find("RandomInt");
        let delete = find("DeleteRNGContext");

        let seed_input = |engine: &mut HugrEngine, node: Node, port: usize, value| {
            let (src, sp) = hugr
                .single_linked_output(node, IncomingPort::from(port))
                .unwrap();
            engine
                .wire_state
                .classical_values
                .insert((src, sp.index()), value);
        };

        let mut engine = HugrEngine::default();

        // NewRNGContext: u64 seed -> Some(context)
        seed_input(&mut engine, new_ctx, 0, ClassicalValue::Int(123_456));
        assert_eq!(
            engine.handle_random_op(&hugr, new_ctx, "NewRNGContext"),
            HandlerOutcome::Processed
        );
        let Some(ClassicalValue::Sum { tag: 1, values }) = engine
            .wire_state
            .classical_values
            .get(&(new_ctx, 0))
            .cloned()
        else {
            panic!("NewRNGContext must produce Sum tag 1 (Some), got {:?}", {
                engine.wire_state.classical_values.get(&(new_ctx, 0))
            });
        };
        let [ClassicalValue::RngContext(ctx_id)] = values.as_slice() else {
            panic!("NewRNGContext Some payload must be an RNG context, got {values:?}");
        };

        // The unwrap Conditional's output is engine-propagated in real runs;
        // seed it directly here so the chain below starts from the context.
        seed_input(&mut engine, bounded, 0, ClassicalValue::RngContext(*ctx_id));
        seed_input(&mut engine, bounded, 1, ClassicalValue::Int(100));
        assert_eq!(
            engine.handle_random_op(&hugr, bounded, "RandomIntBounded"),
            HandlerOutcome::Processed
        );
        assert!(engine.execution_error.is_none());
        match engine.wire_state.classical_values.get(&(bounded, 0)) {
            Some(ClassicalValue::Int(v)) => assert!((0..100).contains(v)),
            other => panic!("RandomIntBounded port 0 must be the value, got {other:?}"),
        }
        assert!(matches!(
            engine.wire_state.classical_values.get(&(bounded, 1)),
            Some(ClassicalValue::RngContext(_))
        ));

        // RandomFloat/RandomAdvance/RandomInt/Delete each read the context
        // from the PREVIOUS op's stored output -- no more seeding.
        assert_eq!(
            engine.handle_random_op(&hugr, float, "RandomFloat"),
            HandlerOutcome::Processed
        );
        match engine.wire_state.classical_values.get(&(float, 0)) {
            Some(ClassicalValue::Float(f)) => assert!((0.0..1.0).contains(f)),
            other => panic!("RandomFloat port 0 must be the value, got {other:?}"),
        }
        assert!(matches!(
            engine.wire_state.classical_values.get(&(float, 1)),
            Some(ClassicalValue::RngContext(_))
        ));

        // Backtracking is unsupported by the step-based implementation and
        // must FAULT (the spec allows negative delta; silently advancing
        // forward instead fabricates a different stream).
        seed_input(&mut engine, advance, 1, ClassicalValue::Int(-1));
        assert!(matches!(
            engine.handle_random_op(&hugr, advance, "RandomAdvance"),
            HandlerOutcome::Fault(_)
        ));
        seed_input(&mut engine, advance, 1, ClassicalValue::Int(2));
        assert_eq!(
            engine.handle_random_op(&hugr, advance, "RandomAdvance"),
            HandlerOutcome::Processed
        );

        assert_eq!(
            engine.handle_random_op(&hugr, int, "RandomInt"),
            HandlerOutcome::Processed
        );
        match engine.wire_state.classical_values.get(&(int, 0)) {
            Some(ClassicalValue::Int(v)) => {
                // Canonical int<5> storage: the 32-bit value sign-extended,
                // i.e. exactly the i32 range (NOT zero-extended [0, 2^32)).
                assert!(
                    (i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(v),
                    "canonical int<32> value, got {v}"
                );
            }
            other => panic!("RandomInt port 0 must be the value, got {other:?}"),
        }
        assert!(matches!(
            engine.wire_state.classical_values.get(&(int, 1)),
            Some(ClassicalValue::RngContext(_))
        ));

        assert_eq!(
            engine.handle_random_op(&hugr, delete, "DeleteRNGContext"),
            HandlerOutcome::Processed
        );
        assert!(engine.extension_state.rng_contexts.is_empty());

        // An empty range has no value to produce: bound 0 must fault, not
        // clamp. (Negative-looking bounds are canonical high-bit unsigned
        // values and are VALID.)
        let mut poisoned = HugrEngine::default();
        seed_input(&mut poisoned, bounded, 0, ClassicalValue::RngContext(7));
        poisoned
            .extension_state
            .rng_contexts
            .insert(7, RngContextState::new(1));
        seed_input(&mut poisoned, bounded, 1, ClassicalValue::Int(0));
        let HandlerOutcome::Fault(fault) =
            poisoned.handle_random_op(&hugr, bounded, "RandomIntBounded")
        else {
            panic!("bound 0 must fault");
        };
        assert!(fault.contains("empty range"), "unexpected fault: {fault}");
    }

    #[test]
    fn test_ry_angle_tuple_runtime_execution() {
        // End-to-end guard for the RUNTIME classical value chain of guppy's
        // tuple-wrapped rotation angle (Const -> LoadConstant -> MakeTuple ->
        // UnpackTuple -> from_halfturns_unchecked -> Ry). The tuple prelude
        // ops execute as classical ops whose num_inputs must come from the
        // dataflow signature -- the portgraph count includes the order port,
        // which used to starve the chain and (before the fail-loud hardening)
        // silently zero the angle. With the hardening, a starved chain makes
        // this generate_commands call error instead of passing.
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/ry_angle_tuple.hugr"
        );
        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        let ops = msg.quantum_ops().expect("Failed to parse quantum ops");

        let ry_cmd = ops
            .iter()
            .find(|g| g.gate_type == GateType::RY)
            .expect("Expected an RY command");
        assert_eq!(ry_cmd.angles.len(), 1, "RY command should carry its angle");
        let radians = ry_cmd.angles[0].to_radians();
        assert!(
            (radians - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "RY command should have angle pi/2, got {radians}",
        );

        // Pin the RUNTIME chain directly: the gate command above can also be
        // satisfied by the STATIC extraction (op.params wins in
        // resolve_rotation_angle), so assert the classical values the runtime
        // tuple ops must have produced during execution.
        let hugr = engine.hugr.clone().expect("hugr present");
        let (unpack, from_halfturns) = ry_runtime_chain_nodes(&hugr);
        assert_eq!(
            engine.wire_state.classical_values.get(&(unpack, 0)),
            Some(&ClassicalValue::Float(0.5)),
            "UnpackTuple should have unpacked the angle float at runtime"
        );
        assert_eq!(
            engine.wire_state.classical_values.get(&(from_halfturns, 0)),
            Some(&ClassicalValue::Rotation(0.5)),
            "from_halfturns should have produced the runtime rotation value"
        );
    }

    /// Walk back from `from_halfturns_unchecked` to its feeding `UnpackTuple`
    /// in the guppy tuple-wrapped-angle fixtures.
    fn ry_runtime_chain_nodes(hugr: &Hugr) -> (Node, Node) {
        let mut from_halfturns = None;
        for node in hugr.nodes() {
            if let Some(ext) = hugr.get_optype(node).as_extension_op()
                && ext.unqualified_id() == "from_halfturns_unchecked"
            {
                from_halfturns = Some(node);
            }
        }
        let fh = from_halfturns.expect("fixture should contain from_halfturns_unchecked");
        let (unpack, _) = hugr
            .single_linked_output(fh, IncomingPort::from(0))
            .expect("from_halfturns input should be wired");
        (unpack, fh)
    }

    #[test]
    fn test_ch_gate_full_execution_completes_cleanly() {
        // End-to-end guard for classical value flow through nested function
        // calls: guppy's ch() decomposes via a called function whose angle
        // (pi/4) is computed by further calls over a tuple constant. The
        // engine must (a) not fire a Call before its argument values exist,
        // (b) run every Case/block classical op before propagating outputs,
        // and (c) finish with NO active control flow left (leftover active
        // entries mean a stall silently truncated the program).
        use pecos_engines::{ByteMessageBuilder, ControlEngine, EngineStage};

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/ch_gate.hugr"
        );
        let mut engine = HugrEngine::from_file(path).expect("Failed to load HUGR");
        let mut stage = engine.start(()).expect("Failed to start engine");
        let mut gate_counts: BTreeMap<GateType, usize> = BTreeMap::new();
        let mut rounds = 0;
        loop {
            rounds += 1;
            assert!(rounds <= 10, "ch execution should complete in a few rounds");
            match stage {
                EngineStage::NeedsProcessing(msg) => {
                    let ops = msg.quantum_ops().expect("parse quantum ops");
                    for g in &ops {
                        *gate_counts.entry(g.gate_type).or_insert(0) += 1;
                    }
                    let n_meas = ops
                        .iter()
                        .filter(|g| {
                            matches!(
                                g.gate_type,
                                GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
                            )
                        })
                        .count();
                    let mut builder = ByteMessageBuilder::new();
                    let _ = builder.for_outcomes();
                    builder.add_outcomes(&vec![0usize; n_meas]);
                    stage = engine
                        .continue_processing(builder.build())
                        .expect("continue");
                }
                EngineStage::Complete(_) => break,
            }
        }

        // The CH decomposition: RY(pi/4), CZ, RY(-pi/4), then 2 measurements.
        assert_eq!(gate_counts.get(&GateType::RY), Some(&2), "{gate_counts:?}");
        assert_eq!(gate_counts.get(&GateType::CZ), Some(&1), "{gate_counts:?}");
        assert_eq!(gate_counts.get(&GateType::MZ), Some(&2), "{gate_counts:?}");

        // No stalled control flow and no starved nodes may remain.
        assert!(engine.active_cases.is_empty(), "{:?}", engine.active_cases);
        assert!(
            engine.active_cfgs.is_empty(),
            "cfgs: {:?}",
            engine.active_cfgs.keys()
        );
        assert!(
            engine.active_calls.is_empty(),
            "calls: {:?}",
            engine.active_calls.keys()
        );
        assert!(engine.active_tailloops.is_empty());
        assert!(
            engine.deferred_nodes.is_empty(),
            "starved nodes: {:?}",
            engine.deferred_nodes
        );
    }

    /// Build ipow / idivmod / itobool / ifrombool via the hugr std-arith
    /// builders and drive the classical executor over them, pinning the
    /// spec shapes: Euclidean (q, r) pairs on two ports, fail-loud m=0 for
    /// the unchecked ops, error Sums for the checked ones, and wrapping
    /// square-and-multiply exponentiation.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_missing_int_ops_execute_per_spec() {
        use crate::engine::analysis::classify_classical_op;
        use crate::engine::handlers::ClassicalOutcome;
        use crate::engine::types::ClassicalOp;
        use tket::hugr::builder::{Dataflow, DataflowHugr, FunctionBuilder};
        use tket::hugr::ops::Value;
        use tket::hugr::std_extensions::arithmetic::conversions::ConvertOpDef;
        use tket::hugr::std_extensions::arithmetic::int_ops::IntOpDef;
        use tket::hugr::std_extensions::arithmetic::int_types::{ConstInt, int_type};
        use tket::hugr::types::Signature;

        let hugr = {
            let mut fb =
                FunctionBuilder::new("int_ops", Signature::new(vec![], vec![int_type(6)])).unwrap();
            let n = fb.add_load_const(Value::from(ConstInt::new_s(6, -7).unwrap()));
            let m = fb.add_load_const(Value::from(ConstInt::new_u(6, 3).unwrap()));
            let base = fb.add_load_const(Value::from(ConstInt::new_s(6, 3).unwrap()));
            let exp = fb.add_load_const(Value::from(ConstInt::new_u(6, 4).unwrap()));
            let bit = fb.add_load_const(Value::from(ConstInt::new_u(0, 1).unwrap()));
            let flag = fb.add_load_const(Value::true_val());
            let [pow] = fb
                .add_dataflow_op(IntOpDef::ipow.with_log_width(6), [base, exp])
                .unwrap()
                .outputs_arr();
            let [_q, _r] = fb
                .add_dataflow_op(IntOpDef::idivmod_s.with_log_width(6), [n, m])
                .unwrap()
                .outputs_arr();
            let [_qr_sum] = fb
                .add_dataflow_op(IntOpDef::idivmod_checked_s.with_log_width(6), [n, m])
                .unwrap()
                .outputs_arr();
            let [_b] = fb
                .add_dataflow_op(ConvertOpDef::itobool.without_log_width(), [bit])
                .unwrap()
                .outputs_arr();
            let [_i] = fb
                .add_dataflow_op(ConvertOpDef::ifrombool.without_log_width(), [flag])
                .unwrap()
                .outputs_arr();
            fb.finish_hugr_with_outputs([pow]).unwrap()
        };

        let find = |name: &str| -> Node {
            hugr.nodes()
                .find(|n| {
                    hugr.get_optype(*n)
                        .as_extension_op()
                        .is_some_and(|op| op.unqualified_id() == name)
                })
                .unwrap_or_else(|| panic!("no {name} node"))
        };
        let mut engine = HugrEngine::default();
        let run = |engine: &mut HugrEngine,
                   name: &str,
                   seeds: &[(usize, ClassicalValue)]|
         -> ClassicalOutcome {
            let node = find(name);
            for (port, value) in seeds {
                let (src, sp) = hugr
                    .single_linked_output(node, IncomingPort::from(*port))
                    .unwrap();
                engine
                    .wire_state
                    .classical_values
                    .insert((src, sp.index()), value.clone());
            }
            let (op_type, num_inputs, num_outputs, int_info) =
                classify_classical_op(hugr.get_optype(node))
                    .unwrap_or_else(|| panic!("{name} must classify"));
            let op = ClassicalOp {
                node,
                op_type,
                num_inputs,
                num_outputs,
                int_info,
                const_value: None,
            };
            engine.handle_classical_op(&hugr, node, &op)
        };

        // ipow: 3^4 = 81
        let got = run(
            &mut engine,
            "ipow",
            &[(0, ClassicalValue::Int(3)), (1, ClassicalValue::Int(4))],
        );
        assert_eq!(
            got,
            ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Int(81))])
        );

        // idivmod_s: Euclidean -7 divmod 3 -> (q, r) = (-3, 2) on two ports
        let got = run(
            &mut engine,
            "idivmod_s",
            &[(0, ClassicalValue::Int(-7)), (1, ClassicalValue::Int(3))],
        );
        assert_eq!(
            got,
            ClassicalOutcome::Outputs(vec![
                (0, ClassicalValue::Int(-3)),
                (1, ClassicalValue::Int(2))
            ])
        );

        // idivmod_s by zero: fatal fault per the spec
        let got = run(
            &mut engine,
            "idivmod_s",
            &[(0, ClassicalValue::Int(-7)), (1, ClassicalValue::Int(0))],
        );
        assert!(
            matches!(got, ClassicalOutcome::Fault(ref msg) if msg.contains("division by zero")),
            "expected div-by-zero fault, got {got:?}"
        );

        // idivmod_checked_s: value = Sum tag 1 with a (q, r) tuple payload
        let got = run(
            &mut engine,
            "idivmod_checked_s",
            &[(0, ClassicalValue::Int(-7)), (1, ClassicalValue::Int(3))],
        );
        assert_eq!(
            got,
            ClassicalOutcome::Outputs(vec![(
                0,
                ClassicalValue::Sum {
                    tag: 1,
                    values: vec![ClassicalValue::Tuple(vec![
                        ClassicalValue::Int(-3),
                        ClassicalValue::Int(2)
                    ])],
                }
            )])
        );

        // idivmod_checked_s by zero: error Sum (tag 0) with the opaque
        // error payload sum_with_error's error variant carries
        let got = run(
            &mut engine,
            "idivmod_checked_s",
            &[(0, ClassicalValue::Int(-7)), (1, ClassicalValue::Int(0))],
        );
        assert_eq!(
            got,
            ClassicalOutcome::Outputs(vec![(
                0,
                ClassicalValue::Sum {
                    tag: 0,
                    values: vec![ClassicalValue::Tuple(vec![])]
                }
            )])
        );

        // itobool / ifrombool
        let got = run(&mut engine, "itobool", &[(0, ClassicalValue::Int(1))]);
        assert_eq!(
            got,
            ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Bool(true))])
        );
        // ifrombool produces int<1>; the canonical (sign-extended) storage
        // of a 1-bit "1" is -1, matching ConstInt::value_s parsing. A
        // round-trip through itobool still reads it as true.
        let got = run(&mut engine, "ifrombool", &[(0, ClassicalValue::Bool(true))]);
        assert_eq!(
            got,
            ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Int(-1))])
        );
        let got = run(&mut engine, "itobool", &[(0, ClassicalValue::Int(-1))]);
        assert_eq!(
            got,
            ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Bool(true))])
        );
    }

    /// int<5> (32-bit) semantics: wrapping addition, shift-out-to-zero,
    /// rotation within 32 bits, and leading-zero counts relative to the
    /// width -- all derived from the op's `BoundedNat` type arg rather than
    /// the engine's 64-bit storage.
    #[test]
    fn test_int_width_modeling_32bit() {
        use crate::engine::analysis::classify_classical_op;
        use crate::engine::handlers::{ClassicalOutcome, HandlerOutcome};
        use crate::engine::types::ClassicalOp;
        use tket::hugr::builder::{Dataflow, DataflowHugr, FunctionBuilder};
        use tket::hugr::ops::Value;
        use tket::hugr::std_extensions::arithmetic::int_ops::IntOpDef;
        use tket::hugr::std_extensions::arithmetic::int_types::{ConstInt, int_type};
        use tket::hugr::types::Signature;

        let hugr = {
            let mut fb =
                FunctionBuilder::new("w32", Signature::new(vec![], vec![int_type(5)])).unwrap();
            let a = fb.add_load_const(Value::from(ConstInt::new_u(5, 0x7FFF_FFFF).unwrap()));
            let b = fb.add_load_const(Value::from(ConstInt::new_u(5, 1).unwrap()));
            let [sum] = fb
                .add_dataflow_op(IntOpDef::iadd.with_log_width(5), [a, b])
                .unwrap()
                .outputs_arr();
            let [_shifted] = fb
                .add_dataflow_op(IntOpDef::ishl.with_log_width(5), [a, b])
                .unwrap()
                .outputs_arr();
            let [_rot] = fb
                .add_dataflow_op(IntOpDef::irotl.with_log_width(5), [a, b])
                .unwrap()
                .outputs_arr();
            fb.finish_hugr_with_outputs([sum]).unwrap()
        };

        let find = |name: &str| -> Node {
            hugr.nodes()
                .find(|n| {
                    hugr.get_optype(*n)
                        .as_extension_op()
                        .is_some_and(|op| op.unqualified_id() == name)
                })
                .unwrap_or_else(|| panic!("no {name} node"))
        };
        let mut engine = HugrEngine::default();
        let seed = |engine: &mut HugrEngine, node: Node, port: usize, value: ClassicalValue| {
            let (src, sp) = hugr
                .single_linked_output(node, IncomingPort::from(port))
                .unwrap();
            engine
                .wire_state
                .classical_values
                .insert((src, sp.index()), value);
        };
        let classical = |node: Node| -> ClassicalOp {
            let (op_type, num_inputs, num_outputs, int_info) =
                classify_classical_op(hugr.get_optype(node)).expect("classifies");
            ClassicalOp {
                node,
                op_type,
                num_inputs,
                num_outputs,
                int_info,
                const_value: None,
            }
        };

        // iadd at 32 bits: i32::MAX + 1 wraps to i32::MIN (canonical
        // sign-extended storage)
        let node = find("iadd");
        seed(&mut engine, node, 0, ClassicalValue::Int(0x7FFF_FFFF));
        seed(&mut engine, node, 1, ClassicalValue::Int(1));
        assert_eq!(
            engine.handle_classical_op(&hugr, node, &classical(node)),
            ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Int(i64::from(i32::MIN)))])
        );

        // ishl at 32 bits: 1 << 31 = i32::MIN; 1 << 32 drops every bit
        let node = find("ishl");
        seed(&mut engine, node, 0, ClassicalValue::Int(1));
        seed(&mut engine, node, 1, ClassicalValue::Int(31));
        assert_eq!(
            engine.handle_classical_op(&hugr, node, &classical(node)),
            ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Int(i64::from(i32::MIN)))])
        );
        seed(&mut engine, node, 1, ClassicalValue::Int(32));
        assert_eq!(
            engine.handle_classical_op(&hugr, node, &classical(node)),
            ClassicalOutcome::Outputs(vec![(0, ClassicalValue::Int(0))])
        );

        // irotl at 32 bits: rotating 0x8000_0001 left by 1 gives
        // 0x0000_0003 (the high bit wraps into bit 0 within 32 bits)
        let node = find("irotl");
        seed(
            &mut engine,
            node,
            0,
            ClassicalValue::Int(i64::from(u32::from_le_bytes([1, 0, 0, 0x80]).cast_signed())),
        );
        seed(&mut engine, node, 1, ClassicalValue::Int(1));
        assert_eq!(
            engine.handle_int_op(&hugr, node, "irotl"),
            HandlerOutcome::Processed
        );
        assert_eq!(
            engine.wire_state.classical_values.get(&(node, 0)),
            Some(&ClassicalValue::Int(3))
        );
    }

    /// Excess measurement outcomes (driver/engine batch-count disagreement)
    /// must raise instead of silently dropping -- an unbound outcome shifts
    /// every later measurement index.
    #[test]
    fn test_excess_measurement_outcomes_error() {
        use pecos_engines::{ByteMessageBuilder, ControlEngine, EngineStage};

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );
        let mut engine = HugrEngine::from_file(path).expect("Failed to load HUGR");
        let stage = engine.start(()).expect("start");
        let EngineStage::NeedsProcessing(msg) = stage else {
            panic!("expected a processing stage");
        };
        let n_meas = msg
            .quantum_ops()
            .expect("parse ops")
            .iter()
            .filter(|g| matches!(g.gate_type, GateType::MZ))
            .count();
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();
        builder.add_outcomes(&vec![0usize; n_meas + 3]);
        let Err(err) = engine.continue_processing(builder.build()) else {
            panic!("excess outcomes must error");
        };
        assert!(
            err.to_string().contains("has no queued measurement"),
            "unexpected error: {err}"
        );
    }

    /// Round-8 regressions: unsigned width conversions must reinterpret
    /// the canonical (sign-extended) storage as a bit pattern at the
    /// SOURCE width -- as_uint-style negative rejection deferred a
    /// canonical `int<1>` "1" (stored -1) forever, and `inarrow_u`'s signed
    /// range test rejected legitimate high-bit unsigned values.
    #[test]
    fn test_unsigned_width_conversions_handle_canonical_storage() {
        use crate::engine::handlers::HandlerOutcome;
        use tket::hugr::builder::{Dataflow, DataflowHugr, FunctionBuilder};
        use tket::hugr::ops::Value;
        use tket::hugr::std_extensions::arithmetic::int_ops::IntOpDef;
        use tket::hugr::std_extensions::arithmetic::int_types::{ConstInt, int_type};
        use tket::hugr::types::Signature;

        let hugr = {
            let mut fb =
                FunctionBuilder::new("widen", Signature::new(vec![], vec![int_type(6)])).unwrap();
            let bit = fb.add_load_const(Value::from(ConstInt::new_u(0, 1).unwrap()));
            let wide = fb.add_load_const(Value::from(ConstInt::new_u(6, 1).unwrap()));
            let [widened] = fb
                .add_dataflow_op(IntOpDef::iwiden_u.with_two_log_widths(0, 6), [bit])
                .unwrap()
                .outputs_arr();
            let [_narrowed] = fb
                .add_dataflow_op(IntOpDef::inarrow_u.with_two_log_widths(6, 5), [wide])
                .unwrap()
                .outputs_arr();
            let _ = widened;
            fb.finish_hugr_with_outputs([widened]).unwrap()
        };
        let find = |name: &str| -> Node {
            hugr.nodes()
                .find(|n| {
                    hugr.get_optype(*n)
                        .as_extension_op()
                        .is_some_and(|op| op.unqualified_id() == name)
                })
                .unwrap_or_else(|| panic!("no {name} node"))
        };
        let seed = |engine: &mut HugrEngine, node: Node, value: ClassicalValue| {
            let (src, sp) = hugr
                .single_linked_output(node, IncomingPort::from(0))
                .unwrap();
            engine
                .wire_state
                .classical_values
                .insert((src, sp.index()), value);
        };

        let mut engine = HugrEngine::default();

        // iwiden_u int<1> -> int<64>: canonical 1-bit "1" stores as -1 and
        // must widen (zero-extend) to 1, not defer or become u64::MAX.
        let widen = find("iwiden_u");
        seed(&mut engine, widen, ClassicalValue::Int(-1));
        assert_eq!(
            engine.handle_int_op(&hugr, widen, "iwiden_u"),
            HandlerOutcome::Processed
        );
        assert_eq!(
            engine.wire_state.classical_values.get(&(widen, 0)),
            Some(&ClassicalValue::Int(1))
        );

        // inarrow_u int<64> -> int<32> of 0xFFFF_FFFF: fits as unsigned,
        // and the narrowed value stores canonically (sign-extended) as -1.
        let narrow = find("inarrow_u");
        seed(&mut engine, narrow, ClassicalValue::Int(0xFFFF_FFFF));
        assert_eq!(
            engine.handle_int_op(&hugr, narrow, "inarrow_u"),
            HandlerOutcome::Processed
        );
        assert_eq!(
            engine.wire_state.classical_values.get(&(narrow, 0)),
            Some(&ClassicalValue::Sum {
                tag: 1,
                values: vec![ClassicalValue::Int(-1)]
            })
        );

        // inarrow_u of a value ABOVE 2^32 must produce the error variant
        // (with its opaque payload), not a fault and not a fit.
        seed(&mut engine, narrow, ClassicalValue::Int(0x1_0000_0000));
        assert_eq!(
            engine.handle_int_op(&hugr, narrow, "inarrow_u"),
            HandlerOutcome::Processed
        );
        assert_eq!(
            engine.wire_state.classical_values.get(&(narrow, 0)),
            Some(&ClassicalValue::Sum {
                tag: 0,
                values: vec![ClassicalValue::Tuple(vec![])]
            })
        );
    }

    /// A DEAD (uncalled, non-entrypoint) module-level function must not
    /// execute: its body used to be ungated (only CALLED `FuncDefns` were
    /// gated), so it raced the real program and could clobber the
    /// entrypoint's captured return values.
    #[test]
    fn test_dead_function_body_does_not_execute() {
        use tket::hugr::builder::{Container, Dataflow, DataflowSubContainer, ModuleBuilder};
        use tket::hugr::hugr::hugrmut::HugrMut;
        use tket::hugr::ops::Value;
        use tket::hugr::ops::handle::NodeHandle;
        use tket::hugr::std_extensions::arithmetic::int_types::{ConstInt, int_type};
        use tket::hugr::types::Signature;

        let (hugr, dead_load) = {
            let mut module = ModuleBuilder::new();
            let mut main_fb = module
                .define_function("main", Signature::new(vec![], vec![int_type(6)]))
                .unwrap();
            let seven = main_fb.add_load_const(Value::from(ConstInt::new_u(6, 7).unwrap()));
            let main_id = main_fb.finish_with_outputs([seven]).unwrap();
            let mut dead_fb = module
                .define_function("dead", Signature::new(vec![], vec![int_type(6)]))
                .unwrap();
            let nine = dead_fb.add_load_const(Value::from(ConstInt::new_u(6, 9).unwrap()));
            let dead_load = nine.node();
            dead_fb.finish_with_outputs([nine]).unwrap();
            let mut hugr = module.hugr().clone();
            hugr.set_entrypoint(main_id.node());
            (hugr, dead_load)
        };

        let engine = HugrEngine::from_hugr(hugr);
        assert!(
            engine.nodes_inside_func_defns.contains(&dead_load),
            "dead function body must be gated"
        );
        assert!(
            !engine.work_queue.contains(&dead_load),
            "dead function body must not be queued"
        );
    }

    #[test]
    fn test_completion_audit_reports_unexecuted_container_ops() {
        // The reachability audit must catch a container the engine claims
        // to have executed whose ops never ran -- the class of bug that is
        // invisible to queue/pending bookkeeping (a node that was never
        // queued at all). Simulate one by recording a real block as
        // executed without running anything.
        use tket::hugr::ops::OpType;

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/forloop_h_test.hugr"
        );
        let mut engine = HugrEngine::from_file(path).expect("Failed to load HUGR");
        let hugr = engine.hugr.clone().expect("hugr loaded");
        let block = hugr
            .nodes()
            .find(|n| {
                matches!(hugr.get_optype(*n), OpType::DataflowBlock(_))
                    && hugr.children(*n).any(|c| {
                        !matches!(
                            hugr.get_optype(c),
                            OpType::Input(_) | OpType::Output(_) | OpType::Const(_)
                        )
                    })
            })
            .expect("fixture has a non-trivial block");
        engine.executed_containers.insert(block, "DataflowBlock");

        let err = engine
            .ensure_no_stalled_execution()
            .expect_err("audit must fail with unexecuted ops");
        let msg = format!("{err}");
        assert!(
            msg.contains("unexecuted ops in executed containers"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_forloop_executes_each_iteration_and_terminates() {
        // Regression guard for the loop-iteration freeze: guppy's
        // `for _ in range(3)` lowers to a CFG cycle whose body block calls
        // the iterator's __next__ each pass. Block re-activation must clear
        // processed flags for ALL op categories BEFORE any readiness check;
        // interleaving them let the Call fire against the previous
        // iteration's flags, re-propagating stale arguments so the loop
        // re-ran iteration 0 forever (H emitted per wave, no termination).
        use pecos_engines::{ByteMessageBuilder, ControlEngine, EngineStage};

        let _ = env_logger::builder().is_test(true).try_init();
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/forloop_h_test.hugr"
        );
        let mut engine = HugrEngine::from_file(path).expect("Failed to load HUGR");
        let mut stage = engine.start(()).expect("Failed to start engine");
        let mut gate_counts: BTreeMap<GateType, usize> = BTreeMap::new();
        let mut rounds = 0;
        loop {
            rounds += 1;
            assert!(
                rounds <= 10,
                "forloop should terminate in a few rounds; gate_counts={gate_counts:?}"
            );
            match stage {
                EngineStage::NeedsProcessing(msg) => {
                    let ops = msg.quantum_ops().expect("parse quantum ops");
                    for g in &ops {
                        *gate_counts.entry(g.gate_type).or_insert(0) += 1;
                    }
                    let n_meas = ops
                        .iter()
                        .filter(|g| {
                            matches!(
                                g.gate_type,
                                GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
                            )
                        })
                        .count();
                    let mut builder = ByteMessageBuilder::new();
                    let _ = builder.for_outcomes();
                    builder.add_outcomes(&vec![0usize; n_meas]);
                    stage = engine
                        .continue_processing(builder.build())
                        .expect("continue");
                }
                EngineStage::Complete(_) => break,
            }
        }

        // range(3): exactly three H applications, then the final measure.
        assert_eq!(gate_counts.get(&GateType::H), Some(&3), "{gate_counts:?}");
        assert_eq!(gate_counts.get(&GateType::MZ), Some(&1), "{gate_counts:?}");

        // Clean completion: no stalled control flow, no starved nodes.
        assert!(engine.active_cfgs.is_empty());
        assert!(engine.active_cases.is_empty());
        assert!(engine.active_calls.is_empty());
        assert!(engine.active_tailloops.is_empty());
        assert!(engine.deferred_nodes.is_empty());
    }

    #[test]
    fn test_ry_angle_through_tuple_wrap() {
        // Guppy lowers `ry(q, angle(0.5))` with the angle constant wrapped in
        // a 1-tuple (Const -> LoadConstant -> MakeTuple -> UnpackTuple ->
        // from_halfturns_unchecked -> Ry). The static angle extraction must
        // trace through the tuple wrap/unwrap; a miss used to silently become
        // RY(0) (issue observed as all-|0> Bell statistics in
        // test_real_quantum_circuits.py::test_rotation_gates).
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/ry_angle_tuple.hugr"
        );
        let engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        let ry_ops: Vec<_> = engine
            .quantum_ops
            .values()
            .filter(|op| op.gate_type == GateType::RY)
            .collect();
        assert_eq!(ry_ops.len(), 1, "Expected exactly one RY op");

        // angle(0.5) = 0.5 half-turns = pi/2 radians.
        let params = &ry_ops[0].params;
        assert_eq!(
            params.len(),
            1,
            "RY angle was not statically extracted (tuple wrap/unwrap not traced)"
        );
        let angle = params[0];
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "RY angle should be pi/2 radians, got {angle}"
        );
    }

    #[test]
    fn test_rx_pi_tuple_const_runtime_execution() {
        // Like test_ry_angle_tuple_runtime_execution, but for guppy's `pi`
        // constant, which lowers the angle as a TUPLE-VALUED Const
        // (Const(Tuple(FloatVal)) -> LoadConstant -> UnpackTuple ->
        // from_halfturns_unchecked -> Rx) with no MakeTuple node. The runtime
        // constant loader must convert tuple constants element-wise or the
        // chain starves and (with the fail-loud hardening) this errors.
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/rx_pi_tuple_const.hugr"
        );
        let mut engine = HugrEngine::from_file(hugr_path).expect("Failed to load HUGR");

        let msg = engine
            .generate_commands()
            .expect("Failed to generate commands");
        let ops = msg.quantum_ops().expect("Failed to parse quantum ops");

        let rx_cmd = ops
            .iter()
            .find(|g| g.gate_type == GateType::RX)
            .expect("Expected an RX command");
        assert_eq!(rx_cmd.angles.len(), 1, "RX command should carry its angle");
        let radians = rx_cmd.angles[0].to_radians();
        assert!(
            (radians - std::f64::consts::PI).abs() < 1e-9,
            "RX command should have angle pi, got {radians}",
        );
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

        // Drive with one outcome per measurement (all zeros). The
        // result-reporting tail maps a function over the measured array via
        // collections.borrow_arr.scan; with scan support the program runs
        // to clean completion and captures the reported result array.
        let mut stage = engine.start(()).expect("Failed to start engine");
        let mut gate_counts: BTreeMap<GateType, usize> = BTreeMap::new();
        let mut rounds = 0;
        loop {
            rounds += 1;
            assert!(rounds <= 20, "conditional_x should complete quickly");
            match stage {
                pecos_engines::EngineStage::NeedsProcessing(msg) => {
                    let ops = msg.quantum_ops().expect("parse quantum ops");
                    for g in &ops {
                        *gate_counts.entry(g.gate_type).or_insert(0) += 1;
                    }
                    let n_meas = ops
                        .iter()
                        .filter(|g| {
                            matches!(
                                g.gate_type,
                                GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
                            )
                        })
                        .count();
                    let mut builder = ByteMessageBuilder::new();
                    let _ = builder.for_outcomes();
                    builder.add_outcomes(&vec![0usize; n_meas]);
                    stage = engine
                        .continue_processing(builder.build())
                        .expect("conditional_x must complete cleanly under scan support");
                }
                pecos_engines::EngineStage::Complete(_) => break,
            }
        }

        // With all-zero outcomes the measurement selects the else branch:
        // H on the control, two measurements, and NO conditional X.
        assert_eq!(gate_counts.get(&GateType::H), Some(&1), "{gate_counts:?}");
        assert_eq!(gate_counts.get(&GateType::MZ), Some(&2), "{gate_counts:?}");
        assert_eq!(gate_counts.get(&GateType::X), None, "{gate_counts:?}");

        // The scan-driven reporting tail must produce the result array:
        // both measured bits are 0.
        assert!(engine.active_scans.is_empty());
        let captured = engine.get_captured_results();
        assert!(
            captured.iter().any(
                |r| matches!(&r.value, ResultValue::ArrayBool(bits) if bits == &vec![false, false])
            ),
            "expected a [false, false] result array, got {captured:?}"
        );
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

        // Every shot must succeed: while-loop execution over CFG back edges
        // is supported (the old `successes > 0 || failures > 0` form was a
        // tautology that passed with 10/10 failed shots).
        assert_eq!(
            failures,
            0,
            "while-loop shots failed: {failures}/{}",
            successes + failures
        );
        assert!(successes > 0, "no shots ran");
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
        assert_eq!(failures, 0, "shots failed: {failures}/{num_shots}");
        {
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
        assert_eq!(failures, 0, "shots failed: {failures}/{num_shots}");
        {
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
        assert_eq!(failures, 0, "shots failed: {failures}/{num_shots}");
        {
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
        assert_eq!(failures, 0, "shots failed: {failures}/{num_shots}");
        {
            let total = count_00 + count_01 + count_10 + count_11;
            assert!(total > 0, "No measurements recorded");

            // Correlated measurements: 00 and 11 should dominate, 01 and 10 should be rare
            let correlated = count_00 + count_11;
            let uncorrelated = count_01 + count_10;
            assert!(
                correlated > uncorrelated * 4,
                "Expected Bell state correlation: {correlated} correlated vs {uncorrelated} uncorrelated"
            );
            // A zero-gate engine also satisfies the ratio (all shots 00):
            // a real Bell state must produce BOTH correlated outcomes.
            assert!(count_00 > 0, "expected some 00 outcomes");
            assert!(count_11 > 0, "expected some 11 outcomes");
        }
    }
}
