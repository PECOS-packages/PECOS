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

//! Higher-order array `scan` execution.
//!
//! `collections.array.scan` / `collections.borrow_arr.scan` fold a function
//! value over an array: `array<N, T1>, (T1, *A -> T2, *A), *A ->
//! array<N, T2>, *A`. The engine runs the scanned function once per element
//! through the same frame machinery as Calls, so quantum ops inside the
//! function (e.g. `measure_array`'s per-qubit measure) go through real
//! measurement rounds. Elements fold left to right, matching the order the
//! measurements were declared in.

use log::debug;
use tket::hugr::ops::OpTrait;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::engine::HugrEngine;
use crate::engine::activation::{ContainerActivation, QueuePolicy};
use crate::engine::analysis::collect_descendants;
use crate::engine::handlers::HandlerOutcome;
use crate::engine::types::{ActiveScanInfo, ClassicalValue};

impl HugrEngine {
    /// Handle a `scan` op: start folding, or defer until every input
    /// (array, function value, accumulators) resolves and the scanned
    /// function's frame is free.
    pub(crate) fn handle_scan_op(&mut self, hugr: &Hugr, node: Node) -> HandlerOutcome {
        if self.active_scans.contains_key(&node) {
            // Already folding: frame completions drive progress; retries of
            // the scan node itself have nothing to do.
            return HandlerOutcome::Defer;
        }

        let Some(ClassicalValue::Array(elements)) = self.get_input_value(hugr, node, 0) else {
            debug!("scan at {node:?}: array not ready, deferring");
            return HandlerOutcome::Defer;
        };
        let Some(ClassicalValue::FuncRef(func_defn_node, func_type_args)) =
            self.get_input_value(hugr, node, 1)
        else {
            debug!("scan at {node:?}: function value not ready, deferring");
            return HandlerOutcome::Defer;
        };
        let Some(sig) = hugr.get_optype(node).dataflow_signature() else {
            return HandlerOutcome::Fault(format!("scan at {node:?} has no dataflow signature"));
        };
        let mut accs = Vec::new();
        for port in 2..sig.input_count() {
            let Some(value) = self.get_input_value(hugr, node, port) else {
                debug!("scan at {node:?}: accumulator {port} not ready, deferring");
                return HandlerOutcome::Defer;
            };
            accs.push(value);
        }

        if !self.func_defns.contains_key(&func_defn_node) {
            return HandlerOutcome::Fault(format!(
                "scan at {node:?} references unknown FuncDefn {func_defn_node:?}"
            ));
        }
        // The scanned function's single execution frame must be free.
        let frame_in_use = self
            .active_calls
            .values()
            .any(|info| info.func_defn_node == func_defn_node)
            || self
                .active_scans
                .values()
                .any(|scan| scan.func_defn_node == func_defn_node);
        if frame_in_use {
            debug!("scan at {node:?}: FuncDefn {func_defn_node:?} frame in use, deferring");
            return HandlerOutcome::Defer;
        }

        let mut state = ActiveScanInfo {
            scan_node: node,
            func_defn_node,
            remaining: elements.into(),
            results: Vec::new(),
            accs,
            type_args: func_type_args,
            frame_ops: std::collections::BTreeSet::new(),
        };
        if state.remaining.is_empty() {
            // Zero-length array: complete immediately.
            self.store_scan_outputs(&state);
            debug!("scan at {node:?}: empty array, completed immediately");
            return HandlerOutcome::Processed;
        }
        debug!(
            "scan at {node:?}: folding {} elements through FuncDefn {func_defn_node:?}",
            state.remaining.len()
        );
        let element = state
            .remaining
            .pop_front()
            .expect("non-empty checked above");
        self.active_scans.insert(node, state);
        if self.launch_scan_iteration(hugr, node, element) {
            // Pure-passthrough frame: fold the remaining elements now.
            self.advance_scan(hugr, node);
        }
        // Like a Call, the scan node stays unprocessed while its frame
        // runs; completion marks it. When advance_scan already completed
        // the whole fold synchronously, the dispatcher's processed guard
        // prevents this Defer from re-parking a finished node.
        HandlerOutcome::Defer
    }

    /// Run one element through the scanned function's frame. Returns true
    /// when the frame completed INSTANTLY (an empty dataflow passthrough
    /// body): the caller must then advance in ITS loop -- calling back into
    /// completion here would recurse once per element.
    fn launch_scan_iteration(
        &mut self,
        hugr: &Hugr,
        scan_node: Node,
        element: ClassicalValue,
    ) -> bool {
        let Some(state) = self.active_scans.get(&scan_node) else {
            // Unreachable from current call sites; if ever reached, a
            // silent false would impersonate a launched frame whose
            // completion never comes.
            self.execution_error = Some(format!(
                "scan {scan_node:?}: launch without an active scan state"
            ));
            return false;
        };
        let func_defn_node = state.func_defn_node;
        let accs = state.accs.clone();
        let Some(func_info) = self.func_defns.get(&func_defn_node).cloned() else {
            self.execution_error = Some(format!(
                "scan {scan_node:?}: FuncDefn {func_defn_node:?} vanished mid-fold"
            ));
            return false;
        };

        // Reset the frame exactly like a Call re-activation: descendants'
        // processed flags AND stale wires clear (previous element's values
        // must not leak into this one), and the previous iteration's
        // executed-container records are invalidated.
        let mut descendants = std::collections::BTreeSet::new();
        collect_descendants(hugr, func_defn_node, &mut descendants);
        let mut act = ContainerActivation::new();
        for node in &descendants {
            self.nodes_inside_func_defns.remove(node);
            act.reset(*node);
            self.executed_containers.remove(node);
        }
        act.keep_wires(func_info.input_node);
        if let Some(cfg_node) = func_info.cfg_node {
            act.reset_processed(cfg_node);
        }
        // A plain dataflow body (no CFG) executes as a tracked op set: queue
        // every non-boundary child; their collective completion finishes
        // the element (check_scan_frame_completion).
        let mut frame_ops = std::collections::BTreeSet::new();
        if func_info.cfg_node.is_none() {
            for child in hugr.children(func_defn_node) {
                let op = hugr.get_optype(child);
                if matches!(
                    op,
                    tket::hugr::ops::OpType::Input(_)
                        | tket::hugr::ops::OpType::Output(_)
                        | tket::hugr::ops::OpType::Const(_)
                ) {
                    continue;
                }
                let policy = match op {
                    tket::hugr::ops::OpType::Conditional(_)
                    | tket::hugr::ops::OpType::TailLoop(_)
                    | tket::hugr::ops::OpType::LoadConstant(_) => QueuePolicy::Always,
                    _ => QueuePolicy::IfReady,
                };
                act.queue(child, policy);
                frame_ops.insert(child);
            }
        }
        if let Some(state) = self.active_scans.get_mut(&scan_node) {
            state.frame_ops = frame_ops;
        }
        self.run_activation(hugr, &act);

        // Element -> Input port 0; accumulators -> Input ports 1..
        let input_wire = (func_info.input_node, 0);
        if let ClassicalValue::QubitRef(qubit_id) = &element {
            self.wire_state.wire_to_qubit.insert(input_wire, *qubit_id);
        } else {
            self.wire_state.wire_to_qubit.remove(&input_wire);
        }
        self.wire_state.classical_values.insert(input_wire, element);
        for (i, acc) in accs.into_iter().enumerate() {
            self.wire_state
                .classical_values
                .insert((func_info.input_node, 1 + i), acc);
        }

        if let Some(cfg_node) = func_info.cfg_node {
            debug!("scan {scan_node:?}: launching frame CFG {cfg_node:?}");
            if !self.work_queue.contains(cfg_node) {
                self.work_queue.push_front(cfg_node);
            }
            false
        } else {
            debug!("scan {scan_node:?}: launched dataflow frame");
            // An EMPTY dataflow body (pure passthrough) completes at once;
            // the caller advances iteratively.
            self.active_scans
                .get(&scan_node)
                .is_some_and(|state| state.frame_ops.is_empty())
        }
    }

    /// Check whether an active scan's DATAFLOW frame finished: every
    /// tracked body op processed and no nested container still active.
    /// Advances the scan when it did. `processed_node` scopes the check to
    /// scans whose frame contains it (or the scan node itself for the
    /// empty-frame case).
    pub(crate) fn check_scan_frame_completion(&mut self, hugr: &Hugr, processed_node: Node) {
        let candidates: Vec<Node> = self
            .active_scans
            .iter()
            .filter(|(scan_node, state)| {
                **scan_node == processed_node || state.frame_ops.contains(&processed_node)
            })
            .filter(|(_, state)| {
                // CFG-bodied frames complete via complete_func_call_if_needed.
                self.func_defns
                    .get(&state.func_defn_node)
                    .is_some_and(|info| info.cfg_node.is_none())
            })
            .map(|(&scan_node, _)| scan_node)
            .collect();
        for scan_node in candidates {
            let Some(state) = self.active_scans.get(&scan_node) else {
                continue;
            };
            let all_done = state.frame_ops.iter().all(|op| {
                self.processed.contains(op)
                    && !self
                        .active_cases
                        .values()
                        .any(|case| case.conditional_node == *op)
                    && !self.active_tailloops.contains_key(op)
                    && !self.active_calls.contains_key(op)
                    && !self.active_cfgs.contains_key(op)
            });
            if all_done {
                debug!("scan {scan_node:?}: dataflow frame complete");
                self.advance_scan(hugr, scan_node);
            }
        }
    }

    /// Route a completed `FuncDefn` CFG to its scan, if one is folding
    /// through it. Returns true when the completion belonged to a scan.
    pub(crate) fn continue_scan_after_frame(&mut self, hugr: &Hugr, cfg_node: Node) -> bool {
        let scan_node = self.active_scans.iter().find_map(|(&scan_node, state)| {
            self.func_defns
                .get(&state.func_defn_node)
                .filter(|info| info.cfg_node == Some(cfg_node))
                .map(|_| scan_node)
        });
        let Some(scan_node) = scan_node else {
            return false;
        };
        self.advance_scan(hugr, scan_node);
        true
    }

    /// One element's frame finished: collect its outputs, then fold the
    /// next element or complete the scan. Iterative: a pure-passthrough
    /// frame completes each launch instantly, and recursing per element
    /// would grow the stack with the array length.
    fn advance_scan(&mut self, hugr: &Hugr, scan_node: Node) {
        loop {
            let Some(state) = self.active_scans.get(&scan_node) else {
                return;
            };
            let Some(func_info) = self.func_defns.get(&state.func_defn_node).cloned() else {
                return;
            };

            // Collect the frame's outputs: port 0 = mapped element, 1.. = accs.
            let read_output = |engine: &Self, port: usize| -> Option<ClassicalValue> {
                let (src, sp) =
                    hugr.single_linked_output(func_info.output_node, IncomingPort::from(port))?;
                let wire = (src, sp.index());
                if let Some(value) = engine.wire_state.classical_values.get(&wire) {
                    return Some(value.clone());
                }
                engine
                    .wire_state
                    .wire_to_qubit
                    .get(&wire)
                    .map(|q| ClassicalValue::QubitRef(*q))
            };
            let Some(mapped) = read_output(self, 0) else {
                // The frame completed without producing the mapped value: a
                // wiring bug this must not paper over.
                self.execution_error = Some(format!(
                    "scan {scan_node:?}: frame completed without an output value"
                ));
                self.active_scans.remove(&scan_node);
                return;
            };
            let num_accs = self.active_scans[&scan_node].accs.len();
            let mut new_accs = Vec::with_capacity(num_accs);
            for i in 0..num_accs {
                if let Some(value) = read_output(self, 1 + i) {
                    new_accs.push(value);
                } else {
                    self.execution_error = Some(format!(
                        "scan {scan_node:?}: frame completed without accumulator {i}"
                    ));
                    self.active_scans.remove(&scan_node);
                    return;
                }
            }

            let state = self
                .active_scans
                .get_mut(&scan_node)
                .expect("checked above");
            state.results.push(mapped);
            state.accs = new_accs;

            if let Some(element) = state.remaining.pop_front() {
                if self.launch_scan_iteration(hugr, scan_node, element) {
                    // Instant (passthrough) completion: fold the next element
                    // in this same loop.
                    continue;
                }
                return;
            }

            // All elements folded: store outputs and complete the scan node.
            let state = self.active_scans.remove(&scan_node).expect("present above");
            self.store_scan_outputs(&state);
            self.processed.insert(scan_node);
            self.deferred_nodes.remove(&scan_node);
            debug!(
                "scan {scan_node:?}: complete with {} results",
                state.results.len()
            );

            // Same completion cascade as a Call.
            self.check_case_completion(hugr, scan_node);
            self.check_cfg_block_completion(hugr, scan_node);
            self.check_tailloop_body_completion(hugr, scan_node);
            self.try_resolve_pending_tailloops();
            self.try_resolve_pending_cfg_branches();
            self.queue_ready_successors(hugr, scan_node);
            self.retry_deferred_nodes();

            // The frame is free now: wake any Call that parked waiting for it
            // (mirrors complete_func_call_if_needed).
            if let Some(pending) = self.pending_func_calls.get_mut(&state.func_defn_node)
                && let Some(next_call) = pending.pop_front()
            {
                debug!(
                    "FuncDefn {:?} free after scan: starting pending Call {next_call:?}",
                    state.func_defn_node
                );
                if !self.work_queue.contains(next_call) {
                    self.work_queue.push_front(next_call);
                }
            }
            return;
        }
    }

    /// Store the scan node's outputs: the mapped array on port 0 and the
    /// final accumulators on ports 1.. .
    fn store_scan_outputs(&mut self, state: &ActiveScanInfo) {
        let scan_node = state.scan_node;
        self.wire_state
            .classical_values
            .insert((scan_node, 0), ClassicalValue::Array(state.results.clone()));
        for (i, acc) in state.accs.iter().enumerate() {
            self.wire_state
                .classical_values
                .insert((scan_node, 1 + i), acc.clone());
        }
    }
}
