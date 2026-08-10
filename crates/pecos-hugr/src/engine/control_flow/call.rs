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

//! Function call handling.
//!
//! Call nodes invoke `FuncDefn` definitions. The engine tracks active calls
//! and manages the call stack for recursive/nested calls.
//!
//! # Structure
//!
//! - Call node: Invokes a `FuncDefn` with input values
//! - `FuncDefn`: Contains the function body (may include CFG)
//! - Return: Outputs from `FuncDefn` propagate back through Call outputs
//!
//! # Execution Flow
//!
//! 1. Call node encountered in work queue
//! 2. `FuncDefn` body activated (typically contains a CFG)
//! 3. CFG executes within `FuncDefn` context
//! 4. On CFG completion, `complete_func_call_if_needed` is triggered
//! 5. Outputs propagate from `FuncDefn` to Call outputs
//! 6. Call's successors are added to work queue

use log::debug;
use tket::hugr::ops::OpType;
use tket::hugr::types::TypeArg;
use tket::hugr::{Hugr, HugrView, Node};

use crate::engine::HugrEngine;

impl HugrEngine {
    /// Fill-only repair of active calls' argument ports.
    ///
    /// A Call launches as soon as it is dispatched; an argument that is a
    /// measurement result still in flight leaves its `FuncDefn` Input port
    /// cleared. Called after each measurement round: writes only ports that
    /// are currently MISSING (never overwrites live frame state), reading
    /// through the tracing layer.
    pub(crate) fn repropagate_active_call_inputs(&mut self, hugr: &Hugr) {
        let targets: Vec<(Node, Node, usize)> = self
            .active_calls
            .iter()
            .filter_map(|(&call_node, info)| {
                self.func_defns
                    .get(&info.func_defn_node)
                    .map(|fi| (call_node, fi.input_node, fi.num_inputs))
            })
            .collect();
        for (call_node, input_node, num_inputs) in targets {
            for port in 0..num_inputs {
                let wire = (input_node, port);
                if self.wire_state.classical_values.contains_key(&wire)
                    || self.wire_state.wire_to_qubit.contains_key(&wire)
                {
                    continue;
                }
                if let Some(qubit_id) = self.get_input_qubit(hugr, call_node, port) {
                    self.wire_state.wire_to_qubit.insert(wire, qubit_id);
                }
                if let Some(value) = self.get_input_value(hugr, call_node, port) {
                    debug!(
                        "Call {call_node:?}: late argument {port} repaired with {value:?} on {wire:?}"
                    );
                    self.wire_state.classical_values.insert(wire, value);
                }
            }
        }
    }

    /// Resolve a type variable used inside a called function body to the
    /// concrete `BoundedNat` from the calling `Call`'s instantiation args.
    ///
    /// Generic function bodies reference their type parameters as variables
    /// (e.g. `prelude.load_nat` of a generic loop bound); only the calling
    /// `Call` op knows the concrete instantiation.
    pub(crate) fn resolve_call_type_arg(
        &self,
        hugr: &Hugr,
        node: Node,
        var_idx: usize,
    ) -> Option<u64> {
        let mut node = node;
        let mut var_idx = var_idx;
        // A generic function may be called from another generic function
        // with the type arg forwarded as a variable (`f<$0>` inside `g<n>`),
        // so resolution walks the active call chain until a concrete
        // BoundedNat appears. Bounded by the active-call count: each hop
        // consumes one distinct call frame.
        for _ in 0..=self.active_calls.len() {
            // Find the enclosing FuncDefn of this node.
            let mut cur = hugr.get_parent(node);
            let func_defn = loop {
                let n = cur?;
                if matches!(hugr.get_optype(n), OpType::FuncDefn(_)) {
                    break n;
                }
                cur = hugr.get_parent(n);
            };
            // Find the active call OR scan executing this FuncDefn and
            // read its arg (a scanned function's frame carries the
            // LoadFunction's instantiation args).
            let (owner_node, arg) = if let Some(info) = self
                .active_calls
                .values()
                .find(|info| info.func_defn_node == func_defn)
            {
                (info.call_node, info.type_args.get(var_idx)?.clone())
            } else {
                let scan = self
                    .active_scans
                    .values()
                    .find(|scan| scan.func_defn_node == func_defn)?;
                (scan.scan_node, scan.type_args.get(var_idx)?.clone())
            };
            match arg {
                TypeArg::BoundedNat(n) => return Some(n),
                TypeArg::Variable(var) => {
                    // Forwarded generic: continue resolution in the CALLER's
                    // frame, at the caller's variable index.
                    node = owner_node;
                    var_idx = var.index();
                }
                _ => return None,
            }
        }
        None
    }

    /// Complete a function call if the completed CFG belongs to an active Call's `FuncDefn`.
    ///
    /// This method is called when a CFG completes. It checks if that CFG belongs
    /// to a `FuncDefn` that was invoked by an active Call, and if so:
    /// 1. Propagates output wires from `FuncDefn` to Call outputs
    /// 2. Marks the Call as processed
    /// 3. Adds Call successors to the work queue
    /// 4. Starts any pending calls to the same `FuncDefn`
    pub(crate) fn complete_func_call_if_needed(
        &mut self,
        hugr: &Hugr,
        cfg_node: Node,
        final_block: Node,
    ) {
        // A scan folding through this FuncDefn owns the frame: route the
        // completion to it (next element, or scan completion).
        if self.continue_scan_after_frame(hugr, cfg_node) {
            return;
        }
        // Find which active Call (if any) has a FuncDefn with this CFG
        let call_to_complete: Option<Node> =
            self.active_calls
                .iter()
                .find_map(|(&call_node, call_info)| {
                    if let Some(func_info) = self.func_defns.get(&call_info.func_defn_node)
                        && func_info.cfg_node == Some(cfg_node)
                    {
                        return Some(call_node);
                    }
                    None
                });

        if let Some(call_node) = call_to_complete {
            // The CFG itself is complete, but its data outputs can still be
            // waiting for a measurement outcome. Keep the Call active and
            // retain enough context to replay CFG-output propagation once the
            // outcome arrives.
            self.pending_call_returns
                .insert(call_node, (cfg_node, final_block));
            self.try_complete_pending_call_return(hugr, call_node);
        }
    }

    /// Retry every Call whose callee CFG completed with unresolved returns.
    pub(crate) fn retry_pending_call_returns(&mut self, hugr: &Hugr) {
        let pending: Vec<Node> = self.pending_call_returns.keys().copied().collect();
        for call_node in pending {
            self.try_complete_pending_call_return(hugr, call_node);
        }
    }

    /// Copy a completed callee frame's returns and release its Call only when
    /// every data port has a runtime representation.
    fn try_complete_pending_call_return(&mut self, hugr: &Hugr, call_node: Node) {
        let Some(&(cfg_node, final_block)) = self.pending_call_returns.get(&call_node) else {
            return;
        };
        let Some(call_info) = self.active_calls.get(&call_node).cloned() else {
            self.pending_call_returns.remove(&call_node);
            return;
        };
        let func_defn_node = call_info.func_defn_node;

        // The first propagation ran as the CFG completed, possibly before a
        // measurement result existed. Replaying it is fill-only in effect:
        // propagate_cfg_outputs only inserts values it can now resolve.
        self.propagate_cfg_outputs(hugr, cfg_node, final_block);

        let Some(func_info) = self.func_defns.get(&func_defn_node).cloned() else {
            return;
        };
        let outputs: Vec<_> = (0..func_info.num_outputs)
            .map(|port| {
                (
                    self.get_input_qubit(hugr, func_info.output_node, port),
                    self.get_input_value(hugr, func_info.output_node, port),
                )
            })
            .collect();
        if outputs
            .iter()
            .any(|(qubit, value)| qubit.is_none() && value.is_none())
        {
            debug!(
                "Call {call_node:?}: CFG {cfg_node:?} complete but return ports unresolved; deferring completion"
            );
            return;
        }

        debug!(
            "Completing Call {call_node:?} after FuncDefn {func_defn_node:?} CFG {cfg_node:?} returns resolved"
        );

        // All return ports are ready; publish them as one completion event.
        for (port, (qubit, value)) in outputs.into_iter().enumerate() {
            let call_output_wire = (call_node, port);
            if let Some(qubit_id) = qubit {
                self.wire_state
                    .wire_to_qubit
                    .insert(call_output_wire, qubit_id);
                debug!(
                    "Call {call_node:?}: mapped FuncDefn output {port} qubit {qubit_id:?} to Call output"
                );
            }
            if let Some(value) = value {
                debug!(
                    "Call {call_node:?}: mapped FuncDefn output {port} classical value {value:?} to Call output"
                );
                self.wire_state
                    .classical_values
                    .insert(call_output_wire, value);
            }
        }

        // Mark Call as processed FIRST so successors can be added correctly.
        self.pending_call_returns.remove(&call_node);
        self.processed.insert(call_node);
        self.active_calls.remove(&call_node);

        // Check if this Call completion allows a parent Case to
        // complete (a case whose FINAL completion event is the Call
        // itself would otherwise stay active forever).
        self.check_scan_frame_completion(hugr, call_node);
        self.check_case_completion(hugr, call_node);

        // Check if this Call completion allows a parent CFG block to complete
        // This is critical for nested function calls
        self.check_cfg_block_completion(hugr, call_node);

        // Check if this Call completion allows a parent TailLoop to complete
        // This is critical for function calls inside TailLoop bodies
        self.check_tailloop_body_completion(hugr, call_node);

        // Add Call's successors to the work queue via the canonical
        // readiness check. (A hand-rolled subset here used to omit
        // classical and extension ops, so e.g. a MakeTuple consuming
        // the Call's result never ran and the enclosing CFG block
        // never completed.)
        self.queue_ready_successors(hugr, call_node);

        // A scan parked on this frame retries through the pending
        // mechanism -- wake the parked set now that the frame freed.
        self.retry_deferred_nodes();

        // Check if there are pending calls to this FuncDefn
        if let Some(pending) = self.pending_func_calls.get_mut(&func_defn_node)
            && let Some(next_call) = pending.pop_front()
        {
            debug!("FuncDefn {func_defn_node:?} free: starting next pending Call {next_call:?}");
            // Add the pending call to the front of the work queue
            // so it gets processed next
            if !self.work_queue.contains(next_call) {
                self.work_queue.push_front(next_call);
            }
        }
    }
}
