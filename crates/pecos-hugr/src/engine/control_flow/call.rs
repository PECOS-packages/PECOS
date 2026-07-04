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
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::engine::HugrEngine;

impl HugrEngine {
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
            // Find the active call executing this FuncDefn and read its arg.
            let info = self
                .active_calls
                .values()
                .find(|info| info.func_defn_node == func_defn)?;
            match info.type_args.get(var_idx)? {
                TypeArg::BoundedNat(n) => return Some(*n),
                TypeArg::Variable(var) => {
                    // Forwarded generic: continue resolution in the CALLER's
                    // frame, at the caller's variable index.
                    node = info.call_node;
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
    pub(crate) fn complete_func_call_if_needed(&mut self, hugr: &Hugr, cfg_node: tket::hugr::Node) {
        // A scan folding through this FuncDefn owns the frame: route the
        // completion to it (next element, or scan completion).
        if self.continue_scan_after_frame(hugr, cfg_node) {
            return;
        }
        // Find which active Call (if any) has a FuncDefn with this CFG
        let call_to_complete: Option<(tket::hugr::Node, tket::hugr::Node)> = self
            .active_calls
            .iter()
            .find_map(|(&call_node, call_info)| {
                if let Some(func_info) = self.func_defns.get(&call_info.func_defn_node)
                    && func_info.cfg_node == Some(cfg_node)
                {
                    return Some((call_node, call_info.func_defn_node));
                }
                None
            });

        if let Some((call_node, func_defn_node)) = call_to_complete {
            debug!(
                "Completing Call {call_node:?} after FuncDefn {func_defn_node:?} CFG {cfg_node:?} finished"
            );

            if let Some(func_info) = self.func_defns.get(&func_defn_node).cloned() {
                // Propagate wires from FuncDefn Output node to Call output ports
                // CFG outputs should already be mapped to FuncDefn Output inputs
                // Now map FuncDefn Output inputs to Call outputs
                for port in 0..func_info.num_outputs {
                    // Check if we have a wire mapping for the FuncDefn Output input
                    // FuncDefn Output receives from CFG outputs
                    let output_in_port = IncomingPort::from(port);
                    if let Some((src_node, src_port)) =
                        hugr.single_linked_output(func_info.output_node, output_in_port)
                    {
                        let src_wire = (src_node, src_port.index());
                        // Map qubits
                        if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&src_wire) {
                            let call_output_wire = (call_node, port);
                            self.wire_state
                                .wire_to_qubit
                                .insert(call_output_wire, qubit_id);
                            debug!(
                                "Call {call_node:?}: mapped FuncDefn output {port} qubit {qubit_id:?} to Call output"
                            );
                        }
                        // Map classical values (including arrays)
                        if let Some(value) =
                            self.wire_state.classical_values.get(&src_wire).cloned()
                        {
                            let call_output_wire = (call_node, port);
                            self.wire_state
                                .classical_values
                                .insert(call_output_wire, value.clone());
                            debug!(
                                "Call {call_node:?}: mapped FuncDefn output {port} classical value {value:?} to Call output"
                            );
                        }
                    }
                }

                // Mark Call as processed FIRST so successors can be added correctly
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
                self.retry_pending_bool_reads();

                // Check if there are pending calls to this FuncDefn
                if let Some(pending) = self.pending_func_calls.get_mut(&func_defn_node)
                    && let Some(next_call) = pending.pop()
                {
                    debug!(
                        "FuncDefn {func_defn_node:?} free: starting next pending Call {next_call:?}"
                    );
                    // Add the pending call to the front of the work queue
                    // so it gets processed next
                    if !self.work_queue.contains(&next_call) {
                        self.work_queue.push_front(next_call);
                    }
                }
            }
        }
    }
}
