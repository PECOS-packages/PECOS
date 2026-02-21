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

//! Wire and value propagation for the HUGR engine.
//!
//! This module handles tracking and propagating values through HUGR wires:
//! - Qubit wire tracking (`wire_to_qubit`)
//! - Classical value propagation (`classical_values`)
//! - Qubit array propagation (`qubit_arrays`)
//!
//! The propagation system follows HUGR's dataflow semantics, where values
//! flow from output ports to connected input ports.

use log::debug;
use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use tket::hugr::ops::OpType;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use crate::engine::GuppyHugrEngine;
use crate::engine::analysis::get_container_type;
use crate::engine::types::{ClassicalValue, ContainerType, QuantumOp, WireKey};

impl GuppyHugrEngine {
    /// Trace through an Input node to find the actual source wire.
    ///
    /// When processing nodes inside containers (DFG, Case, `FuncDefn`, etc.),
    /// the Input node's outputs come from the container's inputs. This method
    /// traces through the Input node to find the actual source wire.
    ///
    /// Different container types have different port mapping semantics:
    /// - DFG/Case/FuncDefn: Input output port N = Container input port N
    /// - Conditional: Port 0 unpacks Sum; ports 1+ are data inputs
    /// - `TailLoop`: Complex handling with CONTINUE/BREAK tags
    ///
    /// Returns the wire key (node, port) of the actual source, or None if not found.
    pub(crate) fn trace_through_input_node(
        &self,
        hugr: &Hugr,
        input_node: Node,
        output_port: usize,
    ) -> Option<WireKey> {
        // Get the parent container of the Input node
        let container = hugr.get_parent(input_node)?;
        let container_type = get_container_type(hugr, container);

        debug!(
            "Tracing Input node {input_node:?}:{output_port} through {container_type:?} container {container:?}"
        );

        // Determine which container input port to check based on container type
        let container_in_port_idx = match container_type {
            ContainerType::Dfg | ContainerType::Case | ContainerType::FuncDefn => {
                // Direct 1:1 mapping: Input output port N = Container input port N
                output_port
            }
            ContainerType::Conditional => {
                // Conditional: Port 0 of Input unpacks Sum fields; subsequent ports are data
                // This is complex - the Input node outputs come from unpacking the Sum
                // For now, skip port 0 (Sum unpacking) and map other ports
                if output_port == 0 {
                    debug!("Skipping Conditional Sum unpacking (port 0)");
                    return None;
                }
                // Data ports start at container input port 1 (after control)
                output_port // Actually maps to same port since control is separate
            }
            ContainerType::TailLoop => {
                // TailLoop is complex - inputs come from both initial values and CONTINUE tag
                // For simplicity, use direct mapping
                output_port
            }
            ContainerType::Call => {
                // Call: Need to trace through to the FuncDefn
                // This is handled separately via static source
                debug!("Call container - tracing not fully implemented");
                output_port
            }
            ContainerType::Cfg => {
                // CFG: Entry block inputs come from CFG inputs
                output_port
            }
            ContainerType::DataflowBlock => {
                // DataflowBlock: Input values are already set by propagate_block_outputs_to_successor
                // The wire key is (input_node, output_port) which was set during block transition
                debug!("DataflowBlock container - checking Input node directly");
                let wire_key = (input_node, output_port);
                if self.wire_state.wire_to_qubit.contains_key(&wire_key) {
                    return Some(wire_key);
                }
                // If not found, try to trace through the CFG structure
                output_port
            }
            ContainerType::Other => {
                // Unknown container type - try direct mapping but warn
                debug!("Unknown container type for {container:?}, trying direct port mapping");
                output_port
            }
        };

        // Check if the container has enough input ports
        let num_container_inputs = hugr.num_inputs(container);
        if container_in_port_idx >= num_container_inputs {
            debug!(
                "Container {container:?} has {num_container_inputs} inputs, but need port {container_in_port_idx} (output_port={output_port})"
            );
            // For containers like Case inside Conditional, the Input node outputs
            // might exceed container inputs - they come from Sum unpacking
            return None;
        }

        // The Input node's output port corresponds to the container's input port
        let container_in_port = IncomingPort::from(container_in_port_idx);

        // Find what's connected to the container's input
        // Use linked_outputs to safely check if there's a connection
        let linked: Vec<_> = hugr.linked_outputs(container, container_in_port).collect();
        if let Some((src_node, src_port)) = linked.first() {
            let wire_key = (*src_node, src_port.index());

            debug!("Container {container:?} input {container_in_port_idx} links to {wire_key:?}");

            // Check if we have a mapping for this wire
            if self.wire_state.wire_to_qubit.contains_key(&wire_key) {
                return Some(wire_key);
            }

            // If the source is also an Input node, recurse
            if matches!(hugr.get_optype(*src_node), OpType::Input(_)) {
                return self.trace_through_input_node(hugr, *src_node, src_port.index());
            }

            // Return the wire key even if we don't have a mapping yet
            // (might be set up later)
            return Some(wire_key);
        }

        None
    }

    /// Propagate all input values to corresponding output ports.
    ///
    /// This is used for pass-through operations that don't modify values.
    pub(crate) fn propagate_all_inputs(&mut self, hugr: &Hugr, node: Node) {
        use tket::hugr::ops::OpTrait;
        let op = hugr.get_optype(node);
        let num_outputs = op.dataflow_signature().map_or(0, |sig| sig.output_count());

        for port in 0..num_outputs {
            if let Some(value) = self.get_input_value(hugr, node, port) {
                self.wire_state.classical_values.insert((node, port), value);
            }
            if let Some(qubit) = self.get_input_qubit(hugr, node, port) {
                self.wire_state.wire_to_qubit.insert((node, port), qubit);
            }
        }
    }

    /// Get a classical value from an input port.
    ///
    /// Follows the wire connected to the specified input port and returns
    /// the classical value at the source, if any.
    pub(crate) fn get_input_value(
        &self,
        hugr: &Hugr,
        node: Node,
        port: usize,
    ) -> Option<ClassicalValue> {
        let in_port = IncomingPort::from(port);
        if let Some((src_node, src_port)) = hugr.single_linked_output(node, in_port) {
            let wire_key = (src_node, src_port.index());
            let value = self.wire_state.classical_values.get(&wire_key).cloned();
            debug!(
                "get_input_value({:?}, {}): src={:?}:{}, wire_key={:?}, value={:?}",
                node,
                port,
                src_node,
                src_port.index(),
                wire_key,
                value
            );
            value
        } else {
            debug!("get_input_value({node:?}, {port}): no linked output");
            None
        }
    }

    /// Get a qubit ID from an input port.
    ///
    /// Follows the wire connected to the specified input port and returns
    /// the qubit ID at the source, if any.
    pub(crate) fn get_input_qubit(&self, hugr: &Hugr, node: Node, port: usize) -> Option<QubitId> {
        let in_port = IncomingPort::from(port);
        if let Some((src_node, src_port)) = hugr.single_linked_output(node, in_port) {
            let wire_key = (src_node, src_port.index());
            self.wire_state.wire_to_qubit.get(&wire_key).copied()
        } else {
            None
        }
    }

    /// Propagate qubit array from input to output (for pass-through operations).
    ///
    /// This handles operations like barriers that pass qubit arrays through unchanged.
    pub(crate) fn propagate_qubit_array(&mut self, hugr: &Hugr, node: Node) {
        // For now, just propagate qubit wire mappings
        let in_port = IncomingPort::from(0);
        if let Some((src_node, src_port)) = hugr.single_linked_output(node, in_port) {
            let src_key = (src_node, src_port.index());

            // Propagate qubit array if present
            if let Some(qubits) = self.wire_state.qubit_arrays.get(&src_key).cloned() {
                self.wire_state.qubit_arrays.insert((node, 0), qubits);
            }

            // Also propagate individual qubit mappings
            if let Some(qubit_id) = self.wire_state.wire_to_qubit.get(&src_key).copied() {
                self.wire_state.wire_to_qubit.insert((node, 0), qubit_id);
            }
        }
    }

    /// Resolve qubit IDs for an operation by following input wires.
    ///
    /// For most operations, this traces input wires to find existing qubit IDs.
    /// For `QAlloc`, it creates a new qubit ID.
    /// Returns a vector of qubit IDs corresponding to the operation's qubit inputs.
    pub(crate) fn resolve_qubits(
        &mut self,
        hugr: &Hugr,
        node: Node,
        op: &QuantumOp,
    ) -> Vec<QubitId> {
        if op.gate_type == GateType::QAlloc {
            // QAlloc creates a new qubit
            let qubit_id = QubitId::from(self.wire_state.next_qubit_id);
            self.wire_state.next_qubit_id += 1;
            self.wire_state.wire_to_qubit.insert((node, 0), qubit_id);
            return vec![qubit_id];
        }

        let mut qubits = Vec::with_capacity(op.num_qubit_inputs);

        for port_idx in 0..op.num_qubit_inputs {
            let in_port = IncomingPort::from(port_idx);

            if let Some((src_node, src_port)) = hugr.single_linked_output(node, in_port) {
                let mut wire_key = (src_node, src_port.index());
                let src_op = hugr.get_optype(src_node);

                // Check if the source is an Input node - if so, trace through it
                if matches!(src_op, OpType::Input(_)) {
                    debug!(
                        "Input node detected: {:?}:{}, attempting trace",
                        src_node,
                        src_port.index()
                    );
                    if let Some(traced_key) =
                        self.trace_through_input_node(hugr, src_node, src_port.index())
                    {
                        debug!(
                            "Traced Input node {:?}:{} -> {:?}",
                            src_node,
                            src_port.index(),
                            traced_key
                        );
                        wire_key = traced_key;
                    } else {
                        debug!(
                            "Failed to trace through Input node {:?}:{}",
                            src_node,
                            src_port.index()
                        );
                    }
                }

                if let Some(&qubit_id) = self.wire_state.wire_to_qubit.get(&wire_key) {
                    qubits.push(qubit_id);

                    // Propagate qubit to output port if this gate has outputs
                    if port_idx < op.num_qubit_outputs {
                        self.wire_state
                            .wire_to_qubit
                            .insert((node, port_idx), qubit_id);
                    }
                } else {
                    // Fallback: create a new qubit ID
                    let fallback = QubitId::from(self.wire_state.next_qubit_id);
                    self.wire_state.next_qubit_id += 1;
                    qubits.push(fallback);
                    if port_idx < op.num_qubit_outputs {
                        self.wire_state
                            .wire_to_qubit
                            .insert((node, port_idx), fallback);
                    }
                    debug!(
                        "Warning: No wire mapping for {wire_key:?}, using fallback {fallback:?}"
                    );
                }
            } else {
                // No linked output - create fallback
                let fallback = QubitId::from(self.wire_state.next_qubit_id);
                self.wire_state.next_qubit_id += 1;
                qubits.push(fallback);
                debug!(
                    "Warning: No linked output for node {node:?} port {port_idx}, using fallback {fallback:?}"
                );
            }
        }

        qubits
    }

    /// Try to load a constant value from a `LoadConstant` node.
    ///
    /// `LoadConstant` nodes have a static edge to a Const node. This method
    /// extracts the value from the Const node and returns it as a `ClassicalValue`.
    ///
    /// Supports integer constants (`ConstInt`), float constants (`ConstF64`),
    /// and boolean constants (`ConstBool`).
    pub(crate) fn try_load_constant(hugr: &Hugr, node: Node) -> Option<ClassicalValue> {
        use tket::extension::bool::ConstBool;
        use tket::hugr::std_extensions::arithmetic::float_types::ConstF64;
        use tket::hugr::std_extensions::arithmetic::int_types::ConstInt;

        // LoadConstant has a static edge from a Const node
        for pred_node in hugr.input_neighbours(node) {
            let pred_op = hugr.get_optype(pred_node);
            if let OpType::Const(const_op) = pred_op {
                let value = const_op.value();
                debug!(
                    "try_load_constant: node {:?}, const value type: {:?}",
                    node,
                    value.get_type()
                );

                // Try to extract as ConstInt
                if let Some(const_int) = value.get_custom_value::<ConstInt>() {
                    // ConstInt can be signed or unsigned
                    let int_value = const_int.value_s();
                    debug!("try_load_constant: found ConstInt with value {int_value}");
                    return Some(ClassicalValue::Int(int_value));
                }

                // Try to extract as ConstF64
                if let Some(const_f64) = value.get_custom_value::<ConstF64>() {
                    let float_value = const_f64.value();
                    debug!("try_load_constant: found ConstF64 with value {float_value}");
                    return Some(ClassicalValue::Float(float_value));
                }

                // Try to extract as ConstBool
                if let Some(const_bool) = value.get_custom_value::<ConstBool>() {
                    let bool_value = const_bool.value();
                    debug!("try_load_constant: found ConstBool with value {bool_value}");
                    return Some(ClassicalValue::Bool(bool_value));
                }

                debug!("try_load_constant: unrecognized const type");
            }
        }

        None
    }
}
