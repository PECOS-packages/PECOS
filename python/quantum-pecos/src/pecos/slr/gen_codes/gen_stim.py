# Copyright 2025 PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Generator for Stim circuit format from SLR programs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.gen_codes.generator import Generator

if TYPE_CHECKING:
    import stim


class StimGenerator(Generator):
    """Generate Stim circuits from SLR programs."""

    def __init__(self, *, add_comments: bool = True):
        """Initialize the Stim generator.

        Args:
            add_comments: Whether to add comments for unsupported operations
        """
        self.circuit = None  # Will be initialized when needed
        self.qubit_map = {}  # Maps (reg_name, index) to qubit_id
        self.next_qubit_id = 0
        self.creg_map = {}  # Tracks classical registers
        self.measurement_count = 0
        self.add_comments = add_comments
        self.current_scope = None
        self.permutation_map = {}

    def get_circuit(self) -> stim.Circuit:
        """Get the generated Stim circuit.

        Returns:
            The generated Stim Circuit object
        """
        if self.circuit is None:
            import stim

            self.circuit = stim.Circuit()
        return self.circuit

    def get_output(self) -> str:
        """Get the string representation of the generated circuit.

        Returns:
            String representation of the Stim circuit
        """
        return str(self.get_circuit())

    def enter_block(self, block):
        """Enter a new block scope."""
        previous_scope = self.current_scope
        self.current_scope = block

        block_name = type(block).__name__

        if block_name == "Main":
            # Initialize Stim circuit if not already done
            if self.circuit is None:
                import stim

                self.circuit = stim.Circuit()

            # Process variable declarations
            for var in block.vars:
                self._process_var_declaration(var)

            # Process any Vars operations in ops
            for op in block.ops:
                if type(op).__name__ == "Vars":
                    for var in op.vars:
                        self._process_var_declaration(var)

        return previous_scope

    def exit_block(self, block) -> None:
        """Exit a block scope."""

    def generate_block(self, block):
        """Generate Stim circuit for a block of operations.

        Parameters:
            block (Block): The block of operations to generate code for.
        """
        # Initialize the circuit and maps
        if self.circuit is None:
            import stim

            self.circuit = stim.Circuit()

        self.qubit_map = {}
        self.next_qubit_id = 0
        self.creg_map = {}
        self.measurement_count = 0
        self.permutation_map = {}

        # Generate the Stim circuit
        self._handle_block(block)

    def _handle_block(self, block):
        """Handle a block of operations."""
        previous_scope = self.enter_block(block)

        block_name = type(block).__name__

        if block_name == "While":
            # While loops can't be directly represented
            if self.add_comments:
                self.circuit.append("TICK")  # Mark boundary
            # Process body once as approximation
            self._handle_block(block)
        elif block_name == "For":
            # For loops - unroll if possible
            if hasattr(block, "count") and isinstance(block.count, int):
                # Static count - can unroll
                for _ in range(block.count):
                    for op in block.ops:
                        self._handle_op(op)
            else:
                # Dynamic count - process once
                if self.add_comments:
                    self.circuit.append("TICK")
                for op in block.ops:
                    self._handle_op(op)
        elif block_name == "Repeat":
            # Repeat blocks can be represented in Stim
            # Repeat uses 'cond' attribute for the count
            repeat_count = getattr(block, "cond", getattr(block, "count", 1))
            if repeat_count > 0:
                import stim

                sub_circuit = stim.Circuit()
                # Temporarily swap circuits to build repeat block
                original_circuit = self.circuit
                self.circuit = sub_circuit
                for op in block.ops:
                    self._handle_op(op)
                self.circuit = original_circuit
                # Add repeat block using CircuitRepeatBlock
                if len(sub_circuit) > 0:
                    self.circuit.append(
                        stim.CircuitRepeatBlock(repeat_count, sub_circuit),
                    )
        elif block_name == "If":
            # Conditional blocks - add tick and process
            if self.add_comments:
                self.circuit.append("TICK")
            if hasattr(block, "then_block"):
                self._handle_block(block.then_block)
            if hasattr(block, "else_block") and block.else_block:
                if self.add_comments:
                    self.circuit.append("TICK")
                self._handle_block(block.else_block)
        elif block_name == "Parallel":
            # Process parallel operations
            for op in block.ops:
                self._handle_op(op)
        else:
            # Default block handling
            for op in block.ops:
                self._handle_op(op)

        self.current_scope = previous_scope
        self.exit_block(block)

    def _handle_op(self, op):
        """Handle a single operation."""
        op_class = type(op).__name__

        # Handle nested blocks
        if hasattr(op, "ops") and not hasattr(op, "is_qgate"):
            self._handle_block(op)
            return

        # Map operations to Stim gates
        if op_class == "Comment":
            # Comments can't be directly added via API
            pass
        elif op_class == "Return":
            # Return is metadata for type checking, not a Stim operation
            pass
        elif op_class == "Barrier":
            self.circuit.append("TICK")
        elif op_class == "Permute":
            # Handle permutation - update mapping
            self._handle_permutation(op)
        elif op_class == "Vars":
            # Variable declarations - already handled
            pass
        else:
            # Quantum operations
            self._handle_quantum_op(op)

    def _handle_quantum_op(self, op):
        """Handle a quantum operation."""
        op_class = type(op).__name__

        # Get qubit indices
        qubits = self._get_qubit_indices(op)
        if not qubits:
            return

        # Map to Stim operations
        if op_class == "H":
            self.circuit.append_operation("H", qubits)
        elif op_class == "X":
            self.circuit.append_operation("X", qubits)
        elif op_class == "Y":
            self.circuit.append_operation("Y", qubits)
        elif op_class == "Z":
            self.circuit.append_operation("Z", qubits)
        elif op_class in ["SZ", "S"]:
            self.circuit.append_operation("S", qubits)
        elif op_class in ["SZdg", "Sdg"]:
            self.circuit.append_operation("S_DAG", qubits)
        elif op_class == "T":
            self.circuit.append_operation("T", qubits)
        elif op_class in ["Tdg", "T_DAG"]:
            self.circuit.append_operation("T_DAG", qubits)
        elif op_class in ["CX", "CNOT"]:
            self._handle_two_qubit_gate("CX", op)
        elif op_class == "CY":
            self._handle_two_qubit_gate("CY", op)
        elif op_class == "CZ":
            self._handle_two_qubit_gate("CZ", op)
        elif op_class == "Measure":
            self.circuit.append_operation("M", qubits)
            self.measurement_count += len(qubits)
        elif op_class == "Prep":
            self.circuit.append_operation("R", qubits)
        elif op_class in ["RX", "RY", "RZ"]:
            # Rotation gates - add as parameterized gates if supported
            if hasattr(op, "angle"):
                # For now, just add a TICK as placeholder
                self.circuit.append("TICK")
            else:
                # Reset in basis
                if op_class == "RX":
                    self.circuit.append_operation("RX", qubits)
                elif op_class == "RY":
                    self.circuit.append_operation("RY", qubits)
                else:
                    self.circuit.append_operation("R", qubits)
        else:
            # Unknown operation
            if self.add_comments:
                self.circuit.append("TICK")

    def _handle_two_qubit_gate(self, gate_name, op):
        """Handle two-qubit gates."""
        qubits = self._get_qubit_indices(op)
        if len(qubits) >= 2:
            # For gates like CX, CY, CZ, the first qubit is control, second is target
            self.circuit.append_operation(gate_name, [qubits[0], qubits[1]])
        elif hasattr(op, "control") and hasattr(op, "target"):
            control_qubits = self._get_qubit_indices_from_target(op.control)
            target_qubits = self._get_qubit_indices_from_target(op.target)
            if control_qubits and target_qubits:
                for c, t in zip(control_qubits, target_qubits):
                    self.circuit.append_operation(gate_name, [c, t])
        elif hasattr(op, "targets"):
            qubits = self._get_qubit_indices(op)
            # Process pairs
            for i in range(0, len(qubits) - 1, 2):
                self.circuit.append_operation(gate_name, [qubits[i], qubits[i + 1]])

    def _handle_permutation(self, op):
        """Handle Permute operation by updating qubit mappings.

        Args:
            op: The permutation operation to handle.
                Currently unused but kept for interface consistency.
        """
        # TODO: Implement proper permutation handling by analyzing op
        # and updating the qubit_map accordingly
        _ = op  # Mark as intentionally unused for now
        if self.add_comments:
            self.circuit.append("TICK")

    def _process_var_declaration(self, var):
        """Process a variable declaration."""
        if var is None:
            return

        var_type = type(var).__name__

        if var_type == "QReg":
            # Allocate qubits for quantum register
            for i in range(var.size):
                self.qubit_map[(var.sym, i)] = self.next_qubit_id
                self.next_qubit_id += 1
        elif var_type == "CReg":
            # Track classical register
            self.creg_map[var.sym] = var.size
        elif var_type == "Qubit":
            # Single qubit
            self.qubit_map[(var.sym, 0)] = self.next_qubit_id
            self.next_qubit_id += 1
        elif var_type == "Bit":
            # Single classical bit
            self.creg_map[var.name] = 1

    def _get_qubit_indices(self, op):
        """Get qubit indices from an operation."""
        if hasattr(op, "qargs"):
            # QGate operations use qargs
            indices = []
            for arg in op.qargs:
                # Check if arg is a tuple of qubits (for multi-qubit gates)
                if isinstance(arg, tuple):
                    # Unwrap tuple and process each qubit
                    for sub_arg in arg:
                        if hasattr(sub_arg, "reg") and hasattr(sub_arg, "index"):
                            key = (sub_arg.reg.sym, sub_arg.index)
                            if key in self.qubit_map:
                                indices.append(self.qubit_map[key])
                elif hasattr(arg, "reg") and hasattr(arg, "index"):
                    # Individual Qubit object
                    key = (arg.reg.sym, arg.index)
                    if key in self.qubit_map:
                        indices.append(self.qubit_map[key])
                elif hasattr(arg, "sym") and hasattr(arg, "size"):
                    # Full QReg object
                    for i in range(arg.size):
                        key = (arg.sym, i)
                        if key in self.qubit_map:
                            indices.append(self.qubit_map[key])
            return indices
        if hasattr(op, "target"):
            return self._get_qubit_indices_from_target(op.target)
        if hasattr(op, "targets"):
            return self._get_qubit_indices_from_target(op.targets)
        return []

    def _get_qubit_indices_from_target(self, target):
        """Extract qubit indices from a target."""
        indices = []

        if hasattr(target, "__iter__") and not isinstance(target, str):
            # Array of targets
            for t in target:
                indices.extend(self._get_qubit_indices_from_target(t))
        elif hasattr(target, "reg") and hasattr(target, "index"):
            # Qubit object with reg and index
            key = (target.reg.sym, target.index)
            if key in self.qubit_map:
                indices.append(self.qubit_map[key])
        elif hasattr(target, "parent") and hasattr(target, "index"):
            # QReg element (legacy support)
            parent_sym = target.parent.sym if hasattr(target.parent, "sym") else None
            if (
                parent_sym
                and hasattr(target, "index")
                and isinstance(target.index, int)
            ):
                key = (parent_sym, target.index)
                if key in self.qubit_map:
                    indices.append(self.qubit_map[key])
        elif hasattr(target, "sym"):
            # Full register or single qubit
            indices.extend(
                self.qubit_map[key] for key in self.qubit_map if key[0] == target.sym
            )
        elif hasattr(target, "name"):
            # Legacy support for name attribute
            indices.extend(
                self.qubit_map[key] for key in self.qubit_map if key[0] == target.name
            )

        return indices
