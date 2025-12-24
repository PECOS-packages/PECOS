# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Generator for PECOS QuantumCircuit format from SLR programs."""

from __future__ import annotations

from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.slr.gen_codes.generator import Generator


class QuantumCircuitGenerator(Generator):
    """Generate PECOS QuantumCircuit from SLR programs."""

    def __init__(self):
        """Initialize the QuantumCircuit generator."""
        self.circuit = QuantumCircuit()
        self.qubit_map = {}  # Maps (reg_name, index) to qubit_id
        self.next_qubit_id = 0
        self.current_tick = {}  # Accumulate operations for current tick
        self.current_scope = None
        self.permutation_map = {}

    def get_circuit(self) -> QuantumCircuit:
        """Get the generated QuantumCircuit.

        Returns:
            The generated QuantumCircuit object
        """
        # Flush any pending operations
        self._flush_tick()
        return self.circuit

    def get_output(self) -> str:
        """Get string representation of the circuit.

        Returns:
            String representation of the QuantumCircuit
        """
        return str(self.get_circuit())

    def enter_block(self, block):
        """Enter a new block scope."""
        previous_scope = self.current_scope
        self.current_scope = block

        block_name = type(block).__name__

        if block_name == "Main":
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
        """Generate QuantumCircuit for a block of operations.

        Parameters:
            block (Block): The block of operations to generate code for.
        """
        # Reset state
        self.circuit = QuantumCircuit()
        self.qubit_map = {}
        self.next_qubit_id = 0
        self.current_tick = {}
        self.permutation_map = {}

        # Generate the circuit
        self._handle_block(block)

        # Flush any remaining operations
        self._flush_tick()

    def _handle_block(self, block):
        """Handle a block of operations."""
        previous_scope = self.enter_block(block)

        block_name = type(block).__name__

        if block_name == "While":
            # While loops cannot be statically unrolled in QuantumCircuit format
            # This would require runtime evaluation which QuantumCircuit doesn't support
            msg = (
                "While loops cannot be converted to QuantumCircuit format as they require "
                "runtime condition evaluation. Use For or Repeat blocks with static bounds instead."
            )
            raise NotImplementedError(
                msg,
            )
        if block_name == "For":
            # For loops - unroll them properly
            self._flush_tick()
            # Check if we can determine the iteration count
            if hasattr(block, "iterable"):
                # For(i, range(n)) or For(i, iterable)
                if hasattr(block.iterable, "__iter__"):
                    # Unroll the loop for each iteration
                    iterations = list(block.iterable)
                    for _ in iterations:
                        for op in block.ops:
                            self._handle_op(op)
                else:
                    msg = f"Cannot unroll For loop with non-iterable: {block.iterable}"
                    raise ValueError(
                        msg,
                    )
            elif hasattr(block, "start") and hasattr(block, "stop"):
                # For(i, start, stop[, step])
                step = getattr(block, "step", 1)
                if not (
                    isinstance(block.start, int)
                    and isinstance(block.stop, int)
                    and isinstance(step, int)
                ):
                    msg = (
                        f"Cannot unroll For loop with non-integer bounds: "
                        f"start={block.start}, stop={block.stop}, step={step}"
                    )
                    raise ValueError(
                        msg,
                    )
                for _ in range(block.start, block.stop, step):
                    for op in block.ops:
                        self._handle_op(op)
            else:
                msg = f"For loop missing required attributes (iterable or start/stop): {block}"
                raise ValueError(
                    msg,
                )
        elif block_name == "Repeat":
            # Repeat blocks - unroll
            self._flush_tick()
            if not hasattr(block, "cond"):
                msg = f"Repeat block missing 'cond' attribute: {block}"
                raise ValueError(msg)
            if not isinstance(block.cond, int):
                msg = f"Cannot unroll Repeat block with non-integer count: {block.cond}"
                raise ValueError(
                    msg,
                )
            if block.cond < 0:
                msg = f"Repeat block has negative count: {block.cond}"
                raise ValueError(msg)
            for _ in range(block.cond):
                for op in block.ops:
                    self._handle_op(op)
        elif block_name == "If":
            # Conditional blocks - process both branches
            self._flush_tick()
            if hasattr(block, "then_block"):
                self._handle_block(block.then_block)
            if hasattr(block, "else_block") and block.else_block:
                self._flush_tick()
                self._handle_block(block.else_block)
        elif block_name == "Parallel":
            # Parallel operations stay in same tick
            for op in block.ops:
                self._handle_op(op, flush=False)
            # Flush after all parallel ops
            self._flush_tick()
        else:
            # Default block handling
            for op in block.ops:
                self._handle_op(op)

        self.current_scope = previous_scope
        self.exit_block(block)

    def _handle_op(self, op, *, flush: bool = True):
        """Handle a single operation."""
        op_class = type(op).__name__

        # Check if this is a Block-like object (has ops attribute and isn't a QGate)
        is_block = hasattr(op, "ops") and not hasattr(op, "is_qgate")

        if is_block:
            # Handle nested blocks
            if flush:
                self._flush_tick()
            self._handle_block(op)
            return

        # Map operations to QuantumCircuit gates
        if op_class == "Comment":
            # Comments don't appear in QuantumCircuit
            pass
        elif op_class == "Return":
            # Return is metadata for type checking, not a QuantumCircuit operation
            pass
        elif op_class == "Barrier":
            self._flush_tick()
        elif op_class == "Permute":
            # Handle permutation - would need to update qubit mapping
            self._flush_tick()
        elif op_class == "Vars":
            # Variable declarations already handled
            pass
        else:
            # Quantum operations (QGate objects)
            self._handle_quantum_op(op)
            if flush:
                # Each operation is its own tick unless in Parallel block
                self._flush_tick()

    def _handle_quantum_op(self, op):
        """Handle a quantum operation."""
        op_class = type(op).__name__

        # Get target qubits
        targets = self._get_targets(op)
        if not targets:
            return

        # Map SLR operations to QuantumCircuit gate names
        gate_map = {
            "H": "H",
            "X": "X",
            "Y": "Y",
            "Z": "Z",
            "SZ": "S",
            "S": "S",
            "SZdg": "SDG",
            "Sdg": "SDG",
            "T": "T",
            "Tdg": "TDG",
            "T_DAG": "TDG",
            "CX": "CX",
            "CNOT": "CX",
            "CY": "CY",
            "CZ": "CZ",
            "Measure": "Measure",
            "Prep": "RESET",
            "RX": "RX",
            "RY": "RY",
            "RZ": "RZ",
        }

        gate_name = gate_map.get(op_class, op_class)

        # Handle two-qubit gates specially
        if op_class in ["CX", "CNOT", "CY", "CZ"]:
            # For PECOS gates, qargs contains both qubits
            if len(targets) >= 2:
                # Take first two as control and target
                self._add_to_tick(gate_name, (targets[0], targets[1]))
            elif hasattr(op, "control") and hasattr(op, "target"):
                control_qubits = self._get_qubit_indices_from_target(op.control)
                target_qubits = self._get_qubit_indices_from_target(op.target)
                for c, t in zip(control_qubits, target_qubits):
                    self._add_to_tick(gate_name, (c, t))
        else:
            # Single qubit gates or measurements
            for qubit in targets:
                self._add_to_tick(gate_name, qubit)

    def _add_to_tick(self, gate_name, target):
        """Add a gate to the current tick."""
        if gate_name not in self.current_tick:
            self.current_tick[gate_name] = set()

        if isinstance(target, tuple):
            self.current_tick[gate_name].add(target)
        else:
            self.current_tick[gate_name].add(target)

    def _flush_tick(self):
        """Flush the current tick to the circuit."""
        if self.current_tick:
            self.circuit.append(dict(self.current_tick))
            self.current_tick = {}

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
        elif var_type == "Qubit":
            # Single qubit
            var_sym = var.sym if hasattr(var, "sym") else str(var)
            self.qubit_map[(var_sym, 0)] = self.next_qubit_id
            self.next_qubit_id += 1

    def _get_targets(self, op):
        """Get target qubit indices from an operation."""
        if hasattr(op, "qargs"):
            # PECOS gate operations use qargs
            return self._get_qubit_indices_from_target(op.qargs)
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
            # Qubit element from QReg (e.g., q[0])
            reg_sym = target.reg.sym if hasattr(target.reg, "sym") else None
            if reg_sym and hasattr(target, "index"):
                key = (reg_sym, target.index)
                if key in self.qubit_map:
                    indices.append(self.qubit_map[key])
        elif hasattr(target, "parent") and hasattr(target, "index"):
            # Alternative format (e.g., from other sources)
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

        return indices
