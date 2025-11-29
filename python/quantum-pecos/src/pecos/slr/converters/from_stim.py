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

"""Convert Stim circuits to SLR format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.qeclib import qubit
from pecos.slr import Comment, CReg, Main, QReg, Repeat

if TYPE_CHECKING:
    import stim


def stim_to_slr(circuit: stim.Circuit) -> Main:
    """Convert a Stim circuit to SLR format.

    Args:
        circuit: A Stim circuit object

    Returns:
        An SLR Main block representing the circuit

    Note:
        - Stim's measurement record and detector/observable annotations are preserved as comments
        - Noise operations are converted to comments (SLR typically handles noise differently)
        - Some Stim-specific features may not have direct SLR equivalents
    """
    import stim  # noqa: F401, PLC0415, RUF100 - Lazy import for optional dependency, used for isinstance checks

    # Determine the number of qubits needed
    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        # Empty circuit
        return Main()

    # Track measurements for creating classical registers
    num_measurements = circuit.num_measurements

    # Create the quantum and classical registers
    ops = []
    q = QReg("q", num_qubits)
    ops.append(q)

    if num_measurements > 0:
        c = CReg("c", num_measurements)
        ops.append(c)
        measurement_count = 0
    else:
        c = None
        measurement_count = 0

    # Process each instruction in the circuit
    for instruction in circuit:
        ops_batch = _convert_instruction(instruction, q, c, measurement_count)
        if ops_batch:
            for op in ops_batch:
                ops.append(op)
                # Track measurement count
                if hasattr(op, "__class__") and op.__class__.__name__ == "Measure":
                    # Count measurements in this operation
                    if hasattr(op, "target") and hasattr(op.target, "__len__"):
                        measurement_count += len(op.target)
                    else:
                        measurement_count += 1

    return Main(*ops)


def _convert_instruction(instruction, q, c, measurement_offset):
    """Convert a single Stim instruction to SLR operations.

    Args:
        instruction: A Stim circuit instruction
        q: The quantum register
        c: The classical register (may be None)
        measurement_offset: Current offset in measurement record

    Returns:
        List of SLR operations
    """
    import stim  # noqa: F401, PLC0415, RUF100 - Lazy import for optional dependency, used for isinstance checks

    ops = []

    # Handle different instruction types
    if isinstance(instruction, stim.CircuitRepeatBlock):
        # Convert repeat block
        block_ops = []
        inner_measurement_offset = measurement_offset
        for inner_inst in instruction.body_copy():
            inner_ops = _convert_instruction(inner_inst, q, c, inner_measurement_offset)
            if inner_ops:
                block_ops.extend(inner_ops)
                # Update measurement offset for inner block
                for op in inner_ops:
                    if hasattr(op, "__class__") and op.__class__.__name__ == "Measure":
                        if hasattr(op, "target") and hasattr(op.target, "__len__"):
                            inner_measurement_offset += len(op.target)
                        else:
                            inner_measurement_offset += 1

        if block_ops:
            # Create repeat block
            repeat = Repeat(instruction.repeat_count)
            repeat.block(*block_ops)
            ops.append(repeat)
    else:
        # Regular instruction
        gate_name = instruction.name.upper()
        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()

        # Map Stim gates to SLR/PECOS operations
        converted = _map_gate(gate_name, targets, args, q, c, measurement_offset)
        if converted:
            ops.extend(converted)

    return ops


def _map_gate(gate_name, targets, args, q, c, measurement_offset):
    """Map a Stim gate to SLR operations.

    Args:
        gate_name: Name of the Stim gate
        targets: List of target qubits/bits
        args: Gate arguments (e.g., rotation angles, error probabilities)
        q: Quantum register
        c: Classical register
        measurement_offset: Current offset in measurement record

    Returns:
        List of SLR operations
    """
    ops = []

    # Extract qubit indices from targets
    qubit_targets = []
    for t in targets:
        if hasattr(t, "value"):
            # Regular qubit target
            if not t.is_measurement_record_target and not t.is_sweep_bit_target:
                qubit_targets.append(t.value)
        elif isinstance(t, int) and t >= 0:
            qubit_targets.append(t)

    # Map common gates
    if gate_name == "H":
        ops.extend(qubit.H(q[idx]) for idx in qubit_targets)
    elif gate_name == "X":
        ops.extend(qubit.X(q[idx]) for idx in qubit_targets)
    elif gate_name == "Y":
        ops.extend(qubit.Y(q[idx]) for idx in qubit_targets)
    elif gate_name == "Z":
        ops.extend(qubit.Z(q[idx]) for idx in qubit_targets)
    elif gate_name == "S":
        ops.extend(qubit.SZ(q[idx]) for idx in qubit_targets)
    elif gate_name == "S_DAG" or gate_name == "SDG":
        ops.extend(qubit.SZdg(q[idx]) for idx in qubit_targets)
    elif gate_name == "T":
        ops.extend(qubit.T(q[idx]) for idx in qubit_targets)
    elif gate_name == "T_DAG" or gate_name == "TDG":
        ops.extend(qubit.Tdg(q[idx]) for idx in qubit_targets)
    elif gate_name in ["CX", "CNOT"]:
        # Process pairs of qubits
        ops.extend(
            qubit.CX(q[qubit_targets[i]], q[qubit_targets[i + 1]])
            for i in range(0, len(qubit_targets), 2)
            if i + 1 < len(qubit_targets)
        )
    elif gate_name == "CY":
        ops.extend(
            qubit.CY(q[qubit_targets[i]], q[qubit_targets[i + 1]])
            for i in range(0, len(qubit_targets), 2)
            if i + 1 < len(qubit_targets)
        )
    elif gate_name == "CZ":
        ops.extend(
            qubit.CZ(q[qubit_targets[i]], q[qubit_targets[i + 1]])
            for i in range(0, len(qubit_targets), 2)
            if i + 1 < len(qubit_targets)
        )
    elif gate_name == "SWAP":
        for i in range(0, len(qubit_targets), 2):
            if i + 1 < len(qubit_targets):
                # Decompose SWAP into 3 CNOTs
                ops.append(qubit.CX(q[qubit_targets[i]], q[qubit_targets[i + 1]]))
                ops.append(qubit.CX(q[qubit_targets[i + 1]], q[qubit_targets[i]]))
                ops.append(qubit.CX(q[qubit_targets[i]], q[qubit_targets[i + 1]]))
    elif gate_name in ["M", "MZ"]:
        # Measurement
        if c is not None:
            for i, idx in enumerate(qubit_targets):
                if measurement_offset + i < len(c):
                    ops.append(qubit.Measure(q[idx]) > c[measurement_offset + i])
    elif gate_name in ["MX", "MY"]:
        # Basis measurements - add basis change before measurement
        if c is not None:
            for i, idx in enumerate(qubit_targets):
                if measurement_offset + i < len(c):
                    if gate_name == "MX":
                        ops.append(qubit.H(q[idx]))
                    else:  # MY
                        ops.append(qubit.SZdg(q[idx]))
                        ops.append(qubit.H(q[idx]))
                    ops.append(qubit.Measure(q[idx]) > c[measurement_offset + i])
    elif gate_name in ["R", "RZ"]:
        # Reset
        ops.extend(qubit.Prep(q[idx]) for idx in qubit_targets)
    elif gate_name in ["RX", "RY"]:
        # Reset in X or Y basis
        for idx in qubit_targets:
            ops.append(qubit.Prep(q[idx]))
            if gate_name == "RX":
                ops.append(qubit.H(q[idx]))
            else:  # RY
                ops.append(qubit.H(q[idx]))
                ops.append(qubit.SZ(q[idx]))
    elif gate_name == "TICK":
        # Timing boundary - add as comment
        ops.append(Comment("TICK"))
    elif "ERROR" in gate_name or gate_name.startswith("E(") or gate_name == "E":
        # Noise operations - add as comment
        error_prob = args[0] if args else 0
        ops.append(
            Comment(f"Noise: {gate_name}({error_prob}) on qubits {qubit_targets}"),
        )
    elif gate_name in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
        # Annotations - add as comment
        ops.append(Comment(f"{gate_name} {targets}"))
    elif gate_name == "QUBIT_COORDS":
        # Coordinate annotation
        ops.append(Comment(f"QUBIT_COORDS {targets} {args}"))
    else:
        # Unknown gate - add as comment
        ops.append(Comment(f"Unsupported Stim gate: {gate_name} {targets} {args}"))

    return ops
