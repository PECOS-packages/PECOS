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

from pecos.slr import Comment, CReg, Main, QReg, Repeat
from pecos.slr.qeclib import qubit

if TYPE_CHECKING:
    import stim


_MEASUREMENT_BASES = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}


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
        ops.extend(_convert_instruction(instruction, q, c, measurement_count))
        measurement_count += instruction.num_measurements

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
        body = instruction.body_copy()
        block_ops = []
        inner_measurement_offset = measurement_offset
        iterations = instruction.repeat_count if body.num_measurements else 1
        for _ in range(iterations):
            for inner_inst in body:
                block_ops.extend(_convert_instruction(inner_inst, q, c, inner_measurement_offset))
                inner_measurement_offset += inner_inst.num_measurements

        if body.num_measurements:
            ops.extend(block_ops)
        elif block_ops:
            repeat = Repeat(instruction.repeat_count)
            repeat.block(*block_ops)
            ops.append(repeat)
    else:
        # Regular instruction
        gate_name = instruction.name.upper()
        if instruction.num_measurements and gate_name not in _MEASUREMENT_BASES:
            msg = f"Unsupported measurement-producing Stim gate {gate_name}: {instruction}"
            raise NotImplementedError(msg)
        targets = instruction.targets_copy()
        gate_data = stim.gate_data(gate_name)
        # Annotations such as DETECTOR reference records legitimately; on a
        # unitary gate a record or sweep target is a classical control.
        if gate_data.is_unitary and any(t.is_measurement_record_target or t.is_sweep_bit_target for t in targets):
            msg = f"Unsupported classical-feedback or sweep-bit targets for Stim gate {gate_name}: {instruction}"
            raise NotImplementedError(msg)
        args = instruction.gate_args_copy()
        is_pure_noise = gate_data.is_noisy_gate and not gate_data.produces_measurements
        is_annotation = not (
            gate_data.is_unitary or gate_data.is_noisy_gate or gate_data.produces_measurements or gate_data.is_reset
        )
        if is_pure_noise or (gate_data.is_noisy_gate and args):
            ops.append(Comment(f"Noise: {instruction}"))

        if not is_pure_noise:
            converted = _map_gate(gate_name, targets, args, q, c, measurement_offset)
            if converted is not None:
                ops.extend(converted)
            elif is_annotation:
                ops.append(Comment(f"{gate_name} {targets} {args}"))
            else:
                msg = f"Unsupported Stim gate {gate_name}: {instruction}"
                raise NotImplementedError(msg)

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

    qubit_targets = [target.value for target in targets]

    # Map common gates
    if gate_name in ["I", "II"]:
        pass  # Identity: nothing to emit
    elif gate_name == "H":
        ops.extend(qubit.H(q[idx]) for idx in qubit_targets)
    elif gate_name == "X":
        ops.extend(qubit.X(q[idx]) for idx in qubit_targets)
    elif gate_name == "Y":
        ops.extend(qubit.Y(q[idx]) for idx in qubit_targets)
    elif gate_name == "Z":
        ops.extend(qubit.Z(q[idx]) for idx in qubit_targets)
    elif gate_name == "S":
        ops.extend(qubit.SZ(q[idx]) for idx in qubit_targets)
    elif gate_name == "S_DAG":
        ops.extend(qubit.SZdg(q[idx]) for idx in qubit_targets)
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
    elif gate_name in _MEASUREMENT_BASES:
        basis = _MEASUREMENT_BASES[gate_name]
        resets = gate_name.startswith("MR")
        flip = qubit.Z if basis == "X" else qubit.X
        for i, target in enumerate(targets):
            idx = target.value
            if target.is_inverted_result_target:
                ops.append(flip(q[idx]))
            if basis == "Y":
                ops.append(qubit.SZdg(q[idx]))
            if basis in ["X", "Y"]:
                ops.append(qubit.H(q[idx]))
            ops.append(qubit.Measure(q[idx]) > c[measurement_offset + i])
            if basis in ["X", "Y"]:
                ops.append(qubit.H(q[idx]))
            if basis == "Y":
                ops.append(qubit.SZ(q[idx]))
            if resets:
                # The reset replaces the trailing inversion Pauli.
                ops.extend(_reset_in_basis(q[idx], basis))
            elif target.is_inverted_result_target:
                ops.append(flip(q[idx]))
    elif gate_name in ["R", "RX", "RY"]:
        basis = {"R": "Z", "RX": "X", "RY": "Y"}[gate_name]
        for idx in qubit_targets:
            ops.extend(_reset_in_basis(q[idx], basis))
    elif gate_name == "TICK":
        # Timing boundary - add as comment
        ops.append(Comment("TICK"))
    elif gate_name in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
        # Annotations - add as comment
        ops.append(Comment(f"{gate_name} {targets}"))
    elif gate_name == "QUBIT_COORDS":
        # Coordinate annotation
        ops.append(Comment(f"QUBIT_COORDS {targets} {args}"))
    else:
        return None

    return ops


def _reset_in_basis(target, basis):
    """Reset a qubit to the +1 eigenstate of the given basis."""
    ops = [qubit.PZ(target)]
    if basis in ["X", "Y"]:
        ops.append(qubit.H(target))
    if basis == "Y":
        ops.append(qubit.SZ(target))
    return ops
