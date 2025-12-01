#!/usr/bin/env python3
# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Example of using ByteMessage to create a Bell state experiment."""

import os
import sys

# Add the parent directory to the path to import _pecos_rslib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _pecos_rslib import ByteMessage


def bell_state_example() -> None:
    """Demonstrate how to create a Bell state experiment using ByteMessage."""
    print("==== Bell State Experiment Example ====")

    # Create a builder for quantum operations
    builder = ByteMessage.quantum_operations_builder()

    print("Building Bell state circuit...")

    # Add gates to create a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    print("- Adding Hadamard gate on qubit 0")
    builder.add_h(0)

    print("- Adding CNOT gate with control=0, target=1")
    builder.add_cx(0, 1)

    # Add measurement operations
    print("- Adding measurement on qubit 0 (result_id=0)")
    builder.add_measurement(0, 0)

    print("- Adding measurement on qubit 1 (result_id=1)")
    builder.add_measurement(1, 1)

    # Build the message
    message = builder.build()

    print("\nDumping the message contents:")
    dump = message.dump_batch()
    print(dump)

    print("\nParsing quantum operations:")
    operations = message.parse_quantum_operations()
    for i, op in enumerate(operations):
        print(f"Operation {i}:")
        for key, value in op.items():
            print(f"  {key} = {value}")

    print("\n==== End of Example ====")

    return message


def build_custom_message() -> None:
    """Build a custom quantum circuit using ByteMessage."""
    print("\n==== Custom Circuit Example ====")

    # Create a builder for quantum operations
    builder = ByteMessage.quantum_operations_builder()

    # Add a set of gates to demonstrate various operations
    print("Building custom circuit...")

    # Single-qubit gates
    builder.add_x(0)  # X gate on qubit 0
    builder.add_y(1)  # Y gate on qubit 1
    builder.add_z(2)  # Z gate on qubit 2
    builder.add_h(3)  # H gate on qubit 3

    # Parameterized gates
    builder.add_rz(0.5, 0)  # RZ(0.5) on qubit 0
    builder.add_r1xy(0.1, 0.2, 1)  # R1XY(0.1, 0.2) on qubit 1

    # Two-qubit gates
    builder.add_cx(0, 1)  # CNOT with control=0, target=1
    builder.add_szz(2, 3)  # SZZ on qubits 2 and 3
    builder.add_rzz(0.25, 0, 2)  # RZZ(0.25) on qubits 0 and 2

    # Measurements
    builder.add_measurement(0, 10)  # Measure qubit 0, result_id=10
    builder.add_measurement(1, 11)  # Measure qubit 1, result_id=11
    builder.add_measurement(2, 12)  # Measure qubit 2, result_id=12
    builder.add_measurement(3, 13)  # Measure qubit 3, result_id=13

    # Build the message
    message = builder.build()

    print("\nDumping the message contents:")
    dump = message.dump_batch()
    print(dump)

    print("\n==== End of Custom Example ====")

    return message


if __name__ == "__main__":
    bell_state_example()
    build_custom_message()
