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

"""Tests for the ByteMessage Python bindings."""

from pecos_rslib import ByteMessage, ByteMessageBuilder


def test_byte_message_builder_basic() -> None:
    """Test basic ByteMessageBuilder functionality."""
    # Create a new builder
    builder = ByteMessageBuilder()
    builder.for_quantum_operations()

    # Add some gates
    builder.add_h(0)
    builder.add_cx(0, 1)

    # Build the message
    message = builder.build()

    # Parse the operations
    ops = message.parse_quantum_operations()

    # Verify the operations
    assert len(ops) == 2
    assert ops[0]["gate_type"] == "H"
    assert ops[0]["qubits"] == [0]
    assert ops[1]["gate_type"] == "CX"
    assert ops[1]["qubits"] == [0, 1]


def test_byte_message_bell_state() -> None:
    """Test creating a Bell state circuit with ByteMessage."""
    # Create a Bell state using the convenience class method
    # message = ByteMessage.create_bell_state()
    builder = ByteMessageBuilder()
    builder.for_quantum_operations()
    builder.add_h(0)
    builder.add_cx(0, 1)
    builder.add_measurement(0, 0)
    builder.add_measurement(1, 1)
    message = builder.build()

    # Parse the operations
    ops = message.parse_quantum_operations()

    # Verify the operations
    assert len(ops) == 4
    assert ops[0]["gate_type"] == "H"
    assert ops[0]["qubits"] == [0]
    assert ops[1]["gate_type"] == "CX"
    assert ops[1]["qubits"] == [0, 1]
    assert ops[2]["gate_type"] == "Measure"
    assert ops[2]["qubits"] == [0]
    # result_id is no longer included in parsed operations (measurements tracked by order)
    assert ops[3]["gate_type"] == "Measure"
    assert ops[3]["qubits"] == [1]
    # result_id is no longer included in parsed operations (measurements tracked by order)

    # Dump the batch for debugging
    batch_dump = message.dump_batch()
    assert "H" in batch_dump
    assert "CX" in batch_dump
    assert "Measure" in batch_dump


def test_byte_message_parameterized_gates() -> None:
    """Test working with parameterized gates in ByteMessage."""
    # Create a new builder
    builder = ByteMessage.quantum_operations_builder()

    # Add some parameterized gates
    builder.add_rz(0.5, 0)
    builder.add_rzz(0.25, 0, 1)
    builder.add_r1xy(0.3, 0.4, 2)

    # Build the message
    message = builder.build()

    # Parse the operations
    ops = message.parse_quantum_operations()

    import math

    # Verify the operations
    # Note: Rotation gate angles are stored in "angles" field (in radians)
    assert len(ops) == 3
    assert ops[0]["gate_type"] == "RZ"
    assert ops[0]["qubits"] == [0]
    assert math.isclose(ops[0]["angles"][0], 0.5)

    assert ops[1]["gate_type"] == "RZZ"
    assert ops[1]["qubits"] == [0, 1]
    assert math.isclose(ops[1]["angles"][0], 0.25)

    assert ops[2]["gate_type"] == "R1XY"
    assert ops[2]["qubits"] == [2]
    assert math.isclose(ops[2]["angles"][0], 0.3)
    assert math.isclose(ops[2]["angles"][1], 0.4)


def test_byte_message_builder_reuse() -> None:
    """Test reusing a ByteMessageBuilder."""
    # Create a new builder
    builder = ByteMessage.quantum_operations_builder()

    # Add some gates
    builder.add_h(0)
    builder.add_cx(0, 1)

    # Build the message
    message1 = builder.build()

    # Clear the builder
    builder.clear()
    builder.for_quantum_operations()

    # Add different gates
    builder.add_x(0)
    builder.add_y(1)

    # Build another message
    message2 = builder.build()

    # Parse both messages
    ops1 = message1.parse_quantum_operations()
    ops2 = message2.parse_quantum_operations()

    # Verify operations in first message
    assert len(ops1) == 2
    assert ops1[0]["gate_type"] == "H"
    assert ops1[1]["gate_type"] == "CX"

    # Verify operations in second message
    assert len(ops2) == 2
    assert ops2[0]["gate_type"] == "X"
    assert ops2[1]["gate_type"] == "Y"


def test_byte_message_with_measurements() -> None:
    """Test ByteMessage with measurements."""
    # Create a new builder
    builder = ByteMessage.quantum_operations_builder()

    # Add gates and measurements
    builder.add_h(0)
    builder.add_measurement(0, 42)

    # Build the message
    message = builder.build()

    # Parse the operations
    ops = message.parse_quantum_operations()

    # Verify the operations
    assert len(ops) == 2
    assert ops[0]["gate_type"] == "H"
    assert ops[1]["gate_type"] == "Measure"
    assert ops[1]["qubits"] == [0]
    # result_id is no longer included in parsed operations (measurements tracked by order)


def example_bell_state_experiment() -> None:
    """Example of using ByteMessage to create a Bell state experiment.

    This is a standalone example that demonstrates:
    1. Creating a ByteMessage encoding a Bell state experiment
    2. Dumping the message to see its interpretation
    """
    print("\n==== Bell State Experiment Example ====")

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
