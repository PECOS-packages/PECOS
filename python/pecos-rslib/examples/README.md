# PECOS Python API Examples

This directory contains examples of using various PECOS Python APIs:

1. **ByteMessage API** - Low-level API for building quantum circuit messages
2. **QASM Simulation API** - High-level API for running QASM quantum circuits with noise models

## Bell State Example

`bell_state_example.py` demonstrates how to create a Bell state circuit using the ByteMessage API. It shows:

1. Creating a `ByteMessageBuilder` and configuring it for quantum operations
2. Adding Hadamard and CNOT gates to create a Bell state
3. Adding measurement operations
4. Building the message
5. Dumping and parsing the message contents

## Bell State Simulator Example

`bell_state_simulator.py` demonstrates running a Bell state circuit using the Python API. It shows:

1. Creating a Bell state circuit using `ByteMessageBuilder`
2. Creating a `StateVectorSimulator` with 2 qubits
3. Processing the Bell state circuit on the simulator
4. Running multiple shots to analyze measurement correlations
5. Running a GHZ state on three qubits

## QASM Simulation Example

`qasm_simulation_examples.py` demonstrates the QASM simulation API with comprehensive examples:

1. Creating and measuring Bell states with various noise models
2. GHZ state preparation with custom depolarizing noise
3. Biased measurement noise demonstration
4. Comparing different quantum engines (StateVector vs SparseStabilizer)
5. Using the builder pattern for reusable simulations
6. Handling large quantum registers (>64 qubits)
7. Parallel execution with multiple workers

## Running the Examples

To run the examples:

```bash
# Make sure you're in the PECOS root directory
cd python/pecos-rslib
python examples/bell_state_example.py
python examples/bell_state_simulator.py
python examples/qasm_simulation_examples.py
```

## API Overview

The ByteMessage Python API provides the following classes:

### `ByteMessage`

The main class for working with byte-encoded quantum messages.

#### Class Methods

- `ByteMessage.builder()`: Create a new `ByteMessageBuilder`
- `ByteMessage.quantum_operations_builder()`: Create a builder pre-configured for quantum operations
- `ByteMessage.outcomes_builder()`: Create a builder pre-configured for measurement outcomes
- `ByteMessage.create_bell_state()`: Create a pre-built Bell state circuit
- `ByteMessage.create_flush()`: Create a flush message
- `ByteMessage.create_empty()`: Create an empty message

#### Instance Methods

- `as_bytes()`: Get the message as bytes
- `is_empty()`: Check if the message is empty
- `parse_quantum_operations()`: Parse quantum operations from the message
- `dump_batch()`: Dump the batch contents for debugging
- `measurement_results()`: Get measurement results as a list of (result_id, outcome) tuples

### `ByteMessageBuilder`

Builder class for creating `ByteMessage` instances.

#### Constructor

- `ByteMessageBuilder()`: Create a new message builder

#### Configuration Methods

- `for_quantum_operations()`: Configure the builder for quantum operations
- `for_outcomes()`: Configure the builder for measurement outcomes

#### Gate Methods

- `x([q0, q1, ...])`: Add X gate(s)
- `y([q0, q1, ...])`: Add Y gate(s)
- `z([q0, q1, ...])`: Add Z gate(s)
- `h([q0, q1, ...])`: Add H gate(s)
- `cx([(c0, t0), ...])`: Add CX (CNOT) gate(s)
- `cy([(c0, t0), ...])`: Add CY gate(s)
- `cz([(c0, t0), ...])`: Add CZ gate(s)
- `rz(theta, [q0, q1, ...])`: Add RZ gate(s)
- `rx(theta, [q0, q1, ...])`: Add RX gate(s)
- `ry(theta, [q0, q1, ...])`: Add RY gate(s)
- `rzz(theta, [(q0, q1), ...])`: Add RZZ gate(s)
- `szz([(q0, q1), ...])`: Add SZZ gate(s)
- `r1xy(theta, phi, [q0, q1, ...])`: Add R1XY gate(s)
- `u(theta, phi, lambda_, [q0, q1, ...])`: Add U gate(s)
- `mz([q0, q1, ...])`: Add Z-basis measurement(s)
- `pz([q0, q1, ...])`: Add PZ (preparation/reset) gate(s)

#### Builder Methods

- `build()`: Build the message and return a `ByteMessage`
- `clear()`: Clear the builder and reset to initial state
- `reset()`: Reset the builder while preserving capacity

### `StateVectorSimulator`

Simulator class for executing quantum circuits encoded as `ByteMessage` objects.

#### Constructor

- `StateVectorSimulator(num_qubits)`: Create a new simulator with the specified number of qubits

#### Simulator Methods

- `reset()`: Reset the simulator state
- `process(message)`: Process a `ByteMessage` circuit and return the measurement results
- `run_bell_state_experiment()`: Execute a Bell state experiment and return the measurement results
- `run_circuit_with_shots(message, shots=1000)`: Execute a circuit multiple times and return the measurement results for each shot
