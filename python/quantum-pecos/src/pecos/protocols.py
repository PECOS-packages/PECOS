# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Common protocols used throughout PECOS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence
    from typing import Any

    from pecos.circuits import LogicalCircuit, QuantumCircuit
    from pecos.error_models.class_errors_circuit import ErrorCircuits
    from pecos.error_models.parent_class_error_gen import ParentErrorModel
    from pecos.misc.symbol_library import JSONDict
    from pecos.typing import (
        ErrorParams,
        LocationSet,
        OutputDict,
        QECCGateParams,
        QECCInstrParams,
    )


class Decoder(Protocol):
    """Protocol for decoder objects."""

    def decode(self, syndrome: set[int]) -> QuantumCircuit | LogicalCircuit:
        """Decode syndrome measurements to determine error correction.

        Args:
            syndrome: Set of syndrome measurement outcomes.

        Returns:
            Quantum or logical circuit implementing the error correction.
        """
        ...


class CCOPProtocol(Protocol):
    """Protocol for CCOP (Classical Co-processor) objects."""

    def exec(self, func_name: str, args: list[int | float | bool]) -> int | None:
        """Execute a function on the classical co-processor.

        Args:
            func_name: Name of the function to execute.
            args: List of arguments to pass to the function.

        Returns:
            Optional integer result from the function execution.
        """
        ...


class SimulatorState(Protocol):
    """Protocol for simulator state objects."""

    # States typically have methods like apply_gate, measure, etc.


class CircuitRunner(Protocol):
    """Protocol for circuit runner objects."""

    def run(
        self,
        state: SimulatorState,
        circuit: QuantumCircuit | LogicalCircuit,
    ) -> tuple[SimulatorState, dict[str, int | list[int]]]:
        """Run a quantum circuit on the given state.

        Args:
            state: Initial quantum state.
            circuit: Quantum or logical circuit to execute.

        Returns:
            Tuple of final state and measurement results.
        """
        ...


class EngineRunner(Protocol):
    """Protocol for engine runner objects."""

    debug: bool
    ccop: CCOPProtocol | None
    circuit: QuantumCircuit


class CircuitInspector(Protocol):
    """Protocol for circuit inspection functionality."""

    def analyze(
        self,
        tick_circuit: QuantumCircuit,
        time: int,
        output: OutputDict,
    ) -> None:
        """Analyze a circuit at a specific time tick.

        Args:
            tick_circuit: Quantum circuit for the current time tick.
            time: Current time step.
            output: Output dictionary to store analysis results.
        """
        ...


class MachineProtocol(Protocol):
    """Protocol for machine implementations.

    This protocol replaces the Machine ABC and defines the interface that all
    machine implementations must follow.
    """

    machine_params: dict | None
    num_qubits: int | None
    metadata: dict | None
    pos: dict | None
    leaked_qubits: set[int]  # Used in leakage noise models
    lost_qubits: set[int]  # Used for tracking lost qubits
    qubit_set: set[int]  # Set of all qubit indices

    def reset(self) -> None:
        """Reset state to initialization state."""
        ...

    def init(self, num_qubits: int | None = None) -> None:
        """Initialize the machine with the given number of qubits."""
        ...

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        ...

    def process(self, op_buffer: list) -> list:
        """Process a buffer of operations."""
        ...


class ErrorModelProtocol(Protocol):
    """Protocol for error model implementations.

    This protocol replaces the ErrorModel ABC and defines the interface that all
    error model implementations must follow.
    """

    error_params: dict
    machine: MachineProtocol | None
    num_qubits: int | None

    def reset(self) -> None:
        """Reset state to initialization state."""
        ...

    def init(self, num_qubits: int, machine: MachineProtocol | None = None) -> None:
        """Initialize the error model with the given number of qubits and optional machine."""
        ...

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        ...

    def process(self, qops: list, call_back: Callable | None = None) -> list | None:
        """Process quantum operations and potentially apply errors."""
        ...


class OpProcessorProtocol(Protocol):
    """Protocol for operation processor implementations.

    This protocol replaces the OpProcessor ABC and defines the interface that all
    operation processor implementations must follow.
    """

    def reset(self) -> None:
        """Reset state to initialization state."""
        ...

    def init(self) -> None:
        """Initialize the operation processor."""
        ...

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot."""
        ...

    def process(self, buffered_ops: list) -> list:
        """Process a buffer of operations."""
        ...

    def process_meas(self, measurements: dict) -> dict:
        """Process measurement operations."""
        ...


class ClassicalInterpreterProtocol(Protocol):
    """Protocol for classical interpreter implementations.

    This protocol replaces the ClassicalInterpreter ABC and defines the interface that all
    classical interpreter implementations must follow.
    """

    program: Any
    foreign_obj: Any

    def reset(self) -> None:
        """Reset state to initialization state."""
        ...

    def init(
        self,
        program: str | dict | QuantumCircuit,
        foreign_classical_obj: object | None = None,
    ) -> int:
        """Initialize the interpreter with a program and optional foreign object."""
        ...

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot."""
        ...

    def execute(self, sequence: Sequence | None) -> Generator:
        """Execute the program with an optional sequence of inputs."""
        ...

    def receive_results(self, qsim_results: list[dict]) -> None:
        """Receive results from quantum simulation."""
        ...

    def results(self) -> dict:
        """Dumps program final results."""
        ...


class ForeignObjectProtocol(Protocol):
    """Protocol for foreign object implementations.

    This protocol replaces the ForeignObject ABC and defines the interface that all
    foreign object implementations must follow.
    """

    def init(self) -> None:
        """Initialize object before a set of simulations."""
        ...

    def new_instance(self) -> None:
        """Create new instance/internal state."""
        ...

    def get_funcs(self) -> list[str]:
        """Get a list of function names available from the object."""
        ...

    def exec(self, func_name: str, args: Sequence) -> tuple:
        """Execute a function given a list of arguments."""
        ...


class SimulatorProtocol(Protocol):
    """Protocol for quantum simulators.

    This protocol defines the interface that all simulator implementations must follow.
    For convenience, simulators can inherit from DefaultSimulator which provides
    default implementations of these methods, or they can implement this protocol
    directly for custom behavior.
    """

    bindings: dict

    def run_gate(
        self,
        symbol: str,
        locations: set[int] | set[tuple[int, ...]],
        **params: dict,
    ) -> dict[int | tuple[int, ...], Any]:
        """Execute a quantum gate on specified locations.

        Args:
            symbol: Gate symbol/name to execute.
            locations: Set of qubit locations to apply the gate to.
            **params: Additional gate parameters.

        Returns:
            Dictionary mapping locations to results.
        """
        ...

    def run_circuit(
        self,
        circuit: QuantumCircuit,
        removed_locations: set | None = None,
    ) -> dict[int | tuple[int, ...], Any]:
        """Execute a quantum circuit.

        Args:
            circuit: Quantum circuit to execute.
            removed_locations: Optional set of locations to exclude.

        Returns:
            Dictionary mapping locations to execution results.
        """
        ...

    def run_circuit_with_errors(
        self,
        circuit: QuantumCircuit,
        error_gen: ParentErrorModel,
        error_params: dict,
    ) -> tuple[dict, OutputDict]:
        """Execute a quantum circuit with error modeling.

        Args:
            circuit: Quantum circuit to execute.
            error_gen: Error model to apply during execution.
            error_params: Parameters for the error model.

        Returns:
            Tuple of execution results and output dictionary.
        """
        ...


class ErrorGenerator(Protocol):
    """Protocol for error generators/models."""

    error_params: ErrorParams

    def start(
        self,
        circuit: QuantumCircuit,
        error_params: ErrorParams,
    ) -> ErrorCircuits:
        """Initialize error generation for a circuit.

        Args:
            circuit: Quantum circuit to generate errors for.
            error_params: Parameters controlling error generation.

        Returns:
            Error circuits object containing generated errors.
        """
        ...

    def generate_tick_errors(
        self,
        time: int,
        gate_time: dict[str, set[int]],
    ) -> QuantumCircuit:
        """Generate errors for a specific time tick.

        Args:
            time: Current time step.
            gate_time: Dictionary mapping gate types to qubit sets.

        Returns:
            Quantum circuit containing the generated errors.
        """
        ...


@runtime_checkable
class LogicalGateProtocol(Protocol):
    """Protocol for logical gate implementations.

    This protocol replaces the LogicalGate parent class and defines the interface
    that all logical gate implementations must follow.
    """

    symbol: str
    qecc: Any  # Reference to the QECC instance
    gate_params: QECCGateParams
    params: QECCGateParams
    instr_symbols: list[str] | None
    instr_instances: list[Any]
    circuits: list[Any]
    error_free: bool
    forced_outcome: bool
    qecc_params_tuple: tuple

    def __eq__(self, other: object) -> bool:
        """Check equality based on protocol implementation."""
        ...

    def __hash__(self) -> int:
        """Return hash value based on protocol implementation."""
        ...


@runtime_checkable
class LogicalInstructionProtocol(Protocol):
    """Protocol for logical instruction implementations.

    This protocol replaces the LogicalInstruction parent class and defines the interface
    that all logical instruction implementations must follow.
    """

    symbol: str
    qecc: Any  # Reference to the QECC instance
    params: QECCInstrParams
    gate_params: QECCInstrParams
    abstract_circuit: Any | None
    circuit: Any | None
    data_qudit_set: set[int]
    ancilla_qudit_set: set[int]
    qudit_set: set[int]
    output_set: set[int]
    gate_params_tuple: tuple

    def items(self) -> Iterator[tuple[str, LocationSet, JSONDict]]:
        """Iterate over instruction items.

        Returns:
            Iterator of tuples containing symbol, locations, and parameters.
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Check equality based on protocol implementation."""
        ...

    def __hash__(self) -> int:
        """Return hash value based on protocol implementation."""
        ...

    def plot(self, figsize: tuple[int, int] | None = None) -> None:
        """Plot the logical instruction.

        Args:
            figsize: Optional figure size as (width, height) tuple.
        """
        ...


@runtime_checkable
class QECCProtocol(Protocol):
    """Protocol for Quantum Error Correcting Codes.

    This protocol defines the interface that all QECC implementations must follow.
    For convenience, QECCs can inherit from DefaultQECC which provides default
    implementations of common methods, or they can implement this protocol
    directly for custom behavior.
    """

    # Required attributes
    name: str | None
    qecc_params: dict
    distance: int | None
    num_data_qudits: int | None
    num_ancilla_qudits: int | None
    num_logical_qudits: int | None
    num_qudits: int

    # Qudit management
    qudit_set: set[int]
    data_qudit_set: set[int]
    ancilla_qudit_set: set[int]

    # Layout and geometry
    layout: dict
    position2qudit: dict
    lattice_dimensions: dict
    sides: dict

    # Gate and instruction management
    sym2gate_class: dict
    sym2instruction_class: dict
    instr_set: set
    gate_set: set

    # Circuit compilation
    circuit_compiler: Any
    mapping: Any

    def gate(
        self,
        symbol: str,
        **gate_params: QECCGateParams,
    ) -> LogicalGateProtocol:
        """Create a logical gate instance.

        Args:
            symbol: Gate symbol/name.
            **gate_params: Gate parameters.

        Returns:
            Logical gate instance implementing the specified gate.
        """
        ...

    def instruction(
        self,
        symbol: str,
        **instr_params: QECCInstrParams,
    ) -> LogicalInstructionProtocol:
        """Create a logical instruction instance.

        Args:
            symbol: Instruction symbol/name.
            **instr_params: Instruction parameters.

        Returns:
            Logical instruction instance implementing the specified instruction.
        """
        ...

    def distance(self, *args: int, **kwargs: int) -> int:
        """Calculate the distance of the quantum error correcting code.

        Args:
            *args: Positional arguments for distance calculation.
            **kwargs: Keyword arguments for distance calculation.

        Returns:
            Distance of the QECC.
        """
        ...

    def plot(self, figsize: tuple[int, int] | None = None) -> None:
        """Plot the quantum error correcting code layout.

        Args:
            figsize: Optional figure size as (width, height) tuple.
        """
        ...


class GuppyCallable(Protocol):
    """Protocol for Guppy-decorated functions."""

    def compile(self) -> dict:
        """Compile the Guppy function to HUGR."""
        ...


class QuantumBackend(Protocol):
    """Protocol for quantum simulator backends with gate operations."""

    def run_1q_gate(self, gate: str, qubit: int, params: dict[str, Any] | None) -> None:
        """Run a single-qubit gate."""
        ...

    def run_2q_gate(
        self,
        gate: str,
        qubits: tuple[int, int],
        params: dict[str, Any] | None,
    ) -> None:
        """Run a two-qubit gate."""
        ...

    def sy_gate(self, qubit: int) -> None:
        """Apply SY gate."""
        ...

    def sydg_gate(self, qubit: int) -> None:
        """Apply SY dagger gate."""
        ...

    def sx_gate(self, qubit: int) -> None:
        """Apply SX gate."""
        ...

    def sxdg_gate(self, qubit: int) -> None:
        """Apply SX dagger gate."""
        ...


__all__ = [
    "CCOPProtocol",
    "CircuitInspector",
    "CircuitRunner",
    "ClassicalInterpreterProtocol",
    "Decoder",
    "EngineRunner",
    "ErrorGenerator",
    "ErrorModelProtocol",
    "ForeignObjectProtocol",
    "GuppyCallable",
    "LogicalGateProtocol",
    "LogicalInstructionProtocol",
    "MachineProtocol",
    "OpProcessorProtocol",
    "QECCProtocol",
    "QuantumBackend",
    "SimulatorProtocol",
    "SimulatorState",
]
