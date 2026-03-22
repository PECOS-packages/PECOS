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

"""Hybrid quantum-classical engine for PECOS.

This module provides the main hybrid engine implementation for executing
quantum-classical algorithms with integrated classical computation support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Union

import pecos as pc
from pecos.classical_interpreters.phir_classical_interpreter import (
    PhirClassicalInterpreter,
)
from pecos.engines import hybrid_engine_multiprocessing
from pecos.machines.generic_machine import GenericMachine
from pecos.noise.error_model import NoErrorModel
from pecos.op_processors.generic_op_processor import GenericOpProc
from pecos.simulators.quantum_simulator import QuantumSimulator

if TYPE_CHECKING:
    from pecos.protocols import (
        ClassicalInterpreterProtocol,
        ErrorModelProtocol,
        ForeignObjectProtocol,
        MachineProtocol,
        OpProcessorProtocol,
    )
    from pecos.reps.pyphir import PyPHIR
    from pecos.typing import GateParams


class PhirConvertible(Protocol):
    """Protocol for objects that can be converted to PHIR dictionary format."""

    def to_phir_dict(self) -> dict[str, Any]:
        """Convert to PHIR dictionary format."""
        ...


PHIRProgram = Union[str, dict[str, Any], "PyPHIR", PhirConvertible]


class HybridEngine:
    """A simulation engine which is capable of running noisy hybrid classical/quantum programs.

    Note:
        Parameters of the quantum simulator are provided as extra keyword arguments passed
        down to ``QuantumSimulator`` as the dictionary ``**params``.
    """

    def __init__(
        self,
        cinterp: ClassicalInterpreterProtocol | None = None,
        qsim: QuantumSimulator | str | None = None,
        machine: MachineProtocol | None = None,
        error_model: ErrorModelProtocol | None = None,
        op_processor: OpProcessorProtocol | None = None,
        **params: GateParams,
    ) -> None:
        """Initialize the hybrid engine with simulation components.

        Args:
            cinterp: Classical interpreter for executing classical operations.
                Defaults to PhirClassicalInterpreter if None.
            qsim: Quantum simulator for executing quantum operations. Can be a
                QuantumSimulator instance or a string specifying the simulator type.
                Defaults to QuantumSimulator if None.
            machine: Machine model defining the quantum hardware constraints.
                Defaults to GenericMachine if None.
            error_model: Error model for simulating noise in quantum operations.
                Defaults to NoErrorModel if None.
            op_processor: Operation processor for handling and transforming operations.
                Defaults to GenericOpProc if None.
            **params: Additional parameters passed to the QuantumSimulator constructor.
        """
        self.seed = None

        self.cinterp: ClassicalInterpreterProtocol | None = cinterp
        if self.cinterp is None:
            self.cinterp: ClassicalInterpreterProtocol = PhirClassicalInterpreter()

        self._internal_cinterp: ClassicalInterpreterProtocol = PhirClassicalInterpreter()
        self._internal_cinterp.phir_validate = self.cinterp.phir_validate

        self.qsim: QuantumSimulator | None = qsim
        if self.qsim is None:
            self.qsim: QuantumSimulator = QuantumSimulator()
        elif isinstance(self.qsim, str):
            self.qsim: QuantumSimulator = QuantumSimulator(self.qsim, **params)

        self.machine: GenericMachine = machine
        if machine is None:
            self.machine: GenericMachine = GenericMachine()

        self.error_model: ErrorModelProtocol | None = error_model
        if self.error_model is None:
            self.error_model: ErrorModelProtocol = NoErrorModel()

        self.op_processor: OpProcessorProtocol | None = op_processor
        if self.op_processor is None:
            self.op_processor: OpProcessorProtocol = GenericOpProc()

        if self.machine:
            self.op_processor.attach_machine(self.machine)

        if self.error_model:
            self.op_processor.attach_error_model(self.error_model)

        self.results = {}
        self.multisim_process_info = {}

    def init(self) -> None:
        """Reset the state of `Engine` before a simulation run."""
        self.results = {}
        self.multisim_process_info = {}

    def reset_all(self) -> None:
        """Reset to the state of initialization."""
        self.cinterp.reset()
        self._internal_cinterp.reset()
        self.qsim.reset()
        self.machine.reset()
        self.error_model.reset()
        self.op_processor.reset()
        self.init()

    def initialize_sim_components(
        self,
        program: PHIRProgram,
        foreign_object: ForeignObjectProtocol | None = None,
    ) -> None:
        """Get objects to initialize before potentially running many simulations."""
        self.init()
        if foreign_object is not None:
            foreign_object.init()
        num_qubits = self.cinterp.init(program, foreign_object)
        self._internal_cinterp.init(program, foreign_object)
        self.machine.init(num_qubits)
        self.error_model.init(num_qubits, self.machine)
        self.op_processor.init()
        # Pass seed to quantum simulator if one was set
        if self.seed is not None:
            self.qsim.qsim_params["seed"] = self.seed
        self.qsim.init(num_qubits)

    def shot_reinit_components(self) -> None:
        """Reinitialize components for a new shot.

        Tells components that a new shot is starting and to run any tasks necessary,
        such as resetting their states.
        """
        self.cinterp.shot_reinit()
        self._internal_cinterp.shot_reinit()
        for i in range(self.machine.num_qubits):
            self._internal_cinterp.add_cvar(f"__q{i}__", pc.dtypes.i64, 1)
        self.machine.shot_reinit()
        self.error_model.shot_reinit()
        self.op_processor.shot_reinit()
        self.qsim.shot_reinit()

    def use_seed(self, seed: int | None = None) -> int:
        """Set a seed for reproducible random number generation.

        This method seeds the global random number generator and stores the seed
        on this engine instance. When run() is called without a seed parameter,
        it will use this stored seed instead of generating a random one.

        Args:
            seed: The seed value to use. If None, a random seed is generated.

        Returns:
            The seed value that was used.

        Example:
            >>> engine = HybridEngine()
            >>> engine.use_seed(42)  # Set seed for reproducibility
            42
            >>> results = engine.run(program, shots=100)  # Uses seed 42
        """
        if seed is None:
            # Use i32::MAX from Rust as max seed value
            seed = int(pc.random.randint(0, pc.dtypes.i32.max, 1)[0])
        pc.random.seed(seed)
        self.seed = seed
        return seed

    def results_accumulator(self, shot_results: dict) -> None:
        """Combines the results of individual runs together."""
        for k, v in shot_results.items():
            self.results.setdefault(k, []).append(v)

    def run(
        self,
        program: PHIRProgram,
        foreign_object: ForeignObjectProtocol | None = None,
        *,
        shots: int = 1,
        seed: int | None = None,
        initialize: bool = True,
        return_int: bool = False,
    ) -> dict:
        """Main method to run simulations.

        Args:
        ----
            program: The quantum program to execute.
            foreign_object: Optional foreign object for external function calls.
            shots: Number of times to run the simulation.
            seed: Random seed for reproducibility. If not provided and use_seed()
                was previously called, that seed will be used. If neither is set,
                a random seed is generated.
            initialize: Whether to initialize the quantum state before running.
            return_int: Whether to return measurement results as integers.

        """
        measurements = MeasData()

        if initialize:
            # Use explicit seed if provided, otherwise use previously set seed,
            # otherwise generate a random seed
            if seed is not None:
                self.use_seed(seed)
            elif self.seed is None:
                # No seed was set via use_seed() and none provided - generate random
                self.use_seed(None)
            else:
                # A seed was previously set via use_seed() - re-seed the RNG with it
                # to ensure reproducibility across multiple run() calls
                pc.random.seed(self.seed)
            self.initialize_sim_components(program, foreign_object)

        for _ in range(shots):
            self.shot_reinit_components()

            # Execute the classical program till quantum sim is needed
            for buffered_ops in self.cinterp.execute(self.cinterp.program.ops):
                # Process ops, e.g., use `machine` and `error_model` to generate noisy qops & cops
                noisy_buffered_ops = self.op_processor.process(buffered_ops)

                # Process noisy operations
                measurements.clear()
                for noisy_qops in self._internal_cinterp.execute(noisy_buffered_ops):
                    temp_meas = self.qsim.run(noisy_qops)
                    self._internal_cinterp.receive_results(temp_meas)
                    measurements.extend(temp_meas)
                transmit_meas = self._internal_cinterp.result_bits(measurements)
                self.cinterp.receive_results([transmit_meas])

            self.results_accumulator(self.cinterp.results(return_int=return_int))

        return self.results

    def run_multisim(
        self,
        program: PHIRProgram,
        foreign_object: ForeignObjectProtocol | None = None,
        shots: int = 1,
        seed: int | None = None,
        pool_size: int = 1,
    ) -> dict:
        """Parallelized running of the sim."""
        return hybrid_engine_multiprocessing.run_multisim(
            self,
            program=program,
            foreign_object=foreign_object,
            shots=shots,
            seed=seed,
            pool_size=pool_size,
        )


class MeasData(list):
    """Class representing a collection of measurements."""
