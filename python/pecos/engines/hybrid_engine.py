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

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np

from pecos.classical_interpreters.phir_classical_interpreter import (
    PHIRClassicalInterpreter,
)
from pecos.engines import hybrid_engine_multiprocessing
from pecos.error_models.error_model import NoErrorModel
from pecos.machines.generic_machine import GenericMachine
from pecos.op_processors.generic_op_processor import GenericOpProc
from pecos.simulators.quantum_simulator import QuantumSimulator

if TYPE_CHECKING:
    from pecos.classical_interpreters.phir_classical_interpreter import (
        ClassicalInterpreter,
    )
    from pecos.error_models.error_model import ErrorModel
    from pecos.foreign_objects.foreign_object_abc import ForeignObject
    from pecos.machines.generic_machine import Machine
    from pecos.op_processors.generic_op_processor import OpProcessor


class HybridEngine:
    """Engine that runs hybrid quantum/classical programs."""

    def __init__(
        self,
        cinterp: ClassicalInterpreter | None = None,
        qsim: QuantumSimulator | str | None = None,
        machine: Machine | None = None,
        error_model: ErrorModel | None = None,
        op_processor: OpProcessor | None = None,
    ) -> None:
        self.seed = None

        self.cinterp = cinterp
        if self.cinterp is None:
            self.cinterp = PHIRClassicalInterpreter()

        self.qsim = qsim
        if self.qsim is None:
            self.qsim = QuantumSimulator()
        elif isinstance(self.qsim, str):
            self.qsim = QuantumSimulator(self.qsim)

        self.machine = machine
        if machine is None:
            self.machine = GenericMachine()

        self.error_model = error_model
        if self.error_model is None:
            self.error_model = NoErrorModel()

        self.op_processor = op_processor
        if self.op_processor is None:
            self.op_processor = GenericOpProc()

        if self.machine:
            self.op_processor.attach_machine(self.machine)

        if self.error_model:
            self.op_processor.attach_error_model(self.error_model)

        self.results = {}
        self.multisim_process_info = {}

    def init(self):
        """Reset the state of `Engine` before a simulation run."""
        self.results = {}
        self.multisim_process_info = {}

    def reset_all(self):
        """Reset to the state of initialization."""
        self.cinterp.reset()
        self.qsim.reset()
        self.machine.reset()
        self.error_model.reset()
        self.op_processor.reset()
        self.init()

    def initialize_sim_components(
        self,
        program: Any,
        foreign_object: ForeignObject | None = None,
    ) -> None:
        """Get objects to initialize before potentially running many simulations."""
        self.init()
        if foreign_object is not None:
            foreign_object.init()
        num_qubits = self.cinterp.init(program, foreign_object)
        self.machine.init(num_qubits)
        self.error_model.init(num_qubits, self.machine)
        self.op_processor.init()
        self.qsim.init(num_qubits)

    def shot_reinit_components(self) -> None:
        """Tells components that a new shot is starting and to run any tasks necessary, such as resetting their
        states.
        """
        self.cinterp.shot_reinit()
        self.machine.shot_reinit()
        self.error_model.shot_reinit()
        self.op_processor.shot_reinit()
        self.qsim.shot_reinit()

    @staticmethod
    def use_seed(seed=None) -> int:
        """Use a seed to set random number generators."""
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        random.seed(seed)
        return seed

    def results_accumulator(self, shot_results: dict) -> None:
        """Combines the results of individual runs together."""
        for k, v in shot_results.items():
            self.results.setdefault(k, []).append(v)

    def run(
        self,
        program,
        foreign_object: ForeignObject = None,
        *,
        shots: int = 1,
        seed: int | None = None,
        initialize: bool = True,
        return_int=False,
    ) -> dict:
        """Main method to run simulations.

        Args:
        ----
            program:
            foreign_object:
            shots:
            seed:
            initialize:
            return_int:

        Returns:
        -------

        """
        # TODO: Qubit loss

        if initialize:
            self.seed = self.use_seed(seed)
            self.initialize_sim_components(program, foreign_object)

        for _ in range(shots):
            self.shot_reinit_components()

            # Execute classical program till quantum sim is needed
            for buffered_ops in self.cinterp.execute(self.cinterp.program.ops):
                # Process ops, e.g., use `machine` and `error_model` to generate noisy qops
                noisy_buffered_qops = self.op_processor.process(buffered_ops)

                measurements = self.qsim.run(noisy_buffered_qops)

                # Allows noise to be dependent on measurement outcomes and to alter measurements
                measurements = self.op_processor.process_meas(measurements)

                # TODO: Consider adding the following to generate/evaluate errors after measurement
                # measurements, residual_noise = self.op_processor.process_meas(measurements)
                # self.qsim.run(residual_noise)

                self.cinterp.receive_results(measurements)

            self.results_accumulator(self.cinterp.results(return_int))

        return self.results

    def run_multisim(
        self,
        program,
        foreign_object: ForeignObject = None,
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
