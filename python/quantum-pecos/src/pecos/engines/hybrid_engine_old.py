# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Legacy hybrid engine implementation for PECOS.

This module provides the legacy hybrid engine implementation maintained
for backward compatibility with existing quantum-classical workflows.
"""

from __future__ import annotations

import os
import random
import struct
from typing import TYPE_CHECKING

import numpy as np

from pecos.engines.cvm.binarray import BinArray
from pecos.engines.cvm.classical import eval_condition, eval_cop, set_output
from pecos.engines.cvm.rng_model import RNGModel
from pecos.engines.cvm.wasm import eval_cfunc, get_ccop
from pecos.error_models.fake_error_model import FakeErrorModel
from pecos.errors import NotSupportedGateError

if TYPE_CHECKING:
    from typing import Protocol

    from pecos.circuits import QuantumCircuit
    from pecos.error_models.parent_class_error_gen import ParentErrorModel
    from pecos.protocols import SimulatorProtocol
    from pecos.typing import GateParams

    class CircuitInspector(Protocol):
        """Protocol for circuit inspector objects."""

        def analyze(
            self,
            tick_circuit: QuantumCircuit,
            time: int,
            output: dict[str, BinArray],
        ) -> None:
            """Analyze a circuit at a specific time tick.

            Args:
                tick_circuit: Quantum circuit for the current time tick.
                time: Current time step.
                output: Output dictionary with binary arrays.
            """
            ...


class HybridEngine:
    """This class represents a standard model for running quantum circuits and adding in errors."""

    def __init__(
        self,
        seed: int | bool | None = None,
        *,
        debug: bool = False,
        regwidth: int = 32,
    ) -> None:
        """Initialize hybrid engine with seed, debug mode, and register width.

        Args:
        seed: Random seed for reproducibility. Can be bool True for random seed, int for specific seed, or None.
        debug: Enable debug mode for additional output.
        regwidth: Width of classical registers in bits.
        """
        self.debug = debug
        self.state = None
        self.circuit = None
        self.regwidth = regwidth

        if isinstance(seed, bool) and seed is True:
            self.seed = struct.unpack("<L", os.urandom(4))[0]

        elif isinstance(seed, int):
            self.seed = seed

        else:
            self.seed = None

        if self.seed:
            self.rng_model = RNGModel(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            self.rng_model = RNGModel(0)

        self.ccop = None

        self.generate_errors = None

    def run(
        self,
        state: SimulatorProtocol,
        circuit: QuantumCircuit,
        shot_id: int,
        error_gen: ParentErrorModel | None = None,
        error_params: dict[str, float | dict[str, float]] | None = None,
        error_circuits: dict[int, dict[str, QuantumCircuit | set[int]]] | None = None,
        output: dict[str, BinArray] | None = None,
        output_spec: dict[str, int] | None = None,
        circ_inspector: CircuitInspector | None = None,
    ) -> tuple[dict, dict]:
        """Run a quantum circuit with optional error modeling.

        Args:
            state: Quantum simulator state.
            circuit: Quantum circuit to execute.
            shot_id: Integer representing current shot.
            error_gen: Optional error model.
            error_params: Parameters for error generation.
            error_circuits: Pre-generated error circuits.
            output: Output dictionary for results.
            output_spec: Specification for output variables.
            circ_inspector: Optional circuit inspector.

        Returns:
            Tuple of final simulator state and output dictionary.
        """
        output = set_output(state, circuit, output_spec, output)
        output["JOB_shotnum"] = shot_id
        output_export = {}

        self.circuit = circuit
        self.ccop = get_ccop(circuit)

        if error_circuits:
            error_gen = FakeErrorModel(error_circuits)
        elif error_gen is None:
            error_gen = FakeErrorModel({})  # No errors

        # Initialize errors...
        # --------------------
        self.generate_errors = True
        error_circuits = error_gen.start(circuit, error_params)
        self.rng_model.count = 0
        self.rng_model.shot_id = shot_id

        # run through the circuits...
        # ---------------------------
        for tick_circuit, time, params in circuit.iter_ticks():
            # ---------------
            # GENERATE ERRORS
            # ---------------
            if params.get("error_free", False):
                errors = {}
            else:
                error_circuits = error_gen.generate_tick_errors(
                    tick_circuit,
                    time,
                    output,
                    **params,
                )
                errors = error_circuits.get(time, {})

            # TODO: Need run the error generator whether we want errors or not because of leakage
            # TODO: Handle applying leakage without generating new errors
            if self.generate_errors:
                before_errors = errors.get("before")
                after_errors = errors.get("after")
            else:
                before_errors = {}
                after_errors = {}

            removed = errors.get("replaced")

            # --------------------
            # RUN QUANTUM CIRCUITS
            # --------------------

            if before_errors:
                self.run_circuit(state, output, output_export, before_errors, error_gen)

            # ideal tick circuit
            # ------------------
            self.run_circuit(
                state,
                output,
                output_export,
                tick_circuit,
                error_gen,
                removed_locations=removed,
            )

            if circ_inspector:
                circ_inspector.analyze(tick_circuit, time, output)

            if after_errors:
                self.run_circuit(state, output, output_export, after_errors, error_gen)

        if output_export:
            output = output_export

        if self.ccop:
            self.ccop.teardown()  # Tear down WASM execution context

        return output, error_circuits

    def run_circuit(
        self,
        state: SimulatorProtocol,
        output: dict[str, BinArray],
        output_export: dict[str, BinArray],
        circuit: QuantumCircuit,
        error_gen: ParentErrorModel,
        removed_locations: set[int] | None = None,
    ) -> None:
        """Run quantum circuit with error generation and classical operations.

        Args:
            state: Quantum state to operate on.
            output: Output object to store results.
            output_export: Dictionary for exported output values.
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.
            error_gen: Error generator object.
            removed_locations: Set of qubit locations to exclude from operations.

        Returns (list): If output is True then the circuit output is returned. Note that this output format may differ
        from what a ``circuit_runner`` will return for the same method named ``run_circuit``.

        """
        self.state = state
        if removed_locations is None:
            removed_locations = set()

        for symbol, locations, params in circuit.items():
            if params.get("skip"):
                continue

            eval_cond2 = (
                eval_condition(params.get("cond2"), output)
                if params.get("cond2")
                else True
            )

            if eval_condition(params.get("cond"), output) and eval_cond2:
                # Run quantum simulator
                if symbol == "cop":
                    if (
                        params.get("cop_type") == "Idle"
                        or params.get("is_transport")
                        or params.get("syn_cvar")
                        or params.get("cop_type") == "Sleep"
                    ):
                        pass

                    elif params.get("cop_type") == "CFunc":
                        cop_name = params.get("func")
                        if "RNG" in cop_name:
                            self.rng_model.eval_func(params, output)
                        else:
                            eval_cfunc(self, params, output)

                    elif params.get("expr"):
                        eval_cop(
                            params.get("expr"),
                            output,
                            width=self.regwidth,
                            shot_id=self.rng_model.shot_id,
                        )

                    elif params.get("cop_type") == "ExportCVar":
                        sym = params["export"]
                        val = output[sym]

                        if isinstance(val, str):
                            output_export[sym] = val
                        elif isinstance(val, BinArray):
                            output_export[sym] = BinArray(str(val))
                        else:
                            msg = (
                                f"This output type `{type(val)}` not handled at export!"
                            )
                            raise Exception(msg)

                    elif (
                        not params.get("comment")
                        and not params.get("linebreak")
                        and not params.get("barrier")
                    ):
                        msg = "A cop must have an `expr`, `comment`, `linebreak`, or `barrier` entry!"
                        raise Exception(msg)

                elif symbol == "eop":  # special error triggering operation
                    pass

                else:  # quantum operation
                    self.run_gate(
                        state,
                        output,
                        symbol,
                        locations - removed_locations,
                        **params,
                    )

                    if symbol == "leak":
                        error_gen.leaked_qubits |= locations

                    elif symbol == {"unleak |0>", "unleak |1>"}:
                        error_gen.leaked_qubits -= locations

    @staticmethod
    def run_gate(
        state: SimulatorProtocol,
        output: dict[str, BinArray],
        symbol: str,
        locations: set[int],
        **params: GateParams,
    ) -> None:
        """Run a single gate operation on the quantum state.

        Args:
        state: Quantum state to operate on.
        output: Output object to store measurement results.
        symbol: Gate symbol identifying the operation.
        locations: Set of qubit locations to apply gate to.
        **params: Additional parameters for the gate operation.

        """
        if params.get("simulate_gate", True):
            for location in locations:
                if params.get("angles") and len(params["angles"]) == 1:
                    params.update({"angle": params["angles"][0]})
                elif "angle" in params and "angles" not in params:
                    params["angles"] = (params["angle"],)

                try:
                    result = state.bindings[symbol](state, location, **params)
                except KeyError:
                    if symbol not in state.bindings:
                        msg = (
                            f'The gate "{symbol}" is not available for this simulator: {type(state)}. '
                            f"Metadata: {params}"
                        )
                        raise NotSupportedGateError(msg) from KeyError
                    raise

                sym = None
                indx = None
                if params.get("var"):
                    sym, indx = params.get("var")
                elif params.get("var_output"):
                    sym, indx = params.get("var_output")[location]

                if sym:
                    if not result:
                        result = 0
                    output[sym][indx] = result
