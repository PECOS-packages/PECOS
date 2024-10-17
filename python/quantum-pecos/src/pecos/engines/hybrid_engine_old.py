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

import os
import random
import struct

import numpy as np

from pecos.engines.cvm.binarray2 import BinArray2 as BinArray
from pecos.engines.cvm.classical import eval_condition, eval_cop, set_output
from pecos.engines.cvm.wasm import eval_cfunc, get_ccop
from pecos.error_models.fake_error_model import FakeErrorModel
from pecos.errors import NotSupportedGateError


class HybridEngine:
    """This class represents a standard model for running quantum circuits and adding in errors."""

    def __init__(self, seed=None, debug=False, regwidth: int = 32) -> None:
        """

        Args:
            seed:
            debug:
            regwidth:
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
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.ccop = None

        self.generate_errors = None

    def run(
        self,
        state,
        circuit,
        error_gen=None,
        error_params=None,
        error_circuits=None,
        output=None,
        output_spec=None,
        circ_inspector=None,
    ):
        output = set_output(state, circuit, output_spec, output)
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
        state,
        output,
        output_export,
        circuit,
        error_gen,
        removed_locations=None,
    ):
        """Args:

            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.
            removed_locations:

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
                        eval_cfunc(self, params, output)

                    elif params.get("expr"):
                        eval_cop(params.get("expr"), output, width=self.regwidth)

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
                        print("received:", symbol, locations, params)
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
    def run_gate(state, output, symbol: str, locations, **params):
        """

        Args:
            state:
            output:
            symbol:
            locations:
            **params:

        Returns:

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
                    else:
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
