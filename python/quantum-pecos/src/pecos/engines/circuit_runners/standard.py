# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
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

from pecos.misc.std_output import StdOutput


class Standard:
    """This class represents a standard model for running quantum circuits and adding in errors."""

    def __init__(self, seed=None) -> None:
        """Args:
        ----
            seed:
        """
        if isinstance(seed, bool) and seed is True:
            self.seed = struct.unpack("<L", os.urandom(4))[0]

        elif isinstance(seed, int):
            self.seed = seed

        else:
            self.seed = None

        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)

    @staticmethod
    def run(
        state,
        circuit,
        error_gen=None,
        error_params=None,
        error_circuits=None,
        output=None,
    ):
        """Args:
        ----
            state:
            circuit:
            error_gen:
            error_params:
            error_circuits:
            output:

        Returns:
        -------

        """
        if output is None:
            output = StdOutput()

        # TODO: Generate errors before running the ticks

        # TODO: Add maps...

        # Initialize errors...
        # --------------------

        if error_gen is None:  # No errors
            generate_errors = False
            if error_circuits is None:
                error_circuits = {}

        else:  # new errors
            generate_errors = True
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
                if generate_errors:
                    error_circuits = error_gen.generate_tick_errors(
                        tick_circuit,
                        time,
                        **params,
                    )
                errors = error_circuits.get(time, {})

            before_errors = errors.get("before")
            after_errors = errors.get("after")
            removed = errors.get("replaced")

            # --------------------
            # RUN QUANTUM CIRCUITS
            # --------------------

            if before_errors:
                state.run_circuit(before_errors)

            # ideal tick circuit
            # ------------------
            result = state.run_circuit(tick_circuit, removed_locations=removed)
            output.record(result, time)

            if after_errors:
                state.run_circuit(after_errors)

        return output, error_circuits
