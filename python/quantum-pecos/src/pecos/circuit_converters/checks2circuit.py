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

"""Check Circuits
==============.

This namespace is for callables that take checks and convert them to physical quantum-circuits.

Callables that can be used for general QECCs should be stored here. If a circuit implementation is specific to the QECC
it may be stored in the QECC's folder.

"""

from pecos.circuits import QuantumCircuit


class Check2Circuits:
    """Converts checks to circuits."""

    def __init__(self) -> None:
        self.name = "Check2Circuits"

    @staticmethod
    def get_num_ancillas(num_checks):
        """Args:
        ----
            num_checks:

        Returns:
        -------

        """
        return num_checks

    def compile(self, instr, abstract_circuit, mapping=None):
        """Converts abstract circuits that have checks into an instance of ``QuantumCircuit``.

        Args:
        ----
            abstract_circuit: Abstract circuit that contains checks.
            mapping (None):

        Returns:
        -------
            ``QuantumCircuit``
        """
        make_ticks = self._check_ticks(abstract_circuit)

        gate_params = instr.gate_params

        if mapping is None:
            mapping = gate_params.get("mapping", NoMap())

        forced_outcome = gate_params.get("forced_outcome", None)

        if forced_outcome is None:
            forced_outcome = abstract_circuit.metadata.get("forced_outcome", True)

        if make_ticks:
            largest_tick = 0

            if make_ticks["max_xdatas"]:
                # init H [data ticks] H meas
                largest_tick = make_ticks["max_xdatas"] + 3

            if make_ticks["max_zdatas"]:
                # init [data ticks] meas
                temp = make_ticks["max_zdatas"] + 1
                largest_tick = max(temp, largest_tick)

            if not make_ticks["max_xdatas"] and not make_ticks["max_zdatas"]:
                msg = "Something very weird happened!"
                raise Exception(msg)

        else:
            largest_tick = 0
            # find maximum tick
            for gate_symbol, _locations, params in abstract_circuit.items():
                if gate_symbol in {"X check", "Z check"}:
                    ancilla_ticks = params[
                        "ancilla_ticks"
                    ]  # Initialization of the ancilla.
                    data_ticks = params["data_ticks"]
                    meas_ticks = params["meas_ticks"]

                    for ticks in [ancilla_ticks, data_ticks, meas_ticks]:
                        tick_list = ticks
                        if isinstance(tick_list, int):
                            tick_list = [ticks]

                        for t in tick_list:
                            largest_tick = max(t, largest_tick)

                else:
                    tick = params["tick"]

                    largest_tick = max(tick, largest_tick)

        # A quantum circuit with number of ticks == ``largest_tick``
        circuit = QuantumCircuit(largest_tick + 1, **gate_params)

        # Add circuits
        # ============
        for gate_symbol, locations, params in abstract_circuit.items():
            if gate_symbol in {"X check", "Z check"}:
                datas = params["datas"]
                ancillas = params["ancillas"]
                if not isinstance(ancillas, int):
                    if len(ancillas) == 1:
                        ancillas = ancillas[0]
                    else:
                        msg = "This circuit compiler only accepts single ancillas per check!"
                        raise Exception(msg)

                if make_ticks:
                    # Come up with ticks if none have been specified.
                    ancilla_ticks, data_ticks, meas_ticks = self.generate_ticks(
                        make_ticks,
                        gate_symbol,
                        locations,
                        params,
                    )

                else:
                    ancilla_ticks = params["ancilla_ticks"]
                    data_ticks = params["data_ticks"]
                    meas_ticks = params["meas_ticks"]

                # Add ancilla init
                # ----------------
                if isinstance(ancilla_ticks, int):
                    circuit.update(
                        {"init |0>": {mapping[ancillas]}},
                        tick=ancilla_ticks,
                    )

                    if gate_symbol == "X check":
                        circuit.update(
                            {"H": {mapping[ancillas]}},
                            tick=ancilla_ticks + 1,
                        )
                        circuit.update({"H": {mapping[ancillas]}}, tick=meas_ticks - 1)
                else:
                    msg = "Can not currently handle multiple ancilla checks!"
                    raise Exception(msg)

                # Add data
                # --------
                if hasattr(data_ticks, "__iter__"):
                    if gate_symbol == "X check":
                        for i, t in enumerate(data_ticks):
                            circuit.update(
                                {"CNOT": {(mapping[ancillas], mapping[datas[i]])}},
                                tick=t,
                            )
                    elif gate_symbol == "Z check":
                        for i, t in enumerate(data_ticks):
                            circuit.update(
                                {"CNOT": {(mapping[datas[i]], mapping[ancillas])}},
                                tick=t,
                            )
                else:
                    msg = "Can not currently handle single data checks!"
                    raise Exception(msg)

                # Add ancilla measurements
                # ------------------------
                if isinstance(meas_ticks, int):
                    if forced_outcome:
                        circuit.update(
                            {"measure Z": {mapping[ancillas]}},
                            tick=meas_ticks,
                        )
                    else:
                        circuit.update(
                            {"measure Z": {mapping[ancillas]}},
                            tick=meas_ticks,
                            forced_outcome=0,
                        )
                else:
                    msg = "Can not currently handle multiple ancilla checks!"
                    raise Exception(msg)

            else:
                tick = params["tick"]
                circuit.update(
                    {gate_symbol: self.mapset(mapping, set(locations))},
                    tick=tick,
                )

        return circuit  # Return QuantumCircuit and number of ancillas used in this circuit.

    @staticmethod
    def mapset(mapping, oldset):
        """Applies a mapping to a set.

        Args:
        ----
            mapping:
            oldset (set):

        Returns:
        -------

        """
        newset = set()

        for e in oldset:
            newset.add(mapping[e])

        return newset

    @staticmethod
    def _check_ticks(abstract_circuit):
        # Determine if any X checks or Z check
        # Determine if ancilla_ticks, data_ticks, or meas_ticks ever set

        has_ticks_set = False
        max_xdatas = 0
        max_zdatas = 0

        for gate_symbol, _, params in abstract_circuit.items():
            if gate_symbol in {"X check", "Z check"}:
                ancilla_ticks = params.get("ancilla_ticks")
                data_ticks = params.get("data_ticks")
                meas_ticks = params.get("meas_ticks")
                datas = params["datas"]

                if ancilla_ticks or data_ticks or meas_ticks:
                    has_ticks_set = True
                    break

                if gate_symbol == "X check":
                    max_xdatas = max(len(datas), max_xdatas)

                elif len(datas) > max_zdatas:
                    max_zdatas = len(datas)

        if has_ticks_set:
            return None

        # ticks init |0>, [H], cnots, [H], meas Z
        return {"max_xdatas": max_xdatas, "max_zdatas": max_zdatas}

    @staticmethod
    def generate_ticks(make_ticks_data, gate_symbol, locations, params):
        # X check: init   H    [all data ticks begin] <- H meas [slide to the left]
        # Z check: [idle] init [all data ticks begin] <- meas [slide to the left]

        max_xdatas = make_ticks_data["max_xdatas"]

        if gate_symbol == "X check":
            ancilla_ticks = 0
            data_ticks = [i + 1 for i in range(len(params["datas"]))]
            meas_ticks = data_ticks[-1] + 2  # One extra to include the H
        elif max_xdatas:  # If there are X checks
            ancilla_ticks = 1
            data_ticks = [i + 2 for i in range(len(params["datas"]))]
            meas_ticks = data_ticks[-1] + 1
        else:
            ancilla_ticks = 0
            data_ticks = [i + 1 for i in range(len(params["datas"]))]
            meas_ticks = data_ticks[-1] + 1
        return ancilla_ticks, data_ticks, meas_ticks


class NoMap:
    """Default Mapping: item -> item."""

    def __init__(self) -> None:
        pass

    def __getitem__(self, item):
        return item
