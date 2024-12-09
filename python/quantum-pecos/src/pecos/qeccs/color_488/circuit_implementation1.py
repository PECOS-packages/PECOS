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

from pecos.circuits import QuantumCircuit


class OneAncillaPerCheck:
    """Class that describes an implementation of the 4.8.8 color code with one ancilla per face."""

    def __init__(
        self,
        square_x_ticks=None,
        square_z_ticks=None,
        octagon_x_ticks=None,
        octagon_z_ticks=None,
    ) -> None:
        """Args:
        ----
            square_x_ticks:
            square_z_ticks:
            octagon_x_ticks:
            octagon_z_ticks:
        """

        if square_x_ticks is None:
            # 8 ticks
            square_x_ticks = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ]  # init, H ticks  # Data ticks  # H, meas ticks

        if square_z_ticks is None:
            # 6 ticks
            square_z_ticks = [12, 13, 14, 15, 16, 17]  # int tick  # meas tick

        if octagon_x_ticks is None:
            # 12 ticks
            octagon_x_ticks = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            ]  # init, H ticks  # Data ticks  # H, meas ticks

        if octagon_z_ticks is None:
            # 10 ticks
            octagon_z_ticks = [
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
            ]  # int tick  # Data ticks  # meas tick

        self.square_x_ticks = square_x_ticks
        self.square_z_ticks = square_z_ticks
        self.octagon_x_ticks = octagon_x_ticks
        self.octagon_z_ticks = octagon_z_ticks

    @staticmethod
    def get_num_ancillas(num_checks):
        """Args:
        ----
            num_checks:

        Returns:
        -------

        """
        return int(num_checks / 2)

    def compile(self, instr, abstract_circuit, mapping=None):
        """Args:
        ----
            abstract_circuit:
            mapping:
            **gate_params:

        Returns:
        -------

        """
        gate_params = instr.gate_params

        if mapping is None:
            mapping = gate_params.get("mapping")

        square_x_ticks = gate_params.get("square_x_ticks", self.square_x_ticks)
        square_z_ticks = gate_params.get("square_z_ticks", self.square_z_ticks)
        octagon_x_ticks = gate_params.get("octagon_x_ticks", self.octagon_x_ticks)
        octagon_z_ticks = gate_params.get("octagon_z_ticks", self.octagon_z_ticks)

        largest_tick = []
        largest_tick.extend(square_x_ticks)
        largest_tick.extend(square_z_ticks)
        largest_tick.extend(octagon_x_ticks)
        largest_tick.extend(octagon_z_ticks)
        largest_tick = max(largest_tick)

        circuit = QuantumCircuit(largest_tick + 1, **gate_params)

        if len(square_x_ticks) != 8:  # data + 4
            msg = "`square_x_ticks` should be of length : init tick, H tick, 4 data ticks, H tick, meas tick"
            raise Exception(msg)

        if len(square_z_ticks) != 6:  # data + 2
            msg = "`square_z_ticks` should be of length : init tick, 4 data ticks, meas tick"
            raise Exception(msg)

        if len(octagon_x_ticks) != 12:  # data + 4
            msg = "`octagon_x_ticks` should be of length : init tick, H tick, 8 data ticks, H tick, meas tick"
            raise Exception(msg)

        if len(octagon_z_ticks) != 10:  # data + 2
            msg = "`octagon_z_ticks` should be of length : init tick, 8 data ticks, meas tick"
            raise Exception(msg)

        for check_type, locations, params in abstract_circuit.items():
            polygon = params.get("polygon")

            if polygon is None:  # This is an actual circuit element
                if mapping:
                    circuit.update(
                        {check_type: self.mapset(mapping, set(locations))},
                        tick=params["tick"],
                    )
                else:
                    circuit.update({check_type: set(locations)}, tick=params["tick"])
            else:
                ancilla = next(iter(locations))
                datas = params["datas"]

                if polygon == "square":
                    ticks = (
                        square_x_ticks if check_type == "X check" else square_z_ticks
                    )

                else:
                    ticks = (
                        octagon_x_ticks if check_type == "X check" else octagon_z_ticks
                    )

                self._create_check(
                    circuit,
                    polygon,
                    ticks,
                    check_type,
                    datas,
                    ancilla,
                    mapping,
                )

        return circuit

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

    def _create_check(
        self,
        circuit,
        polygon,
        ticks,
        check_type,
        datas,
        ancilla,
        mapping,
    ):
        """Args:
        ----
            circuit:
            polygon:
            ticks:
            check_type:
            datas:
            ancilla:
            mapping:

        Returns:
        -------

        """
        if polygon == "square":
            sides = 4

            if len(datas) != sides:
                msg = "Squares must have 4 datas!"
                raise Exception(msg)
        else:
            sides = 8

            if len(datas) != sides:
                msg = "Octagons must have 8 datas!"
                raise Exception(msg)

        if check_type == "X check":
            h1_tick = ticks[1]
            data_ticks = ticks[2 : sides + 2]
            h2_tick = ticks[-2]

            if mapping is None:
                circuit.update({"H": {ancilla}}, tick=h1_tick)
                circuit.update({"H": {ancilla}}, tick=h2_tick)

                # CNOTs...

                for d, t in zip(datas, data_ticks):
                    if d is not None:
                        circuit.update({"CNOT": {(ancilla, d)}}, tick=t)
            else:
                circuit.update({"H": {mapping[ancilla]}}, tick=h1_tick)
                circuit.update({"H": {mapping[ancilla]}}, tick=h2_tick)

                for d, t in zip(datas, data_ticks):
                    if d is not None:
                        circuit.update(
                            {"CNOT": {(mapping[ancilla], mapping[d])}},
                            tick=t,
                        )

        else:  # Z check
            data_ticks = ticks[1 : sides + 1]

            if mapping is None:
                for d, t in zip(datas, data_ticks):
                    if d is not None:
                        circuit.update({"CNOT": {(d, ancilla)}}, tick=t)
            else:
                for d, t in zip(datas, data_ticks):
                    if d is not None:
                        circuit.update(
                            {"CNOT": {(mapping[d], mapping[ancilla])}},
                            tick=t,
                        )

        init_tick = ticks[0]
        meas_tick = ticks[-1]

        if mapping is None:
            circuit.update({"init |0>": {ancilla}}, tick=init_tick)
            circuit.update({"measure Z": {ancilla}}, tick=meas_tick)

        else:
            circuit.update({"init |0>": {mapping[ancilla]}}, tick=init_tick)
            circuit.update({"measure Z": {mapping[ancilla]}}, tick=meas_tick)
