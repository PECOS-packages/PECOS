#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #


class StdOutput(dict):
    """
    Class used to record results of gates (typically, measurements).

    (logical space, logical time) -> time(tick) -> {location: result}
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # self.results = {}

    def record(self, result_dict, logical_time, tick_index):
        """

        Args:
            result_dict:
            logical_coord:
            tick_index:

        Returns:

        """

        # logical_time = (gate_tick_time, instr_index)

        if result_dict:
            # logical_dict = self.setdefault(logical_coord, SortedDict())
            logical_dict = self.setdefault(logical_time, {})

            for value in result_dict.values():
                logical_dict.setdefault(tick_index, {}).update(value)
            # temp_results = self.results.setdefault(time, dict)

    def simplified(self, last=False):
        """
        Gives output in a simplified version. {logical coord=>{set of locations}, ...}


        Outputs the syndromes of the final logical instruction.

        Returns:

        """

        simple = {}
        for logical_time, results in self.items():

            # results=  {tick: {qid: int}} want just qids

            fired = set()
            for _, fired_dict in results.items():
                fired.update(fired_dict.keys())

            simple[logical_time] = fired

        if last and simple:
            # Get the last coordinate
            keys = simple.keys()

            last_id = sorted(keys)[-1]
            simple = simple[last_id]  # just a set of qids

        return simple
