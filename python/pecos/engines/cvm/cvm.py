# Copyright 2022 The PECOS Developers
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


class CVM:
    def __init__(
        self,
        ccop_vm=None,
        cinterpreter=None,
        sim_debug=None,
    ) -> None:
        """Classical Virtual Machine, which is responsible for executing classical functions and statements.

        Attributes:
        ----------
            ccop_vm: A VM representing the computing environment of a classical co-processor. This generally provides
                external classical functions that are usually problem specific.
            cinterpreter: Provides an interpreter for generic classical statements. For example, boolean operations,
                assignments, comparisons, etc.
            sim_debug: A collection of functions used in the simulation environment to provide additional information
                that may not typically be available to a physical quantum device.
        """
        self.state = {}

        self.ccop_vm = ccop_vm
        self.cinterpreter = cinterpreter
        self.sim_debug = sim_debug

    def reset_state(self):
        self.state = ()

    def exec(self, func_name, args): ...
