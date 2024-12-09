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

import random

from pecos.simulators.cointoss import bindings
from pecos.simulators.parent_sim_classes import Simulator


class CoinToss(Simulator):
    """
    Ignore all quantum operations and toss a coin to decide measurement outcomes.
    Meant for stochastical debugging of the classical branches.
    """

    def __init__(self, num_qubits, prob=0.5, seed=None):
        """
        Initialization is trivial, since there is no state.

        Args:
            num_qubits (int): Number of qubits being represented.
            prob (float): Probability of measurements returning |1>.
                Default value is 0.5.
            seed (int): Seed for randomness.

        Returns:

        """

        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)
        if not (prob >= 0 and prob <= 1):
            msg = "``prob`` should be a real number in [0,1]."
            raise ValueError(msg)
        random.seed(seed)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits
        self.prob = prob

    def reset(self):
        """Reset the quantum state for another run without reinitializing."""
        # Do nothing, this simulator does not keep a state!
        return self
