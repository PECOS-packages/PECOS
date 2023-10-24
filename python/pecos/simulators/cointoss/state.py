# TODO: Include license information?

import random
from pecos.simulators.parent_sim_classes import Simulator
from pecos.simulators.cointoss import bindings


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
            prob (float): Probability of measurements returning |0>.
                Default value is 0.5.
            seed (int): Seed for randomness.

        Returns:

        """

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int``.')
        if not (prob >= 0 and prob <= 1):
            raise Exception('``prob`` should be a real number in [0,1].')
        random.seed(seed)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits
        self.prob = prob
