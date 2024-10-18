# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


import numpy as np
from numpy.typing import ArrayLike

from pecos.simulators.basic_sv import bindings
from pecos.simulators.sim_class_types import StateVector


class BasicSV(StateVector):
    """
    Basic state vector simulator using NumPy.

    Notes:
        The purpose of this simulator is to provide a minimum requirement approach
        to test other simulators against. Not suitable for users looking for performance.
        Maximum number of qubits is restricted to 10.
    """

    def __init__(self, num_qubits, seed=None) -> None:
        """
        Initializes the state vector.

        Args:
            num_qubits (int): Number of qubits being represented.
            seed (int): Seed for randomness.

        Raises:
            ValueError: If `num_qubits` is larger than 10.
        """

        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)
        if num_qubits > 10:
            msg = "`num_qubits` cannot be larger than 10."
            raise ValueError(msg)

        super().__init__()
        np.random.seed(seed)

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        self.internal_vector = None
        self.reset()

    def subscript_string(self, qubits: tuple[int], labels: tuple[chr]):
        """Returns a string of subscripts to use with `np.einsum`.

        The string returned identifies each of the qubits (ndarray axes) in
        `self.internal_vector` with a character. Each of the elements in `qubits` is
        assigned the specified character from the `labels` list.

        Args:
            qubits (tuple[int]): The list of qubits with special labels.
            labels (tuple[chr]): The special labels given to the qubits.

        Returns:
            A string of subscripts for `self.internal_vector` to use with `np.einsum`.

        Raises:
            ValueError: If an element in `qubits` is larger than self.num_qubits.
            ValueError: If a label in `labels` would collide with another label.
            ValueError: If the length of `qubits` and `labels` do not match.
        """

        if any(q >= self.num_qubits or q < 0 for q in qubits):
            msg = "Qubit out of range."
            raise ValueError(msg)
        if len(qubits) != len(labels):
            msg = f"Different number of entries in qubits {qubits} and labels {labels}"
            raise ValueError(msg)

        # Each axis in the ndarray corresponds to a qubit. Name each of them with
        # a unique lowercase letter, starting from chr(97)=="a".
        qubit_ids = [chr(97 + q) for q in range(self.num_qubits)]

        # Since the maximum number of qubits is set to 10 and chr(107)=="k", any
        # letter "larger" than "k" is a valid element of `labels`.
        if any(lbl in qubit_ids for lbl in labels) or len(set(labels)) < len(labels):
            msg = "Label already in use, choose another label."
            raise ValueError(msg)

        # Rename the qubits with special labels
        for q, lbl in zip(qubits, labels):
            qubit_ids[q] = lbl

        # Concatenate characters into a string and return
        return "".join(qubit_ids)

    def reset(self):
        """Reset the quantum state for another run without reinitializing."""
        # Initialize state vector to |0>
        self.internal_vector = np.zeros(shape=2**self.num_qubits)
        self.internal_vector[0] = 1
        # Internally use a ndarray representation so that it's easier to apply gates
        # without needing to apply tensor products.
        self.internal_vector = np.reshape(
            self.internal_vector,
            newshape=[2] * self.num_qubits,
        )
        return self

    @property
    def vector(self) -> ArrayLike:
        return np.reshape(self.internal_vector, newshape=2**self.num_qubits)
