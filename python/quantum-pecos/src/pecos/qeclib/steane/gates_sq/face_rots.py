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

from pecos.qeclib import qubit
from pecos.slr import Block, QReg


class F(Block):
    """
    Face rotation on the octahedron.

    X -> Y -> Z -> X

    X -> Y
    Z -> X

    Y -> Z
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            qubit.F4dg(q),
        )


class Fdg(Block):
    """
    The adjoint of the Face rotation on the octahedron.

    X -> Z -> Y -> X

    X -> Z
    Z -> Y

    Y -> X
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            qubit.F4dg(q),
        )
