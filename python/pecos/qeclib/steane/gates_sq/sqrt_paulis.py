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
from pecos.slr import Block, Comment, QReg


class SX(Block):
    """
    Square root of X.

    X -> X
    Z -> -Y

    Y -> Z
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SX"),
            qubit.SXdg(q),
        )


class SXdg(Block):
    """
    Hermitian adjoint of the square root of X.

    X -> X
    Z -> Y

    Y -> -Z
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SXdg"),
            qubit.SX(q),
        )


class SY(Block):
    """
    Square root of Y.

    X -> -Z
    Z -> X

    Y -> Y
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SY"),
            qubit.SY(q),
        )


class SYdg(Block):
    """
    Square root of X.

    X -> Z
    Z -> -X

    Y -> Y
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SYdg"),
            qubit.SYdg(q),
        )


class SZ(Block):
    """
    Square root of Z. Also known as the S gate.
    diag(1, i)

    X -> Y
    Z -> Z

    Y -> -X
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SZ"),
            qubit.SZdg(q),
        )


class SZdg(Block):
    """
    Hermitian adjoint of the square root of Z. Also known as the Sdg gate.
    diag(1, -i)

    X -> -Y
    Z -> Z

    Y -> X
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SZdg"),
            qubit.SZ(q),
        )
