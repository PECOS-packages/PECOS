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

from pecos.circuits.hyqc.fund import Statement


class Include(Statement):
    """Text files to be injected into the program."""

    def __init__(self, txt: str) -> None:
        self.txt = txt


class Define(Statement):
    """Pythonic definition of a gate."""

    def __init__(self) -> None:
        # TODO: ....
        ...


class Assign(Statement):
    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right


class CFunc(Statement):
    def __init__(self, symbol) -> None:
        self.symbol = symbol
        self.cargs = None

    def __call__(self, *cargs):
        f = CFunc(self.symbol)
        f.cargs = cargs
        return f
