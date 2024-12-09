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

from pecos.circuits.hyqc.vars import QVar


class Qubits(QVar):
    def __init__(self, size: int, symbol: str | None = None) -> None:
        super().__init__(symbol)
        self.size = size

    def __getitem__(self, item):
        return Qubit(self, item)

    def __str__(self) -> str:
        return self.symbol


class Qubit(Qubits):
    def __init__(self, qubits: Qubits, idx: int, symbol: str | None = None) -> None:
        if symbol is None:
            symbol = f"{qubits.symbol}[{idx}]"

        super().__init__(size=1, symbol=symbol)

        self.qubits = qubits
        self.index = idx

    def __getitem__(self, item):
        msg = f"'{self.__class__.__name__}' object is not subscriptable"
        raise TypeError(msg)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.index} of {self.qubits.symbol}>"
