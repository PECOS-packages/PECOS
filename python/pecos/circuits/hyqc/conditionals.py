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

from pecos.circuits.hyqc.fund import Block, Statement


class CondStmt(Statement):
    def __init__(self, symbol) -> None:
        self.symbol = symbol
        super().__init__()
        self.cond = None
        self.block = Block()

    def __call__(self, *stmts):
        self.block(*stmts)
        p = self.__class__(self.symbol)
        p.block = self.block
        p.cond = self.cond

        return p

    def __getitem__(self, *stmts):
        self.block(*stmts)
        p = self.__class__(self.symbol)
        p.block = self.block
        p.cond = stmts

        return p


While = CondStmt("While")


class CIf(CondStmt):
    def __init__(self) -> None:
        super().__init__("If")
        self.else_block = None

    def Else(self, *stmts):
        self.else_block = Block(*stmts)


If = CIf()
