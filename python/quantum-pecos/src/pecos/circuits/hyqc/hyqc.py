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

"""A Pythonic representation of hybrid quantum-classical program."""

from pecos.circuits.hyqc import qops
from pecos.circuits.hyqc.fund import Block


class HyQC(Block):
    """Represents a full program for HyQC, a high-level ''language'' for hybrid quantum-classical computing."""

    def __init__(self, *stmts) -> None:
        super().__init__(*stmts)

        self.qops = qops
        self.ast = None

    def get_ast(self):
        """Get the abstract syntax tree for the program."""
        # TODO: get_ast

    def parse(self, txt: str):
        """Parse a text representing HYQC."""
        # TODO: parse
