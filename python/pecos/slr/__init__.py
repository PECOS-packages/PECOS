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

from pecos.slr import std
from pecos.slr.block import Block
from pecos.slr.cond_block import If, Repeat
from pecos.slr.cops import Assign
from pecos.slr.misc import QASM, Barrier, Comment, Permute
from pecos.slr.slr import Main
from pecos.slr.slr import Main as SLR  # noqa: N814
from pecos.slr.vars import Bit, CReg, QReg, Qubit, Vars

__all__ = [
    "Main",
    "SLR",
    "Block",
    "If",
    "Repeat",
    "Assign",
    "QASM",
    "Barrier",
    "Comment",
    "Permute",
    "Vars",
    "Bit",
    "Qubit",
    "CReg",
    "QReg",
    "std",
]
