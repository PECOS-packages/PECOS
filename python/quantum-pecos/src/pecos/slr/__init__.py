# Copyright 2023-2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.slr.block import Block
from pecos.slr.cond_block import If, Repeat
from pecos.slr.loop_block import For, While
from pecos.slr.main import Main
from pecos.slr.main import (
    Main as SLR,
)
from pecos.slr.misc import Barrier, Comment, Parallel, Permute, Return
from pecos.slr.slr_converter import SlrConverter
from pecos.slr.types import Array
from pecos.slr.types import Bit as BitType
from pecos.slr.types import Qubit as QubitType
from pecos.slr.vars import Bit, CReg, QReg, Qubit, Vars

__all__ = [
    "SLR",
    "Array",
    "Barrier",
    "Bit",
    "BitType",
    "Block",
    "CReg",
    "Comment",
    "For",
    "If",
    "Main",
    "Parallel",
    "Permute",
    "QReg",
    "Qubit",
    "QubitType",
    "Repeat",
    "Return",
    "SlrConverter",
    "Vars",
    "While",
]
