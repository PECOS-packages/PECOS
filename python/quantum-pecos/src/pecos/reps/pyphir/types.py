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

"""Consolidated type imports for PyPHIR intermediate representation.

This module provides convenient imports for all PyPHIR (Python PECOS Medium-level Intermediate Representation) types
including blocks, data types, instructions, and operations.
"""

# ruff: noqa: F401

from pecos.reps.pyphir import block_types as block
from pecos.reps.pyphir import data_types as data
from pecos.reps.pyphir import instr_type as instr
from pecos.reps.pyphir import op_types as opt
