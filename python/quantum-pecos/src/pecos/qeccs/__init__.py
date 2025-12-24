"""Quantum Error Correcting Codes (QECCs) for PECOS.

This package provides implementations of various quantum error correcting codes
including surface codes, color codes, and their associated logical operations.
"""

# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.qeccs.color_488.color_488 import Color488
from pecos.qeccs.default_logical_gate import DefaultLogicalGate
from pecos.qeccs.default_logical_instruction import DefaultLogicalInstruction

# Parent Classes:
from pecos.qeccs.default_qecc import DefaultQECC
from pecos.qeccs.surface_4444.surface_4444 import Surface4444
from pecos.qeccs.surface_medial_4444.surface_medial_4444 import SurfaceMedial4444

__all__ = [
    "Color488",
    "DefaultLogicalGate",
    "DefaultLogicalInstruction",
    "DefaultQECC",
    "Surface4444",
    "SurfaceMedial4444",
]
