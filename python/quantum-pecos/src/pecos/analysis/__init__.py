# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Analysis tools for quantum error correction.

This package provides comprehensive analysis tools for quantum error correction,
including threshold estimation, fault tolerance verification, and stabilizer
code analysis.

Submodules:
    threshold: Threshold estimation and code capacity analysis
    fault_tolerance: Fault tolerance checking and verification
    stabilizers: Stabilizer code verification and distance analysis

Example:
    >>> from pecos.analysis import threshold, fault_tolerance
    >>> from pecos.analysis.stabilizer_verification import VerifyStabilizers
"""

from pecos.analysis import fault_tolerance_checks as fault_tolerance
from pecos.analysis import pseudo_threshold_tools as pseudo_threshold
from pecos.analysis import threshold_tools as threshold
from pecos.analysis.stabilizer_verification import VerifyStabilizers
from pecos.analysis.threshold_curve import threshold_fit
from pecos.analysis.threshold_tools import (
    codecapacity_logical_rate,
    codecapacity_logical_rate2,
    codecapacity_logical_rate3,
    threshold_code_capacity,
)
from pecos.analysis.tool_anticommute import anticommute
from pecos.analysis.tool_collection import fault_tolerance_check

__all__ = [
    "VerifyStabilizers",
    "anticommute",
    "codecapacity_logical_rate",
    "codecapacity_logical_rate2",
    "codecapacity_logical_rate3",
    "fault_tolerance",
    "fault_tolerance_check",
    "pseudo_threshold",
    "threshold",
    "threshold_code_capacity",
    "threshold_fit",
]
