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
    >>> from pecos.analysis.stabilizers import VerifyStabilizers
    >>>
    >>> # Threshold analysis
    >>> result = threshold.code_capacity(
    ...     qecc_class=Surface4444,
    ...     error_gen=XModel,
    ...     decoder_class=MWPM2D,
    ...     ps=[0.01, 0.05, 0.10],
    ...     ds=[3, 5, 7],
    ...     runs=1000,
    ... )
    >>>
    >>> # Fault tolerance verification
    >>> passed, weight = fault_tolerance.t_errors_check(qecc)
"""

# Re-export from tools for backwards compatibility
from pecos.tools import fault_tolerance_checks as fault_tolerance
from pecos.tools import pseudo_threshold_tools as pseudo_threshold
from pecos.tools import threshold_tools as threshold
from pecos.tools.stabilizer_verification import VerifyStabilizers
from pecos.tools.threshold_tools import (
    codecapacity_logical_rate,
    codecapacity_logical_rate2,
    codecapacity_logical_rate3,
    threshold_code_capacity,
)

__all__ = [
    # Classes
    "VerifyStabilizers",
    # Threshold functions
    "codecapacity_logical_rate",
    "codecapacity_logical_rate2",
    "codecapacity_logical_rate3",
    # Submodules
    "fault_tolerance",
    "pseudo_threshold",
    "threshold",
    "threshold_code_capacity",
]
