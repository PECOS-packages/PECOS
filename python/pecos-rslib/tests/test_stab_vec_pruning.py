# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""StabVec pruning thresholds reject caller errors as ValueError."""

import re

import pytest
from pecos_rslib import StabVec


@pytest.mark.parametrize(
    ("threshold", "value"),
    [(1.0, "1"), (-1e-3, "-0.001"), (float("nan"), "NaN"), (float("inf"), "inf")],
)
def test_invalid_pruning_threshold_is_value_error(threshold, value):
    message = f"invalid pruning threshold {value}: " "expected a finite value with 0.0 <= threshold < 1.0"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        StabVec(2, pruning_threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_valid_pruning_threshold_constructs(threshold):
    simulator = StabVec(2, pruning_threshold=threshold)
    assert simulator.num_qubits == 2
    assert simulator.num_terms == 1
