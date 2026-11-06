# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Boolean literal recognition excludes arrays and numeric scalars."""

import numpy as np
import pytest
from pecos.slr._literals import is_bool_literal


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, True),
        (np.True_, True),
        (np.bool_(0), True),
        (np.array(True), False),
        (np.array([True]), False),
        (np.int64(1), False),
        (np.uint8(1), False),
        (1, False),
        (0, False),
        (1.0, False),
        (None, False),
    ],
    ids=[
        "true",
        "false",
        "numpy-true",
        "numpy-false",
        "zero-d-array",
        "array",
        "int64",
        "uint8",
        "one",
        "zero",
        "float",
        "none",
    ],
)
def test_is_bool_literal(value, *, expected: bool) -> None:
    assert is_bool_literal(value) is expected
