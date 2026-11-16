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

"""Scalar literal classification shared by SLR conversion and legacy emitters."""

from pecos import dtypes


def is_bool_literal(value: object) -> bool:
    """Recognize Python and NumPy-compatible boolean scalars, excluding arrays.

    PECOS dtypes compare with NumPy dtypes without importing NumPy. The dtype's
    scalar class distinguishes a bool scalar from a zero-dimensional array.
    """
    if isinstance(value, bool):
        return True
    dtype = getattr(value, "dtype", None)
    return dtype is not None and dtypes.bool_ == dtype and isinstance(value, dtype.type)
