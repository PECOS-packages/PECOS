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

"""Deprecated functions and classes for PECOS.

This module contains deprecated APIs that are maintained for backwards compatibility.
These will be removed in a future version.
"""

from __future__ import annotations

import warnings
from typing import Any

from pecos_rslib import BitInt


def BinArray(*args: Any, **kwargs: Any) -> BitInt:  # noqa: N802, ANN401
    """Deprecated: Use BitInt instead.

    BinArray is a deprecated alias for BitInt. It will be removed in a future version.
    Please update your code to use BitInt directly.
    """
    warnings.warn(
        "BinArray is deprecated and will be removed in a future version. "
        "Please use BitInt instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return BitInt(*args, **kwargs)


__all__ = [
    "BinArray",
]
