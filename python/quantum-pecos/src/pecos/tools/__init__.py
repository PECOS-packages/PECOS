"""Deprecated: use pecos.analysis instead."""

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

import warnings

warnings.warn(
    "pecos.tools has been renamed to pecos.analysis. "
    "Please update your imports. pecos.tools will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from pecos.analysis import *  # noqa: E402, F403
from pecos.analysis import __all__  # noqa: E402
