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

"""Deprecated: use pecos.noise instead."""

import warnings

warnings.warn(
    "pecos.error_models has been renamed to pecos.noise. "
    "Please update your imports. pecos.error_models will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from pecos.noise import *  # noqa: E402, F403
