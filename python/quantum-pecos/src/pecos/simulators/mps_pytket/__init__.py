"""Matrix Product State simulator using PyTKET.

This package provides a Matrix Product State simulator using the PyTKET library.
"""

# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.simulators.mps_pytket import bindings
from pecos.simulators.mps_pytket._nvmath_compat import patch_nvmath_cupy_external_stream
from pecos.simulators.mps_pytket.state import MPS

patch_nvmath_cupy_external_stream()
