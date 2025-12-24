# Copyright 2025 The PECOS Developers
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

"""PECOS Qulacs Selene plugin.

This plugin provides a Selene-compatible interface to the Qulacs quantum
circuit simulator through the PECOS wrapper.

Qulacs is developed by the Qulacs team and is available at:
https://github.com/qulacs/qulacs

Qulacs is licensed under the MIT License.
"""

from pecos_selene_qulacs.plugin import QulacsPlugin, SimulatorMode

__all__ = ["QulacsPlugin", "SimulatorMode"]
