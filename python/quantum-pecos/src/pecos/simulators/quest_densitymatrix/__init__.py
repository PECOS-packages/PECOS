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

"""QuEST density matrix simulator for PECOS.

This module provides a quantum density matrix simulator powered by the QuEST quantum simulation library,
enabling efficient simulation of mixed quantum states and noisy quantum circuits.
"""

from pecos.simulators.quest_densitymatrix.state import QuestDensityMatrix

__all__ = ["QuestDensityMatrix"]
