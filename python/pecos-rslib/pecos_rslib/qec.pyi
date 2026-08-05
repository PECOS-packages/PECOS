# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.

"""Typed surface for the dynamically registered ``pecos_rslib.qec`` module."""

from typing import Any

class FaultDistanceResult:
    """A fault distance and one witnessing set of DEM mechanism indices."""

    @property
    def distance(self) -> int: ...
    @property
    def mechanism_indices(self) -> list[int]: ...
    def __repr__(self) -> str: ...

class DetectorErrorModel:
    """Rust-backed detector error model."""

    def graphlike_fault_distance(self) -> FaultDistanceResult | None: ...
    def exhaustive_fault_distance(self, max_weight: int) -> FaultDistanceResult | None: ...
    def __getattr__(self, name: str) -> Any: ...

# The native QEC module predates this focused stub. Preserve the untyped behavior of its other
# classes and functions until that complete API is migrated rather than falsely narrowing them.
def __getattr__(name: str) -> Any: ...
