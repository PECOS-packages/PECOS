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

"""Experimental quantum detector processes and an exact coherent repetition model.

These APIs preserve detector-conditioned quantum maps. They are separate from
the conventional independent-Pauli ``DetectorErrorModel`` API and may change.
"""

from pecos.experimental.detector_models.process import (
    ChannelFactor,
    DetectorDef,
    DetectorInstrument,
    DetectorProcess,
    InstrumentFactor,
    KrausMap,
    SystemSpec,
    compile_instrument,
)
from pecos.experimental.detector_models.repetition import (
    RepetitionConfig,
    RepetitionModel,
    base_angles,
    build_model,
    detector_history,
)

__all__ = [
    "ChannelFactor",
    "DetectorDef",
    "DetectorInstrument",
    "DetectorProcess",
    "InstrumentFactor",
    "KrausMap",
    "RepetitionConfig",
    "RepetitionModel",
    "SystemSpec",
    "base_angles",
    "build_model",
    "compile_instrument",
    "detector_history",
]
