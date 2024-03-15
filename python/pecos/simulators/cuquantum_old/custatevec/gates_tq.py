# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from ..cuconn import cq


def CX(state, location, **params):
    qc, qt = location
    g = cq.PauliX()
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [qc], [qt], False)
    g.free_on_device()


def SqrtZZ(state, location, **params):
    g = cq.SqrtZZ()
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], location, False)
    g.free_on_device()


def RZZ(state, location, **params):
    angle = params["angle"]
    g = cq.RZZ(angle)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], location, False)
    g.free_on_device()
