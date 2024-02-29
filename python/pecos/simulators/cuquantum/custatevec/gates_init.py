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

import numpy as np

from .gates_sq import X


def init_zero(state, location, **params):
    result = state.statevec.batch_measure(state.workspace, [location], np.random.uniform(), True)

    if result == [1]:
        X(state, location)


def init_one(state, location, **params):
    result = state.statevec.batch_measure(state.workspace, [location], np.random.uniform(), True)

    if result == [0]:
        X(state, location)
