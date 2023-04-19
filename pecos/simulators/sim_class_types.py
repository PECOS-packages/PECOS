# -*- coding: utf-8 -*-

#  =========================================================================  #
#   Copyright 2019 Ciarán Ryan-Anderson
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from ._parent_sim_classes import Simulator


class PauliPropagation(Simulator):
    """
    Base class for Pauli-propagation simulators.
    """

    def __init__(self):
        super(PauliPropagation, self).__init__()


class Stabilizer(Simulator):
    """
    Base class for stabilizer simulators.
    """

    def __init__(self):
        super(Stabilizer, self).__init__()


class StateVector(Simulator):
    """
    Base class for state-vector simulators.
    """

    def __init__(self):
        super(StateVector, self).__init__()


class DensityMatrix(Simulator):
    """
    Base class for density-matrix simulators.
    """

    def __init__(self):
        super(DensityMatrix, self).__init__()


class ProcessMatrix(Simulator):
    """
    Base class for process-matrix simulators.
    """

    def __init__(self):
        super(ProcessMatrix, self).__init__()
