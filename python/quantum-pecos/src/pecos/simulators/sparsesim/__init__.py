# Copyright 2018 The PECOS Developers
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

"""A stabilizer simulation that takes advantage of the structure of LDPC codes; more precisely, it utilizes the
row and column-wise sparseness of stabilizer tableau.

.. module:: __init__
   :synopsis: The init file for the package PECOS.

.. moduleauthor:: Ciarán Ryan-Anderson <cryanan@sandia.gov; ciaranra@unm.edu>

------------------------------------------------------------------------------------------------------------------------
Date Created:  12/28/2014
Last modified: 05/23/2017

Revision history

Date        Author  Comment
----------  ------  ----------------------------------------------------------------------------------------------------
12/28/2014  CRA     File created.

05/23/2017  CRA     The stabilizer simulation was separated from the entire QECC tool chain to allow others to more
                    easily utilize the simulator on its own. The __init__ file has been reduced and modified for this
                    purpose.
"""

from pecos.simulators.sparsesim import bindings

# Class that represents the stabilizer state
from pecos.simulators.sparsesim.state import SparseSim
