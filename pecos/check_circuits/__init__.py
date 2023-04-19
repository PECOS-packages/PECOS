#  =========================================================================  #
#   Copyright 2018 Ciarán Ryan-Anderson
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

"""
Check Circuits
==============

This namespace is for callables that take checks and convert them to physical quantum-circuits.

Callables that can be used for general QECCs should be stored here. If a circuit implementation is specific to the QECC
it may be stored in the QECC's folder.

"""

from .checks2circuit import Check2Circuits
