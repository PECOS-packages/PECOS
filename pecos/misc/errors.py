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


class PECOSTypeError(TypeError):
    """General gate error."""
    pass


class GateError(Exception):
    """General gate errors."""
    pass


class GateOverlapError(GateError):
    """Raised when gates act on qudits that are already being acted on."""
    pass
