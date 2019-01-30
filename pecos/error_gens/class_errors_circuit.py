#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
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

# from collections import OrderedDict


class ErrorCircuits(dict):
    """
    Used to store error circuits.
    """

    def __init__(self):

        super().__init__()

    def add_circuits(self, time, before_errors=None, after_errors=None, replaced_locations=None):
        """
        Add error circuits and gate locations to ignore (replaced_locations).

        Args:
            time:
            before_errors:
            after_errors:
            replaced_locations:

        Returns:

        """

        error_dict = {}

        if before_errors and len(before_errors) > 0:
            error_dict['before'] = before_errors

        if after_errors and len(after_errors) > 0:
            error_dict['after'] = after_errors

        if replaced_locations and len(replaced_locations) > 0:
            error_dict['replaced'] = replaced_locations

        if error_dict:
            self[time] = error_dict
