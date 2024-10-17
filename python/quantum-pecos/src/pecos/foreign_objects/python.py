# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from pecos.foreign_objects.foreign_object_abc import ForeignObject

if TYPE_CHECKING:
    from collections.abc import Sequence


class PythonObj(ForeignObject):
    """A Python object with an interface consistent with "foreign objects."."""

    def get_funcs(self) -> list[str]:
        return [attr for attr in dir(self) if inspect.ismethod(getattr(self, attr))]

    def exec(self, func_name: str, args: Sequence) -> tuple:
        return getattr(self, func_name)(*args)
