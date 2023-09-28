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

from typing import Any, List, Optional, Sequence

from .foreign_object_abc import ForeignObject


class NamedObjectPool(ForeignObject):
    """A collection of objections that can be access via this class."""

    def __init__(self, **objects: ForeignObject):
        self.objs = objects
        self.default = objects["default"] if "default" in objects else None

    def new_instance(self) -> None:
        """Create new instance/internal state."""
        for obj in self.objs.values():
            obj.new_instance()

    def init(self) -> None:
        """Initialize object before running a series of experiments."""
        for obj in self.objs.values():
            obj.init()

    def shot_reinit(self) -> None:
        """Call before each shot to, e.g., reset variables."""

        for obj in self.objs.values():
            if "shot_reinit" in obj.get_funcs():
                obj.exec("shot_reinit", [])

    def add(self, namespace: str, obj: ForeignObject):
        if namespace in self.objs:
            raise Exception(f"Object named '{namespace}' already exists!")
        else:
            self.objs[namespace] = obj

    def get_funcs(self) -> List[str]:
        """Get a list of function names available from the object."""
        return []

    def exec(self, func_name: str, args: Sequence, namespace: Optional[str] = None) -> Any:
        """Execute a function given a list of arguments."""

        if namespace is None:
            obj = self.default
        elif namespace not in self.objs:
            raise Exception(f"Object named '{namespace}' not recognized!")
        else:
            obj = self.objs[namespace]

        return obj.exec(func_name, args)
